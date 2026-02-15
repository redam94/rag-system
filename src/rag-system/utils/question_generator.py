"""
Background Question Embedding Generator

Generates hypothetical questions for each document chunk and stores them
as additional embeddings in ChromaDB.  This improves retrieval quality
because user queries (phrased as questions) are more semantically similar
to the generated questions than to the raw chunk text.

The worker:
- Runs in a daemon thread so it doesn't block the main app
- Uses a threading.Event (llm_available) for cooperative pausing: when
  the user needs the LLM for a chat query, the worker waits
- Is resumable: on startup it picks up chunks that still need questions
- Generates 3-5 questions per chunk using a fast LLM model
"""

import asyncio
import re
import threading
import time
from typing import List, Optional

from loguru import logger


class QuestionGenerationWorker:
    """
    Background worker that generates question embeddings for RAG chunks.

    Usage:
        worker = QuestionGenerationWorker(rag, "http://...:11434", "qwen3:8b")
        worker.start()
        # When user needs the LLM:
        worker.pause()
        ... do LLM work ...
        worker.resume()
        # On shutdown:
        worker.stop()
    """

    QUESTION_PROMPT = (
        "Given the following text chunk from a document, generate exactly 3 to 5 "
        "questions that this chunk could answer.  Output ONLY the questions, one per "
        "line, numbered 1-5.  Do not include any other text. /no_think\n\n"
        "Text chunk:\n{chunk_text}"
    )

    # How long to sleep between processing batches (seconds)
    BATCH_SLEEP = 0.5
    # How many chunks to fetch per batch
    BATCH_SIZE = 20
    # How long to sleep when there's nothing to do before re-checking
    IDLE_SLEEP = 30

    def __init__(
        self,
        rag,
        ollama_base_url: str,
        model: str = "qwen3:8b",
        llm_available: Optional[threading.Event] = None,
    ):
        self.rag = rag
        self.ollama_base_url = ollama_base_url
        self.model = model

        # Event: SET means LLM is free, CLEAR means user is using LLM
        self.llm_available = llm_available or threading.Event()
        self.llm_available.set()  # start in "available" state

        # Internal stop signal
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Stats
        self.chunks_processed = 0
        self.questions_generated = 0
        self.errors = 0

    # ── public API ────────────────────────────────────────────────────────

    def start(self):
        """Start the background worker thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("QuestionGenerationWorker already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="question-gen-worker",
            daemon=True,
        )
        self._thread.start()
        logger.info("🧠 QuestionGenerationWorker started")

    def stop(self):
        """Signal the worker to stop and wait for it to finish."""
        self._stop_event.set()
        # Unblock if paused so thread can see stop signal
        self.llm_available.set()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        logger.info("🧠 QuestionGenerationWorker stopped")

    def pause(self):
        """Pause the worker (LLM needed for user query)."""
        self.llm_available.clear()

    def resume(self):
        """Resume the worker (LLM free again)."""
        self.llm_available.set()

    def get_status(self) -> dict:
        """Return current worker status."""
        return {
            "running": self._thread is not None and self._thread.is_alive(),
            "paused": not self.llm_available.is_set(),
            "chunks_processed": self.chunks_processed,
            "questions_generated": self.questions_generated,
            "errors": self.errors,
        }

    # ── internal loop ─────────────────────────────────────────────────────

    def _run(self):
        """Main worker loop — runs in a daemon thread."""
        logger.info("🧠 Question generation worker loop started")

        while not self._stop_event.is_set():
            try:
                chunks = self.rag.get_chunks_needing_questions(
                    limit=self.BATCH_SIZE
                )

                if not chunks:
                    # Nothing to do — sleep and re-check
                    self._stop_event.wait(timeout=self.IDLE_SLEEP)
                    continue

                for chunk in chunks:
                    if self._stop_event.is_set():
                        break

                    # Wait until LLM is available (cooperative pause)
                    while not self.llm_available.is_set():
                        if self._stop_event.is_set():
                            return
                        self.llm_available.wait(timeout=1)

                    self._process_chunk(chunk)

                    # Small sleep between chunks to avoid hammering the LLM
                    time.sleep(self.BATCH_SLEEP)

            except Exception as e:
                logger.error(f"🧠 Worker loop error: {e}")
                self.errors += 1
                # Back off on errors
                self._stop_event.wait(timeout=5)

        logger.info("🧠 Question generation worker loop exiting")

    def _process_chunk(self, chunk: dict):
        """Generate questions for a single chunk and store them."""
        chunk_id = chunk["id"]
        chunk_text = chunk["text"]
        chunk_meta = chunk["metadata"]

        try:
            questions = self._generate_questions(chunk_text)

            if questions:
                self.rag.add_question_embeddings(
                    parent_chunk_id=chunk_id,
                    questions=questions,
                    parent_metadata=chunk_meta,
                )
                self.questions_generated += len(questions)

            # Mark chunk as processed regardless (even if 0 questions)
            self.rag.mark_chunk_has_questions(chunk_id)
            self.chunks_processed += 1

            logger.debug(
                f"🧠 Chunk {chunk_id[:8]}... → {len(questions)} questions"
            )

        except Exception as e:
            logger.error(
                f"🧠 Failed to process chunk {chunk_id[:8]}...: {e}"
            )
            self.errors += 1

    def _generate_questions(self, chunk_text: str) -> List[str]:
        """
        Call the LLM to generate questions for the given chunk text.

        Returns a list of question strings.
        """
        # Truncate very long chunks for the prompt
        max_chunk_len = 2000
        if len(chunk_text) > max_chunk_len:
            chunk_text = chunk_text[:max_chunk_len] + "..."

        prompt = self.QUESTION_PROMPT.format(chunk_text=chunk_text)

        # Run async LLM call in a new event loop (we're in a thread)
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(
                self._call_llm(prompt)
            )
        finally:
            loop.close()

        return self._parse_questions(response)

    async def _call_llm(self, prompt: str) -> str:
        """Make an async LLM call via langchain_ollama."""
        from langchain_ollama import ChatOllama
        from langchain.messages import HumanMessage

        llm = ChatOllama(
            model=self.model,
            temperature=0.3,
            base_url=self.ollama_base_url,
        )

        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return response.content

    @staticmethod
    def _parse_questions(response: str) -> List[str]:
        """
        Parse numbered questions from LLM response.

        Expects lines like:
          1. What is ...?
          2. How does ...?
        """
        if not response:
            return []

        # Strip <think>...</think> blocks if present (qwen3 sometimes adds them)
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

        questions = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            # Remove numbering prefix: "1.", "1)", "- ", "* "
            cleaned = re.sub(r"^\d+[.)]\s*", "", line)
            cleaned = re.sub(r"^[-*]\s*", "", cleaned)
            cleaned = cleaned.strip()

            if cleaned and "?" in cleaned and len(cleaned) > 10:
                questions.append(cleaned)

        return questions[:5]  # Cap at 5 questions
