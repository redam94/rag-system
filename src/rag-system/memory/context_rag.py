"""
Enhanced RAG System with Contextual Chunking

Features:
- Contextual chunking that preserves document structure
- Section-aware segmentation (keeps content under headers together)
- Paragraph preservation as semantic units
- List group preservation
- Code-block aware segmentation (preserves code as atomic units)
- Section context included in chunks for better retrieval
- Web search result storage with enrichment tracking
- Document upload support (PDF, text, URLs)
"""

import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("⚠️ ChromaDB not available. Install with: pip install chromadb")


class TextChunker:
    """
    Contextual text chunking for RAG system.
    
    Features:
    - Preserves document structure (sections, headers)
    - Keeps paragraphs together as semantic units
    - Preserves list items as groups
    - Code blocks treated as atomic units
    - Includes section context in each chunk
    - Semantic boundary detection
    """
    
    # Patterns for structure detection
    CODE_BLOCK_PATTERN = re.compile(r'```[\w]*\n.*?```', re.DOTALL)
    HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    LIST_ITEM_PATTERN = re.compile(r'^(\s*[-*+]|\s*\d+\.)\s+', re.MULTILINE)
    PARAGRAPH_BREAK = re.compile(r'\n\s*\n')
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50, include_context: bool = True):
        """
        Initialize contextual text chunker.
        
        Args:
            chunk_size: Target size for each chunk (in characters)
            overlap: Number of characters to overlap between chunks
            include_context: Whether to include section headers in chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.include_context = include_context
        logger.info(f"📝 TextChunker initialized (chunk_size={chunk_size}, overlap={overlap}, contextual=True)")
    
    def _extract_code_blocks(self, text: str) -> tuple[List[str], str]:
        """Extract code blocks, replacing with placeholders."""
        code_blocks = []
        
        def replace_with_placeholder(match):
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks) - 1}__"
        
        text_with_placeholders = self.CODE_BLOCK_PATTERN.sub(replace_with_placeholder, text)
        return code_blocks, text_with_placeholders
    
    def _restore_code_blocks(self, text: str, code_blocks: List[str]) -> str:
        """Restore code blocks from placeholders."""
        for i, block in enumerate(code_blocks):
            text = text.replace(f"__CODE_BLOCK_{i}__", block)
        return text
    
    def _parse_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse text into hierarchical sections based on headers.
        
        Returns list of sections with header info and content.
        """
        sections = []
        current_section = {"header": None, "level": 0, "content": "", "start": 0}
        
        lines = text.split('\n')
        current_pos = 0
        
        for line in lines:
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            
            if header_match:
                # Save previous section
                if current_section["content"].strip():
                    sections.append(current_section)
                
                # Start new section
                level = len(header_match.group(1))
                header_text = header_match.group(2).strip()
                current_section = {
                    "header": header_text,
                    "level": level,
                    "content": "",
                    "start": current_pos
                }
            else:
                current_section["content"] += line + "\n"
            
            current_pos += len(line) + 1
        
        # Save last section
        if current_section["content"].strip() or current_section["header"]:
            sections.append(current_section)
        
        # If no sections found, treat entire text as one section
        if not sections:
            sections = [{"header": None, "level": 0, "content": text, "start": 0}]
        
        return sections
    
    def _split_into_paragraphs(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into paragraphs, preserving list groups.
        
        Returns list of paragraph objects with type info.
        """
        paragraphs = []
        
        # Split by double newlines (paragraph breaks)
        raw_paragraphs = self.PARAGRAPH_BREAK.split(text)
        
        for para in raw_paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if this is a list
            lines = para.split('\n')
            is_list = all(self.LIST_ITEM_PATTERN.match(line.strip()) or not line.strip() 
                         for line in lines if line.strip())
            
            # Check if contains code placeholder
            has_code_placeholder = '__CODE_BLOCK_' in para
            
            paragraphs.append({
                "text": para,
                "type": "list" if is_list else ("code_ref" if has_code_placeholder else "prose"),
                "size": len(para)
            })
        
        return paragraphs
    
    def _merge_small_paragraphs(self, paragraphs: List[Dict[str, Any]], min_size: int = 100) -> List[Dict[str, Any]]:
        """Merge small consecutive paragraphs of the same type."""
        if not paragraphs:
            return []
        
        merged = []
        current = None
        
        for para in paragraphs:
            if current is None:
                current = para.copy()
            elif (current["type"] == para["type"] == "prose" and 
                  current["size"] + para["size"] < self.chunk_size):
                # Merge small prose paragraphs
                current["text"] += "\n\n" + para["text"]
                current["size"] = len(current["text"])
            else:
                merged.append(current)
                current = para.copy()
        
        if current:
            merged.append(current)
        
        return merged
    
    def _chunk_large_paragraph(self, text: str, context_header: str = None) -> List[str]:
        """
        Chunk a large paragraph by sentences, keeping context.
        """
        # Add context header if provided
        prefix = f"[{context_header}]\n" if context_header and self.include_context else ""
        prefix_size = len(prefix)
        effective_chunk_size = self.chunk_size - prefix_size
        
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_size = len(sentence)
            
            # If single sentence is too large, split it
            if sentence_size > effective_chunk_size:
                if current_chunk:
                    chunks.append(prefix + " ".join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                # Split long sentence by clauses or words
                words = sentence.split()
                word_chunk = []
                word_size = 0
                
                for word in words:
                    if word_size + len(word) + 1 > effective_chunk_size and word_chunk:
                        chunks.append(prefix + " ".join(word_chunk))
                        word_chunk = []
                        word_size = 0
                    word_chunk.append(word)
                    word_size += len(word) + 1
                
                if word_chunk:
                    current_chunk = word_chunk
                    current_size = word_size
                continue
            
            if current_size + sentence_size + 1 > effective_chunk_size and current_chunk:
                chunks.append(prefix + " ".join(current_chunk))
                
                # Overlap: keep last sentence or partial
                if self.overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-1] if len(current_chunk[-1]) < self.overlap else ""
                    current_chunk = [overlap_text] if overlap_text else []
                    current_size = len(overlap_text)
                else:
                    current_chunk = []
                    current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size + 1
        
        if current_chunk:
            chunks.append(prefix + " ".join(current_chunk))
        
        return chunks
    
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Chunk text with contextual awareness.
        
        Preserves:
        - Document structure (sections under headers)
        - Paragraphs as semantic units
        - List groups
        - Code blocks as atomic units
        
        Args:
            text: Text to chunk
            metadata: Base metadata to attach to all chunks
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text:
            return []
        
        # Small text - no chunking needed
        if len(text) <= self.chunk_size:
            has_code = bool(self.CODE_BLOCK_PATTERN.search(text))
            return [{
                "text": text,
                "metadata": {
                    **(metadata or {}),
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "chunk_size": len(text),
                    "has_code": has_code,
                    "chunk_type": "complete"
                }
            }]
        
        # Extract code blocks first
        code_blocks, text_with_placeholders = self._extract_code_blocks(text)
        
        # Parse into sections
        sections = self._parse_sections(text_with_placeholders)
        
        all_chunks = []
        
        for section in sections:
            section_header = section["header"]
            section_content = section["content"]
            
            if not section_content.strip() and not section_header:
                continue
            
            # Split section into paragraphs
            paragraphs = self._split_into_paragraphs(section_content)
            paragraphs = self._merge_small_paragraphs(paragraphs)
            
            for para in paragraphs:
                para_text = para["text"]
                para_type = para["type"]
                
                # Check for code block placeholders
                code_match = re.search(r'__CODE_BLOCK_(\d+)__', para_text)
                
                if code_match:
                    # Handle mixed content with code blocks
                    parts = re.split(r'(__CODE_BLOCK_\d+__)', para_text)
                    
                    for part in parts:
                        part = part.strip()
                        if not part:
                            continue
                        
                        code_placeholder = re.match(r'__CODE_BLOCK_(\d+)__', part)
                        if code_placeholder:
                            # Restore and add code block
                            idx = int(code_placeholder.group(1))
                            code_text = code_blocks[idx]
                            
                            # Add section context to code block if small enough
                            if section_header and self.include_context:
                                code_with_context = f"[{section_header}]\n{code_text}"
                            else:
                                code_with_context = code_text
                            
                            all_chunks.append({
                                "text": code_with_context,
                                "is_code": True,
                                "section": section_header,
                                "chunk_type": "code"
                            })
                        elif len(part) > 20:  # Skip very small fragments
                            # Prose around code
                            if len(part) <= self.chunk_size:
                                prefix = f"[{section_header}]\n" if section_header and self.include_context else ""
                                all_chunks.append({
                                    "text": prefix + part,
                                    "is_code": False,
                                    "section": section_header,
                                    "chunk_type": "prose"
                                })
                            else:
                                sub_chunks = self._chunk_large_paragraph(part, section_header)
                                for sc in sub_chunks:
                                    all_chunks.append({
                                        "text": sc,
                                        "is_code": False,
                                        "section": section_header,
                                        "chunk_type": "prose"
                                    })
                
                elif para_type == "list":
                    # Keep lists together if possible
                    if len(para_text) <= self.chunk_size:
                        prefix = f"[{section_header}]\n" if section_header and self.include_context else ""
                        all_chunks.append({
                            "text": prefix + para_text,
                            "is_code": False,
                            "section": section_header,
                            "chunk_type": "list"
                        })
                    else:
                        # Split large list by items
                        items = re.split(r'\n(?=\s*[-*+]|\s*\d+\.)', para_text)
                        current_list_chunk = []
                        current_size = 0
                        prefix = f"[{section_header}]\n" if section_header and self.include_context else ""
                        
                        for item in items:
                            item = item.strip()
                            if not item:
                                continue
                            if current_size + len(item) > self.chunk_size - len(prefix) and current_list_chunk:
                                all_chunks.append({
                                    "text": prefix + "\n".join(current_list_chunk),
                                    "is_code": False,
                                    "section": section_header,
                                    "chunk_type": "list"
                                })
                                current_list_chunk = []
                                current_size = 0
                            current_list_chunk.append(item)
                            current_size += len(item) + 1
                        
                        if current_list_chunk:
                            all_chunks.append({
                                "text": prefix + "\n".join(current_list_chunk),
                                "is_code": False,
                                "section": section_header,
                                "chunk_type": "list"
                            })
                
                else:
                    # Regular prose paragraph
                    if len(para_text) <= self.chunk_size:
                        prefix = f"[{section_header}]\n" if section_header and self.include_context else ""
                        all_chunks.append({
                            "text": prefix + para_text,
                            "is_code": False,
                            "section": section_header,
                            "chunk_type": "prose"
                        })
                    else:
                        # Chunk large paragraph
                        sub_chunks = self._chunk_large_paragraph(para_text, section_header)
                        for sc in sub_chunks:
                            all_chunks.append({
                                "text": sc,
                                "is_code": False,
                                "section": section_header,
                                "chunk_type": "prose"
                            })
        
        # Build final result with metadata
        total_chunks = len(all_chunks)
        result = []
        
        for idx, chunk_info in enumerate(all_chunks):
            chunk_metadata = {
                **(metadata or {}),
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "chunk_size": len(chunk_info["text"]),
                "has_code": chunk_info.get("is_code", False),
                "chunk_type": chunk_info.get("chunk_type", "unknown"),
                "section": chunk_info.get("section")
            }
            result.append({
                "text": chunk_info["text"],
                "metadata": chunk_metadata
            })
        
        # Log chunking stats
        code_count = sum(1 for c in all_chunks if c.get("is_code"))
        list_count = sum(1 for c in all_chunks if c.get("chunk_type") == "list")
        prose_count = sum(1 for c in all_chunks if c.get("chunk_type") == "prose")
        sections_found = len(set(c.get("section") for c in all_chunks if c.get("section")))
        
        logger.debug(f"📝 Contextual chunking: {total_chunks} chunks "
                    f"({code_count} code, {list_count} lists, {prose_count} prose, {sections_found} sections)")
        
        return result
    
    def chunk_code_only(self, code: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Chunk pure code by logical boundaries (functions, classes).
        
        For storing standalone code snippets in RAG.
        """
        if not code:
            return []
        
        # Try to split by function/class definitions
        definition_pattern = re.compile(r'^((?:def|class|async def)\s+\w+)', re.MULTILINE)
        
        matches = list(definition_pattern.finditer(code))
        
        if not matches or len(code) <= self.chunk_size:
            return [{
                "text": code,
                "metadata": {
                    **(metadata or {}),
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "chunk_size": len(code),
                    "has_code": True,
                    "is_pure_code": True,
                    "chunk_type": "code"
                }
            }]
        
        chunks = []
        
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(code)
            chunk_text = code[start:end].strip()
            
            if chunk_text:
                chunks.append(chunk_text)
        
        # Include any code before the first definition
        if matches and matches[0].start() > 0:
            preamble = code[:matches[0].start()].strip()
            if preamble:
                chunks.insert(0, preamble)
        
        total_chunks = len(chunks)
        result = []
        
        for idx, chunk_text in enumerate(chunks):
            chunk_metadata = {
                **(metadata or {}),
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "chunk_size": len(chunk_text),
                "has_code": True,
                "is_pure_code": True,
                "chunk_type": "code"
            }
            result.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })
        
        logger.debug(f"📝 Chunked code into {total_chunks} segments")
        return result


class ContextRAG:
    """
    Enhanced RAG system with chunking for better context retrieval.
    
    Features:
    - Contextual chunking that preserves document structure
    - Section-aware segmentation with header context
    - Code-aware chunking (preserves code blocks)
    - Web search result storage with enrichment tracking
    - Document upload support (PDF, text, URLs)
    - Focused context for LLM queries
    """
    
    def __init__(
        self,
        collection_name: str = "workflow_context",
        persist_directory: str = "cache/rag_db",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        include_section_context: bool = True
    ):
        """
        Initialize the RAG system.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            include_section_context: Include section headers in chunks for context
        """
        if not CHROMADB_AVAILABLE:
            logger.error("❌ ChromaDB not available. RAG system disabled.")
            self.enabled = False
            return
        
        self.enabled = True
        self.persist_directory = Path(persist_directory).resolve()
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize contextual chunker
        self.chunker = TextChunker(
            chunk_size=chunk_size, 
            overlap=chunk_overlap,
            include_context=include_section_context
        )
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        try:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"📚 ContextRAG initialized: {collection_name}")
            logger.info(f"   Stored documents: {self.collection.count()}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize collection: {e}")
            self.enabled = False
    
    def _generate_id(self, content: str, doc_type: str, chunk_idx: int = 0) -> str:
        """Generate a unique ID for a chunk."""
        hash_input = f"{doc_type}:{content[:100]}:{chunk_idx}:{datetime.now().isoformat()}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean metadata for ChromaDB compatibility.
        ChromaDB only accepts: str, int, float, bool, or None values.
        """
        cleaned = {}
        for key, value in metadata.items():
            if value is None:
                cleaned[key] = 'N/A'
            elif isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            elif isinstance(value, list):
                if len(value) == 0:
                    cleaned[key] = ""
                elif all(isinstance(v, str) for v in value):
                    cleaned[key] = ", ".join(value)
                else:
                    cleaned[key] = str(value)
            elif isinstance(value, dict):
                import json
                cleaned[key] = json.dumps(value)
            else:
                cleaned[key] = str(value)
        
        return cleaned
    
    def _process_batch(self, batch, doc_type: str) -> None:
        """Process and add a batch of chunks to the collection."""
        documents = []
        metadatas = []
        ids = []
        
        for chunk in batch:
            chunk_text = chunk["text"]
            chunk_metadata = self._clean_metadata(chunk["metadata"])
            chunk_id = self._generate_id(chunk_text, doc_type, chunk["metadata"]["chunk_index"])
            
            documents.append(chunk_text)
            metadatas.append(chunk_metadata)
            ids.append(chunk_id)
        
        logger.debug(f"Adding {len(documents)} chunks to collection '{self.collection.name}'")
        logger.debug(f"Sample metadata: {metadatas[0] if metadatas else 'N/A'}")
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def _add_chunks_to_collection(
        self,
        chunks: List[Dict[str, Any]],
        doc_type: str
    ):
        """Helper to add chunks to collection."""
        if not chunks:
            return
        MAX_BATCH_SIZE = 1000
        for i in range(0, len(chunks), MAX_BATCH_SIZE):
            batch = chunks[i:i + MAX_BATCH_SIZE]
            self._process_batch(batch, doc_type)
        
        remaining = chunks[len(chunks) - (len(chunks) % MAX_BATCH_SIZE):]
        if remaining:
            self._process_batch(remaining, doc_type)
        
    
    def add_plot_analysis(
        self,
        plot_name: str,
        plot_path: str,
        analysis: str,
        stage_name: str,
        workflow_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a plot analysis to the RAG system with chunking.
        """
        if not self.enabled:
            return
        
        try:
            base_metadata = {
                "type": "plot_analysis",
                "plot_name": plot_name,
                "plot_path": plot_path,
                "stage_name": stage_name,
                "workflow_id": workflow_id,
                "timestamp": datetime.now().isoformat()
            }
            
            if metadata:
                base_metadata.update(metadata)
            
            doc_text = f"Plot: {plot_name}\n\nAnalysis:\n{analysis}"
            chunks = self.chunker.chunk_text(doc_text, base_metadata)
            
            self._add_chunks_to_collection(chunks, "plot_analysis")
            
            logger.info(f"📊 Added plot analysis to RAG: {plot_name} ({len(chunks)} chunks)")
            
        except Exception as e:
            logger.error(f"❌ Failed to add plot analysis: {e}")
    
    def add_code_execution(
        self,
        code: str,
        stdout: str,
        stderr: str,
        stage_name: str,
        workflow_id: str,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add code execution results to RAG with chunking.
        
        Code is stored separately for better retrieval.
        """
        if not self.enabled:
            return
        
        try:
            base_metadata = {
                "type": "code_execution",
                "stage_name": stage_name,
                "workflow_id": workflow_id,
                "success": success,
                "has_error": bool(stderr),
                "timestamp": datetime.now().isoformat(),
                "code_length": len(code),
                "output_length": len(stdout)
            }
            
            if metadata:
                base_metadata.update(metadata)
            
            # Store code separately using code-aware chunking
            code_metadata = {**base_metadata, "content_type": "code"}
            code_chunks = self.chunker.chunk_code_only(code, code_metadata)
            self._add_chunks_to_collection(code_chunks, "code_execution")
            
            # Store output as prose
            output_text = f"Stage: {stage_name}\nStatus: {'Success' if success else 'Failed'}\n\nOutput:\n{stdout}"
            if stderr:
                output_text += f"\n\nErrors:\n{stderr}"
            
            output_metadata = {**base_metadata, "content_type": "output"}
            output_chunks = self.chunker.chunk_text(output_text, output_metadata)
            self._add_chunks_to_collection(output_chunks, "code_execution")
            
            logger.info(f"💻 Added code execution to RAG: {stage_name} ({len(code_chunks)} code + {len(output_chunks)} output chunks)")
            
        except Exception as e:
            logger.error(f"❌ Failed to add code execution: {e}")
    
    def add_web_result(
        self,
        query: str,
        url: str,
        title: str,
        content: str,
        stage_name: str,
        workflow_id: str,
        source: str = "duckduckgo",
        relevance_score: Optional[float] = None,
        enriched: bool = False,
        query_used: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a web search result to the RAG system.
        
        Args:
            query: Original search query
            url: Source URL
            title: Page/result title
            content: Extracted content
            stage_name: Workflow stage name
            workflow_id: Workflow ID
            source: Search source (e.g., "duckduckgo", "duckduckgo+crawl4ai")
            relevance_score: Optional relevance score (0-1)
            enriched: Whether content was enriched via crawl4ai
            query_used: The specific query that found this result
            metadata: Additional metadata
        """
        if not self.enabled:
            return
        
        try:
            # Auto-detect enriched from source if not explicitly set
            is_enriched = enriched or "crawl4ai" in source.lower()
            
            base_metadata = {
                "type": "web_result",
                "search_query": query,
                "url": url,
                "title": title,
                "source": source,
                "enriched": is_enriched,
                "stage_name": stage_name,
                "workflow_id": workflow_id,
                "timestamp": datetime.now().isoformat(),
                "content_length": len(content)
            }
            
            if relevance_score is not None:
                base_metadata["relevance_score"] = relevance_score
            
            if query_used:
                base_metadata["query_used"] = query_used
            
            if metadata:
                base_metadata.update(metadata)
            
            # Format document text with enrichment indicator
            enriched_marker = " [ENRICHED]" if is_enriched else ""
            doc_text = f"Web Result{enriched_marker}: {title}\nURL: {url}\nQuery: {query}\n\n{content}"
            
            chunks = self.chunker.chunk_text(doc_text, base_metadata)
            self._add_chunks_to_collection(chunks, "web_result")
            
            enriched_str = " (enriched)" if is_enriched else ""
            logger.info(f"🌐 Added web result to RAG{enriched_str}: {title[:50]}... ({len(chunks)} chunks)")
            
        except Exception as e:
            logger.error(f"❌ Failed to add web result: {e}")
    
    def add_web_search_batch(
        self,
        query: str,
        results: List[Dict[str, Any]],
        stage_name: str,
        workflow_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add multiple web search results at once.
        
        Args:
            query: Original search query
            results: List of result dicts with keys: 
                     url, title, content, score (optional), source (optional),
                     enriched (optional), query_used (optional)
            stage_name: Workflow stage name
            workflow_id: Workflow ID
            metadata: Additional metadata for all results
        """
        if not self.enabled:
            return
        
        enriched_count = 0
        for result in results:
            is_enriched = result.get("enriched", False) or "crawl4ai" in result.get("source", "").lower()
            if is_enriched:
                enriched_count += 1
            
            self.add_web_result(
                query=query,
                url=result.get("url", ""),
                title=result.get("title", ""),
                content=result.get("content", ""),
                stage_name=stage_name,
                workflow_id=workflow_id,
                source=result.get("source", "web_search"),
                relevance_score=result.get("score"),
                enriched=is_enriched,
                query_used=result.get("query_used"),
                metadata=metadata
            )
        
        if enriched_count > 0:
            logger.info(f"🌐 Batch stored {len(results)} web results ({enriched_count} enriched)")
    
    def query_enriched_web_context(
        self,
        query: str,
        workflow_id: Optional[str] = None,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query specifically for enriched (crawl4ai) web results.
        
        These contain fuller, more detailed content than basic search snippets.
        """
        if not self.enabled:
            return []
        
        try:
            conditions = [{"type": "web_result"}, {"enriched": True}]
            if workflow_id:
                conditions.append({"workflow_id": workflow_id})
            
            where_filter = {"$and": conditions}
            
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter
            )
            
            if not results['documents'] or not results['documents'][0]:
                return []
            
            contexts = []
            for i, doc in enumerate(results['documents'][0]):
                contexts.append({
                    "document": doc,
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else None
                })
            
            logger.info(f"🔍 Retrieved {len(contexts)} enriched web results")
            return contexts
            
        except Exception as e:
            logger.error(f"❌ Failed to query enriched web context: {e}")
            return []
    
    def add_summary(
        self,
        summary: str,
        stage_name: str,
        workflow_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a summary to RAG with chunking."""
        if not self.enabled:
            return
        
        try:
            base_metadata = {
                "type": "summary",
                "stage_name": stage_name,
                "workflow_id": workflow_id,
                "timestamp": datetime.now().isoformat()
            }
            
            if metadata:
                base_metadata.update(metadata)
            
            doc_text = f"Summary for {stage_name}:\n{summary}"
            chunks = self.chunker.chunk_text(doc_text, base_metadata)
            
            self._add_chunks_to_collection(chunks, "summary")
            
            logger.info(f"📝 Added summary to RAG: {stage_name} ({len(chunks)} chunks)")
            
        except Exception as e:
            logger.error(f"❌ Failed to add summary: {e}")
    
    def query_relevant_context(
        self,
        query: str,
        workflow_id: Optional[str] = None,
        stage_name: Optional[str] = None,
        doc_types: Optional[List[str]] = None,
        code_only: bool = False,
        n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Query for relevant context chunks.
        
        Args:
            query: User query text
            workflow_id: Filter by workflow ID
            stage_name: Filter by stage name
            doc_types: Filter by document types (plot_analysis, code_execution, summary, web_result)
            code_only: If True, only return chunks with has_code=True
            n_results: Number of results to retrieve
            
        Returns:
            List of relevant context chunks with metadata
        """
        if not self.enabled:
            return []
        
        try:
            # Build where filter
            conditions = []
            
            if workflow_id:
                conditions.append({"workflow_id": workflow_id})
            if stage_name:
                conditions.append({"stage_name": stage_name})
            if doc_types:
                conditions.append({"type": {"$in": doc_types}})
            if code_only:
                conditions.append({"has_code": True})
            
            where_filter = None
            if len(conditions) == 1:
                where_filter = conditions[0]
            elif len(conditions) > 1:
                where_filter = {"$and": conditions}
            
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter
            )
            
            if not results['documents'] or not results['documents'][0]:
                logger.info(f"🔍 No relevant context found for query")
                return []
            
            contexts = []
            for i, doc in enumerate(results['documents'][0]):
                context = {
                    "document": doc,
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else None
                }
                contexts.append(context)
            
            logger.info(f"🔍 Retrieved {len(contexts)} relevant context chunks")
            return contexts
            
        except Exception as e:
            logger.error(f"❌ Failed to query context: {e}")
            return []
    
    def query_code_context(
        self,
        query: str,
        workflow_id: Optional[str] = None,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query specifically for code-related context.
        
        Useful when generating new code based on previous executions.
        """
        return self.query_relevant_context(
            query=query,
            workflow_id=workflow_id,
            doc_types=["code_execution"],
            code_only=True,
            n_results=n_results
        )
    
    def query_web_context(
        self,
        query: str,
        workflow_id: Optional[str] = None,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query specifically for web search results.
        
        Useful for finding methodology guidance from previous searches.
        """
        return self.query_relevant_context(
            query=query,
            workflow_id=workflow_id,
            doc_types=["web_result"],
            n_results=n_results
        )
    
    def get_context_summary(
        self,
        query: str,
        workflow_id: Optional[str] = None,
        stage_name: Optional[str] = None,
        max_tokens: int = 2000,
        max_chunks: int = 15
    ) -> str:
        """
        Get a concise context summary relevant to the query.
        
        Combines relevant chunks into a focused context string.
        """
        if not self.enabled:
            return ""
        
        contexts = self.query_relevant_context(
            query=query,
            workflow_id=workflow_id,
            stage_name=stage_name,
            n_results=max_chunks
        )
        
        if not contexts:
            return ""
        
        context_parts = []
        total_length = 0
        max_chars = max_tokens * 4
        
        # Group chunks by document type
        doc_groups = {}
        for ctx in contexts:
            doc_type = ctx['metadata'].get('type', 'unknown')
            stage = ctx['metadata'].get('stage_name', 'unknown')
            chunk_idx = ctx['metadata'].get('chunk_index', 0)
            
            key = f"{doc_type}:{stage}"
            if key not in doc_groups:
                doc_groups[key] = []
            doc_groups[key].append((chunk_idx, ctx))
        
        for key, chunks in doc_groups.items():
            chunks.sort(key=lambda x: x[0])
            
            doc_type, stage = key.split(':', 1)
            context_parts.append(f"\n[{doc_type.upper()} - {stage}]")
            
            for _, ctx in chunks:
                doc_text = ctx['document']
                logger.debug(f"Adding chunk (type={doc_type}, stage={stage}, size={len(doc_text)}) to context")
                if total_length + len(doc_text) > max_chars:
                    remaining = max_chars - total_length
                    if remaining > 100:
                        doc_text = doc_text[:remaining] + "..."
                    else:
                        break
                
                context_parts.append(doc_text)
                total_length += len(doc_text)
                
                if total_length >= max_chars:
                    break
            
            if total_length >= max_chars:
                break
        
        logger.debug(f"Total context length: {total_length} characters")
        return "\n".join(context_parts)
    
    def delete_by_workflow(self, workflow_id: str):
        """Delete all documents for a specific workflow."""
        if not self.enabled:
            return
        
        try:
            results = self.collection.get(
                where={"workflow_id": workflow_id}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"🗑️ Deleted {len(results['ids'])} chunks for workflow: {workflow_id}")
        
        except Exception as e:
            logger.error(f"❌ Failed to delete workflow documents: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        if not self.enabled:
            return {"enabled": False}
        
        try:
            total_count = self.collection.count()
            
            results = self.collection.get()
            type_counts = {}
            chunk_type_counts = {}
            code_chunks = 0
            enriched_web_chunks = 0
            document_chunks = 0
            sections_with_context = 0
            
            if results['metadatas']:
                for metadata in results['metadatas']:
                    # Document type (code_execution, web_result, etc.)
                    doc_type = metadata.get('type', 'unknown')
                    type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
                    
                    # Chunk type (prose, list, code, etc.)
                    chunk_type = metadata.get('chunk_type', 'unknown')
                    chunk_type_counts[chunk_type] = chunk_type_counts.get(chunk_type, 0) + 1
                    
                    if metadata.get('has_code'):
                        code_chunks += 1
                    if metadata.get('enriched') and doc_type == 'web_result':
                        enriched_web_chunks += 1
                    if doc_type == 'document':
                        document_chunks += 1
                    if metadata.get('section'):
                        sections_with_context += 1
            
            return {
                "enabled": True,
                "total_chunks": total_count,
                "type_breakdown": type_counts,
                "chunk_type_breakdown": chunk_type_counts,
                "code_chunks": code_chunks,
                "enriched_web_chunks": enriched_web_chunks,
                "document_chunks": document_chunks,
                "chunks_with_section_context": sections_with_context,
                "persist_directory": str(self.persist_directory),
                "collection_name": self.collection.name,
                "chunk_size": self.chunker.chunk_size,
                "chunk_overlap": self.chunker.overlap,
                "contextual_chunking": self.chunker.include_context
            }
        
        except Exception as e:
            logger.error(f"❌ Failed to get stats: {e}")
            return {"enabled": True, "error": str(e)}
    def add(
            self,
            content: str,
            *args,
            metadata: Optional[Dict[str, Any]] = None
    ):
        ...

    def add_document(
        self,
        content: str,
        title: str,
        source_type: str,
        workflow_id: str,
        source_path: Optional[str] = None,
        url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a document resource to the RAG system.
        
        Args:
            content: Document text content
            title: Document title
            source_type: Type of source (pdf, txt, md, url, etc.)
            workflow_id: Workflow ID
            source_path: Original file path (for uploads)
            url: Source URL (for scraped content)
            metadata: Additional metadata
        """
        if not self.enabled:
            return
        
        try:
            base_metadata = {
                "type": "document",
                "title": title,
                "source_type": source_type,
                "workflow_id": workflow_id,
                "timestamp": datetime.now().isoformat(),
                "content_length": len(content)
            }
            
            if source_path:
                base_metadata["source_path"] = source_path
            if url:
                base_metadata["url"] = url
            logger.debug(f"Metadata sample: {base_metadata}")
            if metadata:
                base_metadata.update(metadata)
            
            # Format document
            doc_text = f"Document: {title}\nType: {source_type}\n\n{content}"
            logger.debug(f"Metadata sample: {base_metadata}")
            chunks = self.chunker.chunk_text(doc_text, base_metadata)
            logger.debug(f"Chunked document into {len(chunks)} chunks")
            
            self._add_chunks_to_collection(chunks, "document")
            
            logger.info(f"📄 Added document to RAG: {title} ({len(chunks)} chunks)")
            return len(chunks)
            
        except Exception as e:
            logger.error(f"❌ Failed to add document: {e}")
            return 0
    
    def add_url_content(
        self,
        url: str,
        content: str,
        title: str,
        workflow_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add scraped URL content to RAG.
        
        Args:
            url: Source URL
            content: Scraped content (markdown)
            title: Page title
            workflow_id: Workflow ID
            metadata: Additional metadata
        """
        return self.add_document(
            content=content,
            title=title,
            source_type="url",
            workflow_id=workflow_id,
            url=url,
            metadata=metadata
        )
    
    def query_documents(
        self,
        query: str,
        workflow_id: Optional[str] = None,
        source_types: Optional[List[str]] = None,
        n_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Query specifically for uploaded/scraped documents.
        
        Args:
            query: Search query
            workflow_id: Filter by workflow
            source_types: Filter by source types (pdf, txt, url, etc.)
            n_results: Max results
        """
        if not self.enabled:
            return []
        
        try:
            conditions = [{"type": "document"}]
            
            if workflow_id:
                conditions.append({"workflow_id": workflow_id})
            
            where_filter = {"$and": conditions} if len(conditions) > 1 else conditions[0]
            
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_filter
            )
            
            if not results['documents'] or not results['documents'][0]:
                return []
            
            contexts = []
            for i, doc in enumerate(results['documents'][0]):
                ctx = {
                    "document": doc,
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else None
                }
                # Filter by source_types if specified
                if source_types:
                    if ctx['metadata'].get('source_type') in source_types:
                        contexts.append(ctx)
                else:
                    contexts.append(ctx)
            
            return contexts
            
        except Exception as e:
            logger.error(f"❌ Failed to query documents: {e}")
            return []
    
    def get_all_documents(self, workflow_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all uploaded/scraped documents."""
        if not self.enabled:
            return []
        
        try:
            if workflow_id:
                where_filter = {"$and": [{"type": "document"}, {"workflow_id": workflow_id}]}
            else:
                where_filter = {"type": "document"}
            
            results = self.collection.get(
                where=where_filter,
                limit=500
            )
            
            if not results['documents']:
                return []
            
            # Group by title/source to get unique documents
            docs = {}
            for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas'])):
                key = meta.get('title', '') + meta.get('url', '') + meta.get('source_path', '')
                if key not in docs:
                    docs[key] = {
                        'title': meta.get('title', 'Unknown'),
                        'source_type': meta.get('source_type', 'unknown'),
                        'url': meta.get('url'),
                        'source_path': meta.get('source_path'),
                        'timestamp': meta.get('timestamp'),
                        'content_length': meta.get('content_length', 0),
                        'chunk_count': 0,
                        'workflow_id': meta.get('workflow_id'),
                        'ids': []
                    }
                docs[key]['chunk_count'] += 1
                docs[key]['ids'].append(results['ids'][i])
            
            return list(docs.values())
            
        except Exception as e:
            logger.error(f"❌ Failed to get documents: {e}")
            return []
    
    def delete_document(self, title: str, workflow_id: Optional[str] = None) -> bool:
        """Delete a document by title."""
        if not self.enabled:
            return False
        
        try:
            conditions = [{"type": "document"}, {"title": title}]
            if workflow_id:
                conditions.append({"workflow_id": workflow_id})
            
            results = self.collection.get(
                where={"$and": conditions}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"🗑️ Deleted document: {title} ({len(results['ids'])} chunks)")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to delete document: {e}")
            return False
    
    def clear_all(self):
        """Clear all documents from the RAG system."""
        if not self.enabled:
            return
        
        try:
            self.client.delete_collection(self.collection.name)
            self.collection = self.client.create_collection(
                name=self.collection.name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("🗑️ Cleared all RAG chunks")
        
        except Exception as e:
            logger.error(f"❌ Failed to clear RAG: {e}")


# ============================================================================
# TESTING
# ============================================================================

def test_chunking():
    """Test contextual text chunking functionality."""
    print("\n" + "="*70)
    print("TESTING CONTEXTUAL TEXT CHUNKING")
    print("="*70)
    
    chunker = TextChunker(chunk_size=300, overlap=50, include_context=True)
    
    # Test 1: Section-aware chunking
    print("\n1. Testing section-aware chunking...")
    doc_with_sections = """
# Introduction

This is the introduction paragraph. It explains the basic concepts 
that will be covered in this document.

# Data Loading

Here's how to load data in pandas. The process is straightforward.

```python
import pandas as pd
df = pd.read_csv('data.csv')
```

This will load your CSV file into a DataFrame.

# Data Cleaning

Cleaning data is essential for accurate analysis.

- Remove null values
- Handle outliers
- Normalize columns
- Convert data types

## Handling Missing Values

Missing values can be handled in several ways:

1. Drop rows with missing values
2. Fill with mean/median
3. Use interpolation
"""
    
    chunks = chunker.chunk_text(doc_with_sections.strip(), metadata={"source": "tutorial"})
    print(f"   Document length: {len(doc_with_sections)} chars")
    print(f"   Chunks created: {len(chunks)}")
    
    for i, c in enumerate(chunks):
        section = c['metadata'].get('section', 'None')
        chunk_type = c['metadata'].get('chunk_type', 'unknown')
        print(f"   Chunk {i}: section='{section}', type={chunk_type}, size={c['metadata']['chunk_size']}")
        # Show first line
        first_line = c['text'].split('\n')[0][:60]
        print(f"            → {first_line}...")
    
    # Test 2: List preservation
    print("\n2. Testing list preservation...")
    list_doc = """
# Installation Steps

Follow these steps to install:

- Download the package from the website
- Extract the archive to your desired location
- Run the setup script with administrator privileges
- Configure the environment variables as described in the docs
- Verify installation by running the test suite

After installation, you can start using the tool.
"""
    
    chunks = chunker.chunk_text(list_doc.strip(), metadata={"source": "guide"})
    print(f"   Document length: {len(list_doc)} chars")
    print(f"   Chunks created: {len(chunks)}")
    
    for i, c in enumerate(chunks):
        chunk_type = c['metadata'].get('chunk_type', 'unknown')
        print(f"   Chunk {i}: type={chunk_type}, size={c['metadata']['chunk_size']}")
        if chunk_type == 'list':
            items = [l for l in c['text'].split('\n') if l.strip().startswith('-')]
            print(f"            Contains {len(items)} list items")
    
    # Test 3: Code block preservation with context
    print("\n3. Testing code block with section context...")
    code_doc = """
# Example Function

Here's a useful function for data processing:

```python
def process_data(df):
    '''Process the dataframe.'''
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df
```

Use this function on your cleaned data.
"""
    
    chunks = chunker.chunk_text(code_doc.strip(), metadata={"source": "example"})
    print(f"   Document length: {len(code_doc)} chars")
    print(f"   Chunks created: {len(chunks)}")
    
    for i, c in enumerate(chunks):
        has_code = c['metadata'].get('has_code', False)
        section = c['metadata'].get('section', 'None')
        print(f"   Chunk {i}: has_code={has_code}, section='{section}'")
        if has_code:
            # Check if context is included
            has_context = c['text'].startswith('[')
            print(f"            Context included: {has_context}")
    
    # Test 4: Pure code chunking
    print("\n4. Testing pure code chunking...")
    code = """
import pandas as pd
import numpy as np

def load_data(path):
    '''Load data from CSV.'''
    return pd.read_csv(path)

def clean_data(df):
    '''Clean the dataframe.'''
    df = df.dropna()
    return df

class DataProcessor:
    def __init__(self, df):
        self.df = df
    
    def process(self):
        return self.df.describe()
"""
    
    chunks = chunker.chunk_code_only(code.strip(), metadata={"source": "script"})
    print(f"   Code length: {len(code)} chars")
    print(f"   Chunks created: {len(chunks)}")
    
    for i, c in enumerate(chunks):
        first_line = c['text'].split('\n')[0]
        print(f"   Chunk {i}: {c['metadata']['chunk_size']} chars - {first_line}")
    
    print("\n" + "="*70)
    print("✅ All contextual chunking tests complete!")
    print("="*70)


if __name__ == "__main__":
    test_chunking()