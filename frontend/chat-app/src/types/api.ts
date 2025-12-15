// =============================================================================
// RE-EXPORT WS TYPES for backwards compatibility
// =============================================================================

export type {
  WSMessage,
  WSMessageType,
  WSMessageData,
  RAGQuery,
  RAGSearchGroup,
  ChatMessage,
  ChatMessageType,
} from './ws'

// =============================================================================
// AUTH TYPES
// =============================================================================

export interface User {
  username: string
  email: string | null
  full_name: string | null
  disabled: boolean
  created_at: string
}

export interface Token {
  access_token: string
  token_type: string
  expires_in: number
  refresh_token?: string
}

export interface LoginRequest {
  username: string
  password: string
}

export interface RegisterRequest {
  username: string
  password: string
  email?: string
  full_name?: string
}

// =============================================================================
// WORKFLOW TYPES
// =============================================================================

export interface WorkflowRequest {
  query: string
  workflow_id: string
  stage_name?: string
  model?: string
  data_path?: string
  web_search_enabled?: boolean
  rag_enabled?: boolean
}

export interface WorkflowMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp?: string
}

export interface WorkflowResponse {
  workflow_id: string
  stage_name: string
  status: string
  message: WorkflowMessage
  action_taken?: string
  code_executed?: string
  plots?: string[]
  summary?: string
  error?: string
  artifacts: string[]
  execution_time_seconds: number
}

export interface DataFileUploadResponse {
  success: boolean
  filename: string
  file_path: string
  file_size: number
  content_type: string
  message: string
}

export interface WorkflowStatus {
  task_id: string
  workflow_id: string
  stage_name: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  current_node?: string
  progress: number
  elapsed_seconds?: number
}

export interface Workflow {
  workflow_id: string
  created_at: string
  stage_count: number
  stages: string[]
}

export interface WorkflowCreate {
  workflow_id: string
  name: string
  description?: string
}

export interface WorkflowInfo {
  workflow_id: string
  name: string
  description?: string
  created_at: string
  updated_at: string
  stage_count: number
  has_results: boolean
}
// =============================================================================
// CHAT REQUEST TYPES (for API calls, not messages)
// =============================================================================

export interface ChatRequest {
  workflow_id: string
  message: string
  history?: Array<{ role: string; content: string }>
  model?: string
}

export interface StreamChunk {
  type: 'start' | 'token' | 'done' | 'error'
  content: string
}

// =============================================================================
// DOCUMENT TYPES
// =============================================================================

export type DocumentSourceType = 'pdf' | 'txt' | 'md' | 'csv' | 'json' | 'url'

export interface DocumentUploadResponse {
  success: boolean
  title: string
  source_type: DocumentSourceType
  chunk_count: number
  content_length: number
  workflow_id: string
  message: string
}

export interface URLScrapeRequest {
  url: string
  workflow_id: string
  title?: string
}

export interface URLScrapeResponse {
  success: boolean
  title: string
  url: string
  chunk_count: number
  content_length: number
  workflow_id: string
  message: string
}

export interface Document {
  title: string
  source_type: DocumentSourceType
  chunk_count: number
  created_at: string
}

// =============================================================================
// RAG TYPES
// =============================================================================

export interface RAGQueryRequest {
  query: string
  workflow_id: string
  n_results?: number
  doc_types?: string[]  // Optional filter by document types
}

export interface RAGChunk {
  content: string
  metadata: Record<string, unknown>
  relevance_score: number  // Match backend field name
}

export interface RAGQueryResponse {
  query: string
  results: RAGChunk[]
  total_found: number  // Match backend field name
}

export interface RAGStats {
  workflow_id: string
  total_documents: number
  total_chunks: number
  sources: string[]
}

// =============================================================================
// HEALTH TYPES
// =============================================================================

export interface HealthResponse {
  status: string
  version: string
  ollama_connected: boolean
  rag_available: boolean
  workflow_ready: boolean
}

// =============================================================================
// IMAGE TYPES
// =============================================================================

export interface ImageInfo {
  filename: string
  path: string
  stage: string
  url: string
  size_bytes: number
}

export interface StageFiles {
  stage_name: string
  images: ImageInfo[]
  data_files: string[]
  code_files: string[]
}

export interface WorkflowImages {
  workflow_id: string
  total_images: number
  images: ImageInfo[]
}