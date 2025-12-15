import type {
  Token,
  User,
  LoginRequest,
  RegisterRequest,
  WorkflowRequest,
  WorkflowResponse,
  ChatRequest,
  DocumentUploadResponse,
  URLScrapeRequest,
  URLScrapeResponse,
  RAGQueryRequest,
  RAGQueryResponse,
  RAGStats,
  HealthResponse,
  Document,
  WorkflowInfo,
  WorkflowCreate,
  DataFileUploadResponse,
  WorkflowImages,
  StageFiles,
} from '../types/api'

const API_BASE = '/api'

// =============================================================================
// TOKEN STORAGE
// =============================================================================

const TOKEN_KEY = 'auth_token'
const REFRESH_KEY = 'refresh_token'

export function getStoredToken(): string | null {
  return localStorage.getItem(TOKEN_KEY)
}

export function getStoredRefreshToken(): string | null {
  return localStorage.getItem(REFRESH_KEY)
}

export function storeTokens(token: Token): void {
  localStorage.setItem(TOKEN_KEY, token.access_token)
  if (token.refresh_token) {
    localStorage.setItem(REFRESH_KEY, token.refresh_token)
  }
}

export function clearTokens(): void {
  localStorage.removeItem(TOKEN_KEY)
  localStorage.removeItem(REFRESH_KEY)
}

// =============================================================================
// BASE FETCH WRAPPER
// =============================================================================

interface FetchOptions extends RequestInit {
  requireAuth?: boolean
}

async function apiFetch<T>(
  endpoint: string,
  options: FetchOptions = {}
): Promise<T> {
  const { requireAuth = true, ...fetchOptions } = options
  
  // 1. Initialize headers from options
  const headers: HeadersInit = {
    ...fetchOptions.headers,
  }
  
  if (requireAuth) {
    const token = getStoredToken()
    if (token) {
      (headers as Record<string, string>)['Authorization'] = `Bearer ${token}`
    }
  }
  
  // 2. CHECK if Content-Type is already set before forcing JSON
  const hasContentType = Object.keys(headers).some(k => k.toLowerCase() === 'content-type');

  if (!hasContentType && fetchOptions.body && typeof fetchOptions.body === 'string') {
    (headers as Record<string, string>)['Content-Type'] = 'application/json'
  }
  
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...fetchOptions,
    headers,
  })
  
  // Handle token refresh on 401
  if (response.status === 401 && requireAuth) {
    const refreshed = await tryRefreshToken()
    if (refreshed) {
      // Retry the request with new token
      const newToken = getStoredToken()
      if (newToken) {
        (headers as Record<string, string>)['Authorization'] = `Bearer ${newToken}`
      }
      const retryResponse = await fetch(`${API_BASE}${endpoint}`, {
        ...fetchOptions,
        headers,
      })
      if (!retryResponse.ok) {
        const error = await retryResponse.json().catch(() => ({ detail: 'Request failed' }))
        throw new Error(error.detail || 'Request failed')
      }
      return retryResponse.json()
    }
    throw new Error('Session expired. Please login again.')
  }
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Request failed' }))
    throw new Error(error.detail || `HTTP ${response.status}`)
  }
  
  // Handle empty responses
  const text = await response.text()
  return text ? JSON.parse(text) : (null as unknown as T)
}

async function tryRefreshToken(): Promise<boolean> {
  const refreshToken = getStoredRefreshToken()
  if (!refreshToken) return false
  
  try {
    const response = await fetch(`${API_BASE}/auth/refresh?refresh_token=${refreshToken}`, {
      method: 'POST',
    })
    
    if (response.ok) {
      const token: Token = await response.json()
      storeTokens(token)
      return true
    }
  } catch {
    // Refresh failed
  }
  
  clearTokens()
  return false
}

// =============================================================================
// AUTH API
// =============================================================================

export const authApi = {
  async login(credentials: LoginRequest): Promise<Token> {
    const formData = new URLSearchParams()
    formData.append('grant_type', 'password')
    formData.append('username', credentials.username)
    formData.append('password', credentials.password)
    formData.append('scope', '')
    formData.append('client_id', '')
    formData.append('client_secret', '')
    console.log('Logging in with', formData.toString())
    const token = await apiFetch<Token>('/auth/token', {
      method: 'POST',
      body: formData.toString(),
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      requireAuth: false,
    })
    
    storeTokens(token)
    return token
  },
  
  async register(data: RegisterRequest): Promise<User> {
    return apiFetch<User>('/auth/register', {
      method: 'POST',
      body: JSON.stringify(data),
      requireAuth: false,
    })
  },
  
  async getCurrentUser(): Promise<User> {
    return apiFetch<User>('/auth/me')
  },
  
  async refresh(refreshToken: string): Promise<Token> {
    const token = await apiFetch<Token>(`/auth/refresh?refresh_token=${refreshToken}`, {
      method: 'POST',
      requireAuth: false,
    })
    storeTokens(token)
    return token
  },
  
  logout(): void {
    clearTokens()
  },
}

// =============================================================================
// WORKFLOW API
// =============================================================================

export const workflowApi = {
  async run(request: WorkflowRequest): Promise<WorkflowResponse> {
    return apiFetch<WorkflowResponse>('/workflow/run', {
      method: 'POST',
      body: JSON.stringify(request),
    })
  },
  
  async getStatus(taskId: string): Promise<{ status: string; result?: WorkflowResponse }> {
    return apiFetch(`/workflow/status/${taskId}`)
  },
  
  async listWorkflows(): Promise<WorkflowInfo[]> {
    return apiFetch<WorkflowInfo[]>('/workflow/list')
  },

  async createWorkflow(data: WorkflowCreate): Promise<WorkflowInfo> {
    return apiFetch<WorkflowInfo>('/workflow/create', {
      method: 'POST',
      body: JSON.stringify(data),
    })
  },
  
  async uploadData(
    file: File,
    onProgress?: (progress: number) => void
  ): Promise<DataFileUploadResponse> {
    const formData = new FormData()
    formData.append('file', file)

    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest()

      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable && onProgress) {
          onProgress((event.loaded / event.total) * 100)
        }
      }

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(JSON.parse(xhr.responseText))
        } else {
          try {
            const error = JSON.parse(xhr.responseText)
            reject(new Error(error.detail || 'Upload failed'))
          } catch {
            reject(new Error('Upload failed'))
          }
        }
      }

      xhr.onerror = () => reject(new Error('Network error'))

      xhr.open('POST', `${API_BASE}/workflow/upload-data`)

      const token = getStoredToken()
      if (token) {
        xhr.setRequestHeader('Authorization', `Bearer ${token}`)
      }

      xhr.send(formData)
    })
  },

  async deleteData(uploadId: string): Promise<{ success: boolean; message: string }> {
    return apiFetch(`/workflow/data/${uploadId}`, {
      method: 'DELETE',
    })
  },
}

// =============================================================================
// CHAT API (Streaming)
// =============================================================================

export const chatApi = {
  streamChat(
    request: ChatRequest,
    onChunk: (chunk: { type: string; content: string }) => void,
    onError?: (error: Error) => void
  ): AbortController {
    const controller = new AbortController()
    const token = getStoredToken()
    
    fetch(`${API_BASE}/chat/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(request),
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`)
        }
        
        const reader = response.body?.getReader()
        if (!reader) throw new Error('No response body')
        
        const decoder = new TextDecoder()
        let buffer = ''
        
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          
          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() || ''
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6))
                onChunk(data)
              } catch {
                // Skip malformed JSON
              }
            }
          }
        }
      })
      .catch((error) => {
        if (error.name !== 'AbortError') {
          onError?.(error)
        }
      })
    
    return controller
  },
  
  streamAgentChat(
    request: ChatRequest,
    onChunk: (chunk: { type: string; content: string }) => void,
    onError?: (error: Error) => void
  ): AbortController {
    const controller = new AbortController()
    const token = getStoredToken()
    
    fetch(`${API_BASE}/chat/agent/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: JSON.stringify(request),
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`)
        }
        
        const reader = response.body?.getReader()
        if (!reader) throw new Error('No response body')
        
        const decoder = new TextDecoder()
        let buffer = ''
        
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          
          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() || ''
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6))
                onChunk(data)
              } catch {
                // Skip malformed JSON
              }
            }
          }
        }
      })
      .catch((error) => {
        if (error.name !== 'AbortError') {
          onError?.(error)
        }
      })
    
    return controller
  },
  
  async queryRAG(request: RAGQueryRequest): Promise<RAGQueryResponse> {
    return apiFetch<RAGQueryResponse>('/chat/query-rag', {
      method: 'POST',
      body: JSON.stringify(request),
    })
  },
  
  async getRAGStats(workflowId: string): Promise<RAGStats> {
    return apiFetch<RAGStats>(`/chat/rag-stats/${workflowId}`)
  },
}

// =============================================================================
// DOCUMENTS API
// =============================================================================

export const documentsApi = {
  async upload(
    file: File,
    workflowId: string,
    title?: string,
    onProgress?: (progress: number) => void
  ): Promise<DocumentUploadResponse> {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('workflow_id', workflowId)
    if (title) formData.append('title', title)
    
    // Use XMLHttpRequest for progress tracking
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable && onProgress) {
          onProgress((event.loaded / event.total) * 100)
        }
      }
      
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          resolve(JSON.parse(xhr.responseText))
        } else {
          try {
            const error = JSON.parse(xhr.responseText)
            reject(new Error(error.detail || 'Upload failed'))
          } catch {
            reject(new Error('Upload failed'))
          }
        }
      }
      
      xhr.onerror = () => reject(new Error('Network error'))
      
      xhr.open('POST', `${API_BASE}/documents/upload`)
      
      const token = getStoredToken()
      if (token) {
        xhr.setRequestHeader('Authorization', `Bearer ${token}`)
      }
      
      xhr.send(formData)
    })
  },
  
  async scrapeURL(request: URLScrapeRequest): Promise<URLScrapeResponse> {
    return apiFetch<URLScrapeResponse>('/documents/scrape-url', {
      method: 'POST',
      body: JSON.stringify(request),
    })
  },
  
  async list(workflowId: string): Promise<Document[]> {
    return apiFetch<Document[]>(`/documents/list/${workflowId}`)
  },
  
  async delete(workflowId: string, title: string): Promise<void> {
    await apiFetch(`/documents/${workflowId}/${encodeURIComponent(title)}`, {
      method: 'DELETE',
    })
  },
}

// =============================================================================
// HEALTH API
// =============================================================================

export const healthApi = {
  async check(): Promise<HealthResponse> {
    return apiFetch<HealthResponse>('/health', { requireAuth: false })
  },
}

// =============================================================================
// FILES API - Add to lib/api.ts
// =============================================================================

export const filesApi = {
  /**
   * List all images for a workflow
   */
  async listImages(workflowId: string, stage?: string): Promise<WorkflowImages> {
    const params = stage ? `?stage=${encodeURIComponent(stage)}` : ''
    return apiFetch<WorkflowImages>(`/files/${workflowId}/images${params}`)
  },

  /**
   * Get image URL with query parameter
   * Returns a URL that can be used to fetch the image
   */
  getImageUrl(workflowId: string, imagePath: string): string {
    return `/files/${workflowId}/image?path=${encodeURIComponent(imagePath)}`
  },

  /**
   * Fetch image as blob (for authenticated access)
   */
  async fetchImageBlob(workflowId: string, imagePath: string): Promise<Blob> {
    const token = getStoredToken()
    const response = await fetch(
      `/files/${workflowId}/image?path=${encodeURIComponent(imagePath)}`,
      {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      }
    )
    if (!response.ok) {
      throw new Error(`Failed to fetch image: ${response.statusText}`)
    }
    return response.blob()
  },

  /**
   * List all stages with their files
   */
  async listStagesWithFiles(workflowId: string): Promise<StageFiles[]> {
    return apiFetch<StageFiles[]>(`/files/${workflowId}/stages`)
  },

  /**
   * Get download URL for any file
   */
  getDownloadUrl(workflowId: string, filePath: string): string {
    return `/files/${workflowId}/download?path=${encodeURIComponent(filePath)}`
  },
}