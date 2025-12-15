import { useState, useEffect } from 'react'
import { useChatStore, selectCurrentWorkflowId } from '../stores/chatStore'
import { workflowApi } from '../lib/api'
import type { WorkflowInfo, WorkflowCreate } from '../types/api'
import {
  FolderOpen,
  RefreshCw,
  Check,
  Layers,
  Plus,
  X,
  Search,
} from 'lucide-react'
import clsx from 'clsx'

// =============================================================================
// CREATE WORKFLOW MODAL
// =============================================================================

interface CreateWorkflowModalProps {
  isOpen: boolean
  onClose: () => void
  onCreated: (workflow: WorkflowInfo) => void
}

function CreateWorkflowModal({ isOpen, onClose, onCreated }: CreateWorkflowModalProps) {
  const [workflowId, setWorkflowId] = useState('')
  const [name, setName] = useState('')
  const [description, setDescription] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Auto-generate workflow_id from name
  const handleNameChange = (value: string) => {
    setName(value)
    // Convert name to slug-style ID
    const slug = value
      .toLowerCase()
      .replace(/[^a-z0-9\s-]/g, '')
      .replace(/\s+/g, '-')
      .substring(0, 64)
    setWorkflowId(slug)
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!workflowId.trim() || !name.trim()) return

    setLoading(true)
    setError(null)

    try {
      const workflow = await workflowApi.createWorkflow({
        workflow_id: workflowId.trim(),
        name: name.trim(),
        description: description.trim() || undefined,
      })
      onCreated(workflow)
      onClose()
      // Reset form
      setWorkflowId('')
      setName('')
      setDescription('')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create workflow')
    } finally {
      setLoading(false)
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
      <div className="bg-white rounded-xl shadow-xl w-full max-w-md mx-4">
        <div className="flex items-center justify-between p-4 border-b">
          <h2 className="text-lg font-semibold">Create Workflow</h2>
          <button
            onClick={onClose}
            className="p-1 hover:bg-gray-100 rounded"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-4 space-y-4">
          {error && (
            <div className="p-3 bg-red-50 text-red-700 rounded-lg text-sm">
              {error}
            </div>
          )}

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Name
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => handleNameChange(e.target.value)}
              placeholder="Q4 Sales Analysis"
              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              required
              maxLength={128}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              ID
            </label>
            <input
              type="text"
              value={workflowId}
              onChange={(e) => setWorkflowId(e.target.value.toLowerCase().replace(/[^a-z0-9_-]/g, ''))}
              placeholder="q4-sales-analysis"
              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent font-mono text-sm"
              required
              maxLength={64}
              pattern="[a-zA-Z0-9_-]+"
            />
            <p className="mt-1 text-xs text-gray-500">
              Letters, numbers, hyphens, and underscores only
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Description (optional)
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Analysis of Q4 2024 sales data..."
              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent resize-none"
              rows={2}
              maxLength={512}
            />
          </div>

          <div className="flex gap-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 px-4 py-2 border rounded-lg hover:bg-gray-50"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading || !workflowId.trim() || !name.trim()}
              className="flex-1 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Creating...' : 'Create'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

// =============================================================================
// WORKFLOWS PAGE
// =============================================================================

export function WorkflowsPage() {
  const workflowId = useChatStore(selectCurrentWorkflowId)
  const setWorkflowId = useChatStore((state) => state.setWorkflowId)
  const clearAllMessages = useChatStore((state) => state.clearAllMessages)

  const [workflows, setWorkflows] = useState<WorkflowInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [search, setSearch] = useState('')
  const [showCreate, setShowCreate] = useState(false)

  const fetchWorkflows = async () => {
    setLoading(true)
    try {
      const data = await workflowApi.listWorkflows()
      setWorkflows(data)
    } catch {
      setWorkflows([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchWorkflows()
  }, [])

  const filteredWorkflows = workflows.filter(
    (w) =>
      w.workflow_id.toLowerCase().includes(search.toLowerCase()) ||
      w.name.toLowerCase().includes(search.toLowerCase())
  )

  const handleLoadWorkflow = (id: string) => {
    console.log('Loading workflow:', id)
    setWorkflowId(id)
    clearAllMessages()
  }

  const handleWorkflowCreated = (workflow: WorkflowInfo) => {
    setWorkflows((prev) => [workflow, ...prev])
    // Optionally auto-load the new workflow
    handleLoadWorkflow(workflow.workflow_id)
  }

  const totalStages = workflows.reduce((sum, w) => sum + w.stage_count, 0)

  return (
    <div className="p-8 max-w-4xl mx-auto overflow-scroll h-full min-h-screen sm:w-full">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">📚 Workflows</h1>
          <p className="mt-1 text-gray-600">
            Browse and manage your analysis workflows
          </p>
        </div>
        <button
          onClick={() => setShowCreate(true)}
          className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
        >
          <Plus className="w-4 h-4" />
          New Workflow
        </button>
      </div>

      {/* Summary */}
      <div className="grid grid-cols-3 gap-4 mb-8">
        <div className="metric-card">
          <div className="text-3xl font-bold mb-1">{workflows.length}</div>
          <div className="text-sm opacity-90">Total Workflows</div>
        </div>
        <div className="metric-card">
          <div className="text-3xl font-bold mb-1">{totalStages}</div>
          <div className="text-sm opacity-90">Total Stages</div>
        </div>
        <div className="metric-card">
          <div className="text-3xl font-bold mb-1">
            {workflows.filter((w) => w.has_results).length}
          </div>
          <div className="text-sm opacity-90">With Results</div>
        </div>
      </div>

      {/* Search & Refresh */}
      <div className="flex gap-4 mb-6">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search workflows..."
            className="w-full pl-10 pr-4 py-2 border rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          />
        </div>
        <button
          onClick={fetchWorkflows}
          disabled={loading}
          className="px-4 py-2 border rounded-lg hover:bg-gray-50 disabled:opacity-50"
        >
          <RefreshCw className={clsx('w-4 h-4', loading && 'animate-spin')} />
        </button>
      </div>

      {/* Workflow List */}
      {loading ? (
        <div className="text-center py-12 text-gray-500">Loading workflows...</div>
      ) : filteredWorkflows.length === 0 ? (
        <div className="text-center py-12">
          <FolderOpen className="w-12 h-12 mx-auto text-gray-300 mb-4" />
          <p className="text-gray-500">
            {search ? 'No workflows match your search' : 'No workflows yet'}
          </p>
          {!search && (
            <button
              onClick={() => setShowCreate(true)}
              className="mt-4 text-purple-600 hover:underline"
            >
              Create your first workflow
            </button>
          )}
        </div>
      ) : (
        <div className="space-y-3">
          {filteredWorkflows.map((w) => (
            <div
              key={w.workflow_id}
              className={clsx(
                'p-4 border rounded-xl hover:border-purple-300 transition-colors cursor-pointer',
                workflowId === w.workflow_id && 'border-purple-500 bg-purple-50'
              )}
              onClick={() => handleLoadWorkflow(w.workflow_id)}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <h3 className="font-medium text-gray-900 truncate">
                      {w.name}
                    </h3>
                    {workflowId === w.workflow_id && (
                      <span className="flex items-center gap-1 text-xs text-purple-600 bg-purple-100 px-2 py-0.5 rounded-full">
                        <Check className="w-3 h-3" />
                        Active
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-gray-500 font-mono">{w.workflow_id}</p>
                  {w.description && (
                    <p className="mt-1 text-sm text-gray-600 line-clamp-2">
                      {w.description}
                    </p>
                  )}
                </div>
                <div className="flex items-center gap-4 text-sm text-gray-500 ml-4">
                  <div className="flex items-center gap-1">
                    <Layers className="w-4 h-4" />
                    {w.stage_count}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Create Modal */}
      <CreateWorkflowModal
        isOpen={showCreate}
        onClose={() => setShowCreate(false)}
        onCreated={handleWorkflowCreated}
      />
    </div>
  )
}