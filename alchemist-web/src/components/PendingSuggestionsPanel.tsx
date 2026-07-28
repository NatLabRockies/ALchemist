// PendingSuggestionsPanel.tsx - show and manage staged suggestions
import { useState } from 'react'
import AddPointDialog from './AddPointDialog'
import { addExperiment } from './api'
import { useVariables } from '../hooks/api/useVariables'

export default function PendingSuggestionsPanel({ sessionId, pending, onRemove, onAdded }:
  { sessionId: string, pending: Array<any>, onRemove: (idx:number)=>void, onAdded?: (resp:any)=>void }) {
  const [dialogOpen, setDialogOpen] = useState(false)
  const [currentSuggestion, setCurrentSuggestion] = useState<any>(null)
  const [currentIndex, setCurrentIndex] = useState<number | null>(null)

  const { data: variablesData } = useVariables(sessionId)
  const variables = variablesData?.variables ?? []

  if (!pending || pending.length === 0) return (
    <div className="pending-panel empty">No pending suggestions</div>
  )

  async function handleAddPoint(s:any, idx:number) {
    setCurrentSuggestion(s)
    setCurrentIndex(idx)
    setDialogOpen(true)
  }

  async function onDialogConfirm(payload: any, options: { saveToFile: boolean, retrain: boolean }) {
    // payload matches API shape: { inputs: {...}, output?: number, noise?: number }
    try {
      const resp = await addExperiment(sessionId, payload, options.retrain)
      onAdded && onAdded(resp)
      // remove the suggestion from pending list in parent
      if (currentIndex !== null) onRemove(currentIndex)
      // Optionally, handle saveToFile by notifying the user (desktop auto-saves to CSV)
      if (options.saveToFile) {
        // Placeholder: frontend currently has no experiment CSV path; inform user
        console.log('Save to file requested (frontend placeholder)')
      }
    } catch (e: any) {
      console.error('Failed to add point', e)
      alert('Failed to add point: '+(e?.message || String(e)))
    } finally {
      setDialogOpen(false)
      setCurrentSuggestion(null)
      setCurrentIndex(null)
    }
  }

  return (
    <div className="pending-panel">
      <h4>Pending Suggestions ({pending.length})</h4>
      <ul>
        {pending.map((s, i) => (
          <li key={i}>
            <div className="suggestion-summary">
              {Object.entries(s).filter(([k])=>!k.startsWith('_')).slice(0,4).map(([k,v])=> (
                <span key={k}>{k}: {String(v)}</span>
              ))}
            </div>
            <div className="suggestion-actions">
              <button onClick={()=>handleAddPoint(s, i)}>Add Point</button>
              <button onClick={()=>onRemove(i)}>Remove</button>
            </div>
          </li>
        ))}
      </ul>

      {dialogOpen && currentSuggestion && (
        <AddPointDialog
          key={currentIndex ?? 0}
          variables={variables}
          suggestion={currentSuggestion}
          index={currentIndex ?? 0}
          total={pending.length}
          onCancel={() => setDialogOpen(false)}
          onConfirm={onDialogConfirm}
          onPrev={currentIndex !== null && currentIndex > 0 ? () => {
            const newIndex = (currentIndex ?? 0) - 1
            setCurrentIndex(newIndex)
            setCurrentSuggestion(pending[newIndex])
          } : undefined}
          onNext={currentIndex !== null && currentIndex < pending.length - 1 ? () => {
            const newIndex = (currentIndex ?? 0) + 1
            setCurrentIndex(newIndex)
            setCurrentSuggestion(pending[newIndex])
          } : undefined}
        />
      )}
    </div>
  )
}
