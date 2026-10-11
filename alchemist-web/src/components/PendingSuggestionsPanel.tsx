// PendingSuggestionsPanel.tsx - show and manage staged suggestions
import { useState } from 'react'
import AddPointDialog from './AddPointDialog'
import { useVariables } from '../hooks/api/useVariables'
import { useAddPointBatch } from '../features/experiments/useAddPointBatch'

/**
 * List of staged suggestions with the shared Add Point dialog. Opening a row
 * starts the dialog at that point; the dialog steps through all of them and
 * saves every entered point in one request (see AddPointDialog).
 */
export default function PendingSuggestionsPanel({ sessionId, pending, onRemove, onPendingChange }:
  { sessionId: string, pending: Array<any>, onRemove: (idx:number)=>void, onPendingChange?: (remaining: any[])=>void }) {
  const [openAt, setOpenAt] = useState<number | null>(null)

  const { data: variablesData } = useVariables(sessionId)
  const variables = variablesData?.variables ?? []
  const addPoint = useAddPointBatch(sessionId, pending, onPendingChange)

  if (!pending || pending.length === 0) return (
    <div className="pending-panel empty">No pending suggestions</div>
  )

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
              <button onClick={()=>setOpenAt(i)}>Add Point</button>
              <button onClick={()=>onRemove(i)}>Remove</button>
            </div>
          </li>
        ))}
      </ul>

      {openAt !== null && (
        <AddPointDialog
          variables={variables}
          suggestions={pending}
          initialIndex={openAt}
          entries={addPoint.entries}
          onEntriesChange={addPoint.onEntriesChange}
          onCancel={() => setOpenAt(null)}
          onSave={async (plan, options) => {
            await addPoint.save(plan, options)
            setOpenAt(null)
          }}
        />
      )}
    </div>
  )
}
