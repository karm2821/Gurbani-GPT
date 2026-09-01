import { useState, useCallback, useRef } from 'react'
import HomeScreen from './components/HomeScreen'
import ChatScreen from './components/ChatScreen'
import './App.css'

export default function App() {
  const [screen, setScreen]       = useState('home') // 'home' | 'chat'
  const [messages, setMessages]   = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [citations, setCitations] = useState([])
  const [confidence, setConfidence] = useState(null)
  const abortRef = useRef(null)

  const sendMessage = useCallback(async (query) => {
    if (!query.trim() || isLoading) return

    const userMsg = { role: 'user', content: query, id: Date.now() }

    // Build history (last 6 messages, exclude system)
    const history = messages
      .filter(m => m.role !== 'system')
      .slice(-6)
      .map(m => ({ role: m.role, content: m.content }))

    // Switch directly to chat
    if (screen === 'home') {
      setScreen('chat')
    }

    const assistantId = Date.now() + 1
    const assistantMsg = { role: 'assistant', content: '', id: assistantId, streaming: true }

    setMessages(prev => [...prev, userMsg, assistantMsg])
    setIsLoading(true)
    setCitations([])
    setConfidence(null)

    // Abort any in-flight request
    if (abortRef.current) abortRef.current.abort()
    const ctrl = new AbortController()
    abortRef.current = ctrl

    try {
      const resp = await fetch('/api/gurbani-chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, history, top_k: 8 }),
        signal: ctrl.signal,
      })

      if (!resp.ok) throw new Error(`Server error ${resp.status}`)

      const reader = resp.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() // keep incomplete line

        for (const line of lines) {
          if (!line.trim()) continue
          try {
            const chunk = JSON.parse(line)

            // Citations metadata chunk
            if (chunk.type === 'citations') {
              setCitations(chunk.citations || [])
              setConfidence(chunk.confidence || null)
              continue
            }

            // Error chunk from backend/API
            if (chunk.error) {
              setMessages(prev =>
                prev.map(m =>
                  m.id === assistantId
                    ? { ...m, content: (m.content ? m.content + '\n\n' : '') + '⚠️ ' + chunk.error }
                    : m
                )
              )
              continue
            }

            // LLM token chunk
            const token = chunk?.message?.content ?? chunk?.response ?? ''
            if (token) {
              setMessages(prev =>
                prev.map(m =>
                  m.id === assistantId
                    ? { ...m, content: m.content + token }
                    : m
                )
              )
            }

            if (chunk.done) break

          } catch {
            // Non-JSON line — ignore
          }
        }
      }

      // Mark streaming complete
      setMessages(prev =>
        prev.map(m => m.id === assistantId ? { ...m, streaming: false } : m)
      )
    } catch (err) {
      if (err.name === 'AbortError') return
      setMessages(prev =>
        prev.map(m =>
          m.id === assistantId
            ? { ...m, content: '⚠️ Connection error. Please check that the server is running.', streaming: false }
            : m
        )
      )
    } finally {
      setIsLoading(false)
    }
  }, [isLoading, messages, screen])

  const handleNewChat = useCallback(() => {
    if (abortRef.current) abortRef.current.abort()
    setMessages([])
    setCitations([])
    setConfidence(null)
    setIsLoading(false)
    setScreen('home')
  }, [])

  return (
    <div className="app-root">
      {/* Global Ambient Background */}
      <div className="ambient-bg" aria-hidden="true">
        <div className="ambient-orb ambient-orb-1" />
        <div className="ambient-orb ambient-orb-2" />
      </div>

      {screen === 'home' && (
        <HomeScreen onSend={sendMessage} />
      )}
      {screen === 'chat' && (
        <ChatScreen
          messages={messages}
          isLoading={isLoading}
          citations={citations}
          confidence={confidence}
          onSend={sendMessage}
          onNewChat={handleNewChat}
        />
      )}
    </div>
  )
}
