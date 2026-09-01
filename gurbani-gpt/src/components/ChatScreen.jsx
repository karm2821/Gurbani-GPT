import { useEffect, useRef } from 'react'
import InputBar from './InputBar'
import './ChatScreen.css'

export default function ChatScreen({
  messages,
  isLoading,
  citations,
  confidence,
  onSend,
  onNewChat
}) {
  const bottomRef = useRef(null)
  const chatContainerRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Enhanced formatter to format Markdown headers, blockquotes, lists, bold, italics, and Gurmukhi
  const formatContent = (content) => {
    if (!content) return null

    const paragraphs = content.split('\n')

    return paragraphs.map((line, idx) => {
      const trimmed = line.trim()
      if (!trimmed) {
        return <div key={idx} className="msg-spacer" />
      }

      // Section Headers like ###
      if (trimmed.startsWith('### ')) {
        return (
          <h3 key={idx} className="msg-heading">
            {trimmed.replace('### ', '')}
          </h3>
        )
      }
      if (trimmed.startsWith('## ')) {
        return (
          <h2 key={idx} className="msg-heading msg-heading--lg">
            {trimmed.replace('## ', '')}
          </h2>
        )
      }

      // Blockquotes
      const isQuote = trimmed.startsWith('>')
      const cleanLine = isQuote ? trimmed.replace(/^>\s*/, '') : trimmed

      // List Items
      const isBullet = cleanLine.startsWith('- ') || cleanLine.startsWith('* ')
      const textToParse = isBullet ? cleanLine.substring(2) : cleanLine

      // Check for Gurmukhi script
      const hasGurmukhi = /[\u0A00-\u0A7F]/.test(textToParse)

      // Format bold and italic markers
      const renderFormattedText = (raw) => {
        // Split by bold (**...**) first
        const boldSegments = raw.split(/(\*\*.*?\*\*)/g)
        return boldSegments.map((seg, sIdx) => {
          if (seg.startsWith('**') && seg.endsWith('**')) {
            return (
              <strong key={sIdx} className="msg-bold">
                {seg.slice(2, -2)}
              </strong>
            )
          }
          // Split by italic (*...*) within non-bold
          const italicSegments = seg.split(/(\*[^*]+?\*)/g)
          return italicSegments.map((iSeg, iIdx) => {
            if (iSeg.startsWith('*') && iSeg.endsWith('*') && iSeg.length > 2) {
              return (
                <em key={`${sIdx}-${iIdx}`} className="msg-italic">
                  {iSeg.slice(1, -1)}
                </em>
              )
            }
            return iSeg
          })
        })
      }

      if (isQuote) {
        return (
          <blockquote key={idx} className={`msg-blockquote ${hasGurmukhi ? 'msg-blockquote--gurmukhi' : ''}`}>
            {renderFormattedText(textToParse)}
          </blockquote>
        )
      }

      if (isBullet) {
        return (
          <div key={idx} className="msg-bullet-item">
            <span className="msg-bullet-dot">•</span>
            <div className="msg-bullet-text">{renderFormattedText(textToParse)}</div>
          </div>
        )
      }

      return (
        <p
          key={idx}
          className={`msg-paragraph ${hasGurmukhi ? 'msg-paragraph--gurmukhi' : ''}`}
        >
          {renderFormattedText(textToParse)}
        </p>
      )
    })
  }

  return (
    <div className="chat-screen">
      {/* Soothing animated glowing background behind chat */}
      <div className="chat-soothing-bg" aria-hidden="true">
        <div className="soothing-blob soothing-blob-1" />
        <div className="soothing-blob soothing-blob-2" />
        <div className="soothing-blob soothing-blob-3" />
        <div className="soothing-blob soothing-blob-4" />
        <div className="soothing-mesh-overlay" />
      </div>

      {/* Top Header Bar */}
      <header className="chat-header">
        <div className="chat-header__left">
          <div className="chat-avatar-mini" aria-hidden="true">
            <span className="gurmukhi">ੴ</span>
          </div>
          <div>
            <h2 className="chat-header__title">Gurbani GPT</h2>
            <span className="chat-header__status">
              <span className="chat-status-dot" />
              Grounded in SGGS
            </span>
          </div>
        </div>

        {/* Close / New Chat pill button */}
        <button
          id="close-chat-btn"
          className="chat-close-btn"
          onClick={onNewChat}
          aria-label="Close chat and start new"
        >
          <span className="chat-close-btn__icon">✕</span>
          <span>Close chat</span>
        </button>
      </header>

      {/* Message Feed Area (Centered on desktop screen) */}
      <main className="chat-feed" ref={chatContainerRef}>
        <div className="chat-feed__inner">
          {messages.map((msg) => {
            const isUser = msg.role === 'user'
            return (
              <div
                key={msg.id}
                className={`chat-message-row ${isUser ? 'chat-message-row--user' : 'chat-message-row--assistant'}`}
              >
                {!isUser && (
                  /* 3D Saffron Metallic Sphere Avatar */
                  <div className="chat-bot-orb" aria-hidden="true">
                    <div className="chat-bot-orb__inner" />
                  </div>
                )}

                <div
                  className={`chat-bubble ${isUser ? 'chat-bubble--user' : 'chat-bubble--assistant'}`}
                >
                  <div className="chat-bubble__content">
                    {formatContent(msg.content)}
                    {msg.streaming && !msg.content && (
                      <div className="chat-typing-indicator">
                        <span></span>
                        <span></span>
                        <span></span>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            )
          })}

          {/* Referenced Shabads Footer Card */}
          {citations && citations.length > 0 && (
            <div className="chat-citations-card">
              <div className="chat-citations-header">
                <span className="chat-citations-title">
                  📖 Referenced Shabads ({citations.length})
                </span>
                {confidence && (
                  <span className={`chat-confidence-badge chat-confidence-badge--${confidence.toLowerCase()}`}>
                    {confidence} Confidence
                  </span>
                )}
              </div>
              <div className="chat-citations-list">
                {citations.slice(0, 4).map((c, idx) => (
                  <div key={idx} className="chat-citation-item">
                    <span className="chat-citation-ang">Ang {c.ang}</span>
                    <span className="chat-citation-meta">
                      {c.author || 'SGGS'} · {c.raag || 'Gurbani'}
                    </span>
                    <span className={`chat-citation-tier chat-citation-tier--${(c.tier || 'SUPPORTING').toLowerCase()}`}>
                      {c.tier || 'SUPPORTING'}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div ref={bottomRef} style={{ height: '12px' }} />
        </div>
      </main>

      {/* Fixed Bottom Input Area */}
      <footer className="chat-footer">
        <InputBar
          onSend={onSend}
          placeholder="Ask a follow-up or another question..."
          disabled={isLoading}
          autoFocus
        />
      </footer>
    </div>
  )
}
