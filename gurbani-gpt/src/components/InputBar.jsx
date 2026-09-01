import { useState, useRef, useEffect } from 'react'
import './InputBar.css'

export default function InputBar({ onSend, placeholder = 'Ask Gurbani a question...', disabled = false, autoFocus = false }) {
  const [value, setValue] = useState('')
  const textareaRef = useRef(null)

  useEffect(() => {
    if (autoFocus && textareaRef.current) {
      textareaRef.current.focus()
    }
  }, [autoFocus])

  // Auto-grow textarea
  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 160) + 'px'
  }, [value])

  const submit = () => {
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setValue('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  const hasValue = value.trim().length > 0

  return (
    <div className={`input-bar ${disabled ? 'input-bar--disabled' : ''}`}>
      <div className="input-bar__inner">
        {/* Khanda icon */}
        <span className="input-bar__icon" aria-hidden="true">🙏</span>

        <textarea
          ref={textareaRef}
          id="gurbani-input"
          className="input-bar__textarea"
          value={value}
          onChange={e => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          rows={1}
          aria-label="Ask Gurbani a question"
        />

        <button
          id="send-btn"
          className={`input-bar__send ${hasValue && !disabled ? 'input-bar__send--active' : ''}`}
          onClick={submit}
          disabled={!hasValue || disabled}
          aria-label="Send message"
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="22" y1="2" x2="11" y2="13"/>
            <polygon points="22 2 15 22 11 13 2 9 22 2"/>
          </svg>
        </button>
      </div>
    </div>
  )
}
