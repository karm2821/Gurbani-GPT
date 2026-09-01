import { useState } from 'react'
import InputBar from './InputBar'
import './HomeScreen.css'

const QUICK_PROMPTS = [
  { id: 'peace',   icon: '🕊️', label: 'Finding inner peace', query: 'How can I find inner peace according to Gurbani?' },
  { id: 'anger',   icon: '🔥', label: 'Overcoming anger',   query: 'What does Gurbani teach about controlling and overcoming anger?' },
  { id: 'grief',   icon: '🌿', label: 'Dealing with grief', query: 'I lost someone close to me. How does Gurbani guide us through grief?' },
  { id: 'anxiety', icon: '🌊', label: 'Calming anxiety',    query: 'I cannot stop worrying about my future. How to calm anxiety through Gurbani?' },
  { id: 'purpose', icon: '✨', label: 'Purpose of life',    query: 'What is the true purpose of human life according to Gurbani?' },
]

export default function HomeScreen({ onSend }) {
  const [isOrbHovered, setIsOrbHovered] = useState(false)

  return (
    <div className="home-screen">
      {/* Top Bar Logo & Brand */}
      <header className="home-header">
        <div className="home-logo">
          <span className="home-logo__symbol gurmukhi">ੴ</span>
        </div>
        <div className="home-header__info">
          <span className="home-header__title">Gurbani GPT</span>
          <span className="home-header__sub">Sri Guru Granth Sahib Ji</span>
        </div>
      </header>

      {/* Main Center Area with Hero Title & Saffron Orb */}
      <main className="home-main">
        {/* Main Heading */}
        <div className="home-headline">
          <div className="home-headline__badge gurmukhi">
            ਵਾਹਿਗੁਰੂ ਜੀ ਕਾ ਖ਼ਾਲਸਾ, ਵਾਹਿਗੁਰੂ ਜੀ ਕੀ ਫ਼ਤਿਹ 🙏
          </div>
          <h1 className="home-headline__text">
            What are you <span className="home-headline__highlight">seeking today?</span>
          </h1>
          <p className="home-headline__sub">
            Spiritual wisdom, emotional grounding & divine guidance from Sri Guru Granth Sahib Ji
          </p>
        </div>

        {/* Central 3D Glowing Saffron Orb (Centered on screen above input) */}
        <div 
          className={`home-orb-wrapper ${isOrbHovered ? 'home-orb-wrapper--hover' : ''}`}
          onMouseEnter={() => setIsOrbHovered(true)}
          onMouseLeave={() => setIsOrbHovered(false)}
          aria-hidden="true"
        >
          <div className="home-orb-glow" />
          <div className="home-orb-ring home-orb-ring--outer" />
          <div className="home-orb-ring home-orb-ring--inner" />
          <div className="home-orb-sphere">
            <div className="home-orb-specular" />
            <div className="home-orb-center-light" />
          </div>
        </div>

        {/* Quick Suggestion Pills */}
        <div className="home-quick-pills" role="list" aria-label="Suggested topics">
          {QUICK_PROMPTS.map((p) => (
            <button
              key={p.id}
              className="quick-pill"
              onClick={() => onSend(p.query)}
              role="listitem"
              aria-label={p.label}
            >
              <span className="quick-pill__icon">{p.icon}</span>
              <span className="quick-pill__label">{p.label}</span>
            </button>
          ))}
        </div>
      </main>

      {/* Bottom Input Area */}
      <footer className="home-footer">
        <InputBar
          onSend={onSend}
          placeholder="Ask Gurbani anything or describe your situation..."
          autoFocus
        />
        <p className="home-footer__note">
          Grounded in Sri Guru Granth Sahib Ji · 842 Shabads indexed
        </p>
      </footer>
    </div>
  )
}
