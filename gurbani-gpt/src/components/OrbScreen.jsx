import './OrbScreen.css'

export default function OrbScreen({ query }) {
  return (
    <div className="orb-screen">
      {/* Glowing orb */}
      <div className="orb-container" aria-hidden="true">
        <div className="orb-outer-ring" />
        <div className="orb-mid-ring" />
        <div className="orb-sphere">
          <div className="orb-highlight" />
        </div>
        <div className="orb-glow" />
      </div>

      {/* Query text below orb */}
      {query && (
        <p className="orb-query">
          {query.length > 120 ? query.slice(0, 120) + '...' : query}
        </p>
      )}

      <p className="orb-status">Seeking Gurbani wisdom...</p>
    </div>
  )
}
