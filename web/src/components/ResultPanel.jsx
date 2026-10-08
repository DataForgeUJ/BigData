import ProvenancePanel from './ProvenancePanel.jsx';

function score(value) {
  return value == null ? 'No score' : value.toFixed(3);
}

function MatchCard({ match, decision }) {
  return (
    <article className="match-card">
      <div className="match-image">
        <img
          src={match.imageUrl || '/gallery/field-placeholder.svg'}
          alt={`Gallery observation ${match.imageId || match.rank}`}
        />
        <span>#{match.rank}</span>
      </div>

      <div className="match-copy">
        <p>
          {decision === 'UNKNOWN' ? 'Nearest visual match' : 'Gallery match'}
        </p>
        <strong>{match.identity || 'Unlabelled individual'}</strong>
        <span>{match.imageId || 'Image reference unavailable'}</span>
      </div>

      <div className="match-score">
        <span>Similarity</span>
        <strong>{score(match.similarity)}</strong>
      </div>
    </article>
  );
}

export default function ResultPanel({ result, queryImage, onNewImage }) {
  const known = result.decision === 'KNOWN';
  const hasMatches = result.matches.length > 0;

  return (
    <section
      className="result-section"
      aria-live="polite"
      data-testid="result-panel"
    >
      <div
        className={`decision-panel decision-panel--${known ? 'known' : 'unknown'}`}
      >
        <div className="decision-query">
          <img src={queryImage} alt="Submitted animal observation" />
          {result.isDemo && (
            <span className="demo-ribbon">Demo data</span>
          )}
        </div>

        <div className="decision-copy">
          <p className="eyebrow">Open-set decision</p>
          <div className="decision-label">
            <span aria-hidden="true">{known ? '✓' : '?'}</span>
            {known ? 'Known individual' : 'Unknown individual'}
          </div>
          <h2>{known ? result.predictedIdentity : 'No confident identity match'}</h2>
          <p>
            {known
              ? 'The strongest gallery match is above the active decision threshold.'
              : hasMatches
                ? 'The nearest images are shown for visual context, but none should be treated as a confirmed identity.'
                : 'The gallery returned no candidate images for this observation.'}
          </p>

          <div className="decision-metrics">
            <div>
              <span>Top similarity</span>
              <strong>{score(result.similarity)}</strong>
            </div>
            <div>
              <span>Threshold</span>
              <strong>{score(result.threshold)}</strong>
            </div>
          </div>

          <button
            className="button button--secondary"
            type="button"
            onClick={onNewImage}
          >
            Identify another image
          </button>
        </div>
      </div>

      <div className="matches-section">
        <div className="section-heading section-heading--compact">
          <span className="step-number">03</span>
          <div>
            <p className="eyebrow">Gallery review</p>
            <h2>
              {hasMatches
                ? `Ranked matches · ${result.matches.length}`
                : 'No candidates returned'}
            </h2>
          </div>
        </div>

        {hasMatches ? (
          <div className="match-list">
            {result.matches.map((match) => (
              <MatchCard
                key={`${match.rank}-${match.imageId}`}
                match={match}
                decision={result.decision}
              />
            ))}
          </div>
        ) : (
          <div className="no-candidates">
            <span aria-hidden="true">∅</span>
            <div>
              <strong>No gallery candidates</strong>
              <p>
                The search completed, but no nearby observations were available
                to rank.
              </p>
            </div>
          </div>
        )}
      </div>

      <ProvenancePanel result={result} />
    </section>
  );
}
