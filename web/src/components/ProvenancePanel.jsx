function readable(value) {
  return value ? String(value).replace(/[_-]/g, ' ') : 'Not provided';
}

export default function ProvenancePanel({ result }) {
  const inference = result.inference || {};
  const values = [
    ['Embedding', readable(inference.embeddingMode)],
    ['Retrieval', readable(inference.retrievalMethod)],
    ['Gallery', readable(inference.galleryVersion)],
    ['Threshold', result.threshold == null ? 'Not available' : result.threshold.toFixed(3)],
  ];

  return (
    <details className="provenance-panel">
      <summary>
        <span>How this result was produced</span>
        <span className="summary-hint">Method details</span>
      </summary>
      <div className="provenance-content">
        <dl>
          {values.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}
        </dl>
        <p><strong>Similarity is not a probability.</strong> It measures how close two image embeddings are. The threshold is the decision boundary selected during validation.</p>
        {inference.modelVersion && <p className="run-note">Model version: {inference.modelVersion} · Threshold version: {readable(inference.thresholdVersion)}</p>}
      </div>
    </details>
  );
}
