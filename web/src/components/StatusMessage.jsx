const COPY = {
  invalid: { title: 'This image cannot be used', action: 'Choose another JPEG, PNG or WebP image and try again.' },
  unavailable: { title: 'The field station service is offline', action: 'The image is safe. Try again when the API is available.' },
  unexpected: { title: 'Identification could not be completed', action: 'Try the image again. If the problem continues, check the API logs.' },
};

export default function StatusMessage({ kind = 'unexpected', message, onRetry }) {
  const content = COPY[kind] || COPY.unexpected;
  return (
    <section className={`status-message status-message--${kind}`} role="alert">
      <span className="status-icon" aria-hidden="true">!</span>
      <div>
        <h2>{content.title}</h2>
        <p>{message || content.action}</p>
        {message && <span>{content.action}</span>}
      </div>
      {onRetry && <button className="button button--secondary" type="button" onClick={onRetry}>Try again</button>}
    </section>
  );
}
