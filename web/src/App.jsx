import { useEffect, useMemo, useState } from 'react';
import { APP_CONFIG } from './config.js';
import { makeApiClient, makeDemoClient } from './adapters/reidentification.js';
import { DEMO_SCENARIOS } from './data/demoResults.js';
import BrandMark from './components/BrandMark.jsx';
import ImageUploader from './components/ImageUploader.jsx';
import MethodNotes from './components/MethodNotes.jsx';
import ProcessingPanel from './components/ProcessingPanel.jsx';
import ResultPanel from './components/ResultPanel.jsx';
import StatusMessage from './components/StatusMessage.jsx';

export default function App() {
  const [image, setImage] = useState(null);
  const [topK, setTopK] = useState(5);
  const [scenario, setScenario] = useState('known');
  const [status, setStatus] = useState('idle');
  const [message, setMessage] = useState('');
  const [result, setResult] = useState(null);

  // Demo mode only picks a saved response. It does not call the Python model.
  const client = useMemo(
    () => APP_CONFIG.useMockApi ? makeDemoClient(scenario) : makeApiClient(APP_CONFIG.apiBaseUrl),
    [scenario],
  );

  useEffect(() => () => { if (image?.previewUrl) URL.revokeObjectURL(image.previewUrl); }, [image]);

  function updateImage(nextImage) {
    if (image?.previewUrl && image.previewUrl !== nextImage?.previewUrl) URL.revokeObjectURL(image.previewUrl);
    setImage(nextImage);
    setResult(null);
    setMessage('');
    setStatus('idle');
  }

  function handleValidationError(errorMessage) {
    setMessage(errorMessage);
    setStatus('invalid');
  }

  async function identify() {
    if (!image) {
      handleValidationError('Choose an animal image before starting identification.');
      return;
    }
    setStatus('loading');
    setResult(null);
    setMessage('');
    try {
      const response = await client.reidentify({ image: image.file, topK });
      setResult(response);
      setStatus('success');
    } catch (error) {
      setMessage(error.message);
      setStatus(error.code === 'unavailable' ? 'unavailable' : 'unexpected');
    }
  }

  function reset() {
    updateImage(null);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  return (
    <div className="app-shell">
      <header className="site-header">
        <a className="brand" href="#top" aria-label="WildlifeReID home"><BrandMark /><span><strong>WildlifeReID</strong><small>Field Station</small></span></a>
        <nav aria-label="Primary navigation"><a href="#workspace">Identify</a><a href="#methods">Methods</a></nav>
        <span className="research-tag">Honours research demo</span>
      </header>

      <main id="top">
        <section className="hero">
          <div className="hero-copy">
            <p className="eyebrow">WildlifeReID-10k · Open-set re-identification</p>
            <h1>Have we seen this <em>individual</em> before?</h1>
            <p className="hero-lede">Compare one field observation with a registered wildlife gallery. The system can suggest a known identity or flag an individual that is not confidently matched.</p>
            <a className="button button--primary" href="#workspace">Start an identification <span aria-hidden="true">↓</span></a>
          </div>
          <div className="hero-plate" aria-label="Illustration of field observation matching">
            <div className="plate-grid" aria-hidden="true" />
            <span className="plate-label">FIELD NOTE · 01</span>
            <div className="track-mark"><BrandMark /></div>
            <div className="plate-question">?</div>
            <p>One image<br />One gallery<br />An open-set decision</p>
          </div>
        </section>

        <section className="workspace" id="workspace" aria-labelledby="workspace-title">
          <div className="workspace-intro">
            <div><p className="eyebrow">Identification workspace</p><h2 id="workspace-title">Submit a field observation</h2><p>Your image stays in the current browser session while the selected adapter processes it.</p></div>
            {APP_CONFIG.useMockApi && (
              <div className="demo-controls">
                <span className="demo-badge">Demo data</span>
                <label htmlFor="demo-scenario">Preview outcome</label>
                <select id="demo-scenario" value={scenario} onChange={(event) => setScenario(event.target.value)} disabled={status === 'loading'}>
                  {DEMO_SCENARIOS.map((item) => <option value={item.value} key={item.value}>{item.label}</option>)}
                </select>
              </div>
            )}
          </div>

          <div className="workspace-grid">
            <ImageUploader image={image} onChange={updateImage} onError={handleValidationError} disabled={status === 'loading'} />
            <aside className="run-card" aria-labelledby="run-title">
              <div className="section-heading section-heading--compact"><span className="step-number">02</span><div><p className="eyebrow">Search settings</p><h2 id="run-title">Run identification</h2></div></div>
              <label className="field-label" htmlFor="top-k">Number of gallery matches</label>
              <select id="top-k" value={topK} onChange={(event) => setTopK(Number(event.target.value))} disabled={status === 'loading'}>
                {[1, 2, 3, 4, 5].map((value) => <option key={value} value={value}>Top {value}</option>)}
              </select>
              <div className="run-summary"><span>Adapter</span><strong>{APP_CONFIG.useMockApi ? 'Local demonstration' : 'WildlifeReID API'}</strong><span>Request</span><strong>Image · Top {topK}</strong></div>
              <button className="button button--primary button--wide" type="button" onClick={identify} disabled={!image || status === 'loading'}>
                {status === 'loading' ? 'Identifying…' : 'Identify animal'}
              </button>
              <p className="privacy-note">The interface sends only the selected image and Top-K value to the configured adapter.</p>
            </aside>
          </div>

          {status === 'loading' && <ProcessingPanel />}
          {status === 'invalid' && <StatusMessage kind="invalid" message={message} />}
          {status === 'unavailable' && <StatusMessage kind="unavailable" message={message} onRetry={identify} />}
          {status === 'unexpected' && <StatusMessage kind="unexpected" message={message} onRetry={identify} />}
          {status === 'success' && result && <ResultPanel result={result} queryImage={image.previewUrl} onNewImage={reset} />}
        </section>

        <MethodNotes />
      </main>

      <footer><div><BrandMark /><span>WildlifeReID-10k Field Station</span></div><p>Research demonstration · Similarity scores are not probabilities.</p></footer>
    </div>
  );
}
