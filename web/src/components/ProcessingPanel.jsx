import { useEffect, useState } from 'react';

const STAGES = [
  ['Preparing image', 'Checking dimensions and applying the model crop.'],
  ['Extracting field marks', 'Creating a 2048-dimensional visual embedding.'],
  ['Searching the gallery', 'Looking for the nearest registered individuals.'],
  ['Applying open-set threshold', 'Deciding whether the closest match is confident enough.'],
];

export default function ProcessingPanel() {
  const [stage, setStage] = useState(0);

  useEffect(() => {
    // TODO: Replace this timer with progress from the live API if it exposes job updates.
    // In demo mode these steps are only a visual wait.
    const timer = window.setInterval(() => {
      setStage((current) => Math.min(current + 1, STAGES.length - 1));
    }, 430);

    return () => window.clearInterval(timer);
  }, []);

  return (
    <section
      className="processing-panel"
      aria-live="polite"
      aria-busy="true"
    >
      <div className="processing-orbit" aria-hidden="true">
        <span />
      </div>
      <div>
        <p className="eyebrow">Identification in progress</p>
        <h2>{STAGES[stage][0]}</h2>
        <p>{STAGES[stage][1]}</p>
        <div className="stage-track" aria-hidden="true">
          {STAGES.map((item, index) => (
            <span
              className={index <= stage ? 'stage-dot stage-dot--active' : 'stage-dot'}
              key={item[0]}
            />
          ))}
        </div>
      </div>
    </section>
  );
}
