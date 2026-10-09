import { useRef, useState } from 'react';
import { validateImage } from '../utils/imageValidation.js';

export default function ImageUploader({
  image,
  onChange,
  onError,
  disabled = false,
}) {
  const inputRef = useRef(null);
  const [dragging, setDragging] = useState(false);

  async function acceptFile(file) {
    try {
      const validated = await validateImage(file);
      onError('');
      onChange(validated);
    } catch (error) {
      onError(error.message);
    }
  }

  function handleInput(event) {
    const [file] = event.target.files;

    if (file) {
      acceptFile(file);
    }

    event.target.value = '';
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragging(false);

    if (disabled) {
      return;
    }

    const [file] = event.dataTransfer.files;

    if (file) {
      acceptFile(file);
    }
  }

  function startDragging(event) {
    event.preventDefault();
    setDragging(true);
  }

  return (
    <section className="upload-card" aria-labelledby="upload-title">
      <div className="section-heading">
        <span className="step-number">01</span>
        <div>
          <p className="eyebrow">Observation image</p>
          <h2 id="upload-title">Choose one animal photograph</h2>
        </div>
      </div>

      <input
        ref={inputRef}
        id="animal-image"
        className="visually-hidden"
        type="file"
        accept="image/jpeg,image/png,image/webp"
        onChange={handleInput}
        disabled={disabled}
      />

      {image ? (
        <div className="preview-shell">
          <div className="preview-frame">
            <img
              src={image.previewUrl}
              alt="Selected animal observation"
            />
            <span className="corner corner--tl" aria-hidden="true" />
            <span className="corner corner--br" aria-hidden="true" />
          </div>

          <div className="preview-meta">
            <div>
              <strong>{image.file.name}</strong>
              <span>
                {image.width} × {image.height}px ·{' '}
                {(image.file.size / 1024 / 1024).toFixed(2)} MB
              </span>
            </div>

            <div className="preview-actions">
              <button
                className="button button--secondary"
                type="button"
                onClick={() => inputRef.current?.click()}
                disabled={disabled}
              >
                Replace
              </button>
              <button
                className="text-button"
                type="button"
                onClick={() => onChange(null)}
                disabled={disabled}
              >
                Remove
              </button>
            </div>
          </div>
        </div>
      ) : (
        <label
          className={`dropzone ${dragging ? 'dropzone--active' : ''}`}
          htmlFor="animal-image"
          onDragEnter={startDragging}
          onDragOver={(event) => event.preventDefault()}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
        >
          <svg viewBox="0 0 48 48" aria-hidden="true">
            <path
              d="M8 35.5 18 24l7 7 5-6 10 10.5M13 13h22a5 5 0 0 1 5 5v17a5 5 0 0 1-5 5H13a5 5 0 0 1-5-5V18a5 5 0 0 1 5-5Zm15 8a3 3 0 1 0 6 0 3 3 0 0 0-6 0Z"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <strong>Drop an image here or browse</strong>
          <span>JPEG, PNG or WebP · up to 10 MB</span>
        </label>
      )}
    </section>
  );
}
