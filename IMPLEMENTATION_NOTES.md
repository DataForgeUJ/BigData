# Implementation notes

## Current app state

The web interface is a local demonstration by default. `web/src/config.js` sets
`VITE_USE_MOCK_API` to `true` unless it is explicitly changed. `App.jsx` keeps
the selected demo scenario in React state. `web/src/adapters/reidentification.js`
returns the matching object from `web/src/data/demoResults.js` after a short
delay. The uploaded image is checked in the browser and shown as a preview, but
is not sent to Python in demo mode.

`web/src/components/ProcessingPanel.jsx` also uses a timer for its progress
steps. That is visual feedback only.

## Pipeline files

| Step | File | Reads | Writes | What is left |
| --- | --- | --- | --- | --- |
| Prepare splits | `scripts/01_prepare_data.py` | `data/raw/metadata.csv` | `data/splits/train.csv`, `validation.csv`, `test.csv` | Put every image at the path listed in the metadata, run the script, and confirm the custom validation split matches the official protocol. |
| Train model | `scripts/02_train_model.py` | split CSVs, raw images | `artifacts/checkpoints/best_resnet50.pth` and `training_checkpoint.pth` | Install Python dependencies and run training. A checkpoint does not currently exist. |
| Export embeddings | `scripts/03_extract_embeddings.py` | checkpoint, split CSVs, raw images | `data/embeddings/<mode>/*_embeddings.npy` and matching metadata CSVs | Run after a checkpoint exists. Use `--mode finetuned` for the selected trained model. |
| Build indexes | `scripts/04_build_indexes.py` | train embeddings and metadata | `artifacts/indexes/<mode>/gallery_*` | Run after embeddings exist. It saves Exact gallery data plus LSH and HNSW indexes. |
| Evaluate | `scripts/05_evaluate.py` | embeddings and built indexes | `artifacts/results/<mode>/*evaluation.csv` and JSON | Run after index building. It calibrates a threshold on validation data and evaluates test queries. |
| Analyse charts | `scripts/06_analyze.py` | evaluation CSVs | result figures and summaries | Run after at least one evaluation result exists. |
| API | `src/api/main.py` | should load selected model/index/metadata/threshold | JSON response for the UI | Only `/health` exists. `POST /reidentify` still needs implementation. |
| Web UI | `web/src/App.jsx` and `web/src/adapters/reidentification.js` | browser image and Top-K | displayed result | It is ready for an API response but defaults to demo data. |

## Connecting the UI to the API

The final request path is:

`browser image -> POST /reidentify -> image embedding -> gallery search -> threshold decision -> response -> result panel`

Implement `POST /reidentify` in `src/api/main.py`. It should accept multipart
form data named `image` and `top_k`, create one normalized embedding, search the
chosen index, apply the saved validation threshold, and return `decision`,
`predictedIdentity`, `similarity`, `threshold`, `matches`, and `inference`.

When that is working, create `web/.env.local` with:

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_USE_MOCK_API=false
```

Start the API with `uvicorn src.api.main:app --reload`. Configure CORS for the
Vite address, normally `http://localhost:5173`, or the browser will block the
request. The returned JSON field names must match the shape checked in
`web/src/adapters/reidentification.js`.

## Current limits

- The local repository has metadata but no recorded trained checkpoint, embeddings,
  indexes, or evaluation output.
- No Python runtime was available in the earlier local check, so the Python steps
  have not been verified in this workspace.
- The real API and its connection to the UI are still unfinished.
