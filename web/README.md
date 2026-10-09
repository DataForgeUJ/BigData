# WildlifeReID Field Station

A React/Vite demonstration interface for the WildlifeReID-10k open-set re-identification workflow. The app runs with labelled demo data by default, so it does not require a checkpoint, gallery, or API server.

## Run locally

```bash
cd web
npm install
npm run dev
```

Open the local URL printed by Vite (normally `http://localhost:5173`).

## Checks

```bash
npm run lint
npm run test
npm run build
```

## Demo modes

The workspace includes scenarios for a known individual, an unknown individual, no gallery candidates, an unavailable API, and an unexpected error. Every simulated result is visibly labelled **Demo data**.

## Connect the real API

Copy `.env.example` to `.env.local`, then configure:

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_USE_MOCK_API=false
```

The real adapter sends `POST /reidentify` as `multipart/form-data` with an `image` file and `top_k` value. Both mock and real modes implement the same adapter interface in `src/adapters/reidentification.js`, so the UI does not need to change when the backend is connected.

The proposed response contract is intentionally defensive: `predictedIdentity` may be `null`, `similarity` may be `null`, and `matches` may be empty. Backend integration should confirm final field names, error shapes, CORS policy, threshold metadata, and gallery-image delivery before production use.
