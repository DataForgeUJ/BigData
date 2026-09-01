# Scalable Open-Set Wildlife Re-Identification

Implementation scaffold for comparing exhaustive cosine search, random-hyperplane LSH, and HNSW on WildlifeReID-10k.

## Project stages

1. Prepare the official WildlifeReID-10k train/validation/test protocol.
2. Fine-tune an ImageNet-pretrained ResNet-50 with metric learning.
3. Freeze the model and extract gallery/query embeddings.
4. Implement exhaustive cosine search, LSH, and HNSW.
5. Select ANN parameters and the open-set threshold using validation data only.
6. Evaluate at 25%, 50%, 75%, and 100% gallery sizes.
7. Report identification, open-set, retrieval-quality, latency, throughput, build-time, and memory metrics.
8. Expose the chosen pipeline through a Python REST API and simple web UI.

## Structure

```text
configs/                 Experiment configuration files
data/
  raw/                   Original dataset (not committed)
  processed/             Processed metadata/images if needed
  splits/                Official/derived split metadata
  embeddings/            Frozen gallery/query embeddings
notebooks/               Exploration only; production logic belongs in src/
src/
  data/                   Dataset loading and preprocessing
  models/                 ResNet-50 embedding model and losses
  retrieval/              Exact cosine, LSH, HNSW
  evaluation/             Closed/open-set and efficiency metrics
  api/                    REST API
  utils/                  Shared utilities
web/
  templates/              Web UI templates
  static/                 CSS/JS/assets
scripts/                  Runnable pipeline entry points
tests/                    Unit/integration tests
artifacts/
  checkpoints/            Trained weights
  indexes/                LSH/HNSW indexes
  results/                CSV/JSON experiment outputs
  logs/                   Training/evaluation logs
docs/                     Project documentation
```

## Initial workflow

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\\Scripts\\activate     # Windows PowerShell
pip install -r requirements.txt
```

The dataset itself should not be committed to Git. Put downloaded source data under `data/raw/` and retain the benchmark's official split metadata.

## Next implementation milestone

Start with dataset inspection and a reproducible split/data-loader pipeline before training the model. Do not implement ANN tuning against the test set; validation data should be used for LSH/HNSW parameters and the open-set similarity threshold.
