"""FastAPI entry point for inference and retrieval."""

from fastapi import FastAPI

app = FastAPI(title="Wildlife Re-ID API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
