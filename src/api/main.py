"""Small FastAPI entry point for the web app."""

from fastapi import FastAPI

app = FastAPI(title="Wildlife Re-ID API")

# TODO: Add POST /reidentify here.
# It needs an uploaded image and top_k, then must load the selected checkpoint,
# gallery index, metadata, and validation threshold before returning the result.
# The response must match web/src/adapters/reidentification.js.


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
