import { DEMO_RESULTS } from '../data/demoResults.js';

export class RequestError extends Error {
  constructor(message, code = 'unexpected') {
    super(message);
    this.name = 'ReidentificationError';
    this.code = code;
  }
}

function tidyResult(payload) {
  if (!payload || !['KNOWN', 'UNKNOWN'].includes(payload.decision)) {
    throw new RequestError('The service returned a response we could not understand.');
  }

  return {
    ...payload,
    predictedIdentity: payload.predictedIdentity ?? null,
    similarity: payload.similarity ?? null,
    threshold: payload.threshold ?? null,
    matches: Array.isArray(payload.matches) ? payload.matches : [],
    inference: payload.inference || {},
  };
}

function wait(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

export function makeDemoClient(scenario = 'known') {
  return {
    async reidentify({ topK = 5 }) {
      await wait(1650);
      if (scenario === 'unavailable') {
        throw new RequestError('The identification service is currently unavailable.', 'unavailable');
      }
      if (scenario === 'error') {
        throw new RequestError('Something unexpected happened while processing the image.');
      }

      const result = DEMO_RESULTS[scenario] || DEMO_RESULTS.known;
      return tidyResult({ ...result, matches: result.matches.slice(0, topK) });
    },
  };
}

// TODO: Keep this response shape aligned with POST /reidentify in src/api/main.py.
// The API still needs to run model inference and return real gallery matches.
export function makeApiClient(apiBaseUrl) {
  return {
    async reidentify({ image, topK = 5 }) {
      const formData = new FormData();
      formData.append('image', image);
      formData.append('top_k', String(topK));

      let response;
      try {
        response = await fetch(`${apiBaseUrl}/reidentify`, { method: 'POST', body: formData });
      } catch {
        throw new RequestError('The identification service is currently unavailable.', 'unavailable');
      }

      let payload;
      try {
        payload = await response.json();
      } catch {
        throw new RequestError('The service returned an unreadable response.');
      }

      if (!response.ok) {
        throw new RequestError(payload.detail || payload.error || 'The image could not be processed.');
      }
      return tidyResult(payload);
    },
  };
}
