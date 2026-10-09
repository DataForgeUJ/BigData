const inference = {
  embeddingMode: 'finetuned',
  modelVersion: 'resnet50-demo-run',
  retrievalMethod: 'hnsw',
  galleryVersion: 'field-gallery-demo',
  thresholdVersion: 'validation-demo',
};

export const DEMO_RESULTS = {
  known: {
    decision: 'KNOWN',
    predictedIdentity: 'zebra_0042',
    similarity: 0.873,
    threshold: 0.781,
    matches: [
      { rank: 1, identity: 'zebra_0042', imageId: 'gallery_1098', similarity: 0.873, imageUrl: '/gallery/zebra-study-1.svg' },
      { rank: 2, identity: 'zebra_0042', imageId: 'gallery_0831', similarity: 0.846, imageUrl: '/gallery/zebra-study-2.svg' },
      { rank: 3, identity: 'zebra_0177', imageId: 'gallery_2410', similarity: 0.724, imageUrl: '/gallery/zebra-study-3.svg' },
      { rank: 4, identity: 'zebra_0093', imageId: 'gallery_1742', similarity: 0.693, imageUrl: '/gallery/zebra-study-4.svg' },
      { rank: 5, identity: 'zebra_0311', imageId: 'gallery_3054', similarity: 0.651, imageUrl: '/gallery/zebra-study-5.svg' },
    ],
    inference,
    latencyMs: 82.7,
    isDemo: true,
  },
  unknown: {
    decision: 'UNKNOWN',
    predictedIdentity: null,
    similarity: 0.642,
    threshold: 0.781,
    matches: [
      { rank: 1, identity: 'zebra_0177', imageId: 'gallery_2410', similarity: 0.642, imageUrl: '/gallery/zebra-study-3.svg' },
      { rank: 2, identity: 'zebra_0042', imageId: 'gallery_1098', similarity: 0.614, imageUrl: '/gallery/zebra-study-1.svg' },
      { rank: 3, identity: 'zebra_0311', imageId: 'gallery_3054', similarity: 0.587, imageUrl: '/gallery/zebra-study-5.svg' },
    ],
    inference,
    latencyMs: 77.4,
    isDemo: true,
  },
  empty: {
    decision: 'UNKNOWN',
    predictedIdentity: null,
    similarity: null,
    threshold: 0.781,
    matches: [],
    inference,
    latencyMs: 68.2,
    isDemo: true,
  },
};

export const DEMO_SCENARIOS = [
  { value: 'known', label: 'Known individual' },
  { value: 'unknown', label: 'Unknown individual' },
  { value: 'empty', label: 'No candidates' },
  { value: 'unavailable', label: 'API unavailable' },
  { value: 'error', label: 'Unexpected error' },
];
