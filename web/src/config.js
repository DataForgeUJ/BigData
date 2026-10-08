export const APP_CONFIG = {
  apiBaseUrl: String(import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/$/, ''),
  useMockApi: String(import.meta.env.VITE_USE_MOCK_API ?? 'true').toLowerCase() !== 'false',
};
