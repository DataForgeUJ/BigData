import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import ResultPanel from './ResultPanel.jsx';
import StatusMessage from './StatusMessage.jsx';

const base = {
  similarity: 0.84,
  threshold: 0.78,
  inference: { embeddingMode: 'finetuned', retrievalMethod: 'hnsw', galleryVersion: 'demo-gallery' },
  matches: [{ rank: 1, identity: 'animal_42', imageId: 'gallery_1', similarity: 0.84 }],
};

describe('ResultPanel', () => {
  it('presents a known identity as a confirmed open-set decision', () => {
    render(<ResultPanel result={{ ...base, decision: 'KNOWN', predictedIdentity: 'animal_42' }} queryImage="query.jpg" onNewImage={() => {}} />);
    expect(screen.getByText('Known individual')).toBeInTheDocument();
    expect(screen.getAllByText('animal_42').length).toBeGreaterThan(0);
    expect(screen.getAllByText('0.840')).toHaveLength(2);
  });

  it('does not present the nearest match as a confirmed identity for unknown results', () => {
    render(<ResultPanel result={{ ...base, decision: 'UNKNOWN', predictedIdentity: null }} queryImage="query.jpg" onNewImage={() => {}} />);
    expect(screen.getByText('Unknown individual')).toBeInTheDocument();
    expect(screen.getByText('No confident identity match')).toBeInTheDocument();
    expect(screen.getByText('Nearest visual match')).toBeInTheDocument();
  });

  it('explains when the gallery returns no candidates', () => {
    render(<ResultPanel result={{ ...base, decision: 'UNKNOWN', predictedIdentity: null, similarity: null, matches: [] }} queryImage="query.jpg" onNewImage={() => {}} />);
    expect(screen.getByText('No gallery candidates')).toBeInTheDocument();
    expect(screen.getByText('No score')).toBeInTheDocument();
  });
});

describe('StatusMessage', () => {
  it('shows an API unavailable state with a retry action', () => {
    const retry = vi.fn();
    render(<StatusMessage kind="unavailable" message="The API cannot be reached." onRetry={retry} />);
    expect(screen.getByRole('alert')).toHaveTextContent('The field station service is offline');
    fireEvent.click(screen.getByRole('button', { name: 'Try again' }));
    expect(retry).toHaveBeenCalledOnce();
  });
});
