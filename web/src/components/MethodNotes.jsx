const METHODS = [
  ['Exact', 'Checks the query against every gallery embedding. It provides the reference nearest-neighbour result.'],
  ['LSH', 'Uses random hyperplane hash buckets to shortlist visually similar candidates before ranking them.'],
  ['HNSW', 'Navigates a layered neighbour graph to find close embeddings without scanning the full gallery.'],
];

export default function MethodNotes() {
  return (
    <section className="methods-section" id="methods" aria-labelledby="methods-title">
      <div><p className="eyebrow">Retrieval notes</p><h2 id="methods-title">Three ways to search the field gallery</h2></div>
      <div className="method-grid">
        {METHODS.map(([name, description], index) => <article key={name}><span>0{index + 1}</span><h3>{name}</h3><p>{description}</p></article>)}
      </div>
    </section>
  );
}
