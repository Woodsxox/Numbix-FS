/**
 * Cosine distance between two embeddings.
 * 0 = identical, 1 = orthogonal, 2 = opposite.
 * Threshold: < 0.5 same person, 0.5–0.7 maybe, > 0.8 different.
 */
export function cosineDistance(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new Error("Embedding length mismatch");
  }
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const norm = Math.sqrt(normA) * Math.sqrt(normB);
  return norm === 0 ? 1 : 1 - dot / norm;
}
  