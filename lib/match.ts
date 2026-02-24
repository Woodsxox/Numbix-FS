/**
 * L2-normalize a vector. Use before storing and before matching for stable cosine distance.
 */
export function normalize(vec: Float32Array): Float32Array {
  const norm = Math.sqrt(
    Array.from(vec).reduce((sum, v) => sum + v * v, 0)
  );
  if (norm === 0) return vec;
  return new Float32Array(vec.map((v) => v / norm));
}

/**
 * Average multiple embeddings then L2-normalize. Used for multi-sample enrollment.
 */
export function averageEmbeddings(embeddings: Float32Array[]): Float32Array {
  if (embeddings.length === 0) throw new Error("Need at least one embedding");
  const length = embeddings[0].length;
  const avg = new Float32Array(length);
  for (const emb of embeddings) {
    if (emb.length !== length) throw new Error("Embedding length mismatch");
    for (let i = 0; i < length; i++) avg[i] += emb[i];
  }
  for (let i = 0; i < length; i++) avg[i] /= embeddings.length;
  return normalize(avg);
}

/**
 * Cosine distance between two embeddings.
 * 0 = identical, 1 = orthogonal, 2 = opposite.
 * For best results, pass L2-normalized vectors.
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
  