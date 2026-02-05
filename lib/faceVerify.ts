/**
 * Compute cosine similarity between two embeddings
 * Range: -1 → 1 (higher = more similar)
 */
export function cosineSimilarity(
  a: Float32Array,
  b: Float32Array
): number {
  if (a.length !== b.length) {
    throw new Error("Embedding lengths do not match");
  }

  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * Face verification decision
 */
export function verifyFace(
  liveEmbedding: Float32Array,
  storedEmbedding: Float32Array,
  threshold = 0.75
) {
  const score = cosineSimilarity(liveEmbedding, storedEmbedding);

  return {
    match: score >= threshold,
    score,
  };
}
