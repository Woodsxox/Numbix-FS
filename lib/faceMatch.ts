export function cosineSimilarity(
  a: Float32Array,
  b: Float32Array
): number {
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

  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * Threshold guide:
 * 0.6   = very strict
 * 0.65  = recommended (FaceNet)
 * 0.7+  = loose (risky)
 */
export function isFaceMatch(
  similarity: number,
  threshold = 0.65
): boolean {
  return similarity >= threshold;
}
