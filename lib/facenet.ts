// lib/facenet.ts
import * as tf from "@tensorflow/tfjs";

type FaceNetModel = tf.GraphModel | tf.LayersModel;
let model: FaceNetModel | null = null;

const LOCAL_MODEL_URL = "/models/facenet/model.json";

/**
 * Optional GitHub fallback: set to the full URL of model.json in your repo.
 * Example: https://raw.githubusercontent.com/USER/REPO/main/public/models/facenet/model.json
 * Weight shards must be in the same directory (same base URL). Fetching from GitHub is safe
 * (model files only, no script execution).
 */
const GITHUB_FALLBACK_URL =
  typeof process !== "undefined" && process.env?.NEXT_PUBLIC_FACENET_GITHUB_URL
    ? process.env.NEXT_PUBLIC_FACENET_GITHUB_URL
    : "";

/**
 * Load FaceNet for 128-D embeddings.
 * Tries local first, then optional GitHub fallback (no external CDN).
 */
export async function loadFaceNet(): Promise<FaceNetModel | null> {
  if (model) return model;

  try {
    model = await tf.loadLayersModel(LOCAL_MODEL_URL);
    console.log("✅ FaceNet loaded (local)");
    return model;
  } catch (e1) {
    console.warn("Local FaceNet load failed:", e1);
  }

  if (GITHUB_FALLBACK_URL) {
    try {
      model = await tf.loadLayersModel(GITHUB_FALLBACK_URL);
      console.log("✅ FaceNet loaded (GitHub fallback)");
      return model;
    } catch (e2) {
      console.warn("GitHub FaceNet load failed:", e2);
    }
  }

  console.error("FaceNet load failed (no remote URL set).");
  return null;
}

export async function getEmbedding(
  face: tf.Tensor
): Promise<number[]> {
  if (!model) throw new Error("FaceNet not loaded. Run loadFaceNet() first.");

  const face4D = face as tf.Tensor4D;

  const embeddingTensor = tf.tidy(() => {
    const pred = model!.predict(face4D) as tf.Tensor;
    const squeezed = pred.rank === 1 ? pred : pred.squeeze();
    return squeezed;
  });

  const embedding = Array.from(await embeddingTensor.data());
  embeddingTensor.dispose();

  return embedding;
}
