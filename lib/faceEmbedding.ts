"use client";

/**
 * Face embedding via @vladmandic/human (TFJS-native, no Keras conversion).
 * Replaces the broken FaceNet graph with a stable, production-ready embedding model.
 * Next.js is configured to resolve @vladmandic/human to the ESM browser build (see next.config).
 */
import Human from "@vladmandic/human";

let human: Human | null = null;

export async function loadEmbeddingModel(): Promise<Human> {
  if (human) return human;

  human = new Human({
    modelBasePath: "https://vladmandic.github.io/human/models",
    face: {
      enabled: true,
      detector: { rotation: true },
      description: { enabled: true },
    },
    body: { enabled: false },
    hand: { enabled: false },
    object: { enabled: false },
    gesture: { enabled: false },
  });

  await human.load();
  console.log("✅ Human embedding model loaded");
  return human;
}

export type FaceEmbeddingInput = HTMLVideoElement | HTMLCanvasElement | HTMLImageElement;

/**
 * Run face detection + embedding on a video frame, canvas, or image.
 * Returns the first face's embedding as Float32Array (length 128 or 512 depending on model).
 */
export async function getEmbedding(input: FaceEmbeddingInput): Promise<Float32Array> {
  const h = await loadEmbeddingModel();
  const result = await h.detect(input);

  if (!result.face || result.face.length === 0) {
    throw new Error("No face detected");
  }

  const embedding = result.face[0].embedding;
  if (!embedding || !Array.isArray(embedding) || embedding.length === 0) {
    throw new Error("Face embedding not available (enable face.description in Human config)");
  }

  return new Float32Array(embedding);
}
