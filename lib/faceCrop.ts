import * as tf from "@tensorflow/tfjs";

export type FaceBox = {
  xMin: number;
  yMin: number;
  width: number;
  height: number;
};

/** Normalize detector box to { xMin, yMin, width, height } with valid numbers. */
function normalizeBox(
  box: Partial<FaceBox> & { xMax?: number; yMax?: number },
  frameHeight: number,
  frameWidth: number
): { x: number; y: number; w: number; h: number } {
  let xMin = Number(box.xMin);
  let yMin = Number(box.yMin);
  let w = Number(box.width);
  let h = Number(box.height);
  if (!Number.isFinite(w) && Number.isFinite(box.xMax)) {
    w = Number(box.xMax) - xMin;
  }
  if (!Number.isFinite(h) && Number.isFinite(box.yMax)) {
    h = Number(box.yMax) - yMin;
  }
  if (!Number.isFinite(xMin)) xMin = 0;
  if (!Number.isFinite(yMin)) yMin = 0;
  if (!Number.isFinite(w) || w <= 0) w = 100;
  if (!Number.isFinite(h) || h <= 0) h = 100;
  const x = Math.max(0, Math.min(xMin, frameWidth - 1));
  const y = Math.max(0, Math.min(yMin, frameHeight - 1));
  const maxW = frameWidth - x;
  const maxH = frameHeight - y;
  return {
    x,
    y,
    w: Math.max(1, Math.min(w, maxW)),
    h: Math.max(1, Math.min(h, maxH)),
  };
}

export type FaceInput = HTMLVideoElement | HTMLCanvasElement;

/**
 * Crop face from video or canvas frame and normalize for FaceNet
 * Output shape: [1, 160, 160, 3]
 */
export function cropAndNormalizeFace(
  input: FaceInput,
  box: Partial<FaceBox> & { xMax?: number; yMax?: number }
): tf.Tensor4D {
  return tf.tidy(() => {
    const width = "videoWidth" in input ? input.videoWidth : input.width;
    const height = "videoHeight" in input ? input.videoHeight : input.height;
    if (!width || !height) {
      throw new Error("Input has no dimensions yet");
    }
    const frame = tf.browser.fromPixels(input);
    const shape = frame.shape;
    if (!shape || shape.length < 2) {
      throw new Error("Invalid frame shape");
    }
    const frameHeight = shape[0];
    const frameWidth = shape[1];
    const { x, y, w, h } = normalizeBox(box, frameHeight, frameWidth);

    // Crop face
    const face = tf.slice(frame, [y, x, 0], [h, w, 3]);

    // Resize to FaceNet expected size
    const resized = tf.image.resizeBilinear(face, [160, 160]);

    // Normalize to [-1, 1]
    const normalized = resized
      .toFloat()
      .div(127.5)
      .sub(1);

    // Add batch dimension → [1, 160, 160, 3]
    return normalized.expandDims(0) as tf.Tensor4D;
  });
}
