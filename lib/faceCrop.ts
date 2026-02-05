import * as tf from "@tensorflow/tfjs";

type FaceBox = {
  xMin: number;
  yMin: number;
  width: number;
  height: number;
};

/**
 * Crop face from video frame and normalize for FaceNet
 * Output shape: [1, 160, 160, 3]
 */
export function cropAndNormalizeFace(
  video: HTMLVideoElement,
  box: FaceBox
): tf.Tensor4D {
  return tf.tidy(() => {
    // Capture frame
    const frame = tf.browser.fromPixels(video);

    // Clamp values for safety
    const x = Math.max(0, box.xMin);
    const y = Math.max(0, box.yMin);
    const w = Math.min(frame.shape[1] - x, box.width);
    const h = Math.min(frame.shape[0] - y, box.height);

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
