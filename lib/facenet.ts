// lib/facenet.ts — Production-safe FaceNet loader (loads once, no double-load)
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import { registerScaleSumLambda } from "./scaleSumLambdaLayer";

registerScaleSumLambda();

let modelPromise: Promise<tf.LayersModel | null> | null = null;
let lastLoadError: string | null = null;

const LOCAL_MODEL_URL = "/models/facenet/model.json";

export async function loadFaceNet(): Promise<tf.LayersModel | null> {
  if (!modelPromise) {
    modelPromise = (async () => {
      try {
        // Force GPU — FaceNet on CPU is extremely slow (minutes)
        await tf.setBackend("webgl");
        await tf.ready();
        const backend = tf.getBackend();
        console.log("TF backend:", backend);
        if (backend !== "webgl") {
          console.warn(
            "⚠️ Backend is",
            backend,
            "— inference will be very slow. WebGL not available; report this."
          );
        }

        const model = await tf.loadLayersModel(LOCAL_MODEL_URL);
        console.log("✅ FaceNet loaded (local)");
        console.log("Number of model inputs:", model.inputs.length);
        console.log("Model inputs:", model.inputs);
        if (model.inputs?.[0]) {
          console.log("Model input shape:", model.inputs[0].shape);
        }
        const weightCount = model.weights.length;
        console.log("Total weights:", weightCount);
        if (weightCount < 100) {
          console.warn(
            "⚠️ Model has very few weights (" + weightCount + "). FaceNet Inception-ResNet typically has hundreds. Model may be incomplete or corrupted."
          );
        }
        return model;
      } catch (e) {
        lastLoadError = (e as Error)?.message ?? String(e);
        console.warn("FaceNet load failed:", e);
        return null;
      }
    })();
  }
  return modelPromise;
}

export function getFaceNetLoadError(): string | null {
  return lastLoadError;
}

const EXPECTED_SHAPE = [1, 160, 160, 3] as const;
const EXPECTED_DTYPE = "float32";

export async function getEmbedding(input: tf.Tensor4D): Promise<Float32Array> {
  const model = await loadFaceNet();
  if (!model) throw new Error("FaceNet not loaded");

  // Validate input (wrong shape/dtype causes .dtype errors inside predict)
  if (!input || !(input instanceof tf.Tensor)) {
    throw new Error("getEmbedding: input is not a tensor");
  }
  const shape = input.shape;
  const dtype = input.dtype;
  console.log("Actual input shape:", shape, "Input dtype:", dtype);
  if (
    shape.length !== 4 ||
    shape[0] !== EXPECTED_SHAPE[0] ||
    shape[1] !== EXPECTED_SHAPE[1] ||
    shape[2] !== EXPECTED_SHAPE[2] ||
    shape[3] !== EXPECTED_SHAPE[3]
  ) {
    throw new Error(
      `FaceNet expects shape [1,160,160,3], got [${shape.join(",")}]`
    );
  }
  if (dtype !== EXPECTED_DTYPE) {
    throw new Error(`FaceNet expects dtype float32, got ${dtype}`);
  }

  // Run inference: predict() can throw .dtype on undefined inside graph; execute() uses a different path
  let output: tf.Tensor | tf.Tensor[];
  try {
    if (model.inputs.length === 1) {
      output = model.predict(input) as tf.Tensor | tf.Tensor[];
    } else if (model.inputs.length === 2) {
      const learningPhase = tf.scalar(0);
      try {
        output = model.predict([input, learningPhase]) as tf.Tensor | tf.Tensor[];
      } finally {
        learningPhase.dispose();
      }
    } else {
      throw new Error(`Unexpected number of model inputs: ${model.inputs.length}`);
    }
  } catch (predictErr) {
    // Fallback: execute() often works when predict() hits internal undefined in the graph
    console.warn("model.predict failed, trying model.execute:", (predictErr as Error)?.message);
    const inputName = model.inputs[0].name;
    if (!inputName) throw new Error("Model input has no name");
    const feedDict: tf.NamedTensorMap = { [inputName]: input };
    if (model.inputs.length === 2) {
      const learningPhase = tf.scalar(0);
      try {
        feedDict[model.inputs[1].name!] = learningPhase;
        const outNames = model.outputs.map((o) => o.name);
        output = model.execute(feedDict, outNames) as tf.Tensor | tf.Tensor[];
      } finally {
        learningPhase.dispose();
      }
    } else {
      const outNames = model.outputs.map((o) => o.name);
      output = model.execute(feedDict, outNames) as tf.Tensor | tf.Tensor[];
    }
  }

  if (!output) {
    throw new Error("Model returned undefined prediction");
  }

  // Handle multi-output models (e.g. Inception-ResNet)
  const tensor = Array.isArray(output) ? output[0] : output;
  if (!tensor || !(tensor instanceof tf.Tensor)) {
    throw new Error("Invalid model output");
  }

  const squeezed =
    tensor.rank === 2 ? tensor.squeeze([0]) : tensor.squeeze();
  const norm = tf.norm(squeezed);
  const normSafe = tf.clipByValue(norm, 1e-12, Infinity);
  const normalized = squeezed.div(normSafe);

  const data = await normalized.data();

  // Dispose in reverse order of creation
  normalized.dispose();
  normSafe.dispose();
  norm.dispose();
  squeezed.dispose();
  tensor.dispose();

  return new Float32Array(data);
}
