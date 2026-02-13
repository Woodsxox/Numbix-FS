"use client";

import { useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import * as faceDetection from "@tensorflow-models/face-detection";

import { cropAndNormalizeFace } from "@/lib/faceCrop";
import { loadFaceNet, getEmbedding, getFaceNetLoadError } from "@/lib/facenet";

type Status =
  | "initializing"
  | "loading_model"
  | "hold_still"
  | "capturing"
  | "saved"
  | "error";

type FaceBox = {
  xMin: number;
  yMin: number;
  width: number;
  height: number;
};

export default function FaceEnrollment({
  onComplete,
}: {
  onComplete?: () => void;
} = {}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const detectorRef = useRef<faceDetection.FaceDetector | null>(null);
  const rafRef = useRef<number | null>(null);

  const embeddingsRef = useRef<Float32Array[]>([]);
  const stableFramesRef = useRef(0);
  const mountedRef = useRef(true);

  const REQUIRED_FRAMES = 90;
  const REQUIRED_SAMPLES = 5;

  const [status, setStatus] = useState<Status>("initializing");
  const [samplesCaptured, setSamplesCaptured] = useState(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  /* ------------------ Setup ------------------ */
  /** Returns true if ready to run detection, false if setup failed (e.g. FaceNet missing). */
  async function setup(): Promise<boolean> {
    if (!videoRef.current) return false;

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
    });

    if (!mountedRef.current) {
      stream.getTracks().forEach((t) => t.stop());
      return false;
    }

    videoRef.current.srcObject = stream;
    try {
      await videoRef.current.play();
    } catch (err) {
      const isAbort = (err as Error)?.name === "AbortError";
      if (isAbort) {
        stream.getTracks().forEach((t) => t.stop());
        return false;
      }
      throw err;
    }
    if (!mountedRef.current) return false;

    await tf.setBackend("webgl");
    await tf.ready();

    detectorRef.current = await faceDetection.createDetector(
      faceDetection.SupportedModels.MediaPipeFaceDetector,
      { runtime: "tfjs", maxFaces: 1 }
    );

    const faceNet = await loadFaceNet();
    if (!mountedRef.current) return false;
    if (!faceNet) {
      const loadErr = getFaceNetLoadError();
      const hint = loadErr?.includes("Lambda")
        ? " Your model uses an unsupported Lambda layer. Use a TF.js–compatible model (see docs/FACENET_SETUP.md)."
        : " Use a model with modelTopology + weightsManifest in public/models/facenet/, or set NEXT_PUBLIC_FACENET_GITHUB_URL to a working model.json URL.";
      setErrorMessage(
        `FaceNet model could not be loaded.${hint}`
      );
      setStatus("error");
      return false;
    }
    setStatus("hold_still");
    return true;
  }

  /* ------------------ Detection Loop ------------------ */
  async function detect() {
    if (!videoRef.current || !detectorRef.current) return;

    const faces = await detectorRef.current.estimateFaces(videoRef.current);

    if (faces.length === 0) {
      stableFramesRef.current = 0;
    } else {
      stableFramesRef.current++;

      if (stableFramesRef.current >= REQUIRED_FRAMES) {
        await captureEmbedding(faces[0].box);
        stableFramesRef.current = 0;
      }
    }

    rafRef.current = requestAnimationFrame(detect);
  }

  /* ------------------ Capture Embedding ------------------ */
  async function captureEmbedding(box: FaceBox) {
    if (!videoRef.current) return;

    setStatus("capturing");

    const faceTensor = cropAndNormalizeFace(videoRef.current, box);
    const embedding = await getEmbedding(faceTensor);

    faceTensor.dispose();
    embeddingsRef.current.push(embedding);
    setSamplesCaptured(embeddingsRef.current.length);

    console.log(
      `🧬 Captured embedding ${embeddingsRef.current.length}/${REQUIRED_SAMPLES}`
    );

    if (embeddingsRef.current.length >= REQUIRED_SAMPLES) {
      finalizeEnrollment();
    }
  }

  /* ------------------ Finalize ------------------ */
  function finalizeEnrollment() {
    const averaged = averageEmbeddings(embeddingsRef.current);

    console.log("✅ Final face embedding:", averaged);

    // TODO: send to backend
    // fetch("/api/face/enroll", {
    //   method: "POST",
    //   body: JSON.stringify({ embedding: Array.from(averaged) }),
    // });

    cleanup();
    setStatus("saved");
    onComplete?.();
  }

  /* ------------------ Utils ------------------ */
  function averageEmbeddings(vectors: Float32Array[]): Float32Array {
    const length = vectors[0].length;
    const result = new Float32Array(length);

    for (const vec of vectors) {
      for (let i = 0; i < length; i++) {
        result[i] += vec[i];
      }
    }

    for (let i = 0; i < length; i++) {
      result[i] /= vectors.length;
    }

    return result;
  }

  function cleanup() {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);

    if (videoRef.current?.srcObject) {
      (videoRef.current.srcObject as MediaStream)
        .getTracks()
        .forEach((t) => t.stop());
    }
  }

  /* ------------------ Init ------------------ */
  useEffect(() => {
    mountedRef.current = true;
    setup().then((ok) => {
      if (mountedRef.current && ok) detect();
    }).catch((err) => {
      if (mountedRef.current) {
        setErrorMessage(err?.message ?? "Setup failed");
        setStatus("error");
      }
    });

    return () => {
      mountedRef.current = false;
      cleanup();
    };
  }, []);

  /* ------------------ UI ------------------ */
  return (
    <div className="space-y-2">
      <video ref={videoRef} autoPlay muted playsInline className="rounded-xl" style={{ transform: "scaleX(-1)" }} />

      <p className="text-sm text-gray-600">
        {status === "initializing" && "Initializing…"}
        {status === "hold_still" && "Hold still…"}
        {status === "capturing" &&
          `Capturing face (${samplesCaptured}/${REQUIRED_SAMPLES})`}
        {status === "saved" && "Enrollment complete ✅"}
        {status === "error" && (
          <>
            {errorMessage ?? "Something went wrong ❌"}
          </>
        )}
      </p>
    </div>
  );
}
