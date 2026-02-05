"use client";

import { useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import * as faceDetection from "@tensorflow-models/face-detection";

import { cropAndNormalizeFace } from "@/lib/faceCrop";
import { loadFaceNet, getEmbedding } from "@/lib/facenet";
import { cosineSimilarity, isFaceMatch } from "@/lib/faceMatch";

type FaceBox = {
  xMin: number;
  yMin: number;
  width: number;
  height: number;
};

/**
 * ⚠️ TEMP
 * Replace this with:
 * - backend fetch
 * - localStorage
 * - database response
 */
const storedEmbedding = new Float32Array(128); // ← load real one later

type Status =
  | "initializing"
  | "hold_still"
  | "verifying"
  | "match"
  | "no_match"
  | "error";

export default function FaceVerify() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const detectorRef = useRef<faceDetection.FaceDetector | null>(null);
  const rafRef = useRef<number | null>(null);

  const stableFramesRef = useRef(0);
  const REQUIRED_FRAMES = 60;

  const [status, setStatus] = useState<Status>("initializing");
  const [score, setScore] = useState<number | null>(null);

  /* ------------------ Setup ------------------ */
  async function setup() {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
    });

    if (!videoRef.current) return;

    videoRef.current.srcObject = stream;
    await videoRef.current.play();

    await tf.setBackend("webgl");
    await tf.ready();

    detectorRef.current = await faceDetection.createDetector(
      faceDetection.SupportedModels.MediaPipeFaceDetector,
      { runtime: "tfjs", maxFaces: 1 }
    );

    await loadFaceNet();
    setStatus("hold_still");
  }

  /* ------------------ Detection ------------------ */
  async function detect() {
    if (!videoRef.current || !detectorRef.current) return;

    const faces = await detectorRef.current.estimateFaces(videoRef.current);

    if (faces.length === 0) {
      stableFramesRef.current = 0;
    } else {
      stableFramesRef.current++;

      if (stableFramesRef.current >= REQUIRED_FRAMES) {
        await verify(faces[0].box);
        return;
      }
    }

    rafRef.current = requestAnimationFrame(detect);
  }

  /* ------------------ Verify ------------------ */
  async function verify(box: FaceBox) {
    try {
      setStatus("verifying");

      if (!videoRef.current) return;

      const faceTensor = cropAndNormalizeFace(videoRef.current, box);
      const rawEmbedding = await getEmbedding(faceTensor);
      faceTensor.dispose();

      const liveEmbedding = new Float32Array(rawEmbedding);
      const similarity = cosineSimilarity(storedEmbedding, liveEmbedding);

      setScore(similarity);

      if (isFaceMatch(similarity, 0.65)) {
        setStatus("match");
      } else {
        setStatus("no_match");
      }
    } catch (err) {
      console.error(err);
      setStatus("error");
    }
  }

  /* ------------------ Init ------------------ */
  useEffect(() => {
    setup().then(detect);

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      if (videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream)
          .getTracks()
          .forEach((t) => t.stop());
      }
    };
  }, []);

  /* ------------------ UI ------------------ */
  return (
    <div className="space-y-2">
      <video ref={videoRef} autoPlay muted playsInline className="rounded-xl" />

      <p className="text-sm text-gray-600">
        {status === "initializing" && "Initializing…"}
        {status === "hold_still" && "Hold still…"}
        {status === "verifying" && "Verifying identity…"}
        {status === "match" && "✅ Identity confirmed"}
        {status === "no_match" && "❌ Face not recognized"}
        {status === "error" && "Something went wrong"}
      </p>

      {score !== null && (
        <p className="text-xs text-gray-400">
          Similarity score: {score.toFixed(3)}
        </p>
      )}
    </div>
  );
}
