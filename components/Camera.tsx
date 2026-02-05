"use client";

import { useEffect, useRef, useState } from "react";
import * as faceDetection from "@tensorflow-models/face-detection";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";

type FaceStatus =
  | "initializing"
  | "camera_ready"
  | "loading_model"
  | "no_face"
  | "face_detected"
  | "error";

export default function Camera() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const detectorRef = useRef<faceDetection.FaceDetector | null>(null);
  const rafRef = useRef<number | null>(null);

  const [status, setStatus] = useState<FaceStatus>("initializing");

  const safeSetStatus = (next: FaceStatus) => {
    setStatus((prev) => (prev === next ? prev : next));
  };

  async function setupCamera(): Promise<void> {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 640 },
        height: { ideal: 480 },
      },
    });

    if (!videoRef.current) return;

    videoRef.current.srcObject = stream;
    await videoRef.current.play();

    safeSetStatus("camera_ready");
  }

  async function loadDetector(): Promise<void> {
    safeSetStatus("loading_model");

    await tf.setBackend("webgl");
    await tf.ready();

    detectorRef.current = await faceDetection.createDetector(
      faceDetection.SupportedModels.MediaPipeFaceDetector,
      {
        runtime: "tfjs",
        maxFaces: 1,
      }
    );

    safeSetStatus("no_face");
  }

  async function detect(): Promise<void> {
    if (!videoRef.current || !detectorRef.current) return;

    try {
      const faces = await detectorRef.current.estimateFaces(
        videoRef.current
      );

      if (faces.length > 0) {
        safeSetStatus("face_detected");
      } else {
        safeSetStatus("no_face");
      }
    } catch (err) {
      console.error("Detection error:", err);
      safeSetStatus("error");
    }

    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(detect);
  }

  useEffect(() => {
    async function init() {
      try {
        await setupCamera();
        await loadDetector();
        detect();
      } catch (err) {
        console.error("Camera init failed:", err);
        safeSetStatus("error");
      }
    }

    init();

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);

      if (videoRef.current?.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach((t) => t.stop());
      }
    };
  }, []);

  return (
    <div className="space-y-2">
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        className="w-full rounded-xl bg-black"
      />

      <p className="text-sm text-gray-500">
        {status === "initializing" && "Initializing…"}
        {status === "camera_ready" && "Camera ready"}
        {status === "loading_model" && "Loading face model…"}
        {status === "no_face" && "No face detected"}
        {status === "face_detected" && "Face detected ✅"}
        {status === "error" && "Something went wrong ❌"}
      </p>
    </div>
  );
}
