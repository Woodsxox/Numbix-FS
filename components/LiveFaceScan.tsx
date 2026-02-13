"use client";

import { useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";

type Status =
  | "initializing"
  | "loading_model"
  | "no_face"
  | "face_detected"
  | "blink_detected"
  | "error";

type Point = { x: number; y: number };

const LEFT_EYE = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE = [263, 387, 385, 362, 380, 373];

const distance = (a: Point, b: Point) =>
  Math.hypot(a.x - b.x, a.y - b.y);

const eyeAspectRatio = (eye: Point[]) => {
  const v1 = distance(eye[1], eye[5]);
  const v2 = distance(eye[2], eye[4]);
  const h = distance(eye[0], eye[3]);
  return (v1 + v2) / (2 * h);
};

export default function LiveFaceScan() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const detectorRef = useRef<any>(null);
  const rafRef = useRef<number | null>(null);
  const blinkedRef = useRef(false);

  const [status, setStatus] = useState<Status>("initializing");

  const BLINK_THRESHOLD = 0.18;

  async function setupCamera() {
    if (!videoRef.current) return;

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
    });

    videoRef.current.srcObject = stream;
    await videoRef.current.play();
  }

  async function loadModel() {
    if (typeof window === "undefined") return;

    setStatus("loading_model");

    await tf.setBackend("webgl");
    await tf.ready();

    const faceLandmarks = await import(
      "@tensorflow-models/face-landmarks-detection"
    );

    detectorRef.current = await faceLandmarks.createDetector(
      faceLandmarks.SupportedModels.MediaPipeFaceMesh,
      {
        runtime: "tfjs",
        refineLandmarks: true,
        maxFaces: 1,
      }
    );
  }

  async function detect() {
    if (!videoRef.current || !detectorRef.current) return;

    try {
      const faces = await detectorRef.current.estimateFaces(
        videoRef.current,
        { flipHorizontal: false }
      );

      if (!faces.length) {
        setStatus("no_face");
      } else {
        setStatus("face_detected");

        const keypoints = faces[0].keypoints;

        const leftEye = LEFT_EYE.map(i => ({
          x: keypoints[i].x,
          y: keypoints[i].y,
        }));

        const rightEye = RIGHT_EYE.map(i => ({
          x: keypoints[i].x,
          y: keypoints[i].y,
        }));

        const ear =
          (eyeAspectRatio(leftEye) + eyeAspectRatio(rightEye)) / 2;

        if (ear < BLINK_THRESHOLD && !blinkedRef.current) {
          blinkedRef.current = true;
          setStatus("blink_detected");
        }

        if (ear >= BLINK_THRESHOLD && blinkedRef.current) {
          blinkedRef.current = false;
        }
      }
    } catch (e) {
      console.error(e);
      setStatus("error");
    }

    rafRef.current = requestAnimationFrame(detect);
  }

  useEffect(() => {
    let mounted = true;

    (async () => {
      try {
        await setupCamera();
        if (!mounted) return;
        await loadModel();
        detect();
      } catch {
        setStatus("error");
      }
    })();

    return () => {
      mounted = false;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      const tracks = (videoRef.current?.srcObject as MediaStream)?.getTracks();
      tracks?.forEach(t => t.stop());
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
        style={{ transform: "scaleX(-1)" }}
      />

      <p className="text-sm text-gray-500">
        {status === "initializing" && "Initializing…"}
        {status === "loading_model" && "Loading face mesh…"}
        {status === "no_face" && "No face detected"}
        {status === "face_detected" && "Face detected"}
        {status === "blink_detected" && "Blink detected ✅ Liveness confirmed"}
        {status === "error" && "Something went wrong ❌"}
      </p>
    </div>
  );
}
