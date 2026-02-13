"use client";

import { useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import * as faceDetection from "@tensorflow-models/face-detection";
import type { FaceBox } from "@/lib/faceCrop";

type Status = "initializing" | "hold_still" | "captured" | "error";

const REQUIRED_FRAMES = 90;

export default function FaceCapture({
  onCaptured,
}: {
  onCaptured: (canvas: HTMLCanvasElement, box: FaceBox) => void;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const detectorRef = useRef<faceDetection.FaceDetector | null>(null);
  const rafRef = useRef<number | null>(null);
  const stableFramesRef = useRef(0);
  const mountedRef = useRef(true);

  const [status, setStatus] = useState<Status>("initializing");

  useEffect(() => {
    mountedRef.current = true;

    async function setup() {
      if (!videoRef.current) return;

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      });

      if (!mountedRef.current) {
        stream.getTracks().forEach((t) => t.stop());
        return;
      }

      videoRef.current.srcObject = stream;
      try {
        await videoRef.current.play();
      } catch (err) {
        if ((err as Error)?.name !== "AbortError") throw err;
      }
      if (!mountedRef.current) return;

      await tf.setBackend("webgl");
      await tf.ready();

      detectorRef.current = await faceDetection.createDetector(
        faceDetection.SupportedModels.MediaPipeFaceDetector,
        { runtime: "tfjs", maxFaces: 1 }
      );

      if (!mountedRef.current) return;
      setStatus("hold_still");
      detect();
    }

    async function detect() {
      if (!videoRef.current || !detectorRef.current) return;

      const faces = await detectorRef.current.estimateFaces(videoRef.current);

      if (faces.length === 0) {
        stableFramesRef.current = 0;
      } else {
        stableFramesRef.current++;
        if (stableFramesRef.current >= REQUIRED_FRAMES) {
          if (!rafRef.current) return;
          cancelAnimationFrame(rafRef.current);
          rafRef.current = null;

          const stream = videoRef.current.srcObject as MediaStream;
          stream?.getTracks().forEach((t) => t.stop());

          const canvas = document.createElement("canvas");
          canvas.width = videoRef.current.videoWidth;
          canvas.height = videoRef.current.videoHeight;
          const ctx = canvas.getContext("2d");
          if (ctx) ctx.drawImage(videoRef.current, 0, 0);

          const box = faces[0].box as FaceBox;
          if (mountedRef.current) {
            setStatus("captured");
            onCaptured(canvas, box);
          }
          return;
        }
      }

      rafRef.current = requestAnimationFrame(detect);
    }

    setup();

    return () => {
      mountedRef.current = false;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      const stream = videoRef.current?.srcObject as MediaStream;
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, [onCaptured]);

  return (
    <div className="space-y-2">
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        className="rounded-xl w-full max-w-full h-auto"
        style={{ transform: "scaleX(-1)" }}
      />
      <p className="text-sm text-gray-600">
        {status === "initializing" && "Initializing…"}
        {status === "hold_still" && "Hold still…"}
        {status === "captured" && "Face captured"}
        {status === "error" && "Error"}
      </p>
    </div>
  );
}
