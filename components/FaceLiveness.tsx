"use client";

import { useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import * as faceLandmarks from "@tensorflow-models/face-landmarks-detection";

/* -------------------------
   Types
--------------------------*/
type Status = "initializing" | "challenge" | "passed" | "error";
type Challenge = "blink" | "turn_left" | "turn_right";

type Point = {
  x: number;
  y: number;
};

/* -------------------------
   Landmarks (MediaPipe)
--------------------------*/
const LEFT_EYE = [33, 160, 158, 133, 153, 144];
const RIGHT_EYE = [263, 387, 385, 362, 380, 373];
const NOSE = 1;
const LEFT_CHEEK = 234;
const RIGHT_CHEEK = 454;

const CHALLENGES: Challenge[] = ["blink", "turn_left", "turn_right"];

/* -------------------------
   Utils
--------------------------*/
const distance = (a: Point, b: Point) =>
  Math.hypot(a.x - b.x, a.y - b.y);

const eyeAspectRatio = (eye: Point[]) => {
  const v1 = distance(eye[1], eye[5]);
  const v2 = distance(eye[2], eye[4]);
  const h = distance(eye[0], eye[3]);
  return (v1 + v2) / (2 * h);
};

/* -------------------------
   Component
--------------------------*/
export default function FaceLiveness({
  onPassed,
}: {
  onPassed?: () => void;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const detectorRef =
    useRef<faceLandmarks.FaceLandmarksDetector | null>(null);
  const rafRef = useRef<number | null>(null);

  const DETECT_WIDTH = 640;
  const DETECT_HEIGHT = 480;

  const blinkedRef = useRef(false);
  const challengeIndexRef = useRef(0);
  const turnHoldFramesRef = useRef(0);

  const [status, setStatus] = useState<Status>("initializing");
  const [current, setCurrent] = useState<Challenge>(CHALLENGES[0]);
  const [earValue, setEarValue] = useState<number | null>(null);
  const [faceDetected, setFaceDetected] = useState(false);

  const BLINK_THRESHOLD = 0.26;
  const TURN_THRESHOLD = 0.15;
  const TURN_HOLD_FRAMES = 25;
  const mountedRef = useRef(true);
  const earUpdateRef = useRef(0);

  /* -------------------------
     Setup
  --------------------------*/
  async function setup() {
    if (!videoRef.current) return;

    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
    });

    if (!mountedRef.current) {
      stream.getTracks().forEach((t) => t.stop());
      return;
    }

    const video = videoRef.current;
    video.srcObject = stream;

    await new Promise<void>((resolve, reject) => {
      const onReady = () => {
        video.removeEventListener("loadeddata", onReady);
        video.removeEventListener("error", onError);
        resolve();
      };
      const onError = (e: Event) => {
        video.removeEventListener("loadeddata", onReady);
        video.removeEventListener("error", onError);
        reject(e);
      };
      video.addEventListener("loadeddata", onReady, { once: true });
      video.addEventListener("error", onError, { once: true });
      if (video.readyState >= 2) onReady();
    });

    try {
      await video.play();
    } catch (err) {
      if ((err as Error)?.name !== "AbortError") throw err;
    }

    if (!mountedRef.current) return;

    await tf.setBackend("webgl");
    await tf.ready();
    if (!mountedRef.current) return;

    detectorRef.current = await faceLandmarks.createDetector(
      faceLandmarks.SupportedModels.MediaPipeFaceMesh,
      { runtime: "tfjs", refineLandmarks: false, maxFaces: 1 }
    );

    if (!mountedRef.current) return;

    setCurrent(CHALLENGES[0]);
    setStatus("challenge");

    for (let i = 0; i < 30 && mountedRef.current; i++) {
      await new Promise((r) => setTimeout(r, 100));
      if (video.videoWidth > 0 && video.videoHeight > 0) break;
    }
    await new Promise((r) => setTimeout(r, 500));
    if (!mountedRef.current) return;
    detect();
  }

  /* -------------------------
     Detection Loop
  --------------------------*/
  async function detect() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !detectorRef.current) return;

    if (video.readyState < 2 || video.videoWidth === 0) {
      rafRef.current = requestAnimationFrame(detect);
      return;
    }

    if (!canvas) {
      rafRef.current = requestAnimationFrame(detect);
      return;
    }

    const ctx = canvas.getContext("2d");
    if (ctx) {
      canvas.width = DETECT_WIDTH;
      canvas.height = DETECT_HEIGHT;
      ctx.drawImage(video, 0, 0, DETECT_WIDTH, DETECT_HEIGHT);
    }

    let faces: Awaited<ReturnType<faceLandmarks.FaceLandmarksDetector["estimateFaces"]>> = [];
    try {
      faces = await detectorRef.current.estimateFaces(
        canvas,
        { flipHorizontal: false, staticImageMode: true }
      );
    } catch (err) {
      console.warn("estimateFaces error:", err);
      rafRef.current = requestAnimationFrame(detect);
      return;
    }

    if (!faces.length) {
      setEarValue(null);
      setFaceDetected(false);
      rafRef.current = requestAnimationFrame(detect);
      return;
    }

    setFaceDetected(true);
    const keypoints = faces[0].keypoints;
    if (!keypoints || keypoints.length < 380) {
      rafRef.current = requestAnimationFrame(detect);
      return;
    }

    const points = keypoints as Array<{ x: number; y: number }>;
    const active = CHALLENGES[challengeIndexRef.current];

    if (active === "blink") {
      try {
        const left = LEFT_EYE.map((i) => points[i]);
        const right = RIGHT_EYE.map((i) => points[i]);
        const valid = (p: Array<{ x?: number; y?: number } | undefined>) =>
          p.every((q) => q != null && typeof q.x === "number" && typeof q.y === "number");
        if (valid(left) && valid(right)) {
          const ear = (eyeAspectRatio(left as Point[]) + eyeAspectRatio(right as Point[])) / 2;
          if (!Number.isNaN(ear) && ear > 0 && ear < 1) {
            const now = Date.now();
            if (now - earUpdateRef.current > 80) {
              earUpdateRef.current = now;
              setEarValue(Math.round(ear * 1000) / 1000);
            }
          }
        }
      } catch {
        // EAR display only; checkBlink runs below
      }
      checkBlink(points as Point[]);
    } else {
      setEarValue(null);
      checkTurn(points as Point[], active);
    }

    rafRef.current = requestAnimationFrame(detect);
  }

  /* -------------------------
     Checks
  --------------------------*/
  function checkBlink(points: Point[]) {
    const left = LEFT_EYE.map((i) => points[i]);
    const right = RIGHT_EYE.map((i) => points[i]);
    const valid = (p: Point[]) =>
      p.every((q) => q != null && typeof q.x === "number" && typeof q.y === "number");
    if (!valid(left) || !valid(right)) return;

    const ear = (eyeAspectRatio(left) + eyeAspectRatio(right)) / 2;
    if (Number.isNaN(ear) || ear <= 0) return;

    if (ear < BLINK_THRESHOLD && !blinkedRef.current) {
      blinkedRef.current = true;
      advance();
    }
  }

  function checkTurn(points: Point[], dir: Challenge) {
    const nose = points[NOSE];
    const left = points[LEFT_CHEEK];
    const right = points[RIGHT_CHEEK];

    const width = right.x - left.x;
    if (width <= 0) return;
    const offset = (nose.x - left.x) / width;

    const turnedLeft = offset < 0.45 - TURN_THRESHOLD;
    const turnedRight = offset > 0.55 + TURN_THRESHOLD;

    if (dir === "turn_left") {
      if (turnedLeft) {
        turnHoldFramesRef.current += 1;
        if (turnHoldFramesRef.current >= TURN_HOLD_FRAMES) {
          turnHoldFramesRef.current = 0;
          advance();
        }
      } else {
        turnHoldFramesRef.current = 0;
      }
    } else if (dir === "turn_right") {
      if (turnedRight) {
        turnHoldFramesRef.current += 1;
        if (turnHoldFramesRef.current >= TURN_HOLD_FRAMES) {
          turnHoldFramesRef.current = 0;
          advance();
        }
      } else {
        turnHoldFramesRef.current = 0;
      }
    }
  }

  function advance() {
    blinkedRef.current = false;
    turnHoldFramesRef.current = 0;
    challengeIndexRef.current += 1;

    if (challengeIndexRef.current >= CHALLENGES.length) {
      setStatus("passed");
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      onPassed?.();
      return;
    }

    setCurrent(CHALLENGES[challengeIndexRef.current]);
  }

  /* -------------------------
     Init
  --------------------------*/
  useEffect(() => {
    mountedRef.current = true;
    setup().catch((err) => {
      console.error("FaceLiveness setup failed:", err);
      if (mountedRef.current) setStatus("error");
    });

    return () => {
      mountedRef.current = false;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      const tracks = (videoRef.current?.srcObject as MediaStream)?.getTracks();
      tracks?.forEach((t) => t.stop());
    };
  }, []);

  /* -------------------------
     UI
  --------------------------*/
  return (
    <div className="space-y-2">
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        width={640}
        height={480}
        className="rounded-xl w-full max-w-full h-auto"
      />
      <canvas
        ref={canvasRef}
        width={DETECT_WIDTH}
        height={DETECT_HEIGHT}
        className="hidden"
        aria-hidden
      />

      <p className="text-sm">
        {status === "initializing" && "Initializing…"}
        {status === "challenge" && (
          <>
            {current === "blink" && (
              <>
                Blink 👀
                <span className="block text-gray-500 text-xs mt-1">
                  {!faceDetected
                    ? "Position your face in the frame."
                    : earValue != null
                      ? `EAR: ${earValue} — close both eyes fully (blink when &lt; ${BLINK_THRESHOLD})`
                      : "Waiting for eye data…"}
                </span>
              </>
            )}
            {current === "turn_left" && "Turn left ⬅️"}
            {current === "turn_right" && "Turn right ➡️"}
          </>
        )}
        {status === "passed" && "Liveness passed ✅"}
        {status === "error" && "Error ❌"}
      </p>
    </div>
  );
}
