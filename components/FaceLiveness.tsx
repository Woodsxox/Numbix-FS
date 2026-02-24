"use client";

import { useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import * as faceLandmarks from "@tensorflow-models/face-landmarks-detection";

/** Mesh edges for drawing wireframe (pairs of keypoint indices). */
const MESH_PAIRS = faceLandmarks.util.getAdjacentPairs(
  faceLandmarks.SupportedModels.MediaPipeFaceMesh
) as Array<[number, number]>;

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
  const meshCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const detectorRef =
    useRef<faceLandmarks.FaceLandmarksDetector | null>(null);
  const rafRef = useRef<number | null>(null);

  const DETECT_WIDTH = 640;
  const DETECT_HEIGHT = 480;

  // Detection flags: refs only (no useState) so frame loop is not affected by React batching
  const eyeClosedRef = useRef(false);
  const blinkConfirmedRef = useRef(false);
  const hasSeenEyesOpenRef = useRef(false);
  const challengeIndexRef = useRef(0);
  const turnHoldFramesRef = useRef(0);

  const [status, setStatus] = useState<Status>("initializing");
  const [current, setCurrent] = useState<Challenge>(CHALLENGES[0]);
  const [earValue, setEarValue] = useState<number | null>(null);
  const [faceDetected, setFaceDetected] = useState(false);
  const [faceTooFar, setFaceTooFar] = useState(false);
  const [faceOffCenter, setFaceOffCenter] = useState(false);

  const BLINK_CLOSE_THRESHOLD = 0.22;
  const BLINK_OPEN_THRESHOLD = 0.26;
  const TURN_THRESHOLD = 0.15;
  const TURN_HOLD_FRAMES = 12;
  const mountedRef = useRef(true);
  const earUpdateRef = useRef(0);
  const frameCountRef = useRef(0);
  const faceUpdateThrottleRef = useRef(0);
  const lastKeypointsRef = useRef<Array<{ x: number; y: number }> | null>(null);
  const faceTooFarRef = useRef(false);
  const FACE_TOO_FAR_RATIO = 0.28;
  const FACE_OFF_CENTER_PX = 70;

  /** Reset all liveness detection state. Call when starting a new scan so first and later scans behave the same. */
  function resetLivenessState() {
    eyeClosedRef.current = false;
    blinkConfirmedRef.current = false;
    hasSeenEyesOpenRef.current = false;
    challengeIndexRef.current = 0;
    turnHoldFramesRef.current = 0;
  }

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

    resetLivenessState();
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

  /** Draw face mesh (dots + lines) on overlay canvas. Coordinates match detection canvas; overlay has scaleX(-1) to match video. */
  function drawMeshOverlay(points: Array<{ x: number; y: number }>) {
    const canvas = meshCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = DETECT_WIDTH;
    canvas.height = DETECT_HEIGHT;
    ctx.clearRect(0, 0, DETECT_WIDTH, DETECT_HEIGHT);

    // Draw mesh lines (purple)
    ctx.strokeStyle = "rgba(168, 85, 247, 0.6)";
    ctx.lineWidth = 1;
    for (const [i, j] of MESH_PAIRS) {
      const a = points[i];
      const b = points[j];
      if (a?.x != null && a?.y != null && b?.x != null && b?.y != null) {
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
    }

    // Draw dots (cyan)
    ctx.fillStyle = "rgba(34, 211, 238, 0.9)";
    ctx.strokeStyle = "rgba(34, 211, 238, 0.5)";
    ctx.lineWidth = 0.5;
    for (const p of points) {
      if (p?.x == null || p?.y == null) continue;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 2, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
    }
  }

  /** Face bbox and center from keypoints (in canvas coords). */
  function getFaceBounds(points: Array<{ x: number; y: number }>) {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const p of points) {
      if (typeof p.x !== "number" || typeof p.y !== "number") continue;
      minX = Math.min(minX, p.x);
      minY = Math.min(minY, p.y);
      maxX = Math.max(maxX, p.x);
      maxY = Math.max(maxY, p.y);
    }
    if (minX === Infinity) return null;
    const width = maxX - minX;
    const height = maxY - minY;
    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    return { minX, minY, maxX, maxY, width, height, centerX, centerY };
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

    const active = CHALLENGES[challengeIndexRef.current];
    const skipInference = active === "blink" && frameCountRef.current % 2 === 1 && lastKeypointsRef.current != null;
    frameCountRef.current += 1;

    let points: Array<{ x: number; y: number }> | null = null;

    if (!skipInference) {
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
        lastKeypointsRef.current = null;
        faceTooFarRef.current = false;
        setEarValue(null);
        setFaceDetected(false);
        setFaceTooFar(false);
        setFaceOffCenter(false);
        const meshCanvas = meshCanvasRef.current;
        if (meshCanvas) {
          meshCanvas.width = DETECT_WIDTH;
          meshCanvas.height = DETECT_HEIGHT;
          const mesh = meshCanvas.getContext("2d");
          if (mesh) mesh.clearRect(0, 0, DETECT_WIDTH, DETECT_HEIGHT);
        }
        rafRef.current = requestAnimationFrame(detect);
        return;
      }

      const keypoints = faces[0].keypoints;
      if (!keypoints || keypoints.length < 380) {
        rafRef.current = requestAnimationFrame(detect);
        return;
      }
      points = keypoints as Array<{ x: number; y: number }>;
      lastKeypointsRef.current = points;
    } else {
      points = lastKeypointsRef.current;
    }

    if (!points || points.length < 380) {
      rafRef.current = requestAnimationFrame(detect);
      return;
    }

    setFaceDetected(true);
    drawMeshOverlay(points);
    const bounds = getFaceBounds(points);
    if (bounds) {
      const tooFar = bounds.width < DETECT_WIDTH * FACE_TOO_FAR_RATIO;
      faceTooFarRef.current = tooFar;
      const centerX = DETECT_WIDTH / 2, centerY = DETECT_HEIGHT / 2;
      const offCenter = Math.hypot(bounds.centerX - centerX, bounds.centerY - centerY) > FACE_OFF_CENTER_PX;
      faceUpdateThrottleRef.current += 1;
      if (faceUpdateThrottleRef.current % 5 === 0) {
        setFaceTooFar(tooFar);
        setFaceOffCenter(!tooFar && offCenter);
      }
    }

    if (active === "blink") {
      try {
        const left = LEFT_EYE.map((i) => points![i]);
        const right = RIGHT_EYE.map((i) => points![i]);
        const valid = (p: Array<{ x?: number; y?: number } | undefined>) =>
          p.every((q) => q != null && typeof q.x === "number" && typeof q.y === "number");
        if (valid(left) && valid(right)) {
          const ear = (eyeAspectRatio(left as Point[]) + eyeAspectRatio(right as Point[])) / 2;
          if (!Number.isNaN(ear) && ear > 0 && ear < 1) {
            const now = Date.now();
            if (now - earUpdateRef.current > 50) {
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
    if (faceTooFarRef.current) return; // require face close enough
    const left = LEFT_EYE.map((i) => points[i]);
    const right = RIGHT_EYE.map((i) => points[i]);
    const valid = (p: Point[]) =>
      p.every((q) => q != null && typeof q.x === "number" && typeof q.y === "number");
    if (!valid(left) || !valid(right)) return;

    const ear = (eyeAspectRatio(left) + eyeAspectRatio(right)) / 2;
    if (Number.isNaN(ear) || ear <= 0) return;

    if (blinkConfirmedRef.current) return; // already passed blink step

    if (ear > BLINK_OPEN_THRESHOLD) {
      hasSeenEyesOpenRef.current = true;
    }
    if (ear < BLINK_CLOSE_THRESHOLD && hasSeenEyesOpenRef.current) {
      eyeClosedRef.current = true;
    }
    if (eyeClosedRef.current && ear > BLINK_OPEN_THRESHOLD) {
      blinkConfirmedRef.current = true;
      advance();
    }
  }

  function checkTurn(points: Point[], dir: Challenge) {
    if (faceTooFarRef.current) return; // require face close enough
    const nose = points[NOSE];
    const left = points[LEFT_CHEEK];
    const right = points[RIGHT_CHEEK];

    const width = right.x - left.x;
    if (width <= 0) return;
    const offset = (nose.x - left.x) / width;

    // In image coords: nose left = small offset, nose right = large offset.
    // Front camera is mirrored, so "turn your head left" = nose moves to image right (large offset).
    const noseTowardImageLeft = offset < 0.45 - TURN_THRESHOLD;
    const noseTowardImageRight = offset > 0.55 + TURN_THRESHOLD;

    if (dir === "turn_left") {
      if (noseTowardImageRight) {
        turnHoldFramesRef.current += 1;
        if (turnHoldFramesRef.current >= TURN_HOLD_FRAMES) {
          turnHoldFramesRef.current = 0;
          advance();
        }
      } else {
        turnHoldFramesRef.current = 0;
      }
    } else if (dir === "turn_right") {
      if (noseTowardImageLeft) {
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
    turnHoldFramesRef.current = 0;
    challengeIndexRef.current += 1;

    if (challengeIndexRef.current >= CHALLENGES.length) {
      setStatus("passed");
      // Hard stop: MUST fully stop before onPassed (prevents identity glitch)
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      if (videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream)
          .getTracks()
          .forEach((track) => track.stop());
      }
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
      <div className="relative w-full max-w-full inline-block p-4">
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          width={640}
          height={480}
          className="rounded-xl w-full max-w-full h-auto"
          style={{ transform: "scaleX(-1)" }}
        />
        {status === "challenge" && (
          <canvas
            ref={meshCanvasRef}
            width={DETECT_WIDTH}
            height={DETECT_HEIGHT}
            className="absolute inset-0 w-full h-full rounded-xl pointer-events-none object-contain"
            style={{ transform: "scaleX(-1)" }}
            aria-hidden
          />
        )}
        {status === "challenge" && (
          <div
            className="absolute inset-0 flex items-center justify-center pointer-events-none rounded-xl"
            aria-hidden
          >
            <div className="border-2 border-white/80 rounded-full w-[55%] max-w-[260px] max-h-[85%] aspect-3/4" />
          </div>
        )}
      </div>
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
                    ? "Position your face in the frame. Improve lighting if needed."
                    : faceTooFar
                      ? "Move closer so your face fills the frame."
                      : faceOffCenter
                        ? "Center your face in the oval."
                        : earValue != null
                          ? `EAR: ${earValue} — blink naturally`
                          : "Waiting for eye data…"}
                </span>
              </>
            )}
            {current === "turn_left" && (
              <>
                Turn left ⬅️
                {faceTooFar && <span className="block text-gray-500 text-xs mt-1">Move closer</span>}
                {faceOffCenter && !faceTooFar && <span className="block text-gray-500 text-xs mt-1">Center your face in the oval</span>}
              </>
            )}
            {current === "turn_right" && (
              <>
                Turn right ➡️
                {faceTooFar && <span className="block text-gray-500 text-xs mt-1">Move closer</span>}
                {faceOffCenter && !faceTooFar && <span className="block text-gray-500 text-xs mt-1">Center your face in the oval</span>}
              </>
            )}
          </>
        )}
        {status === "passed" && "Liveness passed ✅"}
        {status === "error" && "Error ❌"}
      </p>
    </div>
  );
}
