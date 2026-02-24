"use client";

import { useCallback, useRef, useState } from "react";
import FaceLiveness from "@/components/FaceLiveness";
import FaceLivenessChallenge from "@/components/FaceLivenessChallenge";
import FaceCapture from "@/components/FaceCapture";
import FaceIdentity from "@/components/FaceIdentity";
import { cosineDistance, normalize, averageEmbeddings } from "@/lib/match";
import type { FaceBox } from "@/lib/faceCrop";

type Step = "liveness" | "challenge" | "capture" | "identity" | "done";

/** Tune from real data. Lower so sibling (e.g. ~0.346) is rejected; genuine user should stay below this. */
const MATCH_THRESHOLD = 0.32;

const ENROLLMENT_SAMPLES = 3;

/** null = just enrolled, true = match, false = no match (different person) */
type DoneResult = "enrolled" | "match" | "no_match";

export default function FaceAuthFlow() {
  const [step, setStep] = useState<Step>("liveness");
  const [doneResult, setDoneResult] = useState<DoneResult | null>(null);
  const [lastDistance, setLastDistance] = useState<number | null>(null);
  const [identityRunId, setIdentityRunId] = useState(() => crypto.randomUUID());
  const capturedRef = useRef<{ canvas: HTMLCanvasElement; box: FaceBox } | null>(null);
  const storedEmbeddingRef = useRef<Float32Array | null>(null);
  const enrollmentSamplesRef = useRef<Float32Array[]>([]);
  const [enrollmentStep, setEnrollmentStep] = useState<number | null>(null);

  const handleCaptured = useCallback((canvas: HTMLCanvasElement, box: FaceBox) => {
    capturedRef.current = { canvas, box };
    setIdentityRunId(crypto.randomUUID());
    setStep("identity");
  }, []);

  const handleEmbeddingReady = useCallback((embedding: Float32Array) => {
    console.log("Embedding length:", embedding.length);
    const live = normalize(embedding);

    const stored = storedEmbeddingRef.current;
    if (stored) {
      const distance = cosineDistance(stored, live);
      const isMatch = distance < MATCH_THRESHOLD;
      setLastDistance(distance);
      setDoneResult(isMatch ? "match" : "no_match");
      setStep("done");
      console.log("Distance:", distance);
      console.log("Match result:", { distance, isMatch, threshold: MATCH_THRESHOLD });
    } else {
      enrollmentSamplesRef.current.push(live);
      const samples = enrollmentSamplesRef.current;
      if (samples.length < ENROLLMENT_SAMPLES) {
        setEnrollmentStep(samples.length + 1);
        setStep("capture");
      } else {
        storedEmbeddingRef.current = averageEmbeddings(samples);
        enrollmentSamplesRef.current = [];
        setEnrollmentStep(null);
        setLastDistance(null);
        setDoneResult("enrolled");
        setStep("done");
      }
    }
  }, []);

  const handleVerifyAgain = useCallback(() => {
    setDoneResult(null);
    setLastDistance(null);
    setStep("liveness");
  }, []);

  const handleEnrollAgain = useCallback(() => {
    storedEmbeddingRef.current = null;
    enrollmentSamplesRef.current = [];
    setEnrollmentStep(null);
    setDoneResult(null);
    setLastDistance(null);
    setStep("liveness");
  }, []);

  return (
    <div className="max-w-md mx-auto space-y-4">
      {step === "liveness" && (
        <FaceLiveness onPassed={() => setStep("challenge")} />
      )}

      {step === "challenge" && (
        <FaceLivenessChallenge onPassed={() => setStep("capture")} />
      )}

      {step === "capture" && (
        <div className="space-y-2">
          {enrollmentStep != null && (
            <p className="text-sm text-center text-gray-600">
              Enrollment {enrollmentStep}/{ENROLLMENT_SAMPLES} — position your face for the next sample
            </p>
          )}
          <FaceCapture onCaptured={handleCaptured} />
        </div>
      )}

      {step === "identity" && capturedRef.current && (
        <FaceIdentity
          input={capturedRef.current.canvas}
          box={capturedRef.current.box}
          runId={identityRunId}
          onEmbeddingReady={handleEmbeddingReady}
        />
      )}

      {step === "done" && doneResult && (
        <div className="space-y-3 text-center">
          <p className="font-medium">
            {doneResult === "enrolled" && (
              <span className="text-green-600">✅ Identity enrolled</span>
            )}
            {doneResult === "match" && (
              <span className="text-green-600">✅ Identity verified</span>
            )}
            {doneResult === "no_match" && (
              <span className="text-red-600">❌ Identity not verified (different person)</span>
            )}
          </p>
          {lastDistance != null && (
            <p className="text-sm text-gray-500">Distance: {lastDistance.toFixed(4)}</p>
          )}
          <div className="flex flex-wrap justify-center gap-2">
            <button
              type="button"
              onClick={handleVerifyAgain}
              className="rounded-md bg-gray-200 px-4 py-2 text-sm font-medium text-gray-800 hover:bg-gray-300"
            >
              Verify again
            </button>
            <button
              type="button"
              onClick={handleEnrollAgain}
              className="rounded-md bg-gray-200 px-4 py-2 text-sm font-medium text-gray-800 hover:bg-gray-300"
            >
              Enroll again
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
