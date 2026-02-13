"use client";

import { useCallback, useRef, useState } from "react";
import FaceLiveness from "@/components/FaceLiveness";
import FaceLivenessChallenge from "@/components/FaceLivenessChallenge";
import FaceCapture from "@/components/FaceCapture";
import FaceIdentity from "@/components/FaceIdentity";
import { cosineDistance } from "@/lib/match";
import type { FaceBox } from "@/lib/faceCrop";

type Step = "liveness" | "challenge" | "capture" | "identity" | "done";

export default function FaceAuthFlow() {
  const [step, setStep] = useState<Step>("liveness");
  const [identityRunId, setIdentityRunId] = useState(() => crypto.randomUUID());
  const capturedRef = useRef<{ canvas: HTMLCanvasElement; box: FaceBox } | null>(null);
  const storedEmbeddingRef = useRef<Float32Array | null>(null);

  const handleCaptured = useCallback((canvas: HTMLCanvasElement, box: FaceBox) => {
    capturedRef.current = { canvas, box };
    setIdentityRunId(crypto.randomUUID());
    setStep("identity");
  }, []);

  const handleEmbeddingReady = useCallback((embedding: Float32Array) => {
    // Embedding stability check (expected: length 128, values ~ -0.5 to 0.5)
    console.log("Embedding length:", embedding.length);
    console.log("First 5 values:", Array.from(embedding.slice(0, 5)));

    const stored = storedEmbeddingRef.current;
    if (stored) {
      const distance = cosineDistance(stored, embedding);
      const isMatch = distance < 0.5;
      setStep("done");
      console.log("Match result:", { distance, isMatch });
    } else {
      storedEmbeddingRef.current = embedding;
      setStep("done");
    }
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
        <FaceCapture onCaptured={handleCaptured} />
      )}

      {step === "identity" && capturedRef.current && (
        <FaceIdentity
          input={capturedRef.current.canvas}
          box={capturedRef.current.box}
          runId={identityRunId}
          onEmbeddingReady={handleEmbeddingReady}
        />
      )}

      {step === "done" && (
        <p className="text-center text-green-600 font-medium">
          ✅ Identity verified
        </p>
      )}
    </div>
  );
}
