"use client";

import { useEffect, useState } from "react";
import { getEmbedding, type FaceEmbeddingInput } from "@/lib/faceEmbedding";
import type { FaceBox } from "@/lib/faceCrop";

export type FaceIdentityInput = HTMLVideoElement | HTMLCanvasElement;

// Module-level: set only when we complete (so Strict Mode's 2nd mount can run and complete)
let lastCompletedRunId: string | null = null;

export default function FaceIdentity({
  input,
  box,
  runId,
  onEmbeddingReady,
}: {
  input: FaceIdentityInput;
  box: Partial<FaceBox> & { xMax?: number; yMax?: number };
  runId: string;
  onEmbeddingReady: (embedding: Float32Array) => void;
}) {
  const [status, setStatus] = useState("loading");

  useEffect(() => {
    // Skip only if we already completed for this runId (prevents double onEmbeddingReady)
    if (lastCompletedRunId === runId) {
      console.log("Skipping duplicate StrictMode run (already completed)");
      return;
    }

    let cancelled = false;

    async function run() {
      try {
        setStatus("loading");
        setStatus("capturing");
        const embedding = await getEmbedding(input as FaceEmbeddingInput);

        if (!cancelled) {
          lastCompletedRunId = runId;
          setStatus("done");
          onEmbeddingReady(embedding);
        }
      } catch (err) {
        console.error("Identity error:", err);
        if (!cancelled) setStatus("error");
      }
    }

    run();

    return () => {
      cancelled = true;
    };
  }, [runId, input, box, onEmbeddingReady]);

  return <div className="text-sm text-gray-600">Identity: {status}</div>;
}
