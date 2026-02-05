"use client";

import { useState } from "react";
import FaceLiveness from "@/components/FaceLiveness";
import FaceLivenessChallenge from "@/components/FaceLivenessChallenge";
import FaceEnrollment from "@/components/FaceEnrollment";

type Step = "liveness" | "challenge" | "enroll" | "done";

export default function FaceAuthFlow() {
  const [step, setStep] = useState<Step>("liveness");

  return (
    <div className="max-w-md mx-auto space-y-4">
      {step === "liveness" && (
        <FaceLiveness onPassed={() => setStep("challenge")} />
      )}

      {step === "challenge" && (
        <FaceLivenessChallenge onPassed={() => setStep("enroll")} />
      )}

      {step === "enroll" && (
        <FaceEnrollment onComplete={() => setStep("done")} />
      )}

      {step === "done" && (
        <p className="text-center text-green-600 font-medium">
          ✅ Face verified & enrolled
        </p>
      )}
    </div>
  );
}
