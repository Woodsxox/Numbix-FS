"use client";

import FaceLiveness from "@/components/FaceLiveness";

export default function FaceLivenessChallenge({
  onPassed,
}: {
  onPassed: () => void;
}) {
  return <FaceLiveness onPassed={onPassed} />;
}
