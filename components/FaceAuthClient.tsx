"use client";

import dynamic from "next/dynamic";

const FaceAuthFlow = dynamic(
  () => import("./FaceAuthFlow"),
  { ssr: false }
);

export default function FaceAuthClient() {
  return <FaceAuthFlow />;
}