import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  webpack: (config) => {
    // Force browser ESM build of Human (avoids pulling in human.node.js → @tensorflow/tfjs-node)
    config.resolve.alias = {
      ...config.resolve.alias,
      "@vladmandic/human": path.resolve(
        __dirname,
        "node_modules/@vladmandic/human/dist/human.esm.js"
      ),
    };
    return config;
  },
};

export default nextConfig;
