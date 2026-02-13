# Deep dive: Face auth codebase

Technical deep dive: data flow, math, lifecycle, and how it all fits together.

---

## 1. End-to-end data flow

### 1.1 Request path

```
Browser
  → app/page.tsx
  → FaceAuthClient (dynamic import, ssr: false)
  → FaceAuthFlow (useState step: liveness | challenge | enroll | done)
```

- **Why dynamic + ssr: false:** TensorFlow.js and `getUserMedia` only run in the browser. Server-rendering would reference `window`/WebGL and break; dynamic load keeps this off the server bundle and avoids hydration mismatches.

### 1.2 Step flow and ownership

| Step        | Component              | What runs |
|------------|-------------------------|-----------|
| `liveness` | `FaceLiveness`          | Face Mesh (landmarks) → EAR + nose/cheek → blink, turn left, turn right. |
| `challenge`| `FaceLivenessChallenge` | Same as above (currently just `<FaceLiveness onPassed={…} />`). |
| `enroll`   | `FaceEnrollment`        | Face detector (boxes) + FaceNet → 5× 128-D embeddings → average → “saved”. |
| `done`     | Inline in FaceAuthFlow  | “Face verified & enrolled”. |

So: **liveness** = “prove you’re live” (no embeddings). **Enrollment** = “capture and average your face embedding” (no backend yet).

### 1.3 Out-of-flow components

- **FaceVerify:** Same pipeline as enrollment (detector + crop + FaceNet) but compares live embedding to a **stored** one via cosine similarity (threshold 0.65). Not mounted in FaceAuthFlow; intended for login/verify. `storedEmbedding` is a placeholder (zeros).
- **LiveFaceScan:** Single-blink liveness (EAR only). Superseded by FaceLiveness; not used by FaceAuthFlow.
- **Camera:** Face presence only (detector, no challenges). Not used in main flow.

---

## 2. Liveness: math and logic

### 2.1 Models

- **FaceLiveness** uses **face-landmarks-detection** (MediaPipe Face Mesh), not face-detection and not FaceNet. So “FaceNet embedding-based liveness” in DEEP_ANALYSIS.md refers to a different/older design; current liveness is **landmark-based only**.

### 2.2 Challenges (fixed order)

```ts
const CHALLENGES: Challenge[] = ["blink", "turn_left", "turn_right"];
```

- Stored **outside** the component so the order is stable across re-renders (fixes the old “random sort on every render” bug).

### 2.3 Blink: Eye Aspect Ratio (EAR)

- **Landmarks:** Left eye indices `[33, 160, 158, 133, 153, 144]`, right eye `[263, 387, 385, 362, 380, 373]` (MediaPipe mesh).
- **Formula:**
  - Vertical distances: `v1 = distance(eye[1], eye[5])`, `v2 = distance(eye[2], eye[4])`
  - Horizontal: `h = distance(eye[0], eye[3])`
  - `EAR = (v1 + v2) / (2 * h)`
- **Interpretation:** Open eye → larger EAR; closed eye → smaller EAR.
- **Threshold:** `BLINK_THRESHOLD = 0.26`. When `EAR < 0.26` the code counts a blink and advances. No “hold closed for N frames” requirement, so a fast, full blink is enough.

### 2.4 Turn left / turn right

- **Landmarks:** Nose tip `NOSE = 1`, left cheek `LEFT_CHEEK = 234`, right cheek `RIGHT_CHEEK = 454`.
- **Metric:** `width = right.x - left.x`, `offset = (nose.x - left.x) / width` (nose position as fraction of inter-cheek width).
- **Thresholds:** `TURN_THRESHOLD = 0.15`
  - Turn left: `offset < 0.45 - 0.15` → `offset < 0.30`
  - Turn right: `offset > 0.55 + 0.15` → `offset > 0.70`
- **Stability:** Must hold the turn for `TURN_HOLD_FRAMES = 25` consecutive frames (~0.4 s at 60 fps) to advance. Counters live in `turnHoldFramesRef`; reset when not in range.

### 2.5 Frame loop (FaceLiveness)

- **Input:** Video is drawn to a **hidden canvas** at 640×480; `estimateFaces(canvas, { flipHorizontal: false, staticImageMode: true })` runs on that canvas (not the video element). Face Mesh returns keypoints; code assumes ≥380 points.
- **Loop:** `requestAnimationFrame(detect)`; each frame runs detection, then either `checkBlink` or `checkTurn` depending on `challengeIndexRef.current`.
- **Mount safety:** `mountedRef.current` is set true in effect and false in cleanup; setup and async steps check it before calling `setStatus`/advancing so unmount doesn’t update state or leave RAF running.
- **Cleanup:** Cancel `rafRef`, stop all media tracks from `videoRef.current.srcObject`.

---

## 3. Enrollment: frames, crop, FaceNet

### 3.1 Stability and sample count

- **Face detector:** MediaPipe Face Detector (TF.js runtime), `maxFaces: 1`.
- **Stable frames:** Face must be present for **90 consecutive frames** (~1.5 s at 60 fps). Implemented with `stableFramesRef` (correctly a ref, so not reset by re-renders).
- **Samples:** After 90 stable frames, one embedding is captured; repeat until **5 samples**, then finalize.
- **Per-sample flow:** 90 frames → `captureEmbedding(box)` → crop/normalize → FaceNet → push 128-D to `embeddingsRef`; then loop continues for next sample (or stops at 5).

### 3.2 Crop and normalize (lib/faceCrop.ts)

```ts
frame = tf.browser.fromPixels(video)     // [H, W, 3] uint8
face  = tf.slice(frame, [y, x, 0], [h, w, 3])
resized = tf.image.resizeBilinear(face, [160, 160])
normalized = resized.toFloat().div(127.5).sub(1)   // [-1, 1]
return normalized.expandDims(0)                    // [1, 160, 160, 3]
```

- **Bounds:** `x = max(0, box.xMin)`, `y = max(0, box.yMin)`, `w`/`h` clamped so `[y, x]` + `[h, w]` stay inside frame. So out-of-bound boxes are clamped; no unsliced negative indices.
- **FaceNet input:** 160×160, 3 channels, normalized to [-1, 1], batch size 1. Matches the model’s `batch_input_shape: [null, 160, 160, 3]`.

### 3.3 FaceNet and embedding (lib/facenet.ts)

- **Load:** `tf.loadLayersModel(LOCAL_MODEL_URL)` with `LOCAL_MODEL_URL = "/models/facenet/model.json"`. Optional fallback: `NEXT_PUBLIC_FACENET_GITHUB_URL` (full URL to a `model.json`); weight paths in the manifest are relative to that URL.
- **Caching:** Single module-level `model`; first `loadFaceNet()` loads, later calls return the same instance.
- **Inference:** `getEmbedding(faceTensor)` runs `model.predict(face4D)`, squeezes to 1D, then `array from tensor.data()`. No `tf.tidy` in `getEmbedding`; caller (FaceEnrollment) disposes the face tensor after use.

### 3.4 Averaging and “save”

- **Average:** `averageEmbeddings(vectors)` element-wise mean over the 5× 128-D vectors → one 128-D `Float32Array`.
- **Persistence:** Commented-out `fetch("/api/face/enroll", …)`. So the averaged embedding is only in memory and then the step finishes; no server or localStorage yet.

---

## 4. Verification path (FaceVerify)

- Same detector + crop + FaceNet as enrollment. After getting the live 128-D:
  - `cosineSimilarity(storedEmbedding, liveEmbedding)` from `lib/faceMatch.ts`.
  - `isFaceMatch(similarity, 0.65)` → match if similarity ≥ 0.65.
- **storedEmbedding:** Currently `new Float32Array(128)` (zeros). In a real flow this would come from an API or storage after enrollment.

---

## 5. Lib layer: who uses what

| File         | Exports              | Used by                    |
|-------------|----------------------|----------------------------|
| faceCrop.ts | cropAndNormalizeFace  | FaceEnrollment, FaceVerify |
| facenet.ts  | loadFaceNet, getEmbedding | FaceEnrollment, FaceVerify |
| faceMatch.ts| cosineSimilarity, isFaceMatch | FaceVerify only            |
| faceVerify.ts | cosineSimilarity, verifyFace | Nothing (duplicate of faceMatch; different default threshold 0.75) |
| tf.ts       | (empty)               | Nothing                    |

- **Redundancy:** Two similarity helpers (faceMatch vs faceVerify) and two default thresholds (0.65 vs 0.75). Unifying on one module and one threshold would simplify.

---

## 6. Lifecycle and cleanup

### 6.1 FaceLiveness

- **Refs:** `videoRef`, `canvasRef`, `detectorRef`, `rafRef`, `blinkedRef`, `challengeIndexRef`, `turnHoldFramesRef`, `mountedRef`, `earUpdateRef`.
- **Cleanup:** Effect return cancels RAF and stops all tracks. Async path checks `mountedRef.current` before `setStatus` and before calling `onPassed()`. No `stableFrames` in this component; frame counting is only in enrollment/verify.

### 6.2 FaceEnrollment

- **Refs:** `videoRef`, `detectorRef`, `rafRef`, `embeddingsRef`, `stableFramesRef`.
- **Loop:** `setup()` then `detect()`; `detect()` calls `requestAnimationFrame(detect)` except when capturing (capture runs, then `finalizeEnrollment` → `cleanup()` so RAF is cancelled).
- **Cleanup:** `cleanup()` cancels RAF and stops tracks; effect return calls `cleanup`. If the component unmounts during `captureEmbedding`, the effect cleanup still runs, but there’s no explicit “don’t call setState after unmount” guard (unlike FaceLiveness’s `mountedRef`).

### 6.3 Tensor disposal

- **faceCrop:** Entire pipeline inside `tf.tidy()`, so intermediates are disposed; only the returned tensor is long-lived until the caller disposes it.
- **facenet getEmbedding:** No `tf.tidy`; caller must dispose the input face tensor (FaceEnrollment and FaceVerify do dispose it after `getEmbedding`).
- **Detectors:** `estimateFaces` internal tensors are under TF.js/MediaPipe control; not explicitly wrapped in `tf.tidy` in this codebase.

---

## 7. Build and runtime quirks (from ISSUES_AND_FIXES)

- **Build:** `next build` must use **Webpack** (`next build --webpack`) because MediaPipe packages expose globals, not ESM named exports; Turbopack fails on those.
- **Routes:** Only App Router should own `/`; no `pages/index.js` (or other page matching `/`) to avoid “App Router and Pages Router both match path”.
- **FaceNet:** Model must be TF.js Layers format (`modelTopology` + `weightsManifest`); raw Keras JSON won’t load. Conversion: `tensorflowjs_converter` from Keras `.h5` → `public/models/facenet/`.

---

## 8. Security and attack surface (from V1_ATTACK_SIMULATION)

- **Photo/screen attack:** No 3D or texture check; liveness relies on blink + head turn. Static photo can’t pass (no real blink/turn). Replay of a video of someone blinking/turning might pass if the replay looks like a live face to the detector.
- **Blink cheating:** EAR threshold 0.26; very fast or partial blinks might pass if EAR dips below 0.26. No minimum “closed” duration.
- **Turn cheating:** Turn is based on 2D nose position between cheeks; body lean or chin tilt could potentially move the nose in frame; 25-frame hold reduces but doesn’t eliminate that.
- **Embeddings:** Logged in console in enrollment; no encryption or secure channel to a backend yet. Stored embedding in FaceVerify is client-side placeholder.

---

## 9. DEEP_ANALYSIS.md vs current code

- **“stableFrames resets on every render”:** Described as in FaceLiveness; in the **current** code, FaceLiveness does **not** use `stableFrames` at all. FaceEnrollment and FaceVerify **do** use `stableFramesRef` correctly. So that bug appears fixed or was in another component when the analysis was written.
- **“FaceLiveness uses FaceNet”:** Current FaceLiveness uses only face-landmarks (EAR + turn); FaceNet is used in FaceEnrollment and FaceVerify.
- **“cropAndNormalizeFace missing boundary checks”:** faceCrop.ts **does** clamp and bound the slice (see section 3.2 above).

---

## 10. Summary diagram

```
User opens /
  → FaceAuthClient (client-only)
  → FaceAuthFlow
       step === "liveness"
         → FaceLiveness (Face Mesh, EAR + turn)
         → onPassed → step = "challenge"
       step === "challenge"
         → FaceLivenessChallenge (= FaceLiveness again)
         → onPassed → step = "enroll"
       step === "enroll"
         → FaceEnrollment (Face Detector + FaceNet, 5 samples, average)
         → onComplete → step = "done"
       step === "done"
         → "Face verified & enrolled"

Separate (not in flow):
  FaceVerify: same pipeline, compare live vs storedEmbedding (0.65 threshold).
```

This document reflects the codebase as of the deep-dive review; for build/routes and known issues, see ISSUES_AND_FIXES.md and DEEP_ANALYSIS.md.
