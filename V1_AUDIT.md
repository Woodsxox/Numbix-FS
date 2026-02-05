# B️⃣ V1 Audit — KEEP / MERGE / DELETE

**Date:** V1 freeze (pre–V2 CCTV)  
**Purpose:** Label all V1 face-auth files and freeze a clean baseline.

---

## Components

| File | Label | Reason |
|------|--------|--------|
| **FaceAuthFlow.tsx** | **KEEP** | Main V1 flow: liveness → challenge → enroll → done. Single entry point for production. |
| **FaceLiveness.tsx** | **KEEP** | Chained liveness (blink → turn left → turn right). Core of V1 manual test checklist. |
| **FaceLivenessChallenge.tsx** | **KEEP** | Thin wrapper around FaceLiveness for the "challenge" step. No logic duplication. |
| **FaceEnrollment.tsx** | **KEEP** | Multi-sample enrollment, FaceNet embedding, averaging. Used by FaceAuthFlow. |
| **FaceVerify.tsx** | **KEEP** | Live verification vs stored embedding (separate use: login/verify after enrollment). |
| **LiveFaceScan.tsx** | **MERGE / DELETE** | Old single-blink liveness. Superseded by FaceLiveness. Currently used by `app/page.tsx`. **Recommend:** Remove from default page; use FaceAuthFlow. Can DELETE later or keep as minimal demo. |
| **Camera.tsx** | **DELETE** (or keep as util) | Face detection only, no challenges. Not used in FaceAuthFlow or any main flow. Safe to DELETE or keep as shared camera/face-presence utility. |

---

## Lib

| File | Label | Reason |
|------|--------|--------|
| **faceCrop.ts** | **KEEP** | Crop + normalize for FaceNet. Used by FaceEnrollment, FaceVerify. |
| **facenet.ts** | **KEEP** | Load FaceNet, getEmbedding. Used by FaceEnrollment, FaceVerify. |
| **faceMatch.ts** | **KEEP** | cosineSimilarity, isFaceMatch. Used by FaceVerify. |
| **faceVerify.ts** | **MERGE** | Same cosine math as faceMatch + verifyFace(). Only FaceVerify.tsx uses faceMatch. faceVerify.ts is unused. **Recommend:** Later merge into faceMatch or remove faceVerify.ts. For freeze: **KEEP** both (no behavior change). |
| **tf.ts** | **DELETE** | Empty. Safe to delete or leave as stub. |

---

## App entry

| Location | Current | Recommendation |
|----------|---------|-----------------|
| **app/page.tsx** | Renders `LiveFaceScan` | Switch to **FaceAuthFlow** so default route runs full V1 flow (liveness → challenge → enroll → done). |
| **pages/index.js** | Renders `LiveFaceScan` | If using App Router only, remove or redirect to app route. |

---

## V1 freeze summary

**Core V1 (production path):**

- **FaceAuthFlow** → FaceLiveness → FaceLivenessChallenge → FaceEnrollment → done  
- **FaceVerify** for separate “verify identity” flows (e.g. login).  
- **Lib:** faceCrop, facenet, faceMatch (and optionally faceVerify for API consistency).

**Optional cleanup (post-freeze):**

- Remove or replace **LiveFaceScan** on the default page with **FaceAuthFlow**.  
- Delete or repurpose **Camera.tsx** if unused.  
- Delete **tf.ts** or keep as stub.  
- Consolidate **faceVerify.ts** and **faceMatch.ts** in a later pass.

**Hard-fail checks (from V1 checklist):**

- Photo/video must not pass liveness.  
- Blink must require real eye closure (EAR).  
- Head shake must not pass turn challenge.  
- No auto-complete without user actions.

---

**Next:** C️⃣ V2 CCTV architecture (after V1 is confirmed and this audit is accepted).
