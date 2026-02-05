# Issues and code reference

List of issues we hit and where to look in the codebase.

---

## 1. Build failure: “Export FaceDetection / FaceMesh doesn’t exist”

**What you see**

- `next build` (Turbopack) fails with:
  - `Export FaceDetection doesn't exist in target module` in `@mediapipe/face_detection`
  - `Export FaceMesh was not found` in `@mediapipe/face_mesh`

**Why it happens**

- Next.js 16 uses **Turbopack** by default for `next build`.
- Turbopack treats those packages as ESM and looks for **named exports** (`FaceDetection`, `FaceMesh`).
- The MediaPipe packages are **legacy bundles** that attach to the **global object** (e.g. `K("FaceDetection", ...)` in `face_detection.js`), not ESM `export`.
- So Turbopack’s static analysis says “export doesn’t exist.”

**Where it comes from**

- **Package:** `@tensorflow-models/face-detection` (and face-landmarks-detection) import from `@mediapipe/face_detection` and `@mediapipe/face_mesh`.
- **File (in node_modules):**  
  `node_modules/@tensorflow-models/face-detection/dist/face-detection.esm.js` line 17:  
  `import { FaceDetection } from "@mediapipe/face_detection";`
- **MediaPipe file:**  
  `node_modules/@mediapipe/face_detection/face_detection.js` — no ESM exports, only globals.

**Fix (already applied)**

Use **Webpack** for the production build so those packages are bundled in a way that works with globals.

**Code:** `package.json`

```json
"scripts": {
  "build": "next build --webpack"
}
```

- Keep `"dev": "next dev"` as-is (Turbopack for dev is usually fine).
- If you ever need Webpack for dev too: `"dev": "next dev --webpack"`.

**If it comes back**

- Make sure you run `npm run build` (so the script with `--webpack` is used).
- If someone removes `--webpack`, the Turbopack build will fail again with the same “export doesn’t exist” errors.

---

## 2. Route conflict: App Router and Pages Router both match `/`

**What you see**

- Build or dev error:  
  `App Router and Pages Router both match path: /`  
  “Next.js does not support having both App Router and Pages Router routes matching the same path.”

**Why it happens**

- **App Router:** `app/page.tsx` → route `/`
- **Pages Router:** `pages/index.js` → route `/`
- Same path, two routers → conflict.

**Where it was**

- **App route:** `app/page.tsx` (your main page).
- **Pages route (removed):** `pages/index.js` (it imported `LiveFaceScan` and rendered it).

**Fix (already applied)**

- **Deleted** `pages/index.js`.
- `pages/` is now empty; only the App Router serves `/` from `app/page.tsx`.

**If it comes back**

- Don’t add a `pages/index.js` (or any `pages/*` that maps to `/`) if you want to keep using `app/page.tsx` for the home page.
- Or remove `app/page.tsx` and use only Pages Router; one router must “own” `/`.

---

## 3. Dev server: “uv_interface_addresses” / “Unknown system error 1”

**What you see**

- When running `npm run dev`, Next.js may log:
  - `Unhandled Rejection: NodeError [SystemError]: uv_interface_addresses returned Unknown system error 1`
- Sometimes the server still starts (e.g. “Ready”), sometimes it doesn’t.
- More likely in constrained environments (containers, sandboxes, some CI).

**Why it happens**

- Next.js tries to list network interfaces (e.g. to print “Network: http://192.168.x.x:3000”).
- That uses Node’s `os.networkInterfaces()` → `uv_interface_addresses`.
- In some environments that syscall fails (permissions, sandbox, or OS quirk).

**Where it comes from**

- **Next.js internals:**  
  `node_modules/next/dist/lib/get-network-host.js` (and related server startup code).
- Not in your app code.

**Fix (what you can do)**

1. **Ignore if dev works on your machine**  
   If `npm run dev` runs fine locally, you can ignore this in other environments.

2. **Run production build locally**  
   Avoids dev server entirely:
   ```bash
   npm run build
   npm run start
   ```
   Then open http://localhost:3000 (or the port shown).

3. **Environment variable (if needed)**  
   Some setups work around network detection by not relying on it; Next doesn’t document a single “disable network interface check” flag, but keeping Node and Next updated sometimes fixes it.

4. **Docker / CI**  
   Ensure the process has normal network capability and that `os.networkInterfaces()` is allowed.

---

## 4. FaceLiveness: challenge order bug (fixed earlier)

**What it was**

- In `FaceLiveness.tsx`, the challenge list was created **inside the component** with a **random sort**:
  ```ts
  const challenges = ["blink", "turn_left", "turn_right"].sort(() => Math.random() - 0.5);
  ```
- So every render got a **new random order** and `challengeIndexRef` could point at the wrong challenge.

**Fix (already applied)**

- Use a **fixed** list **outside** the component:
  ```ts
  const CHALLENGES: Challenge[] = ["blink", "turn_left", "turn_right"];
  ```
- Use `CHALLENGES` everywhere (initial state, `advance()`, detection loop) so the order is stable.

**Where:** `components/FaceLiveness.tsx` — search for `CHALLENGES`.

---

## 5. Optional: Google Fonts / “Failed to fetch” during build

**What you might see**

- During `next build`, warnings or errors about:
  - `Failed to fetch 'Geist' from Google Fonts`
  - `Failed to fetch 'Geist Mono' from Google Fonts`

**Where it comes from**

- **File:** `app/layout.tsx`
- **Code:**  
  `import { Geist, Geist_Mono } from "next/font/google";`  
  and use in `className={geistSans.variable}` etc.
- Next.js fetches font CSS at build time; in offline or locked-down networks it can fail.

**What you can do**

- Ensure network is available when running `npm run build`.
- Or switch to local fonts / system fonts in `layout.tsx` if you don’t need Geist.

---

## Quick reference: important files

| Purpose              | File |
|----------------------|------|
| Build script (webpack fix) | `package.json` → `"build": "next build --webpack"` |
| Home page            | `app/page.tsx` |
| Client entry for face flow | `components/FaceAuthClient.tsx` |
| Full flow (liveness → challenge → enroll) | `components/FaceAuthFlow.tsx` |
| Chained liveness     | `components/FaceLiveness.tsx` (uses `CHALLENGES`) |
| No more `/` conflict | `pages/` empty (no `pages/index.js`) |

---

## Summary

1. **Build:** Use `next build --webpack` (in `package.json`) so MediaPipe/TF packages build correctly.
2. **Route conflict:** Only one router for `/` — we kept `app/page.tsx`, removed `pages/index.js`.
3. **Dev server error:** Environment-specific; use `npm run dev` locally or `npm run build && npm run start` to test.
4. **FaceLiveness:** Use a fixed `CHALLENGES` array (already in place).
5. **Fonts:** Build needs network for Google Fonts, or change to local/system fonts.

If you hit a new error, compare the message to the “Export doesn’t exist” and “both match path” sections above first; then check the file and script listed for that issue.
