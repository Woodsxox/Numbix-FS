# FaceNet model setup for face enrollment

The enrollment step needs a **TensorFlow.js Layers** FaceNet model that outputs **128‑dimensional face embeddings**. The app loads it from `public/models/facenet/model.json`.

Your repo already has a `model.json` and `.bin` files in that folder, but the current `model.json` is **raw Keras** (architecture only). TensorFlow.js expects a **wrapped** format with `modelTopology` and `weightsManifest`. You need to **convert** a Keras or SavedModel FaceNet into that format and replace the files.

---

## What you need

- **Input:** A FaceNet-style model in one of these forms:
  - Keras `.h5` file (single file with architecture + weights), or
  - Keras SavedModel directory, or
  - TensorFlow SavedModel (then we use a different converter path).
- **Output:** In `public/models/facenet/`:
  - `model.json` — must contain **modelTopology** and **weightsManifest** (paths to the `.bin` files).
  - One or more `.bin` weight shards (e.g. `group1-shard1of2.bin`).

The app uses **`tf.loadLayersModel("/models/facenet/model.json")`**, so the JSON must be the TF.js Layers format, not raw Keras.

---

## Option 1: Convert from a Keras .h5 file (recommended)

### 1. Get a FaceNet Keras .h5 model

You need a pre-trained FaceNet model that:

- Accepts input shape **160×160×3** (or we resize in code).
- Outputs a **128-D** embedding vector.

Examples (use at your own risk; verify license and source):

- **Community Keras .h5:**  
  Some repos provide a single `facenet_keras.h5` (e.g. search GitHub for “facenet keras h5” or “FaceNet keras 128”).  
  Example (no endorsement):  
  `https://github.com/a-m-k-18/Face-Recognition-System` (has `facenet_keras.h5` in the repo; you may need to use Git LFS or a release asset).
- **Build and save yourself:**  
  Use [nyoki-mtl/keras-facenet](https://github.com/nyoki-mtl/keras-facenet) or [davidsandberg/facenet](https://github.com/davidsandberg/facenet) to get a Keras model, then save as `.h5`:
  ```python
  model.save("facenet_keras.h5")
  ```

Download or create the `.h5` and put it somewhere, e.g. `~/Downloads/facenet_keras.h5`.

### 2. Install TensorFlow.js converter

On macOS/Linux, use `python3 -m pip` if `pip` is not found:

```bash
python3 -m pip install tensorflowjs
```

Or with a venv:

```bash
python3 -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install tensorflowjs
```

### 3. Run the converter

From the **project root** (so paths below match):

```bash
# Create output dir
mkdir -p public/models/facenet

# Convert (replace path to your .h5)
tensorflowjs_converter \
  --input_format=keras \
  --output_format=tfjs_layers_model \
  /path/to/facenet_keras.h5 \
  public/models/facenet
```

Example if the .h5 is in your home Downloads:

```bash
tensorflowjs_converter \
  --input_format=keras \
  --output_format=tfjs_layers_model \
  ~/Downloads/facenet_keras.h5 \
  public/models/facenet
```

The converter will:

- Write **model.json** (with `modelTopology` and `weightsManifest`).
- Write one or more **group*-shard*.bin** files into the same folder.

### 4. Verify

- Open `public/models/facenet/model.json` in an editor.
- You should see top-level keys **modelTopology** and **weightsManifest**.
- `weightsManifest` should list paths like `"group1-shard1of2.bin"` that exist in the same directory.

Then run the app: after the liveness step, enrollment should load the model without the previous error.

---

## Option 2: Convert from Keras SavedModel

If you have a Keras model saved as SavedModel (a directory with `saved_model.pb` and `variables/`):

```bash
tensorflowjs_converter \
  --input_format=tf_saved_model \
  --output_format=tfjs_layers_model \
  /path/to/saved_model_directory \
  public/models/facenet
```

If the model is **TensorFlow** (not Keras) SavedModel, use the same `--input_format=tf_saved_model`; for Graph (frozen) models the format differs (see [TF.js conversion guide](https://www.tensorflow.org/js/guide/conversion)).

---

## Option 3: Use the project script (after you have .h5)

We provide a small script that runs the converter and writes output into `public/models/facenet/`. You still need a `.h5` file (or use the one in the repo).

```bash
# From project root
chmod +x scripts/convert-facenet.sh
./scripts/convert-facenet.sh /path/to/facenet_keras.h5
```

**If you have the archive folder:** This repo includes an `archive 2` folder with `facenet_keras.h5`. You can convert that and replace the current model:

```bash
# From project root (quote the path because of the space)
./scripts/convert-facenet.sh "archive 2/facenet_keras.h5"
```

Requires: `pip install tensorflowjs` (or `tensorflowjs[wizard]`).

---

## Input shape and 128-D output

- Our code in **lib/faceCrop.ts** and **components/FaceEnrollment.tsx** crops and resizes the face to the size the model expects (often **160×160**). If your model expects a different size, you’ll need to change the resize dimensions in the code to match.
- **lib/facenet.ts** assumes the model’s output is a **single vector** (e.g. 128-D). If your model has a different output shape (e.g. batch dimension only squeezed), the existing `getEmbedding` may still work; otherwise you’ll need to adjust the squeeze/reshape logic in `lib/facenet.ts`.

---

## Troubleshooting

| Problem | What to do |
|--------|------------|
| “The JSON contains neither model topology or manifest for weights” | Your `model.json` is still raw Keras. Replace it (and the .bin files) with the output of `tensorflowjs_converter` (Option 1 or 2). |
| “Could not load FaceNet” / 404 | App couldn’t load from `/models/facenet/model.json`. Check that `public/models/facenet/model.json` exists and has `modelTopology` and `weightsManifest`, and that all paths in `weightsManifest` point to existing `.bin` files in the same folder. |
| Model loads but embedding dimension is wrong | Inspect `model.predict()` output shape in `lib/facenet.ts` and adjust `getEmbedding` (squeeze/reshape) so the app gets a 128-D (or your chosen size) vector. |
| Converter says “Unknown layer” or “Not supported” | Some Keras layers aren’t supported by TF.js. You may need to use a different FaceNet implementation or a model known to convert cleanly. |

---

## Summary

1. Get a FaceNet Keras `.h5` (or SavedModel) that outputs 128-D embeddings.
2. Install `tensorflowjs` and run `tensorflowjs_converter` with **output_format=tfjs_layers_model** into `public/models/facenet`.
3. Ensure `model.json` has **modelTopology** and **weightsManifest** and that the referenced `.bin` files are in the same directory.
4. Restart the app and run through liveness → enrollment; the previous “Could not load FaceNet” error should be resolved.

---

## Optional: GitHub fallback (no CDN)

If you host your converted model in a GitHub repo, the app can use it as a fallback when the local model fails. This uses a normal fetch from GitHub (model files only; no script execution).

1. Push your `public/models/facenet/` contents (model.json + all `.bin` shards) to a repo.
2. Set the env var (e.g. in `.env.local`):
   ```bash
   NEXT_PUBLIC_FACENET_GITHUB_URL=https://raw.githubusercontent.com/YOUR_USER/YOUR_REPO/main/public/models/facenet/model.json
   ```
3. Rebuild or restart dev. If the local load fails, the loader will try this URL. Weight paths in `model.json` are relative, so shards are fetched from the same directory on GitHub.

For build/runtime issues unrelated to the model, see **ISSUES_AND_FIXES.md**.
