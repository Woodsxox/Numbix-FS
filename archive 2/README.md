# Archive 2 – FaceNet model files

| File | What it is | Use in app? |
|------|------------|--------------|
| **facenet_keras.h5** | Full Keras model (architecture + weights). Input: 160×160×3, output: 128-D embedding. | ✅ **Use this** – convert to TF.js with the project script. |
| **model.h5** | Another Keras model file (may be same or variant). | ✅ Can try if `facenet_keras.h5` fails to convert. |
| **model.json** | Keras **architecture only** (no TF.js weights manifest). Same as the one currently in `public/models/facenet/`. | ❌ Not loadable by the app as-is; need converted output. |
| **weights.h5** | Weights only; needs the architecture from `model.json` to be usable. | ❌ Use `facenet_keras.h5` instead for conversion. |

**To use your model in the app:** From the project root run:

```bash
# If `pip` is not found, use:
python3 -m pip install tensorflowjs

./scripts/convert-facenet.sh "archive 2/facenet_keras.h5"
```

That writes a proper `model.json` + `.bin` shards into `public/models/facenet/` so the enrollment step can load FaceNet.
