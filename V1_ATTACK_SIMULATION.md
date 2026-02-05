# V1 Attack Simulation & Readiness Validation

**Phase A — Hands-on validation.** Run the app locally and test each attack below. Then fill in the results at the bottom.

---

## How to run

```bash
cd /Users/sethofoma/Documents/Face-power/my-face
npm run dev
```

Open **http://localhost:3000**. Complete the flow: **Liveness** (blink → turn left → turn right) → **Challenge** (same again) → **Enroll** (5 samples). For each attack, try to **bypass** or **break** that flow.

**Expected for attacks 1–4:** System should **FAIL** (no pass, no “Liveness verified”).  
**Expected for 5–6:** Pass or fail cleanly; no crashes or stuck states.

---

## 1️⃣ Static photo attack

| Test | Action | Expected |
|------|--------|----------|
| Printed face | Hold a printed photo of a face in front of the camera | **FAIL** — no blink/turn from photo |
| Face on screen | Show a face on another phone/laptop screen | **FAIL** |
| Screenshot | Hold phone showing a face screenshot | **FAIL** |

**Result:** PASS / FAIL  
**Notes:**

---

## 2️⃣ Replay attack

| Test | Action | Expected |
|------|--------|----------|
| Phone video | Play a video of someone blinking on a phone, show to camera | **FAIL** — timing/blink state should block |
| Laptop replay | Play same on laptop screen in front of camera | **FAIL** |

**Result:** PASS / FAIL  
**Notes:**

---

## 3️⃣ Partial compliance (blink cheating)

| Test | Action | Expected |
|------|--------|----------|
| Micro-blink | Blink very quickly / barely close eyes | **FAIL** |
| Slow blink | Close eyes very slowly | **FAIL** (or pass only if EAR drops enough) |
| Eye squint | Squint without full blink | **FAIL** |

**Result:** PASS / FAIL  
**Notes:**

---

## 4️⃣ Head turn cheating

| Test | Action | Expected |
|------|--------|----------|
| Lean body | Lean shoulders/body left or right instead of turning head | **FAIL** |
| Tilt chin | Tilt chin up/down or sideways, no real rotation | **FAIL** |

**Result:** PASS / FAIL  
**Notes:**

---

## 5️⃣ Environmental abuse

| Test | Action | Expected |
|------|--------|----------|
| Low light | Dim the room significantly | Pass **or** fail cleanly; no crash |
| Backlight | Strong light/window behind you | Pass or fail; no crash |
| Glasses | Wear glasses | Pass or fail; no crash |
| Cap / hoodie | Wear cap or hoodie (partial face) | Pass or fail; no crash |

**Result:** PASS / FAIL  
**Notes:** Any crashes or stuck states? Y/N

---

## 6️⃣ Multi-face confusion

| Test | Action | Expected |
|------|--------|----------|
| Two people | Two people in frame | Only one (primary) face processed; no pass unless **you** complete challenge |
| Person passing | Someone walks behind you | No pass unless you complete challenge; no random pass |

**Result:** PASS / FAIL  
**Notes:**

---

## Report template (copy and fill)

```
V1 Attack Simulation Results
- Photo attack:        PASS / FAIL
- Replay attack:       PASS / FAIL
- Partial blink:      PASS / FAIL
- Head turn cheat:    PASS / FAIL
- Low light:          PASS / FAIL
- Multiple faces:     PASS / FAIL

Notes:
- Any weird behavior?
- Any false positives?
```

---

**Interpretation:** For each row, **PASS** here means “the **test** passed” — i.e. the system behaved as expected (blocked the attack or failed cleanly). So e.g. “Photo attack: PASS” = “We tried a photo and the system correctly did **not** verify (good).” If the system wrongly verified on a photo, that’s a **FAIL** for that row.
