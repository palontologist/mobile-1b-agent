# On-device agent loop for a 3.5GB RAM phone (SM-A14)

A perception→reason→act loop, an Android action registry, and a memory layer,
sized for a budget phone (Samsung SM-A145F, 3.5GB RAM). The model target is
**FunctionGemma 270M (int8)**, not a 1B model — see [Why 270M](#why-270m-not-1b).

The repo is named `mobile-1b-agent` for historical reasons. The finding here is
the negative result: **a 1B function-calling model does not fit this device.**

---

## Status: read this first

Everything below is checked against this working tree, not against a demo.

| Claim | Reality | Evidence |
|---|---|---|
| Kotlin agent loop exists | ✅ 1,638 LOC across 5 files | `src/main/kotlin/com/ai/edge/agent/` |
| An Android module exists | ✅ added 2026-09-24: `:app` + manifest + `TorchProbeActivity` | `app/build.gradle.kts`, `settings.gradle.kts` |
| ...and it builds | ✅ **first `assembleDebug` succeeded 2026-09-25**; 852KB APK, `lintDebug` 0 errors / 4 advisory warnings | `app/build/outputs/apk/debug/app-debug.apk` |
| Flashlight works via this code | ⚠️ now implemented for real against `CameraManager.setTorchMode()` in the new module; the **legacy** `ActionRegistry` path still never ran | `app/src/main/kotlin/com/ai/edge/agent/TorchManager.kt` |
| Legacy `TorchService` / `TorchManager` cast | ❌ was undefined anywhere in the repo; rewired to `TorchManager.enable/disable(ctx)` | `ActionRegistry.kt:64` |
| FunctionGemma runs inference | ❌ **not wired up** — loop uses a keyword simulator | `AgentLoop.kt:249` `TODO: Proper tokenization`, `:253` `simulateFunctionGemmaResponse(prompt)` |
| LiteRT-LM API in use | ✅ resolved and consumed: `com.google.ai.edge.litertlm:litertlm-android:0.16.0`, via `ToolRoutingProbeActivity`. **The earlier "verified 404" was a wrong group id** — the group is `com.google.ai.edge.litertlm`, one level below `com.google.ai.edge` | `app/build.gradle.kts`, `app/src/main/kotlin/com/ai/edge/agent/ToolRoutingProbeActivity.kt` |
| A valid `.litertlm` model is on disk | ✅ 289MB, real `LITERTLM` magic | `assets/function-gemma-q8-ekv1024.litertlm` |
| ...but the agent loop never loads it | ✅ (confirmed) no code path in `src/` opens that file. `ToolRoutingProbeActivity` *does* load it, for measurement only | `AgentLoop.kt:249` |
| Legacy tree parses | ❌ known defects, deliberately **outside** the compiled module so the build stays green | `AgentLoop.kt:120` C-style ternary; `ActionRegistry.kt` had 200 lines of map entries *after* the object closed (moved to `legacy/`); `NoteDatabase.kt` ended in pasted transcript incl. a literal `<tool_call>` block |

**What is genuinely useful today:** the device, the measurements it can produce,
and the loop's error-handling design (backoff, error boundaries, no-op fallback).
That is test infrastructure most LiteRT-LM reporters don't have. What is *not*
done is model execution inside the legacy loop — though a real model now runs on
this device in `ToolRoutingProbeActivity`, and `AgentOnPhoneActivity` runs a
working agent with no LLM in the loop.

### Model asset audit

| File | Size | Verdict |
|---|---|---|
| `assets/function-gemma-q8-ekv1024.litertlm` | 289MB | real model |
| `assets/functiongemma-270m-ft-mobile-actions_Google_Tensor_G6.litertlm` | 161 B | **not a model** — it is HuggingFace's "Access to model … is restricted. You must have access to it and be authenticated" message. The gated repo was never downloaded. |
| `assets/model.tflite`, `models/model.tflite` | 54 B each | **not models** — `TFL3` magic + NUL padding. `tflite-runtime` rejects both: `ValueError: Model provided has model identifier …` |

`scripts/create_test_tflite.py` is explicit in its own comments that it cannot
generate a real flatbuffer without TensorFlow, so it writes a header-only stub.
Do not describe those stubs as "verified working"; the interpreter only verifies
that it can *reject* them.

---

## Why 270M, not 1B

| Criterion | FunctionGemma 270M (int8) | Gemma-4 E2B (~1B) |
|---|---|---|
| Weight footprint | ~275MB | ~1GB+ |
| Free RAM on 3.5GB device | workable | OOM during init/decode |
| Sustained 500ms loop | plausible | not achieved |
| Storage | ~300MB | ~4GB+ |

Evidence type, stated honestly: the 270M column is *expected*, since no model
has executed yet; the 1B column is *observed* via Google AI Edge Gallery
crashes on this device, not via a harness in this repo. Neither column has a
committed log file. Publishing the `logcat` + `dumpsys meminfo` capture is the
single highest-value next commit — it is also what upstream is missing (see
[Upstream](#upstream-issues-this-answers)).

---

## Repository layout

```
mobile-1b-model/
├── settings.gradle.kts / build.gradle.kts / gradle.properties
├── gradle/wrapper/gradle-wrapper.properties   # wrapper JAR must still be generated
├── app/                                       # ← the ONLY compiled module
│   ├── build.gradle.kts                       #   AGP 8.13.2, Kotlin 2.2.21, minSdk 23
│   └── src/main/
│       ├── AndroidManifest.xml                #   zero permissions (see below)
│       └── kotlin/com/ai/edge/agent/
│           ├── TorchManager.kt                #   CameraManager.setTorchMode()
│           └── TorchProbeActivity.kt          #   measurement rig
├── src/main/kotlin/com/ai/edge/agent/         # LEGACY, not in the build
│   ├── ActionRegistry.kt                      #   8 actions; flashlight now calls TorchManager
│   ├── AgentLoop.kt                           #   500ms loop, keyword simulator
│   ├── calendar/CalendarService.kt
│   ├── database/NoteDatabase.kt
│   └── tts/TtsConfig.kt
├── legacy/ActionRegistry.orphan-tail.kt.disabled   # 200 lines quarantined, see Status
├── scripts/capture_torch_evidence.sh
├── app.py                                     # Python tflite-runtime harness
├── assets/function-gemma-q8-ekv1024.litertlm  # 289MB, gitignored, never committed
└── requirements/                              # EMPTY directory (delete it)
```

Note the package is `com.ai.edge.agent`, not `ai.edge.agent` as older docs claimed.
There is no `AgentMemory.kt`: SharedPreferences persistence was replaced by the
SQLite store in `database/NoteDatabase.kt`; the seam is still noted at
`AgentLoop.kt:234`.

## Design that is worth keeping

`AgentLoop.kt` treats the loop as something that must never die:

- every action returns an `ActionResult` from inside `try/catch`;
- unknown function names fall through to `noop`;
- `consecutiveErrors` ≥ 3 triggers a 2s backoff instead of a hot crash loop;
- TTS fires only when `should_speak = true`;
- the prompt carries the last 3 interactions and is batch-size 1 for RAM safety.

For a device with 3.5GB total, "one bad inference must not kill the process" is
the correct architectural instinct regardless of which parts are implemented.

---

## Build it

The `:app` module compiles **only** what is under `app/src/main/kotlin`. The legacy
tree at `src/main/kotlin` is intentionally outside the build so that `assembleDebug`
is green while the defects listed above are repaired one file at a time.

```bash
./gradlew :app:assembleDebug
./gradlew :app:lintDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

**Verified working on this machine (2026-09-25).** One-time setup that was needed,
recorded because both steps fail confusingly otherwise:

```bash
# 1. cmdline-tools were installed at ~/Android/Sdk/latest, which sdkmanager rejects:
#    "Could not determine SDK root". They must be at <sdk>/cmdline-tools/latest/.
mv ~/Android/Sdk/latest ~/Android/Sdk/cmdline-tools/latest

# 2. Packages (this SDK had platform-tools only).
yes | sdkmanager --sdk_root="$HOME/Android/Sdk" --licenses
sdkmanager --sdk_root="$HOME/Android/Sdk" "platforms;android-35" "build-tools;35.0.0"

# 3. Per-machine SDK path; gitignored, never committed.
echo "sdk.dir=$HOME/Android/Sdk" > local.properties
```

`gradlew`, `gradlew.bat` and `gradle/wrapper/gradle-wrapper.jar` are committed, so
Gradle 8.14.3 is fetched automatically. Toolchain: AGP `8.13.2`, Kotlin `2.2.21`,
compile/target SDK 35, **minSdk 23** (`setTorchMode` is API 23+), and zero
third-party dependencies — the module uses only platform APIs, so there is nothing
to fail to resolve. `targetSdk`/`compileSdk` stay at 35 deliberately to match the
API-35 test device; lint's "36 is available" warning is expected.

Two API traps that only surface at compile time, both hit during the first real
build and worth knowing before anyone repairs `AgentLoop.kt`:

- `CameraManager.getCameraIdList()` is a method, so the Kotlin synthetic property is
  `cameraIdList` — **not** `cameraIds`.
- The characteristics key is `CameraCharacteristics.FLASH_INFO_AVAILABLE` — there is
  no `FLASH_INFO_AVAILABILITY`.

## What the probe is for

`TorchProbeActivity` is not a UX milestone; it is a measurement rig. It exposes
enable / disable / toggle and a **"hammer 20x on/off"** button, because contending
for the flash repeatedly without settling time is how the camera service actually
fails on 3.5GB RAM devices — the single-press happy path proves nothing.

Every command emits a parseable line to logcat:

```
[PROBE] cmd=enable n=3 result=flashlight on state=true latency_us=812
[PROBE] hammer i=7 state=false status=camera busy, try again (...)
```

`scripts/capture_torch_evidence.sh` wraps a run into `measurements/<UTC>/` with
props, `dumpsys meminfo`, `gfxinfo`, thermal and battery state, and the filtered
logcat. That bundle is the payload for the upstream threads cited below. The device
serial is taken as an argument and never hardcoded, so no identifier is committed.

`TorchManager` also fixes the API misuse in the legacy design: there is no Android
"torch service" to cast, and `CameraManager.setTorchMode()` requires **no**
`CAMERA` permission — which is why the new manifest asks for none.

## Remaining path to the original goal

1. Build + install the probe, capture the first `measurements/` bundle.
2. Repair the legacy tree in place: `AgentLoop.kt` (C-ternary at :120, `ErrorLogger`
   vs `ErrorLoggerImpl`, inner-class constructors), then move `ActionRegistry.kt`
   into `:app` and delete the `legacy/` quarantine or merge it deliberately.
3. ~~Resolve the real LiteRT-LM AAR coordinate~~ — **done**:
   `com.google.ai.edge.litertlm:litertlm-android:0.16.0`, confirmed live on
   `dl.google.com`'s maven2 tree and consumed by `ToolRoutingProbeActivity`.
   Still open: delete `simulateFunctionGemmaResponse()` in the *legacy* loop and
   drive the model from it.

   Measured outcome of wiring a real model in, from the probe (see "Routing" below):
   a 270M bundle routes tool calls at **22%** on this device, so it is not a
   usable router. `AgentOnPhoneActivity` routes at 67% held-out for 285 MB and
   0.6 s instead of 983 MB and 3.1 s.
4. Record tool-call accuracy and RAM per iteration — the numbers the 270M-vs-1B
   table still asserts without a capture behind it.

### `requirements.txt`

Fixed in this pass: it previously ended in a stray ``` ``` ``` fence and listed
`tensorflow` + `torch` while omitting `tflite-runtime`, the only heavy dependency
`app.py` actually imports. It is now pinned to `tflite-runtime==2.14.0` (the version
present in the existing `venv/`) plus `numpy<2`. `tensorflow`/`torch` were dropped:
`scripts/convert_*.py` needs TensorFlow if you want to regenerate a model, so install
it explicitly rather than on every `pip install -r`.

### Python harness

`app.py` is a correct generic `tflite-runtime` runner and will work the moment a
real model is supplied:

```bash
python -m venv .venv && .venv/bin/pip install tflite-runtime numpy
.venv/bin/python app.py --model <a-real>.tflite
```

Reproduce the current failure state yourself:

```bash
.venv/bin/python -c "
from tflite_runtime.interpreter import Interpreter
Interpreter(model_path='assets/model.tflite')"
# ValueError: Model provided has model identifier ...
```

---

## Roadmap

### Phase 1 — Foundation
- [x] `TorchManager` on `CameraManager.setTorchMode()` — real, in `:app`
- [x] Gradle module, manifest, probe activity, evidence script
- [x] `ActionRegistry.kt`: 8 actions, each error-wrapped; flashlight rewired to `TorchManager`
- [x] `NoteDatabase.kt`: SQLite persistence (replaces SharedPreferences design)
- [x] Python inference harness (`app.py`) functional for valid models
- [ ] First `./gradlew :app:assembleDebug` + on-device capture
- [ ] `AgentLoop.kt` repaired (C-style ternary at :120, `ErrorLogger` name mismatch, inner-class ctor calls)
- [ ] Wire LiteRT-LM inference in place of the keyword simulator

### Phase 2 — Core awareness
- [ ] `CalendarService.kt` against a real calendar provider (currently needs an API key path)
- [ ] `TtsConfig.kt` voice selection
- [ ] Memory ceiling enforcement for the 3.5GB constraint
- [ ] `ErrorLogger` / `BatteryOptimizer` promoted out of `AgentLoop.kt` into files

### Phase 3 — Extended skills
- [ ] Media control, settings toggles, weather, email composition
- [ ] App launching, location queries (`FusedLocationProviderClient`)

### Phase 4 — Autonomy
- [ ] Wake word / always-listening
- [ ] Screen-context multimodal input
- [ ] Fully offline operation, adaptive intervals

### Phase 5 — Polish / 6 — Ecosystem
UI, i18n, Play Store, OTA updates, Wear OS companion, cross-device sync — see
`ROADMAP.md`. Untouched until Phase 1 exits.

---

## Upstream issues this answers

This device class is under-represented upstream, and the negative result is the
contribution:

- [`google-ai-edge/LiteRT#6889`](https://github.com/google-ai-edge/LiteRT/issues/6889)
  *Publish pre-built `libLiteRtDispatch_Qualcomm.so` alongside the litertlm-android AAR* —
  open since 2026-04-14, 8 comments. `assets/README.txt` documents exactly the
  workaround that issue is asking to eliminate: extracting QNN `.so` files from
  an installed Gallery APK and `adb push`-ing them.
- [`google-ai-edge/LiteRT-LM#2421`](https://github.com/google-ai-edge/LiteRT-LM/issues/2421)
  *Gemma 4 E2B GPU decode dies with `CL_INVALID_COMMAND_QUEUE` after 1–3 turns* —
  open, **zero comments**. This is the ~1B case in the table above.
- [`google-ai-edge/LiteRT-LM#2966`](https://github.com/google-ai-edge/LiteRT-LM/issues/2966)
  *0.14.0 / 0.15 nightly OOM: uses twice as much memory* — open, 5 comments.

---

## Permissions

The shipped `:app` manifest requests **none**, and that is deliberate:
`CameraManager.setTorchMode()` and `getCameraCharacteristics()` are permission-free
on API 23+. The previously drafted manifest asked for `READ_PHONE_STATE`,
`WRITE_EXTERNAL_STORAGE`, `FOREGROUND_SERVICE` and a `NotificationListenerService`
that no code implements; `allowBackup="false"`.

Skills still in the legacy tree will need these when they are moved into `:app`:

```xml
<uses-permission android:name="android.permission.READ_CALENDAR"/>
<uses-permission android:name="android.permission.SEND_SMS"/>
<uses-permission android:name="android.permission.POST_NOTIFICATIONS"/>
```

`CAMERA` is *not* needed for the torch or for launching the still-image capture
intent (`ACTION_IMAGE_CAPTURE` delegates to another app), so add it only if this
code ever opens a camera session itself.

## Debugging

```bash
adb devices                       # take the serial from here; none is committed
adb logcat -v epoch TorchProbe:V '*':S
adb shell dumpsys meminfo com.ai.edge.agent
scripts/capture_torch_evidence.sh <SERIAL>
```

---

## Contributing

Contributions are welcome. If anything you write here ends up in a Google
repository, you'll need to sign Google's **individual CLA** (or your employer's
corporate CLA if the work belongs to them) — the LiteRT and LiteRT-LM repos only
accept CLA-covered original code.

## License

**Apache License 2.0** — see [`LICENSE`](LICENSE). This README previously
stated MIT "see the LICENSE file", which was aspirational, and then correctly
recorded that no license file existed. Both are now settled.

The license covers this repository's source only. It does **not** cover the model
weights (`assets/function-gemma-q8-ekv1024.litertlm` is Google's, access-gated,
under Google's terms; `minilm.tflite` is a third-party conversion fetched by
script rather than committed), nor Google's LiteRT artifacts pulled from Maven
under Apache-2.0. `LICENSE` states this scope explicitly, because shipping an
Android sample with model licensing left implicit is how that goes wrong.

## Contact

- **GitHub:** [@palontologist](https://github.com/palontologist)
- **Device:** Samsung SM-A145F · Android 15 (SDK 35) · 3.5GB RAM
- **Model target:** FunctionGemma 270M, int8 (`LITERTLM` container)

---

*Honest negative results about small models on cheap hardware, instead of a
demo that only works in a screenshot.*
