# Roadmap: FunctionGemma 270M Stable Autonomous Agent

## Vision
Build a stable, autonomous AI agent on mobile that provides full awareness capabilities (flashlight, calendar, notes, alarms) using FunctionGemma 270M on resource-constrained devices (3.5GB RAM).

## Philosophy
- **Stability over capability**: 270M params for 3.5G RAM vs 1B+ models that crash
- **NPU → CPU fallback**: Graceful degradation, not hard crashes
- **Starting point**: Flashlight is the intended first skill; it is *not* yet implemented (see Known gaps)
- **Modular actions**: Each action independent, one failure won't cascade
- **Memory persistence**: Context survives restarts via SQLite/SharedPreferences

## Known gaps (verified against this tree)

Status after the 2026-09-24 pass:

1. **Nothing has ever been compiled.** ✅ **resolved 2026-09-25.** `:app` (AGP 8.13.2 /
   Kotlin 2.2.21 / minSdk 24 — raised from 23 because the LiteRT-LM AAR declares 24) builds: `assembleDebug` → 852KB APK, `lintDebug` → 0 errors.
   Required fixing the SDK layout (cmdline-tools had to move to `<sdk>/cmdline-tools/latest/`),
   installing `platforms;android-35` + `build-tools;35.0.0`, and generating the wrapper JAR.
   The 289MB `litertlm` is still not loadable — see item 2.
2. **No model inference in the legacy loop.** 🟡 partially resolved. The AAR
   coordinate was *not* unresolved — it was looked up under the wrong group.
   `com.google.ai.edge.litertlm:litertlm-android:0.16.0` serves fine (HTTP 200
   verified 2026-10-05, and pinned by upstream's own
   `samples/litert/qualcomm/gemma3/cpu_gpu/gradle/libs.versions.toml`). It is now
   declared in `app/build.gradle.kts` and loaded by `ToolRoutingProbeActivity`.
   Still open: `AgentLoop.kt` reasons through `simulateFunctionGemmaResponse`
   rather than the model.

   What wiring a real model in showed: **the 270M bundle routes tool calls at 22%
   on this device** (4/18, 72% refusals, 3.12 s median, 983 MB peak RSS). It is not
   a usable router. `AgentOnPhoneActivity` routes at 67% on a held-out set for
   285 MB and 0.6 s using a 22M sentence embedder and no LLM at all. That result
   is why the agent does not call `simulateFunctionGemmaResponse()` either.
3. **Dangling references.** ✅ *fixed for torch*: `TorchManager` now exists and uses
   `CameraManager.setTorchMode()`; the `TorchService` system-service cast is gone. ⬜ still
   open in `AgentLoop.kt`: C-style ternary at :120, `ErrorLogger()` called where the nested
   class is `ErrorLoggerImpl`, and inner classes constructed with the wrong arity.
4. **Pasted tool transcript in a source file.** ✅ `NoteDatabase.kt` truncated to its
   code; the 200 orphan lines that sat *after* `object ActionRegistry` closed (a second
   draft containing Python `True`/`False`) are preserved in
   `legacy/ActionRegistry.orphan-tail.kt.disabled` rather than deleted.
5. **Repo hygiene.** ✅ `.gitignore` excludes model weights, build output, and the
   delegate caches LiteRT-LM writes next to a bundle (`*.xnnpack_cache`,
   `*mldrift*.bin`, `.litertlm-cache/` — ~277MB from the 289MB model, none of which
   the `*.litertlm` rule caught). The 289MB `litertlm` can never be committed anyway
   (GitHub rejects >100MB files), and the two 54-byte `model.tflite` stubs were
   removed from the index. ✅ Apache-2.0 `LICENSE` added; the 90MB router model is
   gitignored and fetched by `scripts/fetch_router_model.sh`.

## Milestone Timeline

### M1: Foundation (Weeks 1-2) — partial, see blockers
- [x] ActionRegistry.kt: 8 functions mapped to Android actions
- [x] AgentLoop.kt: 500ms perceive→reason→act→loop with error boundaries
- [x] Note persistence in SQLite (`NoteDatabase.kt`; replaced the SharedPreferences design)
- [ ] Python ActionRegistry logic verified PASS — no harness or test file exists in this repo
- [ ] LiteRT inference confirmed working on SM-A14 — no real model has been executed
      (`assets/*.tflite` are 54-byte stubs that `tflite-runtime` rejects, and the Kotlin
      loop still uses a keyword simulator at `AgentLoop.kt`)
- [x] AgentLoop stability: 8/8 iterations **in the simulator**, not with model output
- [x] Obtain FunctionGemma 270M `.litertlm` model (`function-gemma-q8-ekv1024.litertlm`, 289MB)
- [x] Place model in `assets/` directory
- [ ] (Optional) Copy NPU libraries from Gallery app
- [ ] Build and run Android app on SM-A14 — blocked: no Gradle project or manifest existed
- [ ] Start AgentLoop - 500ms autonomous cycle begins

### M2: Core Awareness (Weeks 3-4) 🔄 IN PROGRESS
- [x] `CalendarService.kt` created — not tested on device, no real credentials wired
- [x] Note persistence improvement (`NoteDatabase.kt` — SQLite)
- [x] `TtsConfig.kt`: Text-to-Speech configuration
- [ ] `ErrorLogger.kt` as its own file — currently only an inner class `ErrorLoggerImpl`
      in `AgentLoop.kt`; it delegates to a top-level `ErrorLogger` that does not exist
- [ ] `BatteryOptimizer.kt` as its own file — currently an inner class in `AgentLoop.kt`
- [ ] AgentLoop.kt integration with all Phase 2 components
- [ ] Calendar query test on SM-A14
- [ ] Note append/read test on SM-A14
- [ ] TTS speak test on SM-A14
- [ ] Battery optimization test on SM-A14

### M3: Extended Skills (Weeks 5-6) ⏳ PLANNED
- [ ] Media control (volume, playback)
- [ ] Settings toggles (Wi-Fi, Bluetooth, etc.)
- [ ] Weather queries
- [ ] Email composition (intent-based)
- [ ] App launching (package name based)
- [ ] Location queries (FusedLocationProviderClient)

### M4: Autonomy Enhancements (Weeks 7-8) ⏳ PLANNED
- [ ] Wake word detection (always-listening mode)
- [ ] Context-aware prompt engineering (last N interactions)
- [ ] Multi-modal input (voice + screen context)
- [ ] Personalization (user preferences, learned patterns)
- [ ] Offline mode support (no internet required)
- [ ] Battery-friendly scheduling (adaptive intervals)

### M5: Production Polish (Weeks 9-10) ⏳ PLANNED
- [ ] Material Design 3 UI overhaul
- [ ] Multiple language support (i18n)
- [ ] Firebase Analytics integration
- [ ] Play Store release setup
- [ ] Over-the-air (OTA) update mechanism
- [ ] User feedback system
- [ ] Privacy policy and data handling documentation

### M6: Ecosystem (Weeks 11-12) ⏳ PLANNED
- [ ] Wear OS companion app
- [ ] Wear OS voice wake word
- [ ] Smart home integration (IoT control)
- [ ] Cross-device sync (phone ↔ tablet ↔ wear OS)
- [ ] Multi-user support
- [ ] Subscription or one-time purchase model

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **App stability** | >99% crash-free sessions | Logcat monitoring over 24h |
| **Loop stability** | 500ms ± 50ms per iteration | Timing tests every iteration |
| **Function success rate** | >95% per action | Test each skill 20x |
| **RAM usage** | <150MB active | `dumpsys memoryinfo` |
| **Battery impact** | <5%/hour idle | Battery drain monitoring |
| **Voice recognition** | >90% accuracy | Offline STT accuracy |
| **Wake word** | <2s response time | From trigger to action |

## Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Model crashes on init** | High (with 1B models) | Low (using 270M) | Use FunctionGemma 270M exclusively |
| **NPU library compatibility** | Medium | Medium | Start CPU-only, add NPU later |
| **Calendar API key issues** | Medium | Medium | Fallback to hardcoded test data |
| **RAM exhaustion (3.5GB)** | Medium | High | Batch size 1, aggressive GC |
| **TTS spurious output** | Low | Low | Only speak on `should_speak=true` |
| **Invalid command loops** | Low | Medium | No-op fallback + error backoff |

## Dependencies

### External Services
- Google Calendar API (requires API key, fallback to test data)
- Hugging Face (for model download, requires access approval)
- Optional: NPU libraries from Gallery app or Qualcomm AI Hub

### Internal Dependencies
- Kotlin 1.9+ Android SDK
- TensorFlow Lite LiteRT 2.14.0+
- FunctionGemma 270M `.litertlm` model
- Google AI Edge Gallery app (for model download reference)

### System Requirements
- SM-A14 (SM-A145F) or equivalent Android 15 device
- 3.5GB+ RAM
- ADB access for development
- Optional: NPU-enabled Snapdragon device for acceleration

## Contributing Guidelines

### Adding New Skills
1. Add function to `ActionRegistry.kt`:
   ```kotlin
   "new_function_name" to { ctx: Context, params: Map<String, Any> ->
       return ActionResult(
           success = true,
           output = "Your output message",
           should_speak = true,
           speak_text = "Your speak text"
       )
   }
   ```

2. Add to AgentLoop prompt (buildPrompt function):
   ```kotlin
   "new_function_name" → {"name": "new_function_name", "parameters": {}}
   ```

3. Test with Python simulation before integrating into Kotlin

### Pull Request Process
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request with clear description

### Code Style
- Follow Kotlin coding conventions
- All public functions must have KDoc documentation
- All actions must have try/catch → ActionResult
- TTS flags must be set appropriately
- Error messages must be user-friendly

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- **FunctionGemma** - Google's 270M parameter function-calling model
- **LiteRT** - TensorFlow Lite runtime for on-device inference
- **SM-A14** - Test device (Android 15, 3.5GB RAM)
- **Google AI Edge Gallery** - Reference app and model source
- **Kotlin** - Official Android language
- **TensorFlow Lite** - On-device ML framework

---

*Roadmap v1.0 - Created during build phase of autonomous agent project.*
