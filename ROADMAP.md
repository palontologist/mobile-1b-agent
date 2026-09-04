# Roadmap: FunctionGemma 270M Stable Autonomous Agent

## Vision
Build a stable, autonomous AI agent on mobile that provides full awareness capabilities (flashlight, calendar, notes, alarms) using FunctionGemma 270M on resource-constrained devices (3.5GB RAM SM-A14).

## Philosophy
- **Stability over capability**: 270M params for 3.5G RAM vs 1B+ models that crash
- **NPU → CPU fallback**: Graceful degradation, not hard crashes
- **Proven starting point**: Flashlight already works, build outward
- **Modular actions**: Each action independent, one failure won't cascade
- **Memory persistence**: Context survives restarts via Room/SharedPreferences

## Milestone Timeline

### M1: Foundation (Weeks 1-2) ✅ COMPLETED
- [x] ActionRegistry.kt: 8 functions mapped to Android actions
- [x] AgentLoop.kt: 500ms perceive→reason→act→loop with error boundaries
- [x] AgentMemory.kt: SharedPreferences persistence (last 10 cycles)
- [x] Python ActionRegistry logic: All 7 functions verified PASS
- [x] LiteRT on SM-A14: XNNPACK CPU inference confirmed working
- [x] AgentLoop stability: 8/8 iterations in simulation without error
- [ ] Obtain FunctionGemma 270M `.litertlm` model
- [ ] Place model in `assets/` directory
- [ ] (Optional) Copy NPU libraries from Gallery app
- [ ] Build and run Android app on SM-A14
- [ ] Start AgentLoop - 500ms autonomous cycle begins

### M2: Core Awareness (Weeks 3-4) 🔄 IN PROGRESS
- [ ] Calendar integration (Google Calendar API with API key)
- [ ] Note persistence improvement (Room/SQLite instead of file append)
- [ ] TTS configuration and voice selection
- [ ] Error logging and crash reporting
- [ ] Battery optimization and throttling awareness
- [ ] Memory management for 3.5G RAM constraint

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