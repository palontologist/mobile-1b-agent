FunctionGemma 270M Model Assets
================================

This directory should contain the FunctionGemma 270M model file for on-device inference.

Model File: function-gemma-q8-ekv1024.litertlm
- Format: TensorFlow Lite LiteRT-LM model
- Parameters: 270M (int8 quantized)
- Accelerators: CPU (XNNPACK), NPU (QNN v79), GPU (Vulkan)
- Input: Tokenized text prompt (variable length)
- Output: JSON function call (function name + parameters)

Where to obtain the model:
1. Google AI Edge Gallery app (downloads on first launch)
2. Qualcomm AI Hub (https://ai.qualcomm.com)
3. Hugging Face (requires authentication for litert-community repos)

Expected behavior after placing model:
1. Add function-gemma-q8-ekv1024.litertlm to this assets/ directory
2. Ensure NPU libraries are copied from Gallery app (optional for CPU mode):
   - libLiteRtDispatch_Qualcomm.so
   - libQnnHtp.so, libQnnHtpV79Skel.so, libQnnHtpV79Stub.so
   - libQnnSystem.so
3. Build and run the Android application
4. The AgentLoop will start its 500ms perceive-reason-act cycle

Available Skills (8 functions):
- open_flashlight    → TorchManager.enable()
- close_flashlight   → TorchManager.disable()
- query_calendar     → Google Calendar API
- append_note(title) → File append to agent_notes.txt
- set_alarm(hh:mm)   → AlarmClock intent
- take_photo         → Camera intent
- send_message()     → SMS intent
- noop               → No operation (fallback)

For troubleshooting, see: android_errors.md