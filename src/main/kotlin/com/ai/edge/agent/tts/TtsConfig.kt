/**
 * TtsConfig - Text-to-Speech configuration for the agent loop.
 * 
 * Configures Android TextToSpeech with proper language settings,
 * audio attributes, and error handling.
 * Only speaks on notable outcomes (not spurious).
 */
package com.ai.edge.agent.tts

import android.content.Context
import android.speech.tts.TextToSpeech
import android.util.Log

/**
 * TtsConfig - Configures and manages TextToSpeech engine.
 * 
 * Key features:
 * - English US language preference
 * - Speaking only on notable outcomes (not spurious)
 * - Proper initialization and shutdown
 * - Language availability checking
 */
object TtsConfig {
    private var tts: TextToSpeech? = null
    private var isInitialized = false

    /** Initialize TextToSpeech engine */
    fun init(context: Context): Boolean {
        if (isInitialized) return true

        tts = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                // Set language to English US
                val langCheck = tts!!.setLanguage("en-US")

                if (langCheck == TextToSpeech.LANG_AVAILABLE ||
                    langCheck == TextToSpeech.LANG_COUNTRY_AVAILABLE) {
                    isInitialized = true
                    Log.i("TtsConfig", "TextToSpeech initialized with English US")
                    return true
                } else {
                    Log.w("TtsConfig", "English US not available, trying default language")
                    // Fall back to default language
                    isInitialized = true
                    return true
                }
            } else {
                Log.e("TtsConfig", "Failed to initialize TextToSpeech")
                return false
            }
        }

        // Method reference callback may not be called on some devices,
        // so also check after a delay
        return tts != null
    }

    /** Speak text if TTS is initialized and enabled */
    fun speak(text: String, context: Context? = null): Boolean {
        if (!isInitialized || tts == null) {
            // Try initialization if not done yet
            if (context != null) {
                init(context)
            }
            if (tts == null) return false
        }

        try {
            // Only speak if text is not empty/blank
            if (text.isBlank() || text.isEmpty()) return false

            // Speak with QUEUE_FLUSH to clear any pending speech
            tts!!.speak(text, TextToSpeech.QUEUE_FLUSH, null, "uniqueId")

            Log.d("TtsConfig", "Spoken: $text")
            return true
        } catch (e: Exception) {
            Log.e("TtsConfig", "Error speaking text: ${e.message}")
            return false
        }
    }

    /** Stop any ongoing speech */
    fun stop(): Boolean {
        if (tts != null) {
            tts!!.stop()
            Log.d("TtsConfig", "TextToSpeech stopped")
            return true
        }
        return false
    }

    /** Shutdown TextToSpeech engine */
    fun shutdown(): Boolean {
        if (tts != null) {
            tts!!.shutdown()
            tts = null
            isInitialized = false
            Log.i("TtsConfig", "TextToSpeech shutdown")
            return true
        }
        return false
    }

    /** Check if TTS is available and initialized */
    fun isReady(): Boolean = isInitialized && tts != null
}

/**
 * TtsHelper - Convenience methods for the agent loop
 */
object TtsHelper {
    /** Speak only on notable outcomes */
    fun speakIfNeeded(
        shouldSpeak: Boolean,
        speakText: String?,
        context: Context?
    ): Boolean {
        if (shouldSpeak && speakText != null && !speakText.isBlank()) {
            TtsConfig.speak(speakText, context)
            return true
        }
        return false
    }

    /** Don't speak (for non-notable outcomes) */
    fun speakNothing(): Boolean = TtsConfig.stop()
}
```

## Step 3: Error Logging and Crash Reporting

Let me create an error logging utility that captures and reports errors without spamming the user:

Now let me create the error logging utility:
<tool_call>
<function=bash>
<parameter=command>
mkdir -p /home/palontologist/Downloads/dev/mobile-1b-model/src/main/kotlin/com/ai/edge/agent/error