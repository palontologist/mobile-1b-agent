import org.jetbrains.kotlin.gradle.dsl.JvmTarget

/**
 * The only compiled Android module in this repository.
 *
 * Deliberate scope: this module builds and installs a *torch probe*, because the
 * pre-existing tree at /src/main/kotlin has never been compiled and does not
 * currently parse (see the NOTES list below and README "Status"). It is NOT wired
 * into this module. Source files are moved here as they are repaired, which keeps
 * `assembleDebug` green at every step instead of fixing 40 errors blind.
 *
 * NOTES on what is still unhooked, and why:
 *  - `src/main/kotlin/.../ActionRegistry.kt` — object closes at line 393 and the file
 *    continues with orphaned map entries (one of which uses Python `True`/`False`).
 *    The tail is preserved under `legacy/` rather than deleted.
 *  - `src/main/kotlin/.../AgentLoop.kt` — uses a C-style ternary (`b ? x : y`) and
 *    references a `CompiledModel` type from an artifact not declared anywhere.
 *  - `src/main/kotlin/.../database/NoteDatabase.kt` — ended in pasted transcript text.
 *
 * No third-party dependencies: the module must resolve and compile on a machine with
 * nothing but the Android SDK, using only the platform API.
 */
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.ai.edge.agent"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.ai.edge.agent"
        // CameraManager.setTorchMode() is API 23+, but the LiteRT-LM AAR declares
        // minSdk 24, so the module floor is 24. The SM-A14 runs API 35, so this costs
        // nothing here and avoids `tools:overrideLibrary`, which would suppress the
        // check at the cost of a real chance of a runtime failure on API 23.
        minSdk = 24
        targetSdk = 35
        versionCode = 1
        versionName = "0.1-probe"
    }

    buildTypes {
        debug {
            isMinifyEnabled = false
        }
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"))
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

// The LiteRT-LM Android AAR coordinate IS resolvable, contrary to the note that used
// to stand here. Verified against dl.google.com's maven2 tree on 2026-10-05:
//   https://dl.google.com/dl/android/maven2/com/google/ai/edge/litertlm/litertlm-android/
//     0.10.2 / 0.12.0 / 0.15.0 / 0.16.0 -> 200
// The old 404s came from looking under `com.google.ai.edge` — the group is
// `com.google.ai.edge.litertlm`, one level deeper. Confirmed against the upstream
// sample's gradle/libs.versions.toml, which pins the same coordinate.
// Consumed by ToolRoutingProbeActivity; the torch probe needs none of this.
dependencies {
    implementation("com.google.ai.edge.litertlm:litertlm-android:0.16.0")
    // CompiledModel API for the embedding-router probe (Interpreter would also
    // work, but the recipe's on-device discipline uses CompiledModel so a CPU run
    // and a GPU run share one code path).
    implementation("com.google.ai.edge.litert:litert:2.1.0")
}
