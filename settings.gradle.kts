pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        // sherpa-onnx publishes its Android AAR through JitPack rather than to
        // Maven Central, so this is required to resolve the dependency below.
        // Same coordinate speech-to-speech-mobile uses (v1.13.5).
        maven { url = uri("https://jitpack.io") }
    }
}

rootProject.name = "mobile-1b-agent"
include(":app")