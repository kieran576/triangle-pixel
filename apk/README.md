# Tri Camera APK — Build Instructions

## Prerequisites

1. Install [Android Studio](https://developer.android.com/studio) (Hedgehog or newer)
2. Open Android Studio → File → Open → select the `apk/` directory
3. Wait for Gradle sync to complete (uses Aliyun mirrors for China)

## Build

- **Debug APK**: Build → Build Bundle(s) / APK(s) → Build APK(s)
- Output: `apk/app/build/outputs/apk/debug/app-debug.apk`

## Install on Phone

1. Copy the APK to your phone
2. Open it, allow "Install from unknown sources"
3. Grant camera permission when prompted

## Usage

- **Switch Mode button**: cycles RAW direct → borrow RGB → ISP
- **Capture button**: saves current frame as PNG
- FPS counter top-right shows real-time performance

## How it works

1. Camera2 API captures RAW_SENSOR (Bayer raw) or YUV frames
2. `TriSampler` maps Bayer pixels → triangular grid (one value per triangle)
3. `TriBorrow` reconstructs RGB by borrowing from 3 neighbors
4. `TriRenderer` draws triangles to screen Bitmap
5. Zero-latency mode: skip borrow, show RAW directly (eye blends the 2R+2G+2B hexagons)

## File Layout

```
apk/
├── build.gradle.kts
├── settings.gradle.kts
├── app/
│   ├── build.gradle.kts
│   └── src/main/
│       ├── AndroidManifest.xml
│       ├── java/com/trianglepixel/camera/
│       │   ├── MainActivity.kt     — camera + preview loop
│       │   ├── TriGrid.kt          — grid geometry
│       │   ├── TriSampler.kt       — Bayer→triangle sampling
│       │   ├── TriBorrow.kt        — neighbor-borrow RGB
│       │   └── TriRenderer.kt      — triangle→Bitmap render
│       └── res/
│           ├── layout/activity_main.xml
│           └── values/strings.xml
```
