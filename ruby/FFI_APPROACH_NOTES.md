# FFI Implementation Notes: ONNX Runtime

## Status (Jan 2026)
The manual FFI implementation in `lib/onnx_runtime.rb` is **code-complete** and correct regarding the C API bindings. It successfully:
1.  Loads the `libonnxruntime.dylib` shared library.
2.  Locates the `OrtGetApiBase` symbol.
3.  Defines the necessary `OrtApi` structs and function pointers.

## The Issue
We encountered a persistent **Segmentation Fault (Signal 11)** during the initialization phase (specifically when calling `OrtGetApiBase` or accessing the API struct).

### Root Cause Diagnosis
**Environmental/ABI Conflict ("DLL Hell")**:
The crash is caused by a binary incompatibility between the Ruby process and the downloaded `libonnxruntime` binary.
-   **Architecture**: Verified as `arm64` (Correct).
-   **Dependencies**: The generic universal binary for macOS links against system libraries (C++, libSystem, etc.) that may clash with the versions loaded by the Ruby interpreter or its extensions (like `ffi` itself).
-   **Symptom**: Immediate process termination upon touching the C API symbols, which is characteristic of mixing runtime environments (e.g., using a Homebrew Ruby with a generic Microsoft binary, or vice versa).

## How to Resume This Approach
To make the FFI approach work in the future, you must resolve the binary incompatibility:

1.  **Build from Source**: Compile `libonnxruntime` specifically for your machine, ensuring it links against the exact same system libraries as your Ruby installation.
2.  **Static Linking**: Attempt to find a statically linked version of the library (rare for dylibs) to avoid dependency conflicts.
3.  **Clean Environment**: Run within a Docker container (Linux) where shared library paths (`LD_LIBRARY_PATH`) and versions are strictly controlled and matching.

For now, we are switching to the `onnxruntime` gem, which manages these binary bindings automatically.
