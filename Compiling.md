
# Compiling KataGo
KataGo is written in C++. It should compile on Linux or OSX via g++ that supports at least C++14, or on Windows via MSVC 15 (2017) and later or MinGW. Other compilers and systems have not been tested yet. This is recommended if you want to run the full KataGo self-play training loop on your own and/or do your own research and experimentation, or if you want to run KataGo on an operating system for which there is no precompiled executable available.

### Building for Distributed
As also mentioned in the instructions below but repeated here for visibility, if you also are building KataGo with the intent to use it in distributed training on https://katagotraining.org, then keep in mind:
* You'll need to specify `-DBUILD_DISTRIBUTED=1` or `BUILD_DISTRIBUTED` and have OpenSSL installed.
* Building will need to happen within a Git clone of the KataGo repo, rather than a zipped copy of the source (such as what you might download from a packaged release).
* The version will need to be supported for distributed training. **The `master` branch will NOT work** - instead please use the either latest release tag or the tip of the `stable` branch, these should both work.
* Please do NOT attempt to bypass any versioning or safety checks - if you feel you need to do so, please first reach out by opening an issue or messaging in [discord](https://discord.gg/bqkZAz3). There is an alternate site [test.katagodistributed.org](test.katagodistributed.org) you can use if you are working on KataGo development or want to test things more freely, ask in the KataGo channel of discord to set up a test account.

## Linux
   * TLDR (if you have a working GPU):
     ```
     git clone https://github.com/lightvector/KataGo.git
     cd KataGo/cpp
     # If you get missing library errors, install the appropriate packages using your system package manager and try again.
     # -DBUILD_DISTRIBUTED=1 is only needed if you want to contribute back to public training.
     cmake . -DUSE_BACKEND=OPENCL -DBUILD_DISTRIBUTED=1
     make -j 4
     ```
   * TLDR (building the slow pure-CPU version):
     ```
     git clone https://github.com/lightvector/KataGo.git
     cd KataGo/cpp
     # If you get missing library errors, install the appropriate packages using your system package manager and try again.
     cmake . -DUSE_BACKEND=EIGEN -DUSE_AVX2=1
     make -j 4
     ```
   * Requirements
      * CMake with a minimum version of 3.18.2 - for example `sudo apt install cmake` on Debian, or download from https://cmake.org/download/ if that doesn't give you a recent-enough version.
      * Some version of g++ that supports at least C++14.
      * If using the OpenCL backend, a modern GPU that supports OpenCL 1.2 or greater, or else something like [this](https://software.intel.com/en-us/opencl-sdk) for CPU. But if using CPU, Eigen should be better.
      * If using the CUDA backend, CUDA 11 or later and a compatible version of CUDNN based on your CUDA version (https://developer.nvidia.com/cuda-toolkit) (https://developer.nvidia.com/cudnn) and a GPU capable of supporting them.
      * If using the TensorRT backend, in addition to a compatible CUDA Toolkit (https://developer.nvidia.com/cuda-toolkit), you also need TensorRT (https://developer.nvidia.com/tensorrt) that is at least version 8.5.
      * If using the Eigen backend, Eigen3. With Debian packages, (i.e. apt or apt-get), this should be `libeigen3-dev`.
      * If using the ONNX backend, ONNX Runtime headers/libs and ONNX protobuf dependencies (`onnx/onnx_pb.h`, `onnx_proto`, `protobuf-lite`) for `.bin.gz` model conversion support.
      * zlib, libzip. With Debian packages (i.e. apt or apt-get), these should be `zlib1g-dev`, `libzip-dev`.
      * If you want to do self-play training and research, probably Google perftools `libgoogle-perftools-dev` for TCMalloc or some other better malloc implementation. For unknown reasons, the allocation pattern in self-play with large numbers of threads and parallel games causes a lot of memory fragmentation under glibc malloc that will eventually run your machine out of memory, but better mallocs handle it fine.
      * If compiling to contribute to public distributed training runs, OpenSSL is required (`libssl-dev`).
   * Clone this repo:
      * `git clone https://github.com/lightvector/KataGo.git`
   * Compile using CMake and make in the cpp directory:
      * `cd KataGo/cpp`
      * `cmake . -DUSE_BACKEND=OPENCL` or `cmake . -DUSE_BACKEND=CUDA` or `cmake . -DUSE_BACKEND=TENSORRT` or `cmake . -DUSE_BACKEND=EIGEN` or `cmake . -DUSE_BACKEND=ONNX` depending on which backend you want. (The `WINML` backend is Windows-only and not available on Linux.)
         * Specify also `-DUSE_TCMALLOC=1` if using TCMalloc.
         * Compiling will also call git commands to embed the git hash into the compiled executable, specify also `-DNO_GIT_REVISION=1` to disable it if this is causing issues for you.
         * Specify `-DUSE_AVX2=1` to also compile Eigen with AVX2 and FMA support, which will make it incompatible with old CPUs but much faster. (If you want to go further, you can also add `-DCMAKE_CXX_FLAGS='-march=native'` which will specialize to precisely your machine's CPU, but the exe might not run on other machines at all).
         * Specify `-DBUILD_DISTRIBUTED=1` to compile with support for contributing data to public distributed training runs.
            * If building distributed, you will also need to build with Git revision support, including building within a clone of the repo, as opposed to merely an unzipped copy of its source.
            * Only builds from specific tagged versions or branches can contribute, in particular, instead of the `master` branch, use either the latest [release](https://github.com/lightvector/KataGo/releases) tag or the tip of the `stable` branch. To minimize the chance of any data incompatibilities or bugs, please do NOT attempt to contribute with custom changes or circumvent these limitations.
      * `make`
   * Done! You should now have a compiled `katago` executable in your working directory.
   * Pre-trained neural nets are available at [the main training website](https://katagotraining.org/).
   * You will probably want to edit `configs/gtp_example.cfg` (see "Tuning for Performance" above).
   * If using OpenCL, you will want to verify that KataGo is picking up the correct device when you run it (e.g. some systems may have both an Intel CPU OpenCL and GPU OpenCL, if KataGo appears to pick the wrong one, you can correct this by specifying `openclGpuToUse` in `configs/gtp_example.cfg`).

##### ONNX Runtime Backend (Linux)
The ONNX backend uses ONNX Runtime for inference, and supports both:
* `.onnx` models loaded directly.
* `.bin.gz` KataGo models via internal conversion to ONNX graph (requires ONNX protobuf dependencies in CMake).

##### Linux Intel NPU (OpenVINO EP) Setup
1. Install Intel NPU driver on Linux:
   * https://github.com/intel/linux-npu-driver
2. Install OpenVINO via system package manager (APT example):
   * https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-apt.html
3. Build ONNX Runtime with OpenVINO EP for NPU (same ORT flow as Windows):
   * https://onnxruntime.ai/docs/build/eps.html#openvino
   * Set OpenVINO EP build option so `use_openvino` is `NPU` (for example `--use_openvino NPU` in ORT build.py).

##### Prepare `ONNXRUNTIME_ROOT` in KataGo (Linux)
Use package root:
* `cpp/external/onnxruntime-linux-x64-openvino`

Linux one-to-one mapping (`<ORT_PACKAGE_ROOT>` -> KataGo):

Include files:

| Source (`<ORT_PACKAGE_ROOT>`) | Destination (KataGo) |
| --- | --- |
| `include/core/*` | `cpp/external/onnxruntime-linux-x64-openvino/include/core/` |
| `include/cpu_provider_factory.h` | `cpp/external/onnxruntime-linux-x64-openvino/include/cpu_provider_factory.h` |
| `include/provider_options.h` | `cpp/external/onnxruntime-linux-x64-openvino/include/provider_options.h` |
| `include/onnxruntime_c_api.h` | `cpp/external/onnxruntime-linux-x64-openvino/include/onnxruntime_c_api.h` |
| `include/onnxruntime_cxx_api.h` | `cpp/external/onnxruntime-linux-x64-openvino/include/onnxruntime_cxx_api.h` |
| `include/onnxruntime_cxx_inline.h` | `cpp/external/onnxruntime-linux-x64-openvino/include/onnxruntime_cxx_inline.h` |
| `include/onnxruntime_env_config_keys.h` | `cpp/external/onnxruntime-linux-x64-openvino/include/onnxruntime_env_config_keys.h` |
| `include/onnxruntime_ep_c_api.h` | `cpp/external/onnxruntime-linux-x64-openvino/include/onnxruntime_ep_c_api.h` |
| `include/onnxruntime_ep_device_ep_metadata_keys.h` | `cpp/external/onnxruntime-linux-x64-openvino/include/onnxruntime_ep_device_ep_metadata_keys.h` |
| `include/onnxruntime_float16.h` | `cpp/external/onnxruntime-linux-x64-openvino/include/onnxruntime_float16.h` |
| `include/onnxruntime_lite_custom_op.h` | `cpp/external/onnxruntime-linux-x64-openvino/include/onnxruntime_lite_custom_op.h` |
| `include/onnxruntime_run_options_config_keys.h` | `cpp/external/onnxruntime-linux-x64-openvino/include/onnxruntime_run_options_config_keys.h` |
| `include/onnxruntime_session_options_config_keys.h` | `cpp/external/onnxruntime-linux-x64-openvino/include/onnxruntime_session_options_config_keys.h` |

Library/config/pkgconfig files:

| Source (`<ORT_PACKAGE_ROOT>`) | Destination (KataGo) |
| --- | --- |
| `lib/libonnxruntime_providers_openvino.so` | `cpp/external/onnxruntime-linux-x64-openvino/lib/libonnxruntime_providers_openvino.so` |
| `lib/libonnxruntime_providers_shared.so` | `cpp/external/onnxruntime-linux-x64-openvino/lib/libonnxruntime_providers_shared.so` |
| `lib/libonnxruntime.so.1.24.3` | `cpp/external/onnxruntime-linux-x64-openvino/lib/libonnxruntime.so.1.24.3` |
| `lib/libonnxruntime.so.1` (symlink to `.1.24.3`) | `cpp/external/onnxruntime-linux-x64-openvino/lib/libonnxruntime.so.1` |
| `lib/libonnxruntime.so` (symlink to `.1`) | `cpp/external/onnxruntime-linux-x64-openvino/lib/libonnxruntime.so` |
| `lib/cmake/onnxruntime/onnxruntimeConfig.cmake` | `cpp/external/onnxruntime-linux-x64-openvino/lib/cmake/onnxruntime/onnxruntimeConfig.cmake` |
| `lib/cmake/onnxruntime/onnxruntimeConfigVersion.cmake` | `cpp/external/onnxruntime-linux-x64-openvino/lib/cmake/onnxruntime/onnxruntimeConfigVersion.cmake` |
| `lib/cmake/onnxruntime/onnxruntimeTargets.cmake` | `cpp/external/onnxruntime-linux-x64-openvino/lib/cmake/onnxruntime/onnxruntimeTargets.cmake` |
| `lib/cmake/onnxruntime/onnxruntimeTargets-release.cmake` | `cpp/external/onnxruntime-linux-x64-openvino/lib/cmake/onnxruntime/onnxruntimeTargets-release.cmake` |
| `lib/pkgconfig/libonnxruntime.pc` | `cpp/external/onnxruntime-linux-x64-openvino/lib/pkgconfig/libonnxruntime.pc` |

##### Minimal KataGo Build Commands (Linux, ONNX backend)
On Linux, `KATAGO_AUTO_FETCH_DEPS=ON` can auto-fetch missing `zlib`, `onnx`, and `protobuf` dependencies via vcpkg into `cpp/build/deps/vcpkg`.

```bash
cmake -S cpp -B cpp/build -G Ninja -DUSE_BACKEND=ONNX -DONNXRUNTIME_ROOT=cpp/external/onnxruntime-linux-x64-openvino
cmake --build cpp/build -j
```

If you want to disable auto-fetch and provide dependencies manually:
* `-DKATAGO_AUTO_FETCH_DEPS=OFF`
* plus `-DONNX_INCLUDE_DIR=... -DONNX_PROTO_LIB=... -DPROTOBUF_INCLUDE_DIR=... -DPROTOBUF_LIB=... -DZLIB_INCLUDE_DIR=... -DZLIB_LIBRARY=...`

Typical run config for Intel NPU:
* `onnxProvider = openvino`
* `onnxOpenVINODeviceType = NPU`
* `onnxOpenVINOEnableNPUFastCompile = true` (optional; may be ignored on ORT builds that do not support this key)

Multi-device assignment is mainly for `onnxProvider=cuda/tensorrt/migraphx` (`onnxDeviceToUseThread*`).
For `onnxProvider=openvino` on Intel NPU, a single device is typically used.


## Windows
   * TLDR:
      * Building from source on Windows is actually a bit tricky, depending on what version you're building, there's not necessarily a super-fast way.
   * Requirements
      * CMake with a minimum version of 3.18.2, GUI version strongly recommended (https://cmake.org/download/)
      * Microsoft Visual Studio for C++. Version 15 (2017) has been tested and should work, MinGW version also should work but only with Eigen and OpenCL backends (CUDA and TensorRT MinGW backends are [not supported by NVIDIA](https://forums.developer.nvidia.com/t/cuda-with-mingw-how-to-get-cuda-running-under-mingw)).
      * If using the OpenCL backend, a modern GPU that supports OpenCL 1.2 or greater, or else something like [this](https://software.intel.com/en-us/opencl-sdk) for CPU. But if using CPU, Eigen should be better.
      * If using the CUDA backend, CUDA 11 or later and a compatible version of CUDNN based on your CUDA version (https://developer.nvidia.com/cuda-toolkit) (https://developer.nvidia.com/cudnn) and a GPU capable of supporting them. I'm unsure how version compatibility works with CUDA, there's a good chance that later versions than these work just as well, but they have not been tested.
      * If using the TensorRT backend, in addition to a compatible CUDA Toolkit (https://developer.nvidia.com/cuda-toolkit), you also need TensorRT (https://developer.nvidia.com/tensorrt) that is at least version 8.5.
      * If using the Eigen backend, Eigen3, version 3.3.x. (http://eigen.tuxfamily.org/index.php?title=Main_Page#Download).
      * If using the ONNX backend, ONNX Runtime package (headers + import libs + runtime DLLs).
      * If using the WinML backend, the Windows App SDK Machine Learning NuGet package is auto-downloaded by CMake. ONNX protobuf dependencies are also needed for `.bin.gz` model support.
      * On Windows, missing `zlib` and ONNX model-conversion dependencies (`onnx`, `protobuf`) can be auto-fetched by CMake into `cpp/build/deps/vcpkg` (default `KATAGO_AUTO_FETCH_DEPS=ON`).
      * libzip (optional, needed only for self-play training) - for example https://github.com/kiyolee/libzip-win-build
      * For MinGW it's recommended to use [MSYS2](https://www.msys2.org/) building platform to get necessary zlib and libzip dependencies:
        * Install MSYS2 according to the instruction on the official site
        * Run `mingw64.exe` app from Console
        * Install zlib/libzip dependencies using pacman package manager:
          * `pacman -S mingw-w64-x86_64-libzip`
          * `pacman -S mingw-w64-x86_64-xz`
          * `pacman -S mingw-w64-x86_64-bzip2`
          * `pacman -S mingw-w64-x86_64-zstd`
      * If compiling to contribute to public distributed training runs, OpenSSL is required (https://www.openssl.org/, https://wiki.openssl.org/index.php/Compilation_and_Installation).
   * Download/clone this repo to some folder `KataGo`.
   * Configure using CMake GUI and compile in an IDE:
      * Select `KataGo/cpp` as the source code directory in [CMake GUI](https://cmake.org/runningcmake/).
      * Set the build directory to wherever you would like the built executable to be produced.
      * Click "Configure". For the generator select your generator (MSVC or MinGW), and also select "x64" for the optional platform if you're on 64-bit windows, don't use win32.
      * If you get errors where CMake has not automatically found ZLib, point it to the appropriate places according to the error messages:
        * `ZLIB_INCLUDE_DIR` - point this to the directory containing `zlib.h` and other headers
        * `ZLIB_LIBRARY` - point this to the `libz.lib` (`libz.a` for MinGW) resulting from building zlib. Note that "*_LIBRARY" expects to be pointed to the ".lib" file, whereas the ".dll" file is the file that needs to be included with KataGo at runtime.
        * For MinGW zlib/libzip CMake options should look like the following way:
          ```
          -DZLIB_INCLUDE_DIR="C:/msys64/mingw64/include"
          -DZLIB_LIBRARY="C:/msys64/mingw64/lib/libz.a"
          -DLIBZIP_INCLUDE_DIR_ZIP:PATH="C:/msys64/mingw64/include"
          -DLIBZIP_INCLUDE_DIR_ZIPCONF:PATH="C:/msys64/mingw64/include"
          -DLIBZIP_LIBRARY:FILEPATH="C:/msys64/mingw64/lib/libzip.dll.a"
          ```
      * Also set `USE_BACKEND` to `OPENCL`, or `CUDA`, or `TENSORRT`, or `EIGEN`, or `ONNX`, or `WINML` depending on what backend you want to use.
      * Set any other options you want and re-run "Configure" again as needed after setting them. Such as:
         * `NO_GIT_REVISION` if you don't have Git or if cmake is not finding it.
         * `NO_LIBZIP` if you don't care about running self-play training and you don't have libzip.
         * `USE_AVX2` if you want to compile with AVX2 and FMA instructions, which will fail on some CPUs but speed up Eigen greatly on CPUs that support them.
         * `BUILD_DISTRIBUTED` to compile with support for contributing data to public distributed training runs.
            * If building distributed, you will also need to build with Git revision support, including building within a clone of the repo, as opposed to merely an unzipped copy of its source.
            * Only builds from specific tagged versions or branches can contribute, in particular, instead of the `master` branch, use either the latest [release](https://github.com/lightvector/KataGo/releases) tag or the tip of the `stable` branch. To minimize the chance of any data incompatibilities or bugs, please do NOT attempt to contribute with custom changes or circumvent these limitations.
      * Once running "Configure" looks good, run "Generate" and then open the project in Visual Studio or CLion and build it as usual.
   * For MinGW it's recommended to configure the project in the following ways:
     * Use the default MinGW toolchain in [CLion IDE](https://www.jetbrains.com/clion/) (free for Non-Commercial use)
     * Use [MSYS2](https://www.msys2.org/) MinGW toolchain. Befor configuring, install gcc compiler using pacman package manager: `pacman -S mingw-w64-x86_64-gcc`
   * Done! You should now have a compiled `katago.exe` executable in your working directory.
   * Note: You may need to copy the ".dll" files corresponding to the various ".lib" (".a") files you compiled with into the directory containing katago.exe.
     * MinGW has different dlls. If you use pacman, the necessary dlls (`libbz2-1.dll`, `libzip.dll`, `libzstd.dll`, `liblzma-5.dll`) should be copied from MinGW bin directory (like `C:\msys64\mingw64\bin`).
   * Note: If you had to update or install CUDA or GPU drivers, you will likely need to reboot before they will work.
   * Pre-trained neural nets are available at [the main training website](https://katagotraining.org/).
   * You will probably want to edit `configs/gtp_example.cfg` (see "Tuning for Performance" above).
   * If using OpenCL, you will want to verify that KataGo is picking up the correct device (e.g. some systems may have both an Intel CPU OpenCL and GPU OpenCL, if KataGo appears to pick the wrong one, you can correct this by specifying `openclGpuToUse` in `configs/gtp_example.cfg`).

##### ONNX Runtime Backend
The ONNX backend uses ONNX Runtime for inference, and supports both:
* `.onnx` models loaded directly.
* `.bin.gz` KataGo models via internal conversion to ONNX graph (requires ONNX protobuf dependencies in CMake).

##### Windows Intel NPU (OpenVINO EP) Setup
1. Install Visual Studio 2026 Community or Visual Studio 2026 Build Tools:
   * https://visualstudio.microsoft.com/zh-hans/downloads/
   * In installer workloads, select **Desktop development with C++**.
2. Install Intel NPU driver:
   * https://www.intel.com/content/www/us/en/download/794734/intel-npu-driver-windows.html
3. Install OpenVINO 2026 archive package on Windows:
   * https://docs.openvino.ai/2026/get-started/install-openvino/install-openvino-archive-windows.html
   * Typical install root looks like: `C:\Program Files (x86)\Intel\openvino_2026.0`
4. Add these to System PATH:
   * `C:\Program Files (x86)\Intel\openvino_2026.0\runtime\bin\intel64\Release`
   * `C:\Program Files (x86)\Intel\openvino_2026.0\runtime\3rdparty\tbb\bin`
5. Build ONNX Runtime with OpenVINO EP for NPU (follow official docs):
   * https://onnxruntime.ai/docs/build/eps.html#openvino
   * Set OpenVINO EP build option so `use_openvino` is `NPU` (for example `--use_openvino NPU` in ORT build.py).

##### Prepare `ONNXRUNTIME_ROOT` in KataGo (Windows)
Use package root:
* `cpp/external/onnxruntime-win-x64-openvino`

Windows one-to-one mapping (`<ORT_PACKAGE_ROOT>` -> KataGo):

Windows and Linux generally share the same ORT include/config layout.
The main differences are binary file names/extensions (`.dll/.lib` vs `.so`).

Include files:

| Source (`<ORT_PACKAGE_ROOT>`) | Destination (KataGo) |
| --- | --- |
| `include/core/*` | `cpp/external/onnxruntime-win-x64-openvino/include/core/` |
| `include/cpu_provider_factory.h` | `cpp/external/onnxruntime-win-x64-openvino/include/cpu_provider_factory.h` |
| `include/provider_options.h` | `cpp/external/onnxruntime-win-x64-openvino/include/provider_options.h` |
| `include/onnxruntime_c_api.h` | `cpp/external/onnxruntime-win-x64-openvino/include/onnxruntime_c_api.h` |
| `include/onnxruntime_cxx_api.h` | `cpp/external/onnxruntime-win-x64-openvino/include/onnxruntime_cxx_api.h` |
| `include/onnxruntime_cxx_inline.h` | `cpp/external/onnxruntime-win-x64-openvino/include/onnxruntime_cxx_inline.h` |
| `include/onnxruntime_env_config_keys.h` | `cpp/external/onnxruntime-win-x64-openvino/include/onnxruntime_env_config_keys.h` |
| `include/onnxruntime_ep_c_api.h` | `cpp/external/onnxruntime-win-x64-openvino/include/onnxruntime_ep_c_api.h` |
| `include/onnxruntime_ep_device_ep_metadata_keys.h` | `cpp/external/onnxruntime-win-x64-openvino/include/onnxruntime_ep_device_ep_metadata_keys.h` |
| `include/onnxruntime_float16.h` | `cpp/external/onnxruntime-win-x64-openvino/include/onnxruntime_float16.h` |
| `include/onnxruntime_lite_custom_op.h` | `cpp/external/onnxruntime-win-x64-openvino/include/onnxruntime_lite_custom_op.h` |
| `include/onnxruntime_run_options_config_keys.h` | `cpp/external/onnxruntime-win-x64-openvino/include/onnxruntime_run_options_config_keys.h` |
| `include/onnxruntime_session_options_config_keys.h` | `cpp/external/onnxruntime-win-x64-openvino/include/onnxruntime_session_options_config_keys.h` |

Library/config/pkgconfig files:

| Source (`<ORT_PACKAGE_ROOT>`) | Destination (KataGo) |
| --- | --- |
| `lib/onnxruntime.lib` | `cpp/external/onnxruntime-win-x64-openvino/lib/onnxruntime.lib` |
| `lib/onnxruntime.dll` | `cpp/external/onnxruntime-win-x64-openvino/lib/onnxruntime.dll` |
| `lib/onnxruntime_providers_shared.dll` | `cpp/external/onnxruntime-win-x64-openvino/lib/onnxruntime_providers_shared.dll` |
| `lib/onnxruntime_providers_openvino.dll` | `cpp/external/onnxruntime-win-x64-openvino/lib/onnxruntime_providers_openvino.dll` |
| `lib/onnxruntime_providers_shared.lib` (optional import lib) | `cpp/external/onnxruntime-win-x64-openvino/lib/onnxruntime_providers_shared.lib` |
| `lib/onnxruntime_providers_openvino.lib` (optional import lib) | `cpp/external/onnxruntime-win-x64-openvino/lib/onnxruntime_providers_openvino.lib` |
| `lib/cmake/onnxruntime/onnxruntimeConfig.cmake` | `cpp/external/onnxruntime-win-x64-openvino/lib/cmake/onnxruntime/onnxruntimeConfig.cmake` |
| `lib/cmake/onnxruntime/onnxruntimeConfigVersion.cmake` | `cpp/external/onnxruntime-win-x64-openvino/lib/cmake/onnxruntime/onnxruntimeConfigVersion.cmake` |
| `lib/cmake/onnxruntime/onnxruntimeTargets.cmake` | `cpp/external/onnxruntime-win-x64-openvino/lib/cmake/onnxruntime/onnxruntimeTargets.cmake` |
| `lib/cmake/onnxruntime/onnxruntimeTargets-release.cmake` | `cpp/external/onnxruntime-win-x64-openvino/lib/cmake/onnxruntime/onnxruntimeTargets-release.cmake` |
| `lib/pkgconfig/libonnxruntime.pc` (optional) | `cpp/external/onnxruntime-win-x64-openvino/lib/pkgconfig/libonnxruntime.pc` |

##### Minimal KataGo Build Commands (Windows, ONNX backend)
On Windows, `KATAGO_AUTO_FETCH_DEPS=ON` by default, so missing `zlib`, `onnx`, and `protobuf` dependencies are auto-fetched via vcpkg into `cpp/build/deps/vcpkg`.

```
cmake -S cpp -B cpp/build -G "Visual Studio 18 2026" -A x64 -DUSE_BACKEND=ONNX -DONNXRUNTIME_ROOT=cpp/external/onnxruntime-win-x64-openvino
cmake --build cpp/build --config Release -j
```

If you want to disable auto-fetch and provide dependencies manually:
* `-DKATAGO_AUTO_FETCH_DEPS=OFF`
* plus `-DONNX_INCLUDE_DIR=... -DONNX_PROTO_LIB=... -DPROTOBUF_INCLUDE_DIR=... -DPROTOBUF_LIB=... -DZLIB_INCLUDE_DIR=... -DZLIB_LIBRARY=...`

Typical run config for Intel NPU:
* `onnxProvider = openvino`
* `onnxOpenVINODeviceType = NPU`
* `onnxOpenVINOEnableNPUFastCompile = true` (optional; may be ignored on ORT builds that do not support this key)

Multi-device assignment is mainly for `onnxProvider=cuda/tensorrt/migraphx` (`onnxDeviceToUseThread*`).
For `onnxProvider=openvino` on Intel NPU, a single device is typically used.

##### WinML (Windows App SDK) Backend
The WinML backend uses the [Windows App SDK](https://learn.microsoft.com/en-us/windows/ai/) Machine Learning APIs with an EP (Execution Provider) catalog. The EP catalog automatically discovers, downloads, and registers hardware-specific execution providers at startup. This backend is Windows-only.

WinML supports both:
* `.onnx` models loaded directly.
* `.bin.gz` KataGo models via internal conversion to ONNX graph (same as the ONNX backend, requires ONNX protobuf dependencies in CMake).

Supported execution providers (configured via `winmlProvider`, **required** — there is no default,
and if it is unset or refers to a provider not installed on the machine, KataGo will print the list
of providers actually available on the machine and exit):
* `dml` — DirectML (works with most GPUs on Windows 10+)
* `openvino` — Intel OpenVINO (CPU/GPU/NPU). Also requires `winmlOpenVINODeviceType` to be set
  explicitly to `cpu`, `gpu`, or `npu` — if omitted, KataGo prints the OpenVINO hardware types
  actually detected on the machine and exits.
* `nvtensorrtrtx` — NVIDIA TensorRT RTX
* `migraphx` — AMD MIGraphX (also distributed under the Store package name "AMD GPU EP")
* `vitisai` — AMD VitisAI (NPU; also distributed under the Store package name "AMD NPU EP")
* `qnn` — Qualcomm QNN
* `cpu` — CPU fallback

`openvino`/`nvtensorrtrtx`/`migraphx`/`vitisai`/`qnn` are downloaded and updated automatically by
Windows ML from the Microsoft Store the first time they're used; `dml` and `cpu` are always
available. The Windows App SDK Machine Learning NuGet package is automatically downloaded by CMake
during the build. No manual SDK installation is required beyond having a compatible GPU/NPU driver.

EP discovery happens entirely at process startup (`NeuralNet::globalInitialize()`), not at compile
time: primarily via the `WinMLEpCatalog` C API (`EnsureReady` + `GetLibraryPath`), and as a fallback
if that fails, via a scan of `C:\Program Files\WindowsApps\` performed on whichever machine actually
runs `katago.exe`. Neither path bakes an absolute path into the binary at build time, so a
`katago.exe` built on one machine keeps working correctly if copied to another machine with a
different EP package version installed (or after Microsoft renames/updates a package, e.g.
`AMD.MIGraphX.EP` -> `AMD.GPU.EP`, `Xilinx.VitisAI.EP` -> `AMD.NPU.EP`).

##### Minimal KataGo Build Commands (Windows, WinML backend)
On Windows, `KATAGO_AUTO_FETCH_DEPS=ON` by default, so missing `zlib`, `onnx`, and `protobuf` dependencies are auto-fetched via vcpkg into `cpp/build/deps/vcpkg`.

Always use the Visual Studio generator (adjust `-G` to whatever Visual Studio version/generator name
you actually have installed, e.g. `"Visual Studio 17 2022"`) for configure, and `cmake --build ...`
(not a hand-invoked `ninja`/`msbuild`) for the actual build:

```
cmake -S cpp -B cpp/build -G "Visual Studio 18 2026" -A x64 -DUSE_BACKEND=WINML
cmake --build cpp/build --config Release --parallel
```

No manual `vcvars64.bat`/`vcvarsall.bat` sourcing is needed before either command, even in a brand
new shell: unlike the Ninja generator (used by, e.g., the ROCm backend on Windows, where the actual
build step is a separate `ninja` process that does not inherit whatever environment `cmake` saw
during configure), the Visual Studio generator's `cmake --build` step delegates to `msbuild.exe`,
which resolves its own MSVC toolchain from the `.vcxproj` files/registry/VS installation metadata
regardless of the invoking shell's environment. `Release`/`Debug` builds both live under this same
single `cpp/build` directory (selected via `--config`), so there is no need for separate build
directories per configuration.

Typical run config for Intel NPU via OpenVINO:
* `winmlProvider = openvino`
* `winmlOpenVINODeviceType = NPU`
* `winmlOpenVINOEnableNPUFastCompile = true` (optional)

Typical run config for DirectML (GPU):
* `winmlProvider = dml`
* `winmlDeviceToUse = 0` (GPU device index)

Compile caching (VitisAI / MIGraphX): both EPs compile the model into a hardware-specific format
the first time a given model/shape is used (VitisAI NPU compiles can take on the order of 15
minutes; MIGraphX GPU compiles are faster but still non-trivial). Both are configured to
persistently cache their compiled output on disk under `<dir containing katago.exe>/KataGoData/EPCache/`
(`vitisai` / `migraphx` subfolders) so this cost is only paid once per model, not on every process
launch. This cache location is hardcoded (not a config option) since there's no good reason for a
user to want a different one.

For `migraphx`, KataGo pins the ONNX graph's batch dimension to a fixed static size so MIGraphX
compiles exactly once regardless of how the real batch fill level fluctuates at runtime (instead
of recompiling from scratch, taking minutes, every time it sees a new batch fill level). This
fixed size defaults to 8 and can be changed via `winmlMigraphxBatchSize` (must not exceed
`nnMaxBatchSize`; larger values are automatically clamped down).

##### VitisAI (NPU) requires a pre-quantized model

The VitisAI EP only accelerates INT8 ops on the XDNA NPU. An FP32 graph (as KataGo builds on the
fly from a `.bin.gz` model for every other provider) would otherwise get almost entirely fallen
back to CPU by VitisAI, silently, defeating the point of using the NPU. To prevent that trap,
`winmlProvider = vitisai` **requires** a model file whose name ends in `-int8.onnx` -- KataGo
refuses to start (with a clear error) if given a `.bin.gz` or a plain `.onnx` instead. KataGo also
sets the ONNX Runtime session option `session.disable_cpu_ep_fallback=1` for this provider, so if
VitisAI can't claim every node in the graph, session creation fails loudly instead of silently
running (some or all of) the network on CPU while reporting success.

There's no automated in-process quantization step (this was tried and removed: automating a
Python/`amd-quark` environment inside KataGo added a large, fragile dependency footprint for a
one-time step). Quantize offline instead, on any machine (no NPU or Ryzen AI driver needed for the
quantization step itself -- it only needs to run the FP32 graph on CPU to collect calibration
statistics; can be a different machine/OS than the one that will actually run katago.exe, including
Linux, since only step 4 below needs the Python/`amd-quark` environment and nothing else from
KataGo):

1. **Build KataGo with the WinML backend** (see "Minimal KataGo Build Commands" above) -- this
   gives you `katago.exe` with the `exportonnx` and `dumpcalibrationdata` subcommands needed below.

2. **Export the FP32 ONNX graph** from your `.bin.gz` model:
   ```
   katago.exe exportonnx -model kata1-xxx.bin.gz -o kata1-xxx-fp32.onnx -x 19 -y 19
   ```

3. **Produce a calibration dataset** with the `dumpcalibrationdata` subcommand -- samples NN input
   feature tensors from real games, using the exact same feature-encoding code (`NNInputs::fillRowV7`)
   used at real inference time, so calibration data is guaranteed to match what the network
   actually sees (do not reimplement feature extraction separately, e.g. in Python -- any mismatch
   silently degrades quantization quality without erroring):
   ```
   katago.exe dumpcalibrationdata -model kata1-xxx.bin.gz -sgfsdir games.sgfs \
     -output calib.npz -x 19 -y 19 -positions-per-game 1 -max-positions 128
   ```
   (`-sgfdir`/`-sgf` also accepted for individual `.sgf` files; `-sgfsdir` takes a `.sgfs`
   multi-game-per-file bundle or a directory of them.) Use *real* games, not synthetic/test
   positions -- calibration quality directly determines quantized accuracy. Prefer more distinct
   games over more positions per game (`-positions-per-game 1`, one per game, maximizes position
   diversity for a given total count) -- positions from the same game are highly correlated. Keep
   the total position count modest: empirically, quantization consumes **on the order of 0.5GB of
   disk cache per calibration position** (quark caches intermediate activations per-sample to
   disk), so a few hundred positions can consume 100+GB of temp disk space and tens of minutes.
   64-128 positions is a reasonable default (also `dumpcalibrationdata`'s example above); only go
   higher if you have the disk/time budget for it.

4. **Set up a Python 3.12 environment** (on whichever machine will run the quantization script;
   does not need to be Windows or have NPU/Ryzen AI drivers) with:
   ```
   pip install numpy onnx onnxruntime
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install "amd-quark==0.11.2" --extra-index-url https://pypi.amd.com/quark/cpu/simple
   ```
   Pin `amd-quark` to `0.11.2`, not latest -- newer releases (`0.12.x` at time of writing) hard-fail
   at import time if their custom-ops C++ extension can't be JIT-compiled (which additionally
   requires `ninja` plus a working C/C++ toolchain on PATH); `0.11.2` degrades this to a harmless
   warning and falls back to a pure-Python path, which is all a standard CNN like KataGo's needs.

5. **Run the quantization script** (`python/quantize_vitisai.py` in this repo) against the FP32
   onnx + calibration npz from steps 2-3:
   ```
   python quantize_vitisai.py --input kata1-xxx-fp32.onnx --calibration calib.npz --output kata1-xxx-int8.onnx
   ```
   This uses `quark.onnx`'s `ModelQuantizer` with the `XINT8_QCONFIG` preset (AMD's recommended
   power-of-two-scale INT8 scheme for Ryzen AI NPU CNN deployment), falling back to the lower-level
   `quantize_static` if that preset isn't available in your installed `amd-quark` version.

6. **Copy the resulting `kata1-xxx-int8.onnx`** to the machine running katago.exe and use it
   directly as the model (the filename must end in `-int8.onnx`, per the requirement above):
   ```
   katago.exe gtp -config gtp.cfg -model kata1-xxx-int8.onnx -override-config "winmlProvider=vitisai"
   ```

**Known issue, confirmed in testing: this EP build does not actually offload to the NPU even for a
correctly quantized model.** A model quantized via the steps above (verified: valid QDQ INT8 graph,
all major ops converted, numerically sane inference output -- win/loss probabilities correctly
normalized, no NaNs) still shows **zero NPU utilization** when checked with AMD's own `xrt-smi
examine -r aie-partitions` (run continuously during a sustained `kata-analyze` workload -- every
HW context present belonged to an unrelated Windows system process, none to katago.exe). With
`session.disable_cpu_ep_fallback=1` (which KataGo now always sets for this provider), this
manifests as a session-creation error ("this session contains graph nodes that are assigned to the
default CPU EP...") rather than a silent full-CPU run -- confirming VitisAI genuinely does not
claim (at least some) nodes in the graph, not a false negative in the utilization check. No
per-node diagnostic detail is available: neither ONNX Runtime's `SetLogSeverityLevel(VERBOSE)` nor
the standard `GLOG_minloglevel`/`GLOG_v` environment variables (VitisAI's internal `vaip_core`
compiler appears to use glog, based on the format of its crash logs) surface which node(s) or op
type(s) caused the rejection. This looks like an AMD-side EP/quantizer compatibility gap rather
than anything fixable from KataGo's side; reported upstream. Until resolved, `winmlProvider =
vitisai` should be expected to fail outright (loudly, thanks to `disable_cpu_ep_fallback`) rather
than provide a working NPU speedup on this EP build (`VitisAIExecutionProvider` v1.8.63.0).

##### Known WinML/MIGraphX limitations (Store EP package v1.8.57.0)

* **Compile cache does not actually persist to disk.** KataGo sets both the documented
  `ORT_MIGRAPHX_*` environment variables and (previously) attempted `migraphx_save_compiled_model`
  / `migraphx_load_compiled_model` provider options, but this Store-distributed EP build honors
  neither: the `EPCache/migraphx/` directory is created but no `.mxr` file is ever written, and
  every process launch recompiles from scratch. Worse, passing the provider-option keys causes
  this specific EP build to hard-fail session creation with `EP_FAIL : Unknown provider option`
  (unlike most ORT EPs, which silently ignore options they don't recognize) — so KataGo
  intentionally only sets the env vars, not the provider options, even though neither currently
  achieves persistence. If a future EP package version fixes this, no KataGo code change should be
  needed to pick it up.
* **Output tensors can come back in the wrong slot.** This EP build was observed returning the
  three `[N,C,1,1]`-shaped outputs (`OutputPolicyPass`, `OutputValue`, `OutputScoreValue` — all
  rank 4 with H=W=1, indistinguishable to the EP's own bookkeeping) permuted relative to the
  requested output-name order, even though `Ort::Session::Run()` is documented to return tensors
  positionally matching the requested names. ONNX Runtime itself logs a `VerifyOutputSizes`
  warning for each swapped output but still proceeds, so this is silent data corruption unless
  handled explicitly. KataGo detects and corrects this after every inference call by matching each
  returned tensor's channel-dimension count against what's expected for
  PolicyPass/Value/ScoreValue (which differ for any realistic model version), and prints a
  one-time note to stderr when it has to remap. This is a no-op when the EP returns outputs in the
  requested order.
* **Deterministic crash inside the EP's bundled HIP runtime on at least one discrete-GPU (RX 7900
  XTX / gfx1100) machine.** First-time compile of a large network (e.g. b40) shows several GB of
  host RAM usage and 100% CPU while MIGraphX compiles on the host side, VRAM usage staying at 0%
  the entire time, then the process hard-crashes (access violation, uncatchable — not a C++
  exception) right as CPU usage drops back to 0%. Windows Event Viewer (Application log, event ID
  1000) pins this to `amdhip64_7.dll` v7.2.2606.20 — the HIP runtime bundled *inside* the Store EP
  package itself (`.../AMD.GPU.EP.1.8_.../ExecutionProvider/amdhip64_7.dll`), not the machine's
  regular AMD/ROCm driver install — with exception code `0xc0000005` at the exact same faulting
  offset (`0x465007`) across independent process runs (both `gtp` and standalone `benchmark`
  subcommands). Same offset every time means this is a reproducible bug in that bundled runtime
  build interacting with this GPU, not a memory-pressure/race condition — RAM exhaustion was ruled
  out (64GB system RAM, 55%+ free). Could not be reproduced on an integrated-GPU+NPU machine, which
  instead just took ~9 minutes to compile successfully via the same EP package. No known KataGo-side
  workaround (the crash happens inside third-party closed-source code before session creation
  returns); on affected discrete-AMD-GPU machines, prefer a native ROCm/OpenCL KataGo build over
  `winmlProvider=migraphx` until Microsoft/AMD ship a fixed EP package version.
* **ORT's default console logger writes to stdout**, which corrupts the GTP protocol stream (GTP
  clients like Sabaki read engine responses from stdout and will report a dead/failed connection
  if anything else is interleaved in it) — this was especially visible with `migraphx` since the
  shape-mismatch warning above fires on every single inference call. KataGo now installs a custom
  ORT logging callback that routes all ONNX Runtime log output to stderr instead, so stdout stays
  protocol-clean regardless of what ORT logs.

## MacOS
   * TLDR (Metal backend - recommended for most users, hybrid CPU+GPU+Neural Engine for maximum throughput):
     ```
     git clone https://github.com/lightvector/KataGo.git
     cd KataGo/cpp
     # If you get missing library errors, install the appropriate packages using your system package manager and try again.
     # -DBUILD_DISTRIBUTED=1 is only needed if you want to contribute back to public training.
     cmake -G Ninja -DUSE_BACKEND=METAL -DBUILD_DISTRIBUTED=1
     ninja
     ```
   * Requirements
      * [Homebrew](https://brew.sh): `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
      * CMake with a minimum version of 3.18.2: `brew install cmake`.
      * AppleClang and Swift compilers: `xcode-select --install`.
      * If using the Metal backend, [Ninja](https://ninja-build.org): `brew install ninja`
      * If using the Metal backend, protobuf and abseil: `brew install protobuf abseil`
      * libzip: `brew install libzip`.
      * If you want to do self-play training and research, probably Google perftools `brew install gperftools` for TCMalloc or some other better malloc implementation. For unknown reasons, the allocation pattern in self-play with large numbers of threads and parallel games causes a lot of memory fragmentation under glibc malloc that will eventually run your machine out of memory, but better mallocs handle it fine.
      * If compiling to contribute to public distributed training runs, OpenSSL is required (`brew install openssl`).
   * Clone this repo:
      * `git clone https://github.com/lightvector/KataGo.git`
   * Compile using CMake and make in the cpp directory:
      * `cd KataGo/cpp`
      * `cmake . -G Ninja -DUSE_BACKEND=METAL` or `cmake . -DUSE_BACKEND=OPENCL` or `cmake . -DUSE_BACKEND=EIGEN` depending on which backend you want.
         * Specify also `-DUSE_TCMALLOC=1` if using TCMalloc.
         * Compiling will also call git commands to embed the git hash into the compiled executable, specify also `-DNO_GIT_REVISION=1` to disable it if this is causing issues for you.
         * Specify `-DUSE_AVX2=1` to also compile Eigen with AVX2 and FMA support, which will make it incompatible with old CPUs but much faster. Intel-based Macs with new processors support AVX2, but Apple Silicon Macs do not support AVX2 natively. (If you want to go further, you can also add `-DCMAKE_CXX_FLAGS='-march=native'` which will specialize to precisely your machine's CPU, but the exe might not run on other machines at all).
         * Specify `-DBUILD_DISTRIBUTED=1` to compile with support for contributing data to public distributed training runs.
            * If building distributed, you will also need to build with Git revision support, including building within a clone of the repo, as opposed to merely an unzipped copy of its source.
            * Only builds from specific tagged versions or branches can contribute, in particular, instead of the `master` branch, use either the latest [release](https://github.com/lightvector/KataGo/releases) tag or the tip of the `stable` branch. To minimize the chance of any data incompatibilities or bugs, please do NOT attempt to contribute with custom changes or circumvent these limitations.
      * `ninja` for Metal backend, or `make` for other backends.
   * Done! You should now have a compiled `katago` executable in your working directory.
   * Pre-trained neural nets are available at [the main training website](https://katagotraining.org/).
   * You will probably want to edit `configs/gtp_example.cfg` (see "Tuning for Performance" above).
   * If using OpenCL, you will want to verify that KataGo is picking up the correct device when you run it (e.g. some systems may have both an Intel CPU OpenCL and GPU OpenCL, if KataGo appears to pick the wrong one, you can correct this by specifying `openclGpuToUse` in `configs/gtp_example.cfg`).
