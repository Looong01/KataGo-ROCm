// Windows ML backend for KataGo.
// Uses ONNX Runtime with the Windows ML EP catalog for execution provider management.
// Supports: CPU, DML (DirectML), OpenVINO, NvTensorRtRtx, MIGraphX, QNN, VitisAI
//
// The EP catalog discovers, downloads, and registers hardware-specific execution
// providers at startup. After registration, the selected EP is appended to the
// ORT session for inference.

#include "../neuralnet/nninterface.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/modelversion.h"

#include <onnxruntime_cxx_api.h>
#include "../neuralnet/onnxmodelbuilder.h"

#ifdef WINML_HAS_EP_CATALOG
#include <WinMLEpCatalog.h>
#endif

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include <fstream>
#include <unordered_map>
#include <iostream>
#include <mutex>

using namespace std;

//--------------------------------------------------------------

// Auto-detect modelVersion from introspected channel counts.
// Same heuristic as the ONNX backend.
static int detectModelVersion(
  int numInputChannels, int numInputGlobalChannels,
  int numPolicyChannels, int numScoreValueChannels,
  int configModelVersion
) {
  if(configModelVersion >= 0)
    return configModelVersion;
  if(numInputChannels == NNInputs::NUM_FEATURES_SPATIAL_V7 &&
     numInputGlobalChannels == NNInputs::NUM_FEATURES_GLOBAL_V7) {
    if(numScoreValueChannels == 6 && numPolicyChannels == 2)
      return 15;
    if(numScoreValueChannels == 6 && numPolicyChannels == 1)
      return 10;
    if(numScoreValueChannels == 4)
      return 8;
    return 15;
  }
  return NNModelVersion::defaultModelVersion;
}

//--------------------------------------------------------------

struct LoadedModel {
  ModelDesc modelDesc;
  bool isRawOnnx;
  string rawOnnxBytes;

  LoadedModel(const string& fileName, const string& expectedSha256, bool rawOnnx)
    : isRawOnnx(rawOnnx)
  {
    if(!rawOnnx) {
      ModelDesc::loadFromFileMaybeGZipped(fileName, modelDesc, expectedSha256);
      return;
    }

    // Read raw .onnx file bytes
    {
      std::ifstream in(fileName, std::ios::binary | std::ios::ate);
      if(!in.good())
        throw StringError("WinML backend: could not open raw ONNX file: " + fileName);
      std::streamsize size = in.tellg();
      if(size < 0)
        throw StringError("WinML backend: could not determine size of ONNX file: " + fileName);
      in.seekg(0, std::ios::beg);
      rawOnnxBytes.resize(size);
      if(!in.read(rawOnnxBytes.data(), size))
        throw StringError("WinML backend: failed to read raw ONNX file: " + fileName);
    }

    // Create a temporary CPU session to introspect shapes
    Ort::Env tmpEnv(ORT_LOGGING_LEVEL_WARNING, "KataGoWinMLIntrospect");
    Ort::SessionOptions tmpOpts;
    tmpOpts.SetIntraOpNumThreads(1);
    Ort::Session tmpSession(tmpEnv, rawOnnxBytes.data(), rawOnnxBytes.size(), tmpOpts);

    Ort::AllocatorWithDefaultOptions allocator;

    int numInputChannels = 0;
    int numInputGlobalChannels = 0;
    int numInputMetaChannels = 0;
    size_t numInputs = tmpSession.GetInputCount();
    for(size_t i = 0; i < numInputs; i++) {
      Ort::AllocatedStringPtr namePtr = tmpSession.GetInputNameAllocated(i, allocator);
      string name = namePtr.get();
      auto typeInfo = tmpSession.GetInputTypeInfo(i);
      auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
      auto shape = tensorInfo.GetShape();
      if(name.find("spatial") != string::npos) {
        if(shape.size() >= 2)
          numInputChannels = (int)shape[1];
      } else if(name.find("global") != string::npos) {
        if(shape.size() >= 2)
          numInputGlobalChannels = (int)shape[1];
      } else if(name.find("meta") != string::npos) {
        if(shape.size() >= 2)
          numInputMetaChannels = (int)shape[1];
      } else if(shape.size() == 4) {
        numInputChannels = (int)shape[1];
      } else if(shape.size() == 2) {
        if(numInputGlobalChannels == 0)
          numInputGlobalChannels = (int)shape[1];
        else
          numInputMetaChannels = (int)shape[1];
      } else {
        cerr << "WinML backend warning: unrecognized input tensor '" << name
             << "' with " << shape.size() << "D shape, ignoring" << "\n";
      }
    }

    int numPolicyChannels = 0;
    int numValueChannels = 0;
    int numScoreValueChannels = 0;
    int numOwnershipChannels = 0;
    size_t numOutputs = tmpSession.GetOutputCount();
    for(size_t i = 0; i < numOutputs; i++) {
      Ort::AllocatedStringPtr namePtr = tmpSession.GetOutputNameAllocated(i, allocator);
      string name = namePtr.get();
      auto typeInfo = tmpSession.GetOutputTypeInfo(i);
      auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
      auto shape = tensorInfo.GetShape();

      if(name.find("policy") != string::npos) {
        if(shape.size() >= 2)
          numPolicyChannels = (int)shape[1];
      } else if(name.find("miscvalue") != string::npos) {
        if(shape.size() >= 2)
          numScoreValueChannels = (int)shape[1];
      } else if(name.find("value") != string::npos) {
        if(shape.size() >= 2)
          numValueChannels = (int)shape[1];
      } else if(name.find("ownership") != string::npos) {
        if(shape.size() >= 2)
          numOwnershipChannels = (int)shape[1];
      }
    }

    modelDesc.numInputChannels = numInputChannels;
    modelDesc.numInputGlobalChannels = numInputGlobalChannels;
    modelDesc.numInputMetaChannels = numInputMetaChannels;
    modelDesc.numPolicyChannels = numPolicyChannels;
    modelDesc.numValueChannels = numValueChannels;
    modelDesc.numScoreValueChannels = numScoreValueChannels;
    modelDesc.numOwnershipChannels = numOwnershipChannels;

    {
      size_t lastSlash = fileName.find_last_of("/\\");
      string basename = (lastSlash != string::npos) ? fileName.substr(lastSlash + 1) : fileName;
      size_t dotPos = basename.find('.');
      modelDesc.name = (dotPos != string::npos) ? basename.substr(0, dotPos) : basename;
    }

    modelDesc.modelVersion = detectModelVersion(
      numInputChannels, numInputGlobalChannels,
      numPolicyChannels, numScoreValueChannels,
      -1
    );
  }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

LoadedModel* NeuralNet::loadModelFile(const string& file, const string& expectedSha256) {
  bool isRawOnnx = Global::isSuffix(file, ".onnx");
  return new LoadedModel(file, expectedSha256, isRawOnnx);
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}

const ModelDesc& NeuralNet::getModelDesc(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc;
}

//--------------------------------------------------------------

// EP library paths discovered at startup (via catalog or compile-time fallback).
static std::wstring g_openvinoEpLibPath;
static std::wstring g_nvtrtRtxEpLibPath;
static std::wstring g_migraphxEpLibPath;
static std::wstring g_vitisaiEpLibPath;
static std::wstring g_qnnEpLibPath;

// Whether the EP was initialized through EnsureReady (vs direct fallback loading).
static bool g_openvinoEpCatalogReady = false;
static bool g_nvtrtRtxEpCatalogReady = false;
static bool g_migraphxEpCatalogReady = false;
static bool g_vitisaiEpCatalogReady = false;
static bool g_qnnEpCatalogReady = false;

// Add a directory to the DLL search path so EP plugin dependencies can be found.
static void addDllSearchDir(const wchar_t* dir) {
  if(dir && dir[0])
    AddDllDirectory(dir);
}
static void addDllSearchDirFromPath(const std::wstring& dllPath) {
  // Extract directory from full DLL path.
  size_t pos = dllPath.find_last_of(L"/\\");
  if(pos != std::wstring::npos)
    addDllSearchDir(dllPath.substr(0, pos).c_str());
}

//--------------------------------------------------------------
// Dynamic Dependencies support for unpackaged desktop apps (Windows 11 22H2+).
//
// Store EP packages (OpenVINO, NvTRT-RTX) register as uap17:PackageExtension
// with DependencyTarget=true. For unpackaged apps, we must add them to the
// process's package graph via Dynamic Dependencies so the EP Catalog can find
// and properly initialize them (via EnsureReady).
//--------------------------------------------------------------
#ifdef _WIN32

// Known Store EP package family names.
static const wchar_t* kStoreEpPackageFamilies[] = {
  L"MicrosoftCorporationII.WinML.Intel.OpenVINO.EP.1.8_8wekyb3d8bbwe",
  L"MicrosoftCorporationII.WinML.NVIDIA.TRT-RTX.EP.2_8wekyb3d8bbwe",
  L"MicrosoftCorporationII.WinML.NVIDIA.TRT-RTX.EP.1.8_8wekyb3d8bbwe",
  L"MicrosoftCorporationII.WinML.AMD.MIGraphX.EP_8wekyb3d8bbwe",
  L"MicrosoftCorporationII.WinML.AMD.MIGraphX.EP.1.8_8wekyb3d8bbwe",
  L"MicrosoftCorporationII.WinML.Xilinx.VitisAI.EP_8wekyb3d8bbwe",
  L"MicrosoftCorporationII.WinML.Xilinx.VitisAI.EP.1.8_8wekyb3d8bbwe",
  L"MicrosoftCorporationII.WinML.Qualcomm.QNN.EP_8wekyb3d8bbwe",
  L"MicrosoftCorporationII.WinML.Qualcomm.QNN.EP.1.8_8wekyb3d8bbwe",
};

// We define the Dynamic Dependencies types ourselves to avoid requiring
// a specific SDK target version. Functions are loaded via GetProcAddress.
struct KG_PACKAGE_VERSION {
  union {
    UINT64 Version;
    struct { USHORT Revision; USHORT Build; USHORT Minor; USHORT Major; } Parts;
  };
};

// See: https://learn.microsoft.com/en-us/windows/win32/api/appmodel/ne-appmodel-packagedependencyprocessorarchitectures
enum KG_PackageDependencyProcessorArchitectures {
  KG_PDP_None = 0, KG_PDP_Neutral = 0x1, KG_PDP_X86 = 0x2, KG_PDP_X64 = 0x4,
  KG_PDP_Arm = 0x8, KG_PDP_Arm64 = 0x10, KG_PDP_X86A64 = 0x20
};
enum KG_PackageDependencyLifetimeKind {
  KG_PDLK_Process = 0, KG_PDLK_FilePath = 1, KG_PDLK_RegistryKey = 2
};
enum KG_CreatePackageDependencyOptions {
  KG_CPDO_None = 0, KG_CPDO_DoNotVerifyDependencyResolution = 0x1, KG_CPDO_ScopeIsSystem = 0x2
};
enum KG_AddPackageDependencyOptions {
  KG_APDO_None = 0, KG_APDO_PrependIfRankCollision = 0x1
};

typedef HRESULT (WINAPI* PFN_TryCreatePackageDependency)(
  PSID user, PCWSTR packageFamilyName, KG_PACKAGE_VERSION minVersion,
  KG_PackageDependencyProcessorArchitectures arch,
  KG_PackageDependencyLifetimeKind lifetimeKind, PCWSTR lifetimeArtifact,
  KG_CreatePackageDependencyOptions options, PWSTR* packageDependencyId);

typedef HRESULT (WINAPI* PFN_AddPackageDependency)(
  PCWSTR packageDependencyId, INT32 rank,
  KG_AddPackageDependencyOptions options, void** context, PWSTR* packageFullName);

// Attempt to activate Store EP packages via Dynamic Dependencies.
// This makes them discoverable by the WinMLEpCatalog.
static void tryActivateStoreEpPackages() {
  HMODULE mod = GetModuleHandleW(L"kernelbase.dll");
  if(!mod) return;

  auto pfnCreate = (PFN_TryCreatePackageDependency)GetProcAddress(mod, "TryCreatePackageDependency");
  auto pfnAdd = (PFN_AddPackageDependency)GetProcAddress(mod, "AddPackageDependency");
  if(!pfnCreate || !pfnAdd) {
    cout << "WinML backend: Dynamic Dependencies API not available (requires Windows 11 22H2+)" << endl;
    return;
  }

  cout << "WinML backend: activating Store EP packages via Dynamic Dependencies..." << endl;

  for(const auto* familyName : kStoreEpPackageFamilies) {
    KG_PACKAGE_VERSION minVer = {};
    PWSTR depId = nullptr;

    HRESULT hr = pfnCreate(
      nullptr,        // current user
      familyName,
      minVer,
      KG_PDP_X64,
      KG_PDLK_Process,
      nullptr,        // no lifetime artifact
      KG_CPDO_DoNotVerifyDependencyResolution,
      &depId
    );

    if(FAILED(hr) || !depId) {
      // Package not installed or not available — this is normal for EPs the user doesn't have
      continue;
    }

    void* ctx = nullptr;
    PWSTR fullName = nullptr;
    hr = pfnAdd(depId, 0, KG_APDO_None, &ctx, &fullName);
    if(SUCCEEDED(hr)) {
      if(fullName) {
        wprintf(L"WinML backend: activated Store EP package: %s\n", fullName);
        CoTaskMemFree(fullName);
      }
    } else {
      wprintf(L"WinML backend: AddPackageDependency failed for %s (hr=0x%08lx)\n", familyName, hr);
    }
    CoTaskMemFree(depId);
  }
}

#endif // _WIN32

struct ComputeContext {
  Ort::Env env;
  int nnXLen;
  int nnYLen;
  string providerName;

  // OpenVINO-specific options
  string openvinoDeviceType;
  string openvinoDeviceId;
  bool openvinoEnableNPUFastCompile;
  string openvinoCacheDir;

  // Configurable input/output node names
  string inputSpatialName;
  string inputGlobalName;
  string inputMetaName;
  string outputPolicyName;
  string outputValueName;
  string outputMiscvalueName;
  string outputOwnershipName;

  // Config override for model version (-1 means auto-detect)
  int configModelVersion;

  ComputeContext(int xLen, int yLen, const string& provider)
    : env(ORT_LOGGING_LEVEL_WARNING, "KataGoWinML"),
      nnXLen(xLen),
      nnYLen(yLen),
      providerName(provider),
      openvinoDeviceType("NPU"),
      openvinoDeviceId(""),
      openvinoEnableNPUFastCompile(false),
      openvinoCacheDir(""),
      inputSpatialName("input_spatial"),
      inputGlobalName("input_global"),
      inputMetaName("input_meta"),
      outputPolicyName("out_policy"),
      outputValueName("out_value"),
      outputMiscvalueName("out_miscvalue"),
      outputOwnershipName("out_ownership"),
      configModelVersion(-1)
  {
    // Register Store EP libraries on this Env so AppendExecutionProvider can find them.
    if(provider == "openvino" && !g_openvinoEpLibPath.empty()) {
      try {
        env.RegisterExecutionProviderLibrary("OpenVINO", g_openvinoEpLibPath);
      } catch(const Ort::Exception& e) {
        cerr << "WinML backend: failed to register OpenVINO EP: " << e.what() << endl;
      }
    }
    if(provider == "nvtensorrtrtx" && !g_nvtrtRtxEpLibPath.empty()) {
      try {
        env.RegisterExecutionProviderLibrary("NvTensorRtRtx", g_nvtrtRtxEpLibPath);
      } catch(const Ort::Exception& e) {
        cerr << "WinML backend: failed to register NvTensorRtRtx EP: " << e.what() << endl;
      }
    }
    if(provider == "migraphx" && !g_migraphxEpLibPath.empty()) {
      try {
        env.RegisterExecutionProviderLibrary("MIGraphX", g_migraphxEpLibPath);
      } catch(const Ort::Exception& e) {
        cerr << "WinML backend: failed to register MIGraphX EP: " << e.what() << endl;
      }
    }
    if(provider == "vitisai" && !g_vitisaiEpLibPath.empty()) {
      try {
        env.RegisterExecutionProviderLibrary("VitisAI", g_vitisaiEpLibPath);
      } catch(const Ort::Exception& e) {
        cerr << "WinML backend: failed to register VitisAI EP: " << e.what() << endl;
      }
    }
    if(provider == "qnn" && !g_qnnEpLibPath.empty()) {
      try {
        env.RegisterExecutionProviderLibrary("QNN", g_qnnEpLibPath);
      } catch(const Ort::Exception& e) {
        cerr << "WinML backend: failed to register QNN EP: " << e.what() << endl;
      }
    }
  }
};

//--------------------------------------------------------------

struct ComputeHandle {
  ComputeContext* context;
  std::unique_ptr<Ort::Session> session;
  int modelVersion;
  int numInputChannels;
  int numInputGlobalChannels;
  int numPolicyChannels;
  int numValueChannels;
  int numScoreValueChannels;
  int numOwnershipChannels;
  int numInputMetaChannels;
  int policyResultLen;
  bool fixedBatchOne;  // NPU requires batch=1; getOutput must loop

  vector<string> inputNames;
  vector<string> outputNames;
  vector<const char*> inputNamePtrs;
  vector<const char*> outputNamePtrs;

  ComputeHandle(ComputeContext* ctx, const LoadedModel& loadedModel, Logger* logger, int deviceIdxForThread)
    : context(ctx),
      modelVersion(loadedModel.modelDesc.modelVersion),
      numInputChannels(loadedModel.modelDesc.numInputChannels),
      numInputGlobalChannels(loadedModel.modelDesc.numInputGlobalChannels),
      numPolicyChannels(loadedModel.modelDesc.numPolicyChannels),
      numValueChannels(loadedModel.modelDesc.numValueChannels),
      numScoreValueChannels(loadedModel.modelDesc.numScoreValueChannels),
      numOwnershipChannels(loadedModel.modelDesc.numOwnershipChannels),
      numInputMetaChannels(loadedModel.modelDesc.numInputMetaChannels),
      policyResultLen(ctx->nnXLen * ctx->nnYLen + 1),
      fixedBatchOne(false)
  {
    if(ctx->configModelVersion >= 0)
      modelVersion = ctx->configModelVersion;

    const char* onnxData;
    size_t onnxSize;
    string builtOnnxBytes;
    if(loadedModel.isRawOnnx) {
      if(logger != NULL)
        logger->write("WinML backend: using raw ONNX model (" +
                       Global::uint64ToString(loadedModel.rawOnnxBytes.size()) + " bytes)");
      onnxData = loadedModel.rawOnnxBytes.data();
      onnxSize = loadedModel.rawOnnxBytes.size();
    } else {
      builtOnnxBytes = OnnxModelBuilder::buildOnnxModel(loadedModel.modelDesc, ctx->nnXLen, ctx->nnYLen);
      if(logger != NULL)
        logger->write("WinML backend: ONNX graph built from .bin.gz (" +
                       Global::uint64ToString(builtOnnxBytes.size()) + " bytes)");
      onnxData = builtOnnxBytes.data();
      onnxSize = builtOnnxBytes.size();
    }

    if(logger != NULL)
      logger->write("WinML backend: creating session...");

    Ort::SessionOptions sessionOpts;
    sessionOpts.SetIntraOpNumThreads(1);

    // Enable thread spinning for better latency (disabled by default in Windows ML for battery life)
    sessionOpts.AddConfigEntry("session.intra_op.allow_spinning", "1");
    sessionOpts.AddConfigEntry("session.inter_op.allow_spinning", "1");

    // Select execution provider
    const string& provider = ctx->providerName;
    int deviceIdx = deviceIdxForThread >= 0 ? deviceIdxForThread : 0;

    // NPU requires static shapes — override the dynamic batch dimension "N" to 1.
    // This must be set before appending any execution provider.
    if(provider == "openvino") {
      string upperDevType = ctx->openvinoDeviceType;
      for(auto& c : upperDevType) c = toupper(c);
      if(upperDevType == "NPU") {
        sessionOpts.AddFreeDimensionOverrideByName("N", 1);
        fixedBatchOne = true;
        if(logger != NULL)
          logger->write("WinML backend: fixed batch dimension N=1 for NPU");
      }
    }

    if(provider == "dml") {
      // DirectML - always available on Windows 10+
      std::unordered_map<std::string, std::string> dmlOpts;
      dmlOpts["device_id"] = Global::intToString(deviceIdx);
      sessionOpts.AppendExecutionProvider("DML", dmlOpts);
      if(logger != NULL)
        logger->write("WinML backend: DirectML execution provider enabled, device_id=" + Global::intToString(deviceIdx));

    } else if(provider == "openvino") {
      // Use V2 API for plugin-loaded EPs (old AppendExecutionProvider does not work with plugin EPs).
      // V2 flow: GetEpDevices() -> filter by EP name + hardware type -> AppendExecutionProvider_V2()
      auto allEpDevices = ctx->env.GetEpDevices();

      // Map user device_type string to OrtHardwareDeviceType for filtering
      OrtHardwareDeviceType desiredHwType = OrtHardwareDeviceType_GPU;
      string upperDevType = ctx->openvinoDeviceType;
      for(auto& c : upperDevType) c = toupper(c);
      if(upperDevType == "CPU") desiredHwType = OrtHardwareDeviceType_CPU;
      else if(upperDevType == "NPU") desiredHwType = OrtHardwareDeviceType_NPU;
      else desiredHwType = OrtHardwareDeviceType_GPU;

      // Find OpenVINO devices matching the desired hardware type.
      // Exclude ".AUTO" variants which may cause conflicts.
      std::vector<Ort::ConstEpDevice> matchedDevices;
      std::vector<Ort::ConstEpDevice> allOpenvinoDevices;
      for(size_t di = 0; di < allEpDevices.size(); di++) {
        const char* epName = nullptr;
        try { epName = allEpDevices[di].EpName(); } catch(...) { continue; }
        if(!epName) continue;
        string name(epName);
        // Match EP name containing "OpenVINO" but not ".AUTO" variants
        if(name.find("OpenVINO") != string::npos && name.find(".AUTO") == string::npos) {
          allOpenvinoDevices.push_back(allEpDevices[di]);
          try {
            if(allEpDevices[di].Device().Type() == desiredHwType)
              matchedDevices.push_back(allEpDevices[di]);
          } catch(...) {}
        }
      }

      // Fallback to any OpenVINO device if no hardware type match
      if(matchedDevices.empty())
        matchedDevices = allOpenvinoDevices;

      if(matchedDevices.empty()) {
        throw StringError("WinML backend: no OpenVINO EP devices found. GetEpDevices returned " +
          Global::uint64ToString(allEpDevices.size()) + " total device(s). "
          "Make sure the OpenVINO EP is installed from the Microsoft Store.");
      }

      // V2 API: pass single device (first match). Passing multiple devices can crash.
      // Do NOT include device_type in options — it conflicts with the V2 device selection.
      // cache_dir is safe to pass for model compilation caching.
      std::vector<Ort::ConstEpDevice> singleDevice = { matchedDevices[0] };
      std::unordered_map<std::string, std::string> v2Opts;
      if(!ctx->openvinoCacheDir.empty())
        v2Opts["cache_dir"] = ctx->openvinoCacheDir;
      if(ctx->openvinoEnableNPUFastCompile)
        v2Opts["enable_npu_fast_compile"] = "true";

      sessionOpts.AppendExecutionProvider_V2(ctx->env, singleDevice, v2Opts);

      if(logger != NULL) {
        logger->write("WinML backend: OpenVINO execution provider enabled via V2 API, device_type=" +
          ctx->openvinoDeviceType +
          (ctx->openvinoDeviceId.empty() ? "" : (", device_id=" + ctx->openvinoDeviceId)) +
          (ctx->openvinoCacheDir.empty() ? "" : (", cache_dir=" + ctx->openvinoCacheDir)));
      }

    } else if(provider == "nvtensorrtrtx") {
      std::unordered_map<std::string, std::string> nvidiaOpts;
      nvidiaOpts["device_id"] = Global::intToString(deviceIdx);

      // Use V2 API for plugin-loaded EPs
      if(logger != NULL)
        logger->write("WinML backend: enumerating EP devices for NvTensorRtRtx...");
      auto allEpDevices = ctx->env.GetEpDevices();
      std::vector<Ort::ConstEpDevice> nvDevices;
      for(const auto& dev : allEpDevices) {
        const char* epName = dev.EpName();
        if(epName != nullptr) {
          string name(epName);
          if(name.find("NvTensorRt") != string::npos || name.find("TensorRT") != string::npos) {
            nvDevices.push_back(dev);
            if(logger != NULL)
              logger->write(string("  -> selected NV device: ") + epName);
          }
        }
      }

      if(!nvDevices.empty()) {
        sessionOpts.AppendExecutionProvider_V2(ctx->env, nvDevices, nvidiaOpts);
      } else {
        // Fallback to old API if no V2 devices found (e.g. built-in EP)
        sessionOpts.AppendExecutionProvider("NvTensorRtRtx", nvidiaOpts);
      }
      if(logger != NULL)
        logger->write("WinML backend: NvTensorRtRtx execution provider enabled, device_id=" + Global::intToString(deviceIdx));

    } else if(provider == "migraphx") {
      std::unordered_map<std::string, std::string> migraphxOpts;
      migraphxOpts["device_id"] = Global::intToString(deviceIdx);

      if(g_migraphxEpCatalogReady) {
        if(logger != NULL)
          logger->write("WinML backend: enumerating EP devices for MIGraphX...");
        auto allEpDevices = ctx->env.GetEpDevices();
        std::vector<Ort::ConstEpDevice> mgxDevices;
        for(const auto& dev : allEpDevices) {
          const char* epName = dev.EpName();
          if(epName != nullptr) {
            string name(epName);
            if(name.find("MIGraphX") != string::npos) {
              mgxDevices.push_back(dev);
              if(logger != NULL)
                logger->write(string("  -> selected MIGraphX device: ") + epName);
            }
          }
        }
        if(!mgxDevices.empty())
          sessionOpts.AppendExecutionProvider_V2(ctx->env, mgxDevices, migraphxOpts);
        else
          sessionOpts.AppendExecutionProvider("MIGraphX", migraphxOpts);
      } else {
        sessionOpts.AppendExecutionProvider("MIGraphX", migraphxOpts);
      }
      if(logger != NULL)
        logger->write("WinML backend: MIGraphX execution provider enabled, device_id=" + Global::intToString(deviceIdx));

    } else if(provider == "qnn") {
      std::unordered_map<std::string, std::string> qnnOpts;

      if(g_qnnEpCatalogReady) {
        if(logger != NULL)
          logger->write("WinML backend: enumerating EP devices for QNN...");
        auto allEpDevices = ctx->env.GetEpDevices();
        std::vector<Ort::ConstEpDevice> qnnDevices;
        for(const auto& dev : allEpDevices) {
          const char* epName = dev.EpName();
          if(epName != nullptr) {
            string name(epName);
            if(name.find("QNN") != string::npos) {
              qnnDevices.push_back(dev);
              if(logger != NULL)
                logger->write(string("  -> selected QNN device: ") + epName);
            }
          }
        }
        if(!qnnDevices.empty())
          sessionOpts.AppendExecutionProvider_V2(ctx->env, qnnDevices, qnnOpts);
        else
          sessionOpts.AppendExecutionProvider("QNN", qnnOpts);
      } else {
        sessionOpts.AppendExecutionProvider("QNN", qnnOpts);
      }
      if(logger != NULL)
        logger->write("WinML backend: QNN execution provider enabled");

    } else if(provider == "vitisai") {
      std::unordered_map<std::string, std::string> vitisOpts;

      if(g_vitisaiEpCatalogReady) {
        if(logger != NULL)
          logger->write("WinML backend: enumerating EP devices for VitisAI...");
        auto allEpDevices = ctx->env.GetEpDevices();
        std::vector<Ort::ConstEpDevice> vitisDevices;
        for(const auto& dev : allEpDevices) {
          const char* epName = dev.EpName();
          if(epName != nullptr) {
            string name(epName);
            if(name.find("VitisAI") != string::npos) {
              vitisDevices.push_back(dev);
              if(logger != NULL)
                logger->write(string("  -> selected VitisAI device: ") + epName);
            }
          }
        }
        if(!vitisDevices.empty())
          sessionOpts.AppendExecutionProvider_V2(ctx->env, vitisDevices, vitisOpts);
        else
          sessionOpts.AppendExecutionProvider("VitisAI", vitisOpts);
      } else {
        sessionOpts.AppendExecutionProvider("VitisAI", vitisOpts);
      }
      if(logger != NULL)
        logger->write("WinML backend: VitisAI execution provider enabled");

    } else if(provider == "cpu" || provider.empty()) {
      if(logger != NULL)
        logger->write("WinML backend: using CPU execution provider");

    } else {
      throw StringError("WinML backend: unknown winmlProvider '" + provider +
        "', expected 'cpu', 'dml', 'openvino', 'nvtensorrtrtx', 'migraphx', 'qnn', or 'vitisai'");
    }

    // Create session from in-memory bytes
    if(logger != NULL)
      logger->write("WinML backend: calling Ort::Session constructor...");
    try {
      session = std::make_unique<Ort::Session>(ctx->env, onnxData, onnxSize, sessionOpts);
    } catch(const Ort::Exception& e) {
      throw StringError(string("WinML backend: ORT session creation failed: ") + e.what());
    } catch(const std::exception& e) {
      throw StringError(string("WinML backend: session creation std::exception: ") + e.what());
    } catch(...) {
      throw StringError("WinML backend: session creation unknown exception");
    }

    // Query and store input names
    Ort::AllocatorWithDefaultOptions allocator;
    size_t numInputs = session->GetInputCount();
    for(size_t i = 0; i < numInputs; i++) {
      Ort::AllocatedStringPtr name = session->GetInputNameAllocated(i, allocator);
      inputNames.push_back(name.get());
    }
    for(auto& n : inputNames)
      inputNamePtrs.push_back(n.c_str());

    // Query and store output names
    size_t numOutputs = session->GetOutputCount();
    for(size_t i = 0; i < numOutputs; i++) {
      Ort::AllocatedStringPtr name = session->GetOutputNameAllocated(i, allocator);
      outputNames.push_back(name.get());
    }
    for(auto& n : outputNames)
      outputNamePtrs.push_back(n.c_str());

    if(logger != NULL)
      logger->write("WinML backend: session created, inputs=" + Global::uint64ToString(numInputs) +
                     " outputs=" + Global::uint64ToString(numOutputs));
  }

  ComputeHandle() = delete;
  ComputeHandle(const ComputeHandle&) = delete;
  ComputeHandle& operator=(const ComputeHandle&) = delete;
};

//--------------------------------------------------------------

struct InputBuffers {
  int maxBatchSize;
  size_t singleInputElts;
  size_t singleInputGlobalElts;
  size_t singleInputMetaElts;
  vector<float> spatialInput;
  vector<float> globalInput;
  vector<float> metaInput;

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;
    maxBatchSize = maxBatchSz;
    singleInputElts = (size_t)m.numInputChannels * nnXLen * nnYLen;
    singleInputGlobalElts = (size_t)m.numInputGlobalChannels;
    singleInputMetaElts = (size_t)m.numInputMetaChannels;
    spatialInput.resize(singleInputElts * maxBatchSize, 0.0f);
    globalInput.resize(singleInputGlobalElts * maxBatchSize, 0.0f);
    if(m.numInputMetaChannels > 0)
      metaInput.resize(singleInputMetaElts * maxBatchSize, 0.0f);
  }

  ~InputBuffers() {}

  InputBuffers() = delete;
  InputBuffers(const InputBuffers&) = delete;
  InputBuffers& operator=(const InputBuffers&) = delete;
};

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  return new InputBuffers(loadedModel, maxBatchSize, nnXLen, nnYLen);
}
void NeuralNet::freeInputBuffers(InputBuffers* inputBuffers) {
  delete inputBuffers;
}

//--------------------------------------------------------------

// Helper: convert narrow UTF-8 string to wide string.
static std::wstring toWide(const char* s) {
  if(!s || !*s) return {};
  int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
  if(n <= 0) return {};
  std::wstring w(n - 1, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s, -1, w.data(), n);
  return w;
}

#ifdef WINML_HAS_EP_CATALOG
// Enumeration callback: prints all available EPs in the catalog for diagnostics.
static BOOL CALLBACK enumEpCallback(WinMLEpHandle ep, const WinMLEpInfo* info, void* context) {
  (void)ep;
  (void)context;
  cout << "  EP: name='" << (info->name ? info->name : "?")
       << "' version='" << (info->version ? info->version : "?")
       << "' pkgFamily='" << (info->packageFamilyName ? info->packageFamilyName : "?")
       << "' readyState=" << (int)info->readyState
       << " path='" << (info->libraryPath ? info->libraryPath : "?") << "'" << endl;
  return TRUE; // continue enumeration
}

// Use WinMLEpCatalog C API to find an EP by name, ensure it is ready,
// and return its library path. Returns empty string on failure.
// Tries multiple name variants for robust matching.
static std::wstring findEpLibraryPath(WinMLEpCatalogHandle catalog, const char* epName,
                                       const vector<string>& alternateNames, bool& outCatalogReady) {
  outCatalogReady = false;

  // Try the primary name first, then alternates
  vector<string> namesToTry;
  namesToTry.push_back(epName);
  for(const auto& alt : alternateNames) {
    if(alt != epName) namesToTry.push_back(alt);
  }

  WinMLEpHandle ep = nullptr;
  HRESULT hr = E_FAIL;
  string matchedName;

  for(const auto& name : namesToTry) {
    hr = WinMLEpCatalogFindProvider(catalog, name.c_str(), nullptr, &ep);
    if(SUCCEEDED(hr) && ep) {
      matchedName = name;
      break;
    }
    cout << "WinML backend: EP '" << name << "' not found in catalog (hr=0x"
         << std::hex << hr << std::dec << ")" << endl;
  }

  if(!ep)
    return {};

  cout << "WinML backend: EP found via name '" << matchedName << "', calling EnsureReady..." << endl;
  hr = WinMLEpEnsureReady(ep);
  if(FAILED(hr)) {
    cout << "WinML backend: EP '" << matchedName << "' not ready (hr=0x"
         << std::hex << hr << std::dec << ")" << endl;
    return {};
  }
  cout << "WinML backend: EP '" << matchedName << "' is ready." << endl;
  outCatalogReady = true;

  size_t pathSize = 0;
  hr = WinMLEpGetLibraryPathSize(ep, &pathSize);
  if(FAILED(hr) || pathSize == 0) {
    cerr << "WinML backend: could not get library path size for EP '" << matchedName << "'" << endl;
    return {};
  }
  std::string pathBuf(pathSize, '\0');
  size_t used = 0;
  hr = WinMLEpGetLibraryPath(ep, pathSize, pathBuf.data(), &used);
  if(FAILED(hr)) {
    cerr << "WinML backend: could not get library path for EP '" << matchedName << "'" << endl;
    return {};
  }
  while(!pathBuf.empty() && pathBuf.back() == '\0')
    pathBuf.pop_back();

  cout << "WinML backend: EP '" << matchedName << "' library path: " << pathBuf << endl;
  return toWide(pathBuf.c_str());
}
#endif // WINML_HAS_EP_CATALOG

// Find the EP plugin DLL inside a Store EP directory.
// Store EP dirs contain files like onnxruntime_providers_openvino_plugin.dll.
static std::wstring findEpDllInDir(const char* dirPath, const char* dllPattern) {
  string dir(dirPath);
  // Normalize path separators
  for(auto& c : dir) { if(c == '\\') c = '/'; }
  if(!dir.empty() && dir.back() != '/') dir += '/';

  WIN32_FIND_DATAW findData;
  std::wstring searchPattern = toWide((dir + dllPattern).c_str());
  HANDLE hFind = FindFirstFileW(searchPattern.c_str(), &findData);
  if(hFind == INVALID_HANDLE_VALUE)
    return {};
  std::wstring result = toWide(dir.c_str()) + findData.cFileName;
  FindClose(hFind);
  return result;
}

void NeuralNet::globalInitialize() {
  // Step 1: For unpackaged desktop apps, activate Store EP packages via Dynamic Dependencies.
  // This adds them to the process's package graph so the EP Catalog can discover them.
#ifdef _WIN32
  tryActivateStoreEpPackages();
#endif

  // Step 2: Try WinMLEpCatalog C API to discover and initialize EPs.
#ifdef WINML_HAS_EP_CATALOG
  {
    WinMLEpCatalogHandle catalog = nullptr;
    HRESULT hr = WinMLEpCatalogCreate(&catalog);
    if(SUCCEEDED(hr) && catalog) {
      cout << "WinML backend: EP catalog created, enumerating all execution providers..." << endl;
      WinMLEpCatalogEnumProviders(catalog, enumEpCallback, nullptr);

      // Try to find OpenVINO EP with multiple name variants
      // (manifest says ProviderName="OpenVINOExecutionProvider", but catalog may use short name)
      g_openvinoEpLibPath = findEpLibraryPath(catalog, "OpenVINO",
        {"OpenVINOExecutionProvider", "OpenVINOEP", "openvino"}, g_openvinoEpCatalogReady);

      // Try to find NvTensorRtRtx EP
      g_nvtrtRtxEpLibPath = findEpLibraryPath(catalog, "NvTensorRtRtx",
        {"NvTensorRtRtxExecutionProvider", "NvTensorRtRtxEP"}, g_nvtrtRtxEpCatalogReady);

      // Try to find MIGraphX EP
      g_migraphxEpLibPath = findEpLibraryPath(catalog, "MIGraphX",
        {"MIGraphXExecutionProvider", "MIGraphXEP", "migraphx"}, g_migraphxEpCatalogReady);

      // Try to find VitisAI EP
      g_vitisaiEpLibPath = findEpLibraryPath(catalog, "VitisAI",
        {"VitisAIExecutionProvider", "VitisAIEP", "vitisai"}, g_vitisaiEpCatalogReady);

      // Try to find QNN EP
      g_qnnEpLibPath = findEpLibraryPath(catalog, "QNN",
        {"QNNExecutionProvider", "QNNEP", "qnn"}, g_qnnEpCatalogReady);

      WinMLEpCatalogRelease(catalog);
    } else {
      cout << "WinML backend: EP catalog unavailable (hr=0x" << std::hex << hr << std::dec
           << "), falling back to compile-time paths." << endl;
    }
  }
#endif

  // Step 3: Fallback — use compile-time discovered Store EP directories.
  // NOTE: EPs loaded via fallback were NOT initialized by EnsureReady,
  // which may cause crashes. The Dynamic Dependencies + Catalog path above is preferred.
#ifdef WINML_STORE_OPENVINO_EP_DIR
  if(g_openvinoEpLibPath.empty()) {
    g_openvinoEpLibPath = findEpDllInDir(WINML_STORE_OPENVINO_EP_DIR, "onnxruntime_providers_openvino*.dll");
    if(!g_openvinoEpLibPath.empty())
      cout << "WinML backend: Found OpenVINO EP DLL via compile-time path (no EnsureReady!)" << endl;
  }
#endif
#ifdef WINML_STORE_NVTRT_RTX_EP_DIR
  if(g_nvtrtRtxEpLibPath.empty()) {
    g_nvtrtRtxEpLibPath = findEpDllInDir(WINML_STORE_NVTRT_RTX_EP_DIR, "onnxruntime_providers_nv_tensorrt_rtx*.dll");
    if(!g_nvtrtRtxEpLibPath.empty())
      cout << "WinML backend: Found NvTensorRtRtx EP DLL via compile-time path (no EnsureReady!)" << endl;
  }
#endif

  // Step 4: Add EP directories to DLL search path so plugin dependencies are found.
  if(!g_openvinoEpLibPath.empty()) {
    addDllSearchDirFromPath(g_openvinoEpLibPath);
    // Also prepend to PATH for maximum compatibility with implicit LoadLibrary calls
    std::wstring epDir = g_openvinoEpLibPath.substr(0, g_openvinoEpLibPath.find_last_of(L"/\\"));
    std::wstring oldPath(32768, L'\0');
    DWORD len = GetEnvironmentVariableW(L"PATH", oldPath.data(), (DWORD)oldPath.size());
    oldPath.resize(len);
    std::wstring newPath = epDir + L";" + oldPath;
    SetEnvironmentVariableW(L"PATH", newPath.c_str());
  } else {
    cout << "WinML backend: OpenVINO EP not available." << endl;
  }
  if(!g_nvtrtRtxEpLibPath.empty()) {
    addDllSearchDirFromPath(g_nvtrtRtxEpLibPath);
    std::wstring epDir = g_nvtrtRtxEpLibPath.substr(0, g_nvtrtRtxEpLibPath.find_last_of(L"/\\"));
    std::wstring oldPath(32768, L'\0');
    DWORD len = GetEnvironmentVariableW(L"PATH", oldPath.data(), (DWORD)oldPath.size());
    oldPath.resize(len);
    std::wstring newPath = epDir + L";" + oldPath;
    SetEnvironmentVariableW(L"PATH", newPath.c_str());
  } else {
    cout << "WinML backend: NvTensorRtRtx EP not available." << endl;
  }
  if(!g_migraphxEpLibPath.empty()) {
    addDllSearchDirFromPath(g_migraphxEpLibPath);
    std::wstring epDir = g_migraphxEpLibPath.substr(0, g_migraphxEpLibPath.find_last_of(L"/\\"));
    std::wstring oldPath(32768, L'\0');
    DWORD len = GetEnvironmentVariableW(L"PATH", oldPath.data(), (DWORD)oldPath.size());
    oldPath.resize(len);
    std::wstring newPath = epDir + L";" + oldPath;
    SetEnvironmentVariableW(L"PATH", newPath.c_str());
  } else {
    cout << "WinML backend: MIGraphX EP not available." << endl;
  }
  if(!g_vitisaiEpLibPath.empty()) {
    addDllSearchDirFromPath(g_vitisaiEpLibPath);
    std::wstring epDir = g_vitisaiEpLibPath.substr(0, g_vitisaiEpLibPath.find_last_of(L"/\\"));
    std::wstring oldPath(32768, L'\0');
    DWORD len = GetEnvironmentVariableW(L"PATH", oldPath.data(), (DWORD)oldPath.size());
    oldPath.resize(len);
    std::wstring newPath = epDir + L";" + oldPath;
    SetEnvironmentVariableW(L"PATH", newPath.c_str());
  } else {
    cout << "WinML backend: VitisAI EP not available." << endl;
  }
  if(!g_qnnEpLibPath.empty()) {
    addDllSearchDirFromPath(g_qnnEpLibPath);
    std::wstring epDir = g_qnnEpLibPath.substr(0, g_qnnEpLibPath.find_last_of(L"/\\"));
    std::wstring oldPath(32768, L'\0');
    DWORD len = GetEnvironmentVariableW(L"PATH", oldPath.data(), (DWORD)oldPath.size());
    oldPath.resize(len);
    std::wstring newPath = epDir + L";" + oldPath;
    SetEnvironmentVariableW(L"PATH", newPath.c_str());
  } else {
    cout << "WinML backend: QNN EP not available." << endl;
  }
}

void NeuralNet::globalCleanup() {
  // Nothing to clean up.
}

//--------------------------------------------------------------

ComputeContext* NeuralNet::createComputeContext(
  const std::vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const string& backendExtraParam,
  const string& homeDataDirOverride,
  bool openCLReTunePerBoardSize,
  enabled_t useFP16Mode,
  enabled_t useNHWCMode,
  const LoadedModel* loadedModel
) {
  (void)gpuIdxs;
  (void)homeDataDirOverride;
  (void)openCLReTunePerBoardSize;
  (void)useFP16Mode;
  (void)useNHWCMode;
  (void)loadedModel;

  // Parse backendExtraParam as "key=value;key=value;..."
  string providerName = "dml";  // Default to DirectML
  map<string, string> params;
  if(!backendExtraParam.empty()) {
    vector<string> parts = Global::split(backendExtraParam, ';');
    for(const string& part : parts) {
      size_t eq = part.find('=');
      if(eq != string::npos) {
        string key = Global::trim(part.substr(0, eq));
        string val = Global::trim(part.substr(eq + 1));
        params[key] = val;
      } else {
        string trimmed = Global::trim(part);
        if(!trimmed.empty())
          providerName = trimmed;
      }
    }
    if(params.count("provider"))
      providerName = params["provider"];
  }
  providerName = Global::toLower(providerName);

  if(logger != NULL)
    logger->write("WinML backend: creating compute context for " +
                   Global::intToString(nnXLen) + "x" + Global::intToString(nnYLen) +
                   " with provider '" + providerName + "'");

  ComputeContext* ctx = new ComputeContext(nnXLen, nnYLen, providerName);

  // Apply configured node names
  if(params.count("inputSpatial")) ctx->inputSpatialName = params["inputSpatial"];
  if(params.count("inputGlobal")) ctx->inputGlobalName = params["inputGlobal"];
  if(params.count("inputMeta")) ctx->inputMetaName = params["inputMeta"];
  if(params.count("outputPolicy")) ctx->outputPolicyName = params["outputPolicy"];
  if(params.count("outputValue")) ctx->outputValueName = params["outputValue"];
  if(params.count("outputMiscvalue")) ctx->outputMiscvalueName = params["outputMiscvalue"];
  if(params.count("outputOwnership")) ctx->outputOwnershipName = params["outputOwnership"];
  if(params.count("openvinoDeviceType")) ctx->openvinoDeviceType = params["openvinoDeviceType"];
  if(params.count("openvinoDeviceId")) ctx->openvinoDeviceId = params["openvinoDeviceId"];
  if(params.count("openvinoEnableNPUFastCompile")) {
    string v = Global::toLower(params["openvinoEnableNPUFastCompile"]);
    ctx->openvinoEnableNPUFastCompile = (v == "1" || v == "true" || v == "yes" || v == "on");
  }
  if(params.count("openvinoCacheDir")) ctx->openvinoCacheDir = params["openvinoCacheDir"];
  if(params.count("modelVersion")) {
    int v = Global::stringToInt(params["modelVersion"]);
    if(v >= 0)
      ctx->configModelVersion = v;
  }

  return ctx;
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

//--------------------------------------------------------------

ComputeHandle* NeuralNet::createComputeHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int gpuIdxForThisThread,
  int serverThreadIdx
) {
  (void)maxBatchSize;
  (void)requireExactNNLen;
  if(inputsUseNHWC)
    throw StringError("WinML backend: inputsUseNHWC = true not supported, must use NCHW");

  if(logger != NULL) {
    logger->write("WinML backend thread " + Global::intToString(serverThreadIdx) +
                  ": createComputeHandle entered");
    logger->write("WinML backend thread " + Global::intToString(serverThreadIdx) +
                  ": Model version " + Global::intToString(loadedModel->modelDesc.modelVersion));
    logger->write("WinML backend thread " + Global::intToString(serverThreadIdx) +
                  ": Model name: " + loadedModel->modelDesc.name);
    string deviceInfo =
      context->providerName == "openvino"
      ? "n/a (use winmlOpenVINODeviceType/winmlOpenVINODeviceId)"
      : Global::intToString(gpuIdxForThisThread);
    logger->write("WinML backend thread " + Global::intToString(serverThreadIdx) +
                  ": provider=" + context->providerName +
                  " deviceIdx=" + deviceInfo);
  }

  return new ComputeHandle(context, *loadedModel, logger, gpuIdxForThisThread);
}

void NeuralNet::freeComputeHandle(ComputeHandle* computeHandle) {
  delete computeHandle;
}

bool NeuralNet::isUsingFP16(const ComputeHandle* handle) {
  (void)handle;
  return false;
}

//--------------------------------------------------------------

static int findNameIndex(const vector<string>& names, const vector<string>& targets) {
  for(size_t i = 0; i < names.size(); i++) {
    for(const auto& t : targets) {
      if(names[i] == t)
        return (int)i;
    }
  }
  return -1;
}

void NeuralNet::getOutput(
  ComputeHandle* computeHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs
) {
  assert(numBatchEltsFilled <= inputBuffers->maxBatchSize);
  assert(numBatchEltsFilled > 0);
  const int batchSize = numBatchEltsFilled;
  const int nnXLen = computeHandle->context->nnXLen;
  const int nnYLen = computeHandle->context->nnYLen;
  const int numSpatialFeatures = computeHandle->numInputChannels;
  const int numGlobalFeatures = computeHandle->numInputGlobalChannels;
  const int numPolicyChannels = computeHandle->numPolicyChannels;

  // Fill input buffers
  for(int nIdx = 0; nIdx < batchSize; nIdx++) {
    float* rowSpatialInput = inputBuffers->spatialInput.data() + (inputBuffers->singleInputElts * nIdx);
    float* rowGlobalInput = inputBuffers->globalInput.data() + (inputBuffers->singleInputGlobalElts * nIdx);

    const float* rowGlobal = inputBufs[nIdx]->rowGlobalBuf.data();
    const float* rowSpatial = inputBufs[nIdx]->rowSpatialBuf.data();
    std::copy(rowGlobal, rowGlobal + numGlobalFeatures, rowGlobalInput);
    SymmetryHelpers::copyInputsWithSymmetry(rowSpatial, rowSpatialInput, 1, nnYLen, nnXLen, numSpatialFeatures, false, inputBufs[nIdx]->symmetry);

    if(computeHandle->numInputMetaChannels > 0) {
      float* rowMetaInput = inputBuffers->metaInput.data() + (inputBuffers->singleInputMetaElts * nIdx);
      const float* rowMeta = inputBufs[nIdx]->rowMetaBuf.data();
      std::copy(rowMeta, rowMeta + computeHandle->numInputMetaChannels, rowMetaInput);
    }
  }

  // Helper lambda to run inference for a contiguous sub-batch starting at `startRow` with `subBatchSize` rows,
  // then write results into `outputs`.
  const ComputeContext* ctx = computeHandle->context;
  const int policyResultLen = computeHandle->policyResultLen;
  const int spatialPolicyLen = nnXLen * nnYLen;

  int spatialIdx = findNameIndex(computeHandle->inputNames, {ctx->inputSpatialName});
  int globalIdx = findNameIndex(computeHandle->inputNames, {ctx->inputGlobalName});
  if(spatialIdx < 0 || globalIdx < 0)
    throw StringError("WinML backend: could not find expected input names");

  int metaIdx = -1;
  if(computeHandle->numInputMetaChannels > 0) {
    metaIdx = findNameIndex(computeHandle->inputNames, {ctx->inputMetaName});
    if(metaIdx < 0)
      throw StringError("WinML backend: model has metadata channels but could not find input_meta");
  }

  int policyOutputIdx = findNameIndex(computeHandle->outputNames, {ctx->outputPolicyName});
  int valueOutputIdx = findNameIndex(computeHandle->outputNames, {ctx->outputValueName});
  int miscvalueOutputIdx = findNameIndex(computeHandle->outputNames, {ctx->outputMiscvalueName});
  int ownershipOutputIdx = findNameIndex(computeHandle->outputNames, {ctx->outputOwnershipName});
  if(policyOutputIdx < 0)
    throw StringError("WinML backend: could not find policy output node '" + ctx->outputPolicyName + "'");
  if(valueOutputIdx < 0)
    throw StringError("WinML backend: could not find value output node '" + ctx->outputValueName + "'");
  if(miscvalueOutputIdx < 0)
    throw StringError("WinML backend: could not find miscvalue output node '" + ctx->outputMiscvalueName + "'");
  if(ownershipOutputIdx < 0)
    throw StringError("WinML backend: could not find ownership output node '" + ctx->outputOwnershipName + "'");

  assert((int)outputs.size() == batchSize);

  // When fixedBatchOne is true (NPU), we must run inference one sample at a time
  // because the session was compiled with a fixed batch dimension of 1.
  const int inferBatchSize = computeHandle->fixedBatchOne ? 1 : batchSize;
  const int numInferCalls = computeHandle->fixedBatchOne ? batchSize : 1;

  float policyProbsTmp[NNPos::MAX_NN_POLICY_SIZE];

  for(int inferIdx = 0; inferIdx < numInferCalls; inferIdx++) {
    const int startRow = inferIdx * inferBatchSize;

    // Create ONNX tensors
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    std::array<int64_t, 4> spatialShape = {inferBatchSize, numSpatialFeatures, nnYLen, nnXLen};
    Ort::Value spatialTensor = Ort::Value::CreateTensor<float>(
      memInfo, inputBuffers->spatialInput.data() + (inputBuffers->singleInputElts * startRow),
      inputBuffers->singleInputElts * inferBatchSize,
      spatialShape.data(), spatialShape.size()
    );

    std::array<int64_t, 2> globalShape = {inferBatchSize, numGlobalFeatures};
    Ort::Value globalTensor = Ort::Value::CreateTensor<float>(
      memInfo, inputBuffers->globalInput.data() + (inputBuffers->singleInputGlobalElts * startRow),
      inputBuffers->singleInputGlobalElts * inferBatchSize,
      globalShape.data(), globalShape.size()
    );

    Ort::Value metaTensor(nullptr);
    if(computeHandle->numInputMetaChannels > 0) {
      std::array<int64_t, 2> metaShape = {inferBatchSize, computeHandle->numInputMetaChannels};
      metaTensor = Ort::Value::CreateTensor<float>(
        memInfo, inputBuffers->metaInput.data() + (inputBuffers->singleInputMetaElts * startRow),
        inputBuffers->singleInputMetaElts * inferBatchSize,
        metaShape.data(), metaShape.size()
      );
    }

    vector<Ort::Value> inputTensors;
    inputTensors.reserve(computeHandle->inputNames.size());
    for(size_t i = 0; i < computeHandle->inputNames.size(); i++) {
      if((int)i == spatialIdx)
        inputTensors.push_back(std::move(spatialTensor));
      else if((int)i == globalIdx)
        inputTensors.push_back(std::move(globalTensor));
      else if((int)i == metaIdx)
        inputTensors.push_back(std::move(metaTensor));
      else {
        throw StringError("WinML backend: unexpected input node '" + computeHandle->inputNames[i] +
                           "' -- only spatial, global, and meta inputs are supported");
      }
    }

    // Run inference
    auto outputTensors = computeHandle->session->Run(
      Ort::RunOptions{nullptr},
      computeHandle->inputNamePtrs.data(),
      inputTensors.data(),
      inputTensors.size(),
      computeHandle->outputNamePtrs.data(),
      computeHandle->outputNamePtrs.size()
    );

    const float* policyData = outputTensors[policyOutputIdx].GetTensorData<float>();
    const float* valueData = outputTensors[valueOutputIdx].GetTensorData<float>();
    const float* miscvalueData = outputTensors[miscvalueOutputIdx].GetTensorData<float>();
    const float* ownershipData = outputTensors[ownershipOutputIdx].GetTensorData<float>();

    assert(policyData != nullptr);
    assert(valueData != nullptr);
    assert(miscvalueData != nullptr);
    assert(ownershipData != nullptr);

    for(int subRow = 0; subRow < inferBatchSize; subRow++) {
      const int row = startRow + subRow;
      NNOutput* output = outputs[row];
      assert(output->nnXLen == nnXLen);
      assert(output->nnYLen == nnYLen);
      float policyOptimism = (float)inputBufs[row]->policyOptimism;

      // Policy: [N, C, H*W+1]
      {
        const float* policyRowBase = policyData + subRow * numPolicyChannels * policyResultLen;
        float* policyProbs = output->policyProbs;

        if(numPolicyChannels >= 2) {
          const float* ch0 = policyRowBase;
          const float* ch1 = policyRowBase + policyResultLen;
          for(int i = 0; i < spatialPolicyLen; i++) {
            float p = ch0[i];
            float pOpt = ch1[i];
            policyProbsTmp[i] = p + (pOpt - p) * policyOptimism;
          }
          SymmetryHelpers::copyOutputsWithSymmetry(policyProbsTmp, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
          policyProbs[spatialPolicyLen] = ch0[spatialPolicyLen] + (ch1[spatialPolicyLen] - ch0[spatialPolicyLen]) * policyOptimism;
        } else {
          assert(numPolicyChannels == 1);
          const float* ch0 = policyRowBase;
          SymmetryHelpers::copyOutputsWithSymmetry(ch0, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
          policyProbs[spatialPolicyLen] = ch0[spatialPolicyLen];
        }
      }

      // Value: [N, 3]
      {
        int numVC = computeHandle->numValueChannels;
        assert(numVC == 3);
        output->whiteWinProb = valueData[subRow * numVC];
        output->whiteLossProb = valueData[subRow * numVC + 1];
        output->whiteNoResultProb = valueData[subRow * numVC + 2];
      }

      // MiscValue
      {
        int numScoreValueChannels = computeHandle->numScoreValueChannels;
        if(computeHandle->modelVersion >= 9) {
          assert(numScoreValueChannels >= 6);
          output->whiteScoreMean = miscvalueData[subRow * numScoreValueChannels];
          output->whiteScoreMeanSq = miscvalueData[subRow * numScoreValueChannels + 1];
          output->whiteLead = miscvalueData[subRow * numScoreValueChannels + 2];
          output->varTimeLeft = miscvalueData[subRow * numScoreValueChannels + 3];
          output->shorttermWinlossError = miscvalueData[subRow * numScoreValueChannels + 4];
          output->shorttermScoreError = miscvalueData[subRow * numScoreValueChannels + 5];
        }
        else if(computeHandle->modelVersion >= 8) {
          assert(numScoreValueChannels >= 4);
          output->whiteScoreMean = miscvalueData[subRow * numScoreValueChannels];
          output->whiteScoreMeanSq = miscvalueData[subRow * numScoreValueChannels + 1];
          output->whiteLead = miscvalueData[subRow * numScoreValueChannels + 2];
          output->varTimeLeft = miscvalueData[subRow * numScoreValueChannels + 3];
          output->shorttermWinlossError = 0;
          output->shorttermScoreError = 0;
        }
        else if(computeHandle->modelVersion >= 4) {
          assert(numScoreValueChannels >= 2);
          output->whiteScoreMean = miscvalueData[subRow * numScoreValueChannels];
          output->whiteScoreMeanSq = miscvalueData[subRow * numScoreValueChannels + 1];
          output->whiteLead = output->whiteScoreMean;
          output->varTimeLeft = 0;
          output->shorttermWinlossError = 0;
          output->shorttermScoreError = 0;
        }
        else if(computeHandle->modelVersion >= 3) {
          assert(numScoreValueChannels >= 1);
          output->whiteScoreMean = miscvalueData[subRow * numScoreValueChannels];
          output->whiteScoreMeanSq = output->whiteScoreMean * output->whiteScoreMean;
          output->whiteLead = output->whiteScoreMean;
          output->varTimeLeft = 0;
          output->shorttermWinlossError = 0;
          output->shorttermScoreError = 0;
        }
        else {
          ASSERT_UNREACHABLE;
        }
      }

      // Ownership: [N, 1, H, W]
      if(output->whiteOwnerMap != NULL) {
        assert(computeHandle->numOwnershipChannels == 1);
        const float* ownershipRowBuf = ownershipData + subRow * nnXLen * nnYLen;
        SymmetryHelpers::copyOutputsWithSymmetry(ownershipRowBuf, output->whiteOwnerMap, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
      }
    }
  }
}

#ifdef WINML_HAS_EP_CATALOG
static BOOL CALLBACK printEpInfoCallback(WinMLEpHandle /*ep*/, const WinMLEpInfo* info, void* /*ctx*/) {
  string stateStr;
  switch(info->readyState) {
    case WinMLEpReadyState_Ready: stateStr = "Ready"; break;
    case WinMLEpReadyState_NotReady: stateStr = "NotReady"; break;
    case WinMLEpReadyState_NotPresent: stateStr = "NotPresent"; break;
    default: stateStr = "Unknown"; break;
  }
  cout << "  " << (info->name ? info->name : "?") << " v" << (info->version ? info->version : "?")
       << " - " << stateStr << endl;
  return TRUE;
}
#endif

void NeuralNet::printDevices() {
  cout << "WinML backend: supported execution providers:" << endl;
  cout << "  cpu           - CPU (always available)" << endl;
  cout << "  dml           - DirectML (GPU, always available on Windows 10+)" << endl;
  cout << "  openvino      - Intel OpenVINO (CPU/GPU/NPU)" << endl;
  cout << "  nvtensorrtrtx - NVIDIA TensorRT RTX (GPU)" << endl;
  cout << "  migraphx      - AMD MIGraphX (GPU)" << endl;
  cout << "  qnn           - Qualcomm QNN (NPU/GPU)" << endl;
  cout << "  vitisai       - AMD Vitis AI (NPU)" << endl;
  cout << endl;
  cout << "Use winmlProvider config key to select an EP." << endl;

#ifdef WINML_HAS_EP_CATALOG
  cout << endl;
  cout << "EP Catalog available. Querying installed providers..." << endl;
  WinMLEpCatalogHandle catalog = nullptr;
  HRESULT hr = WinMLEpCatalogCreate(&catalog);
  if(SUCCEEDED(hr) && catalog) {
    WinMLEpCatalogEnumProviders(catalog, printEpInfoCallback, nullptr);
    WinMLEpCatalogRelease(catalog);
  } else {
    cout << "EP Catalog query failed (hr=0x" << std::hex << hr << std::dec << ")" << endl;
  }
#else
  cout << "EP Catalog not available (built without WINML_HAS_EP_CATALOG)." << endl;
#endif
}

//--------------------------------------------------------------
// FOR TESTING -- all return false (not implemented for this backend)

bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc* desc, int batchSize, int nnXLen, int nnYLen,
  bool useFP16, bool useNHWC, const std::vector<float>& inputBuffer, std::vector<float>& outputBuffer
) {
  (void)desc; (void)batchSize; (void)nnXLen; (void)nnYLen;
  (void)useFP16; (void)useNHWC; (void)inputBuffer; (void)outputBuffer;
  return false;
}

bool NeuralNet::testEvaluateBatchNorm(
  const BatchNormLayerDesc* desc, int batchSize, int nnXLen, int nnYLen,
  bool useFP16, bool useNHWC, const std::vector<float>& inputBuffer,
  const std::vector<float>& maskBuffer, std::vector<float>& outputBuffer
) {
  (void)desc; (void)batchSize; (void)nnXLen; (void)nnYLen;
  (void)useFP16; (void)useNHWC; (void)inputBuffer; (void)maskBuffer; (void)outputBuffer;
  return false;
}

bool NeuralNet::testEvaluateResidualBlock(
  const ResidualBlockDesc* desc, int batchSize, int nnXLen, int nnYLen,
  bool useFP16, bool useNHWC, const std::vector<float>& inputBuffer,
  const std::vector<float>& maskBuffer, std::vector<float>& outputBuffer
) {
  (void)desc; (void)batchSize; (void)nnXLen; (void)nnYLen;
  (void)useFP16; (void)useNHWC; (void)inputBuffer; (void)maskBuffer; (void)outputBuffer;
  return false;
}

bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(
  const GlobalPoolingResidualBlockDesc* desc, int batchSize, int nnXLen, int nnYLen,
  bool useFP16, bool useNHWC, const std::vector<float>& inputBuffer,
  const std::vector<float>& maskBuffer, std::vector<float>& outputBuffer
) {
  (void)desc; (void)batchSize; (void)nnXLen; (void)nnYLen;
  (void)useFP16; (void)useNHWC; (void)inputBuffer; (void)maskBuffer; (void)outputBuffer;
  return false;
}
