#ifdef USE_MGX_BACKEND

#include <migraphx/migraphx.hpp>
#include <unistd.h>
// MIGraphX can parse several serialized model formats such as ONNX, JSON, and
// MsgPack. For now we will use ONNX as the interchange format when converting
// KataGo's raw neural net model to something MIGraphX can consume.

#include "mgxconvert.h"

#include "../core/global.h"
#include "../neuralnet/desc.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nninterface.h"
#include "../neuralnet/nninputs.h"

using namespace std;

void NeuralNet::globalInitialize() {
  // No global initialization required for MIGraphX
}

void NeuralNet::globalCleanup() {
  // No global cleanup required for MIGraphX
}

struct ComputeContext {
  int nnXLen;
  int nnYLen;
};

ComputeContext* NeuralNet::createComputeContext(
  const vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const string& openCLTunerFile,
  const string& homeDataDirOverride,
  bool openCLReTunePerBoardSize,
  enabled_t useFP16Mode,
  enabled_t useNHWCMode,
  const LoadedModel* loadedModel) {
  (void)gpuIdxs;
  (void)logger;
  (void)openCLTunerFile;
  (void)homeDataDirOverride;
  (void)openCLReTunePerBoardSize;
  (void)useFP16Mode;
  (void)useNHWCMode;
  (void)loadedModel;
  ComputeContext* context = new ComputeContext();
  context->nnXLen = nnXLen;
  context->nnYLen = nnYLen;
  return context;
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

// -----------------------------------------------------------------------------
// Model loading and conversion
// -----------------------------------------------------------------------------

struct LoadedModel {
  ModelDesc modelDesc;
  string rawFile;
  LoadedModel(const string& file, const string& expectedSha256) {
    ModelDesc::loadFromFileMaybeGZipped(file, modelDesc, expectedSha256);
    modelDesc.applyScale8ToReduceActivations();
    rawFile = file;
  }
};

LoadedModel* NeuralNet::loadModelFile(const string& file, const string& expectedSha256) {
  return new LoadedModel(file, expectedSha256);
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}

const ModelDesc& NeuralNet::getModelDesc(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc;
}

struct MGXModel {
  migraphx::program program;
  vector<string> paramNames;
};

static void convertRawModelToONNX(const LoadedModel* rawModel, const string& outFile, int nnXLen, int nnYLen) {
  (void)nnXLen;
  (void)nnYLen;

  string modelPath = rawModel->rawFile;
  if(!MGXConvert::convertRawToOnnx(modelPath, outFile))
    throw StringError("Error converting model to ONNX");
}

struct ModelParser {
  unique_ptr<MGXModel> build(const LoadedModel* rawModel, int nnXLen, int nnYLen) {
    string tmpFile = Global::strprintf("/tmp/katago-mgx-%d.onnx", getpid());
    convertRawModelToONNX(rawModel, tmpFile, nnXLen, nnYLen);

    migraphx::onnx_options options;
    options.set_default_dim_value(1);
    options.set_default_loop_iterations(10);
    options.set_limit_loop_iterations(65535);

    migraphx::program program = migraphx::parse_onnx(tmpFile.c_str(), options);
    migraphx::target t("gpu");
    program.compile(t);
    unlink(tmpFile.c_str());

    unique_ptr<MGXModel> model = make_unique<MGXModel>();
    model->program = std::move(program);
    auto paramShapes = model->program.get_parameter_shapes();
    vector<const char*> names = paramShapes.names();
    for(const char* n : names)
      model->paramNames.emplace_back(n);
    return model;
  }
};

struct ComputeHandle {
  unique_ptr<MGXModel> model;
  int nnXLen;
  int nnYLen;
};

ComputeHandle* NeuralNet::createComputeHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int gpuIdxForThisThread,
  int serverThreadIdx) {
  (void)context;
  (void)logger;
  (void)maxBatchSize;
  (void)requireExactNNLen;
  (void)inputsUseNHWC;
  (void)gpuIdxForThisThread;
  (void)serverThreadIdx;
  ModelParser parser;
  auto mgxModel = parser.build(loadedModel, context->nnXLen, context->nnYLen);
  ComputeHandle* handle = new ComputeHandle();
  handle->model = std::move(mgxModel);
  handle->nnXLen = context->nnXLen;
  handle->nnYLen = context->nnYLen;
  return handle;
}

void NeuralNet::freeComputeHandle(ComputeHandle* computeHandle) {
  delete computeHandle;
}

bool NeuralNet::isUsingFP16(const ComputeHandle* computeHandle) {
  (void)computeHandle;
  return false;
}

void NeuralNet::printDevices() {
  // MIGraphX does not currently provide a simple device enumeration API
}

struct InputBuffers {
  vector<float> spatial;
  vector<float> global;
};

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  (void)loadedModel;
  InputBuffers* buffers = new InputBuffers();
  size_t spatialSize = (size_t)maxBatchSize * NNInputs::NUM_FEATURES * nnXLen * nnYLen;
  buffers->spatial.resize(spatialSize);
  buffers->global.resize((size_t)maxBatchSize * NNInputs::NUM_GLOBAL_FEATURES);
  return buffers;
}

void NeuralNet::freeInputBuffers(InputBuffers* buffers) {
  delete buffers;
}

void NeuralNet::getOutput(
  ComputeHandle* computeHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs
) {
  (void)inputBufs;
  migraphx::program_parameters params;
  // Assume first param is spatial input and second is global input
  if(computeHandle->model->paramNames.size() >= 1) {
    migraphx::shape spatialShape(
      migraphx_shape_float_type,
      {(size_t)numBatchEltsFilled,
       (size_t)NNInputs::NUM_FEATURES,
       (size_t)computeHandle->nnXLen,
       (size_t)computeHandle->nnYLen});
    migraphx::argument spatialArg(spatialShape, inputBuffers->spatial.data());
    params.add(computeHandle->model->paramNames[0].c_str(), spatialArg);
  }
  if(computeHandle->model->paramNames.size() >= 2) {
    migraphx::shape globalShape(
      migraphx_shape_float_type,
      {(size_t)numBatchEltsFilled,
       (size_t)NNInputs::NUM_GLOBAL_FEATURES});
    migraphx::argument globalArg(globalShape, inputBuffers->global.data());
    params.add(computeHandle->model->paramNames[1].c_str(), globalArg);
  }

  migraphx::arguments results = computeHandle->model->program.eval(params);

  const float* policyData = nullptr;
  const float* valueData = nullptr;

  if(results.size() >= 1)
    policyData = reinterpret_cast<const float*>(results[0].data());
  if(results.size() >= 2)
    valueData = reinterpret_cast<const float*>(results[1].data());

  for(int b = 0; b < numBatchEltsFilled; b++) {
    NNOutput* out = outputs[b];
    if(policyData != nullptr) {
      const float* src = policyData + b * NNPos::MAX_NN_POLICY_SIZE;
      for(int i = 0; i < NNPos::MAX_NN_POLICY_SIZE; i++)
        out->policyProbs[i] = src[i];
    }
    if(valueData != nullptr) {
      const float* v = valueData + b * 3;
      out->whiteWinProb = v[0];
      out->whiteLossProb = v[1];
      out->whiteNoResultProb = v[2];
    }
  }
}

bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  vector<float>& outputBuffer
) {
  (void)desc;
  (void)batchSize;
  (void)nnXLen;
  (void)nnYLen;
  (void)useFP16;
  (void)useNHWC;
  (void)inputBuffer;
  (void)outputBuffer;
  return false;
}

bool NeuralNet::testEvaluateBatchNorm(
  const BatchNormLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer
) {
  (void)desc;
  (void)batchSize;
  (void)nnXLen;
  (void)nnYLen;
  (void)useFP16;
  (void)useNHWC;
  (void)inputBuffer;
  (void)maskBuffer;
  (void)outputBuffer;
  return false;
}

bool NeuralNet::testEvaluateResidualBlock(
  const ResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer
) {
  (void)desc;
  (void)batchSize;
  (void)nnXLen;
  (void)nnYLen;
  (void)useFP16;
  (void)useNHWC;
  (void)inputBuffer;
  (void)maskBuffer;
  (void)outputBuffer;
  return false;
}

bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(
  const GlobalPoolingResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer
) {
  (void)desc;
  (void)batchSize;
  (void)nnXLen;
  (void)nnYLen;
  (void)useFP16;
  (void)useNHWC;
  (void)inputBuffer;
  (void)maskBuffer;
  (void)outputBuffer;
  return false;
}

#endif // USE_MGX_BACKEND
