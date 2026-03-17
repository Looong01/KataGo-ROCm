#include "../neuralnet/nninterface.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/modelversion.h"

#include <openvino/openvino.hpp>
#include <openvino/opsets/opset16.hpp>
#include <openvino/runtime/properties.hpp>

#include <algorithm>
#include <map>
#include <memory>
#include <vector>

using namespace std;

namespace {
namespace ovop = ov::opset16;
using OVOut = ov::Output<ov::Node>;

static shared_ptr<ovop::Constant> makeF32Const(const ov::Shape& shape, const vector<float>& values) {
  return make_shared<ovop::Constant>(ov::element::f32, shape, values.data());
}

static shared_ptr<ovop::Constant> makeI64Const(const vector<int64_t>& values) {
  return ovop::Constant::create(ov::element::i64, ov::Shape{values.size()}, values);
}

static shared_ptr<ovop::Constant> makeI64Scalar(int64_t value) {
  return ovop::Constant::create(ov::element::i64, ov::Shape{}, {value});
}

static shared_ptr<ovop::Constant> makeF32Scalar(float value) {
  return ovop::Constant::create(ov::element::f32, ov::Shape{}, {value});
}

static OVOut addConvNode(
  const OVOut& input,
  const ConvLayerDesc& desc,
  const string& prefix
) {
  const ov::Shape wShape = {
    (size_t)desc.outChannels,
    (size_t)desc.inChannels,
    (size_t)desc.convYSize,
    (size_t)desc.convXSize
  };
  auto weights = makeF32Const(wShape, desc.weights);
  weights->set_friendly_name(prefix + "/w");

  const ov::Strides strides = {1, 1};
  const ov::CoordinateDiff padsBegin = {desc.convYSize / 2, desc.convXSize / 2};
  const ov::CoordinateDiff padsEnd = {desc.convYSize / 2, desc.convXSize / 2};
  const ov::Strides dilations = {(size_t)desc.dilationY, (size_t)desc.dilationX};

  auto conv = make_shared<ovop::Convolution>(input, weights, strides, padsBegin, padsEnd, dilations);
  conv->set_friendly_name(prefix + "/conv");
  return conv;
}

static OVOut addMergedBNNode(
  const OVOut& input,
  const BatchNormLayerDesc& desc,
  const string& prefix
) {
  const int c = desc.numChannels;
  auto scale = makeF32Const({(size_t)c, 1, 1}, desc.mergedScale);
  scale->set_friendly_name(prefix + "/scale");
  auto bias = makeF32Const({(size_t)c, 1, 1}, desc.mergedBias);
  bias->set_friendly_name(prefix + "/bias");

  auto scaled = make_shared<ovop::Multiply>(input, scale);
  scaled->set_friendly_name(prefix + "/scaled");
  auto out = make_shared<ovop::Add>(scaled, bias);
  out->set_friendly_name(prefix + "/bn_out");
  return out;
}

static OVOut addActivationNode(
  const OVOut& input,
  int activation,
  const string& prefix
) {
  if(activation == ACTIVATION_RELU) {
    auto relu = make_shared<ovop::Relu>(input);
    relu->set_friendly_name(prefix + "/relu");
    return relu;
  }

  if(activation == ACTIVATION_MISH) {
    auto softplus = make_shared<ovop::SoftPlus>(input);
    softplus->set_friendly_name(prefix + "/softplus");

    auto tanhNode = make_shared<ovop::Tanh>(softplus);
    tanhNode->set_friendly_name(prefix + "/tanh");
    auto mish = make_shared<ovop::Multiply>(input, tanhNode);
    mish->set_friendly_name(prefix + "/mish");
    return mish;
  }

  if(activation == ACTIVATION_MISH_SCALE8) {
    auto c8 = makeF32Scalar(8.0f);
    c8->set_friendly_name(prefix + "/c8");
    auto xScaled = make_shared<ovop::Multiply>(input, c8);
    xScaled->set_friendly_name(prefix + "/x8");

    auto softplus = make_shared<ovop::SoftPlus>(xScaled);
    softplus->set_friendly_name(prefix + "/softplus8");

    auto tanhNode = make_shared<ovop::Tanh>(softplus);
    tanhNode->set_friendly_name(prefix + "/tanh");
    auto mish = make_shared<ovop::Multiply>(input, tanhNode);
    mish->set_friendly_name(prefix + "/mish_scale8");
    return mish;
  }

  // ACTIVATION_IDENTITY
  return input;
}

static OVOut addBNActivationMask(
  const OVOut& input,
  const BatchNormLayerDesc& bnDesc,
  const ActivationLayerDesc& actDesc,
  const OVOut& mask,
  const string& prefix
) {
  OVOut bn = addMergedBNNode(input, bnDesc, prefix + "/bn");
  OVOut act = addActivationNode(bn, actDesc.activation, prefix + "/act");
  auto masked = make_shared<ovop::Multiply>(act, mask);
  masked->set_friendly_name(prefix + "/masked");
  return masked;
}

static OVOut addMatMulNode(
  const OVOut& input,
  const MatMulLayerDesc& desc,
  const string& prefix
) {
  auto weights = makeF32Const({(size_t)desc.inChannels, (size_t)desc.outChannels}, desc.weights);
  weights->set_friendly_name(prefix + "/w");
  auto matmul = make_shared<ovop::MatMul>(input, weights, false, false);
  matmul->set_friendly_name(prefix + "/matmul");
  return matmul;
}

static OVOut addBiasNode(
  const OVOut& input,
  const MatBiasLayerDesc& desc,
  const string& prefix
) {
  auto bias = makeF32Const({(size_t)desc.numChannels}, desc.weights);
  bias->set_friendly_name(prefix + "/b");
  auto out = make_shared<ovop::Add>(input, bias);
  out->set_friendly_name(prefix + "/biased");
  return out;
}

static OVOut addGlobalPool(
  const OVOut& input,
  const OVOut& mask,
  const OVOut& maskSumHW,
  const string& prefix
) {
  auto xMasked = make_shared<ovop::Multiply>(input, mask);
  xMasked->set_friendly_name(prefix + "/xm");

  auto axes23 = makeI64Const({2, 3});
  axes23->set_friendly_name(prefix + "/axes23");
  auto sumOut = make_shared<ovop::ReduceSum>(xMasked, axes23, false);
  sumOut->set_friendly_name(prefix + "/sum");

  auto reshapeShape = makeI64Const({0, 1});
  reshapeShape->set_friendly_name(prefix + "/shape_n1");
  auto maskSumFlat = make_shared<ovop::Reshape>(maskSumHW, reshapeShape, true);
  maskSumFlat->set_friendly_name(prefix + "/msf");

  auto mean = make_shared<ovop::Divide>(sumOut, maskSumFlat);
  mean->set_friendly_name(prefix + "/mean");

  auto sqrtMs = make_shared<ovop::Sqrt>(maskSumFlat);
  sqrtMs->set_friendly_name(prefix + "/sqrt");

  auto c14 = makeF32Scalar(14.0f);
  c14->set_friendly_name(prefix + "/c14");
  auto sqrtMsSub = make_shared<ovop::Subtract>(sqrtMs, c14);
  sqrtMsSub->set_friendly_name(prefix + "/sqrtsub");

  auto c01 = makeF32Scalar(0.1f);
  c01->set_friendly_name(prefix + "/c01");
  auto scaledSqrt = make_shared<ovop::Multiply>(sqrtMsSub, c01);
  scaledSqrt->set_friendly_name(prefix + "/ssm");

  auto pool2 = make_shared<ovop::Multiply>(mean, scaledSqrt);
  pool2->set_friendly_name(prefix + "/p2");

  auto cNeg1 = makeF32Scalar(-1.0f);
  cNeg1->set_friendly_name(prefix + "/cn1");
  auto maskBias = make_shared<ovop::Add>(mask, cNeg1);
  maskBias->set_friendly_name(prefix + "/mb");

  auto xShifted = make_shared<ovop::Add>(input, maskBias);
  xShifted->set_friendly_name(prefix + "/xs");

  auto axes23b = makeI64Const({2, 3});
  axes23b->set_friendly_name(prefix + "/axes23b");
  auto pool3 = make_shared<ovop::ReduceMax>(xShifted, axes23b, false);
  pool3->set_friendly_name(prefix + "/p3");

  auto concat = make_shared<ovop::Concat>(ov::OutputVector{mean, pool2, pool3}, 1);
  concat->set_friendly_name(prefix + "/out");
  return concat;
}

static OVOut addValueHeadGPool(
  const OVOut& input,
  const OVOut& mask,
  const OVOut& maskSumHW,
  const string& prefix
) {
  auto xMasked = make_shared<ovop::Multiply>(input, mask);
  xMasked->set_friendly_name(prefix + "/xm");

  auto axes23 = makeI64Const({2, 3});
  axes23->set_friendly_name(prefix + "/axes23");
  auto sumOut = make_shared<ovop::ReduceSum>(xMasked, axes23, false);
  sumOut->set_friendly_name(prefix + "/sum");

  auto reshapeShape = makeI64Const({0, 1});
  reshapeShape->set_friendly_name(prefix + "/shape_n1");
  auto maskSumFlat = make_shared<ovop::Reshape>(maskSumHW, reshapeShape, true);
  maskSumFlat->set_friendly_name(prefix + "/msf");

  auto mean = make_shared<ovop::Divide>(sumOut, maskSumFlat);
  mean->set_friendly_name(prefix + "/mean");

  auto sqrtMs = make_shared<ovop::Sqrt>(maskSumFlat);
  sqrtMs->set_friendly_name(prefix + "/sqrt");

  auto c14 = makeF32Scalar(14.0f);
  c14->set_friendly_name(prefix + "/c14");
  auto sqrtMsSub = make_shared<ovop::Subtract>(sqrtMs, c14);
  sqrtMsSub->set_friendly_name(prefix + "/ss");

  auto c01 = makeF32Scalar(0.1f);
  c01->set_friendly_name(prefix + "/c01");
  auto scaledSqrt = make_shared<ovop::Multiply>(sqrtMsSub, c01);
  scaledSqrt->set_friendly_name(prefix + "/ssm");

  auto pool2 = make_shared<ovop::Multiply>(mean, scaledSqrt);
  pool2->set_friendly_name(prefix + "/p2");

  auto sqrtMsSubSq = make_shared<ovop::Multiply>(sqrtMsSub, sqrtMsSub);
  sqrtMsSubSq->set_friendly_name(prefix + "/sq");

  auto cp01 = makeF32Scalar(0.01f);
  cp01->set_friendly_name(prefix + "/cp01");
  auto sqScaled = make_shared<ovop::Multiply>(sqrtMsSubSq, cp01);
  sqScaled->set_friendly_name(prefix + "/sqs");

  auto cn01 = makeF32Scalar(-0.1f);
  cn01->set_friendly_name(prefix + "/cn01");
  auto sqShifted = make_shared<ovop::Add>(sqScaled, cn01);
  sqShifted->set_friendly_name(prefix + "/sqsh");

  auto pool3 = make_shared<ovop::Multiply>(mean, sqShifted);
  pool3->set_friendly_name(prefix + "/p3");

  auto concat = make_shared<ovop::Concat>(ov::OutputVector{mean, pool2, pool3}, 1);
  concat->set_friendly_name(prefix + "/out");
  return concat;
}

static OVOut addResidualBlock(
  const OVOut& input,
  const OVOut& mask,
  const ResidualBlockDesc& desc,
  const string& prefix
);

static OVOut addGPoolResidualBlock(
  const OVOut& input,
  const OVOut& mask,
  const OVOut& maskSumHW,
  const GlobalPoolingResidualBlockDesc& desc,
  const string& prefix
);

static OVOut addNestedBottleneckResidualBlock(
  const OVOut& input,
  const OVOut& mask,
  const OVOut& maskSumHW,
  const NestedBottleneckResidualBlockDesc& desc,
  const string& prefix
);

static OVOut addResidualBlock(
  const OVOut& input,
  const OVOut& mask,
  const ResidualBlockDesc& desc,
  const string& prefix
) {
  OVOut pre = addBNActivationMask(input, desc.preBN, desc.preActivation, mask, prefix + "/pre");
  OVOut mid = addConvNode(pre, desc.regularConv, prefix + "/conv1");
  OVOut midAct = addBNActivationMask(mid, desc.midBN, desc.midActivation, mask, prefix + "/mid");
  OVOut finalOut = addConvNode(midAct, desc.finalConv, prefix + "/conv2");
  auto res = make_shared<ovop::Add>(input, finalOut);
  res->set_friendly_name(prefix + "/resadd");
  return res;
}

static OVOut addGPoolResidualBlock(
  const OVOut& input,
  const OVOut& mask,
  const OVOut& maskSumHW,
  const GlobalPoolingResidualBlockDesc& desc,
  const string& prefix
) {
  OVOut pre = addBNActivationMask(input, desc.preBN, desc.preActivation, mask, prefix + "/pre");

  OVOut regOut = addConvNode(pre, desc.regularConv, prefix + "/reg");

  OVOut gpoolConv = addConvNode(pre, desc.gpoolConv, prefix + "/gconv");
  OVOut gpoolBNAct = addBNActivationMask(gpoolConv, desc.gpoolBN, desc.gpoolActivation, mask, prefix + "/gbn");
  OVOut gpool = addGlobalPool(gpoolBNAct, mask, maskSumHW, prefix + "/gpool");

  OVOut gpoolBias = addMatMulNode(gpool, desc.gpoolToBiasMul, prefix + "/g2b");
  auto biasShape = makeI64Const({0, -1, 1, 1});
  biasShape->set_friendly_name(prefix + "/shape_nc11");
  auto gpoolBiasReshaped = make_shared<ovop::Reshape>(gpoolBias, biasShape, true);
  gpoolBiasReshaped->set_friendly_name(prefix + "/gbr");

  auto regPlusBias = make_shared<ovop::Add>(regOut, gpoolBiasReshaped);
  regPlusBias->set_friendly_name(prefix + "/rpb");

  OVOut midAct = addBNActivationMask(regPlusBias, desc.midBN, desc.midActivation, mask, prefix + "/mid");
  OVOut finalOut = addConvNode(midAct, desc.finalConv, prefix + "/conv2");

  auto res = make_shared<ovop::Add>(input, finalOut);
  res->set_friendly_name(prefix + "/resadd");
  return res;
}

static OVOut addNestedBottleneckResidualBlock(
  const OVOut& input,
  const OVOut& mask,
  const OVOut& maskSumHW,
  const NestedBottleneckResidualBlockDesc& desc,
  const string& prefix
) {
  OVOut pre = addBNActivationMask(input, desc.preBN, desc.preActivation, mask, prefix + "/pre");
  OVOut midOut = addConvNode(pre, desc.preConv, prefix + "/preconv");

  for(int i = 0; i < desc.numBlocks; i++) {
    const int kind = desc.blocks[i].first;
    const string sub = prefix + "/sub" + to_string(i);
    if(kind == ORDINARY_BLOCK_KIND) {
      midOut = addResidualBlock(
        midOut,
        mask,
        *((const ResidualBlockDesc*)desc.blocks[i].second.get()),
        sub
      );
    }
    else if(kind == GLOBAL_POOLING_BLOCK_KIND) {
      midOut = addGPoolResidualBlock(
        midOut,
        mask,
        maskSumHW,
        *((const GlobalPoolingResidualBlockDesc*)desc.blocks[i].second.get()),
        sub
      );
    }
    else if(kind == NESTED_BOTTLENECK_BLOCK_KIND) {
      midOut = addNestedBottleneckResidualBlock(
        midOut,
        mask,
        maskSumHW,
        *((const NestedBottleneckResidualBlockDesc*)desc.blocks[i].second.get()),
        sub
      );
    }
    else {
      throw StringError("OpenVINO backend: unknown sub-block kind " + to_string(kind));
    }
  }

  OVOut post = addBNActivationMask(midOut, desc.postBN, desc.postActivation, mask, prefix + "/post");
  OVOut postOut = addConvNode(post, desc.postConv, prefix + "/postconv");

  auto res = make_shared<ovop::Add>(input, postOut);
  res->set_friendly_name(prefix + "/resadd");
  return res;
}

static shared_ptr<ov::Model> buildOpenVinoModel(const ModelDesc& modelDesc, int nnXLen, int nnYLen, int maxBatchSize) {
  const int modelVersion = modelDesc.modelVersion;
  const int numInputChannels = modelDesc.numInputChannels;
  const int numInputGlobalChannels = modelDesc.numInputGlobalChannels;
  const int numPolicyChannels = modelDesc.numPolicyChannels;

  const TrunkDesc& trunk = modelDesc.trunk;
  const PolicyHeadDesc& policyHead = modelDesc.policyHead;
  const ValueHeadDesc& valueHead = modelDesc.valueHead;

  auto inputSpatial = make_shared<ovop::Parameter>(
    ov::element::f32,
    ov::PartialShape{maxBatchSize, numInputChannels, nnYLen, nnXLen}
  );
  inputSpatial->set_friendly_name("input_spatial");
  inputSpatial->output(0).get_tensor().set_names({"input_spatial"});

  auto inputGlobal = make_shared<ovop::Parameter>(
    ov::element::f32,
    ov::PartialShape{maxBatchSize, numInputGlobalChannels}
  );
  inputGlobal->set_friendly_name("input_global");
  inputGlobal->output(0).get_tensor().set_names({"input_global"});

  shared_ptr<ovop::Parameter> inputMeta;
  if(modelDesc.numInputMetaChannels > 0) {
    inputMeta = make_shared<ovop::Parameter>(
      ov::element::f32,
      ov::PartialShape{maxBatchSize, modelDesc.numInputMetaChannels}
    );
    inputMeta->set_friendly_name("input_meta");
    inputMeta->output(0).get_tensor().set_names({"input_meta"});
  }

  auto maskIndices = makeI64Const({0});
  maskIndices->set_friendly_name("mask_indices");
  auto axis1 = makeI64Scalar(1);
  axis1->set_friendly_name("axis1");
  auto mask = make_shared<ovop::Gather>(inputSpatial, maskIndices, axis1);
  mask->set_friendly_name("mask");

  auto sumAxes = makeI64Const({2, 3});
  sumAxes->set_friendly_name("mask_sum_axes");
  auto maskSumHW = make_shared<ovop::ReduceSum>(mask, sumAxes, true);
  maskSumHW->set_friendly_name("maskSumHW");

  OVOut trunkOut = addConvNode(inputSpatial, trunk.initialConv, "trunk/init_conv");

  OVOut globalBias = addMatMulNode(inputGlobal, trunk.initialMatMul, "trunk/init_matmul");
  auto trunkBiasShape = makeI64Const({0, -1, 1, 1});
  trunkBiasShape->set_friendly_name("trunk_bias_shape");
  auto globalBiasReshaped = make_shared<ovop::Reshape>(globalBias, trunkBiasShape, true);
  globalBiasReshaped->set_friendly_name("trunk/gbr");

  auto trunkCombined = make_shared<ovop::Add>(trunkOut, globalBiasReshaped);
  trunkCombined->set_friendly_name("trunk/combined");
  trunkOut = trunkCombined;

  if(trunk.metaEncoderVersion > 0) {
    if(!inputMeta)
      throw StringError("OpenVINO backend: model requires metadata encoder but input_meta is missing");

    const SGFMetadataEncoderDesc& enc = trunk.sgfMetadataEncoder;
    OVOut metaOut = addMatMulNode(inputMeta, enc.mul1, "trunk/meta_mul1");
    metaOut = addBiasNode(metaOut, enc.bias1, "trunk/meta_b1");
    metaOut = addActivationNode(metaOut, enc.act1.activation, "trunk/meta_a1");
    metaOut = addMatMulNode(metaOut, enc.mul2, "trunk/meta_mul2");
    metaOut = addBiasNode(metaOut, enc.bias2, "trunk/meta_b2");
    metaOut = addActivationNode(metaOut, enc.act2.activation, "trunk/meta_a2");
    metaOut = addMatMulNode(metaOut, enc.mul3, "trunk/meta_mul3");

    auto metaBiasShape = makeI64Const({0, -1, 1, 1});
    metaBiasShape->set_friendly_name("trunk_meta_bias_shape");
    auto metaBiasReshaped = make_shared<ovop::Reshape>(metaOut, metaBiasShape, true);
    metaBiasReshaped->set_friendly_name("trunk/mbr");

    auto trunkWithMeta = make_shared<ovop::Add>(trunkOut, metaBiasReshaped);
    trunkWithMeta->set_friendly_name("trunk/with_meta");
    trunkOut = trunkWithMeta;
  }

  for(int i = 0; i < trunk.numBlocks; i++) {
    const int blockKind = trunk.blocks[i].first;
    const string blockPrefix = "trunk/block" + to_string(i);

    if(blockKind == ORDINARY_BLOCK_KIND) {
      const ResidualBlockDesc& blockDesc = *((const ResidualBlockDesc*)trunk.blocks[i].second.get());
      trunkOut = addResidualBlock(trunkOut, mask, blockDesc, blockPrefix);
    }
    else if(blockKind == GLOBAL_POOLING_BLOCK_KIND) {
      const GlobalPoolingResidualBlockDesc& blockDesc = *((const GlobalPoolingResidualBlockDesc*)trunk.blocks[i].second.get());
      trunkOut = addGPoolResidualBlock(trunkOut, mask, maskSumHW, blockDesc, blockPrefix);
    }
    else if(blockKind == NESTED_BOTTLENECK_BLOCK_KIND) {
      const NestedBottleneckResidualBlockDesc& blockDesc = *((const NestedBottleneckResidualBlockDesc*)trunk.blocks[i].second.get());
      trunkOut = addNestedBottleneckResidualBlock(trunkOut, mask, maskSumHW, blockDesc, blockPrefix);
    }
    else {
      throw StringError("OpenVINO backend: unknown block kind " + to_string(blockKind));
    }
  }

  trunkOut = addBNActivationMask(trunkOut, trunk.trunkTipBN, trunk.trunkTipActivation, mask, "trunk/tip");

  OVOut p1Out = addConvNode(trunkOut, policyHead.p1Conv, "policy/p1conv");

  OVOut g1Out = addConvNode(trunkOut, policyHead.g1Conv, "policy/g1conv");
  OVOut g1BNAct = addBNActivationMask(g1Out, policyHead.g1BN, policyHead.g1Activation, mask, "policy/g1bn");
  OVOut g1Pool = addGlobalPool(g1BNAct, mask, maskSumHW, "policy/g1pool");

  OVOut policyBias = addMatMulNode(g1Pool, policyHead.gpoolToBiasMul, "policy/g2b");
  auto pBiasShape = makeI64Const({0, -1, 1, 1});
  pBiasShape->set_friendly_name("policy/bias_shape");
  auto policyBiasReshaped = make_shared<ovop::Reshape>(policyBias, pBiasShape, true);
  policyBiasReshaped->set_friendly_name("policy/pbr");

  auto p1PlusBias = make_shared<ovop::Add>(p1Out, policyBiasReshaped);
  p1PlusBias->set_friendly_name("policy/p1pb");

  OVOut p1BNAct = addBNActivationMask(p1PlusBias, policyHead.p1BN, policyHead.p1Activation, mask, "policy/p1bn");
  OVOut p2Out = addConvNode(p1BNAct, policyHead.p2Conv, "policy/p2conv");

  auto pSpatialShape = makeI64Const({0, numPolicyChannels, -1});
  pSpatialShape->set_friendly_name("policy/spat_shape");
  auto policySpatial = make_shared<ovop::Reshape>(p2Out, pSpatialShape, true);
  policySpatial->set_friendly_name("policy/spatial");

  OVOut passOut;
  if(modelVersion >= 15) {
    OVOut passMul1 = addMatMulNode(g1Pool, policyHead.gpoolToPassMul, "policy/pass_mul1");
    OVOut passBiased = addBiasNode(passMul1, policyHead.gpoolToPassBias, "policy/pass_bias");
    OVOut passAct = addActivationNode(passBiased, policyHead.passActivation.activation, "policy/pass_act");
    passOut = addMatMulNode(passAct, policyHead.gpoolToPassMul2, "policy/pass_mul2");
  }
  else {
    passOut = addMatMulNode(g1Pool, policyHead.gpoolToPassMul, "policy/pass_mul");
  }

  auto passShape = makeI64Const({0, numPolicyChannels, 1});
  passShape->set_friendly_name("policy/pass_shape");
  auto passReshaped = make_shared<ovop::Reshape>(passOut, passShape, true);
  passReshaped->set_friendly_name("policy/pass_r");

  auto outPolicy = make_shared<ovop::Concat>(ov::OutputVector{policySpatial, passReshaped}, 2);
  outPolicy->set_friendly_name("out_policy_node");

  OVOut v1Out = addConvNode(trunkOut, valueHead.v1Conv, "value/v1conv");
  OVOut v1BNAct = addBNActivationMask(v1Out, valueHead.v1BN, valueHead.v1Activation, mask, "value/v1bn");

  OVOut v1Pool = addValueHeadGPool(v1BNAct, mask, maskSumHW, "value/vpool");

  OVOut v2Out = addMatMulNode(v1Pool, valueHead.v2Mul, "value/v2mul");
  OVOut v2Biased = addBiasNode(v2Out, valueHead.v2Bias, "value/v2bias");
  OVOut v2Act = addActivationNode(v2Biased, valueHead.v2Activation.activation, "value/v2act");

  OVOut v3Out = addMatMulNode(v2Act, valueHead.v3Mul, "value/v3mul");
  OVOut outValue = addBiasNode(v3Out, valueHead.v3Bias, "value/v3bias");

  OVOut sv3Out = addMatMulNode(v2Act, valueHead.sv3Mul, "value/sv3mul");
  OVOut outMiscvalue = addBiasNode(sv3Out, valueHead.sv3Bias, "value/sv3bias");

  OVOut outOwnership = addConvNode(v1BNAct, valueHead.vOwnershipConv, "value/own_conv");

  auto resultPolicy = make_shared<ovop::Result>(outPolicy);
  resultPolicy->set_friendly_name("out_policy_result");
  resultPolicy->output(0).get_tensor().set_names({"out_policy"});

  auto resultValue = make_shared<ovop::Result>(outValue);
  resultValue->set_friendly_name("out_value_result");
  resultValue->output(0).get_tensor().set_names({"out_value"});

  auto resultMiscvalue = make_shared<ovop::Result>(outMiscvalue);
  resultMiscvalue->set_friendly_name("out_miscvalue_result");
  resultMiscvalue->output(0).get_tensor().set_names({"out_miscvalue"});

  auto resultOwnership = make_shared<ovop::Result>(outOwnership);
  resultOwnership->set_friendly_name("out_ownership_result");
  resultOwnership->output(0).get_tensor().set_names({"out_ownership"});

  ov::ParameterVector params{inputSpatial, inputGlobal};
  if(inputMeta)
    params.push_back(inputMeta);

  auto model = make_shared<ov::Model>(
    ov::ResultVector{resultPolicy, resultValue, resultMiscvalue, resultOwnership},
    params,
    modelDesc.name
  );
  model->validate_nodes_and_infer_types();
  return model;
}

static string buildDeviceName(const string& type, const string& id) {
  if(id.empty())
    return type;
  if(type.find(':') != string::npos)
    return type;
  if(type.find('.') != string::npos)
    return type;
  return type + "." + id;
}

static map<string, string> parseBackendExtraParam(const string& backendExtraParam) {
  map<string, string> params;
  if(backendExtraParam.empty())
    return params;

  vector<string> parts = Global::split(backendExtraParam, ';');
  for(const string& part : parts) {
    size_t eq = part.find('=');
    if(eq == string::npos)
      continue;
    string key = Global::trim(part.substr(0, eq));
    string val = Global::trim(part.substr(eq + 1));
    if(!key.empty())
      params[key] = val;
  }
  return params;
}

static bool parseBool(const string& s) {
  string v = Global::toLower(Global::trim(s));
  return v == "1" || v == "true" || v == "yes" || v == "on";
}

static bool parsePerformanceMode(const string& s, ov::hint::PerformanceMode& mode) {
  string v = Global::toUpper(Global::trim(s));
  if(v == "LATENCY") {
    mode = ov::hint::PerformanceMode::LATENCY;
    return true;
  }
  if(v == "THROUGHPUT") {
    mode = ov::hint::PerformanceMode::THROUGHPUT;
    return true;
  }
  if(v == "CUMULATIVE_THROUGHPUT") {
    mode = ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT;
    return true;
  }
  return false;
}

static bool deviceTypeSupportsExplicitGpuSelection(const string& deviceType) {
  const string v = Global::toUpper(Global::trim(deviceType));
  return v.find("GPU") != string::npos;
}

} // namespace

struct LoadedModel {
  ModelDesc modelDesc;

  LoadedModel(const string& fileName, const string& expectedSha256) {
    ModelDesc::loadFromFileMaybeGZipped(fileName, modelDesc, expectedSha256);
    modelDesc.applyScale8ToReduceActivations();
  }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

struct ComputeContext {
  ov::Core core;
  int nnXLen;
  int nnYLen;
  enabled_t useFP16Mode;

  string deviceType;
  string deviceId;
  bool enableNPUFastCompile;
  string cacheDir;
  int numStreams;
  bool hasPerformanceMode;
  ov::hint::PerformanceMode performanceMode;

  vector<string> availableDevices;

  ComputeContext(int xLen, int yLen)
    : core(),
      nnXLen(xLen),
      nnYLen(yLen),
      useFP16Mode(enabled_t::Auto),
      deviceType("NPU"),
      deviceId(""),
      enableNPUFastCompile(false),
      cacheDir(""),
      numStreams(0),
      hasPerformanceMode(false),
      performanceMode(ov::hint::PerformanceMode::LATENCY),
      availableDevices() {}

  string resolveDeviceName(int deviceIdxForThread) const {
    if(deviceIdxForThread >= 0) {
      if((size_t)deviceIdxForThread >= availableDevices.size()) {
        throw StringError(
          "OpenVINO backend: device index " + Global::intToString(deviceIdxForThread) +
          " out of range, available devices=" + Global::uint64ToString(availableDevices.size())
        );
      }
      return availableDevices[deviceIdxForThread];
    }
    return buildDeviceName(deviceType, deviceId);
  }
};

struct ComputeHandle {
  ComputeContext* context;
  std::shared_ptr<ov::Model> model;
  ov::CompiledModel compiledModel;
  ov::InferRequest inferRequest;

  int modelVersion;
  int numInputChannels;
  int numInputGlobalChannels;
  int numInputMetaChannels;

  int numPolicyChannels;
  int numValueChannels;
  int numScoreValueChannels;
  int numOwnershipChannels;
  int policyResultLen;
  int fixedBatchSize;

  bool usingFP16;
  string deviceName;

  ComputeHandle(ComputeContext* ctx, const LoadedModel& loadedModel, Logger* logger, int deviceIdxForThread, int maxBatchSize)
    : context(ctx),
      model(),
      compiledModel(),
      inferRequest(),
      modelVersion(loadedModel.modelDesc.modelVersion),
      numInputChannels(loadedModel.modelDesc.numInputChannels),
      numInputGlobalChannels(loadedModel.modelDesc.numInputGlobalChannels),
      numInputMetaChannels(loadedModel.modelDesc.numInputMetaChannels),
      numPolicyChannels(loadedModel.modelDesc.numPolicyChannels),
      numValueChannels(loadedModel.modelDesc.numValueChannels),
      numScoreValueChannels(loadedModel.modelDesc.numScoreValueChannels),
      numOwnershipChannels(loadedModel.modelDesc.numOwnershipChannels),
      policyResultLen(ctx->nnXLen * ctx->nnYLen + 1),
      fixedBatchSize(maxBatchSize),
      usingFP16(false),
      deviceName(ctx->resolveDeviceName(deviceIdxForThread))
  {
    if(logger != NULL) {
      logger->write("OpenVINO backend: building model graph from .bin/.bin.gz weights...");
    }
    model = buildOpenVinoModel(loadedModel.modelDesc, ctx->nnXLen, ctx->nnYLen, maxBatchSize);

    ov::AnyMap config;
    if(!ctx->cacheDir.empty())
      config.emplace(ov::cache_dir.name(), ctx->cacheDir);

    if(ctx->numStreams > 0)
      config.emplace(ov::num_streams.name(), ov::streams::Num(ctx->numStreams));

    if(ctx->hasPerformanceMode)
      config.emplace(ov::hint::performance_mode.name(), ctx->performanceMode);

    if(ctx->enableNPUFastCompile) {
      // Plugin-specific key. If unsupported, compile_model will throw and be reported.
      config.emplace("NPU_COMPILATION_MODE_PARAMS", string("enable-fast-compilation=true"));
    }

    if(ctx->useFP16Mode == enabled_t::True) {
      usingFP16 = true;
      config.emplace(ov::hint::inference_precision.name(), ov::element::f16);
    }

    if(logger != NULL) {
      logger->write("OpenVINO backend: compiling model for device '" + deviceName + "'...");
    }
    compiledModel = ctx->core.compile_model(model, deviceName, config);
    inferRequest = compiledModel.create_infer_request();

    if(logger != NULL) {
      logger->write("OpenVINO backend: compile complete");
    }
  }

  ComputeHandle() = delete;
  ComputeHandle(const ComputeHandle&) = delete;
  ComputeHandle& operator=(const ComputeHandle&) = delete;
};

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

  InputBuffers() = delete;
  InputBuffers(const InputBuffers&) = delete;
  InputBuffers& operator=(const InputBuffers&) = delete;
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

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  return new InputBuffers(loadedModel, maxBatchSize, nnXLen, nnYLen);
}

void NeuralNet::freeInputBuffers(InputBuffers* inputBuffers) {
  delete inputBuffers;
}

void NeuralNet::globalInitialize() {
}

void NeuralNet::globalCleanup() {
}

ComputeContext* NeuralNet::createComputeContext(
  const vector<int>& gpuIdxs,
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
  (void)loadedModel;

  if(useNHWCMode == enabled_t::True) {
    throw StringError("OpenVINO backend: useNHWC = false required, other configurations not supported");
  }

  ComputeContext* ctx = new ComputeContext(nnXLen, nnYLen);
  ctx->useFP16Mode = useFP16Mode;

  map<string, string> params = parseBackendExtraParam(backendExtraParam);
  if(params.count("deviceType")) ctx->deviceType = params["deviceType"];
  if(params.count("device")) ctx->deviceType = params["device"];
  if(params.count("openvinoDeviceType")) ctx->deviceType = params["openvinoDeviceType"];

  if(params.count("deviceId")) ctx->deviceId = params["deviceId"];
  if(params.count("openvinoDeviceId")) ctx->deviceId = params["openvinoDeviceId"];

  if(params.count("cacheDir")) ctx->cacheDir = params["cacheDir"];
  if(params.count("openvinoCacheDir")) ctx->cacheDir = params["openvinoCacheDir"];

  if(params.count("openvinoEnableNPUFastCompile"))
    ctx->enableNPUFastCompile = parseBool(params["openvinoEnableNPUFastCompile"]);

  if(params.count("numStreams")) {
    int n = Global::stringToInt(params["numStreams"]);
    if(n > 0)
      ctx->numStreams = n;
  }

  if(params.count("performanceMode")) {
    ov::hint::PerformanceMode mode;
    if(parsePerformanceMode(params["performanceMode"], mode)) {
      ctx->hasPerformanceMode = true;
      ctx->performanceMode = mode;
    }
    else {
      throw StringError(
        "OpenVINO backend: invalid performanceMode '" + params["performanceMode"] +
        "', expected LATENCY, THROUGHPUT, or CUMULATIVE_THROUGHPUT"
      );
    }
  }

  if(
    !ctx->deviceId.empty() &&
    !deviceTypeSupportsExplicitGpuSelection(ctx->deviceType)
  ) {
    delete ctx;
    throw StringError(
      "OpenVINO backend: 'openvinoDeviceId' is only supported when openvinoDeviceType targets GPU. "
      "For NPU, remove openvinoDeviceId (use openvinoDeviceType = NPU)."
    );
  }

  try {
    ctx->availableDevices = ctx->core.get_available_devices();
  }
  catch(const std::exception& e) {
    delete ctx;
    throw StringError(string("OpenVINO backend: failed to enumerate available devices: ") + e.what());
  }

  if(logger != NULL) {
    logger->write(
      "OpenVINO backend: creating compute context for " +
      Global::intToString(nnXLen) + "x" + Global::intToString(nnYLen) +
      " with deviceType='" + ctx->deviceType + "'"
    );

    string devList;
    for(size_t i = 0; i < ctx->availableDevices.size(); i++) {
      if(i > 0)
        devList += ", ";
      devList += Global::intToString((int)i) + ":" + ctx->availableDevices[i];
    }
    logger->write("OpenVINO backend: available devices [index:name] = [" + devList + "]");
  }

  return ctx;
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

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
    throw StringError("OpenVINO backend: inputsUseNHWC = true not supported, must use NCHW");

  if(
    gpuIdxForThisThread >= 0 &&
    !deviceTypeSupportsExplicitGpuSelection(context->deviceType)
  ) {
    throw StringError(
      "OpenVINO backend: openvinoDeviceToUse / openvinoDeviceToUseThread* are only supported when "
      "openvinoDeviceType targets GPU. For NPU, remove per-thread device mapping."
    );
  }

  if(logger != NULL) {
    logger->write("OpenVINO backend thread " + Global::intToString(serverThreadIdx) +
                  ": Model version " + Global::intToString(loadedModel->modelDesc.modelVersion));
    logger->write("OpenVINO backend thread " + Global::intToString(serverThreadIdx) +
                  ": Model name: " + loadedModel->modelDesc.name);

    string deviceMsg;
    if(gpuIdxForThisThread >= 0) {
      if((size_t)gpuIdxForThisThread < context->availableDevices.size())
        deviceMsg = context->availableDevices[gpuIdxForThisThread];
      else
        deviceMsg = "invalid-index-" + Global::intToString(gpuIdxForThisThread);
    }
    else {
      deviceMsg = buildDeviceName(context->deviceType, context->deviceId);
    }

    logger->write("OpenVINO backend thread " + Global::intToString(serverThreadIdx) +
                  ": device=" + deviceMsg +
                  " fp16Mode=" + context->useFP16Mode.toString());
  }

  return new ComputeHandle(context, *loadedModel, logger, gpuIdxForThisThread, maxBatchSize);
}

void NeuralNet::freeComputeHandle(ComputeHandle* computeHandle) {
  delete computeHandle;
}

bool NeuralNet::isUsingFP16(const ComputeHandle* computeHandle) {
  return computeHandle->usingFP16;
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

  for(int nIdx = 0; nIdx < batchSize; nIdx++) {
    float* rowSpatialInput = inputBuffers->spatialInput.data() + (inputBuffers->singleInputElts * nIdx);
    float* rowGlobalInput = inputBuffers->globalInput.data() + (inputBuffers->singleInputGlobalElts * nIdx);

    const float* rowGlobal = inputBufs[nIdx]->rowGlobalBuf.data();
    const float* rowSpatial = inputBufs[nIdx]->rowSpatialBuf.data();

    std::copy(rowGlobal, rowGlobal + numGlobalFeatures, rowGlobalInput);
    SymmetryHelpers::copyInputsWithSymmetry(
      rowSpatial,
      rowSpatialInput,
      1,
      nnYLen,
      nnXLen,
      numSpatialFeatures,
      false,
      inputBufs[nIdx]->symmetry
    );

    if(computeHandle->numInputMetaChannels > 0) {
      float* rowMetaInput = inputBuffers->metaInput.data() + (inputBuffers->singleInputMetaElts * nIdx);
      const float* rowMeta = inputBufs[nIdx]->rowMetaBuf.data();
      std::copy(rowMeta, rowMeta + computeHandle->numInputMetaChannels, rowMetaInput);
    }
  }

  // The compiled model has a fixed batch size. Fill unused rows with a valid
  // copy to avoid divide-by-zero instability in masked global pooling paths.
  if(batchSize < computeHandle->fixedBatchSize) {
    const size_t spatialStride = inputBuffers->singleInputElts;
    const size_t globalStride = inputBuffers->singleInputGlobalElts;
    const size_t metaStride = inputBuffers->singleInputMetaElts;
    for(int nIdx = batchSize; nIdx < computeHandle->fixedBatchSize; nIdx++) {
      std::copy(
        inputBuffers->spatialInput.data(),
        inputBuffers->spatialInput.data() + spatialStride,
        inputBuffers->spatialInput.data() + spatialStride * nIdx
      );
      std::copy(
        inputBuffers->globalInput.data(),
        inputBuffers->globalInput.data() + globalStride,
        inputBuffers->globalInput.data() + globalStride * nIdx
      );
      if(computeHandle->numInputMetaChannels > 0) {
        std::copy(
          inputBuffers->metaInput.data(),
          inputBuffers->metaInput.data() + metaStride,
          inputBuffers->metaInput.data() + metaStride * nIdx
        );
      }
    }
  }

  ov::Tensor spatialTensor(
    ov::element::f32,
    ov::Shape{(size_t)computeHandle->fixedBatchSize, (size_t)numSpatialFeatures, (size_t)nnYLen, (size_t)nnXLen},
    inputBuffers->spatialInput.data()
  );
  computeHandle->inferRequest.set_tensor("input_spatial", spatialTensor);

  ov::Tensor globalTensor(
    ov::element::f32,
    ov::Shape{(size_t)computeHandle->fixedBatchSize, (size_t)numGlobalFeatures},
    inputBuffers->globalInput.data()
  );
  computeHandle->inferRequest.set_tensor("input_global", globalTensor);

  if(computeHandle->numInputMetaChannels > 0) {
    ov::Tensor metaTensor(
      ov::element::f32,
      ov::Shape{(size_t)computeHandle->fixedBatchSize, (size_t)computeHandle->numInputMetaChannels},
      inputBuffers->metaInput.data()
    );
    computeHandle->inferRequest.set_tensor("input_meta", metaTensor);
  }

  computeHandle->inferRequest.infer();

  ov::Tensor policyTensor = computeHandle->inferRequest.get_tensor("out_policy");
  ov::Tensor valueTensor = computeHandle->inferRequest.get_tensor("out_value");
  ov::Tensor miscvalueTensor = computeHandle->inferRequest.get_tensor("out_miscvalue");
  ov::Tensor ownershipTensor = computeHandle->inferRequest.get_tensor("out_ownership");

  const float* policyData = policyTensor.data<const float>();
  const float* valueData = valueTensor.data<const float>();
  const float* miscvalueData = miscvalueTensor.data<const float>();
  const float* ownershipData = ownershipTensor.data<const float>();

  assert(policyData != nullptr);
  assert(valueData != nullptr);
  assert(miscvalueData != nullptr);
  assert(ownershipData != nullptr);
  assert((int)outputs.size() == batchSize);

  const int policyResultLen = computeHandle->policyResultLen;
  const int spatialPolicyLen = nnXLen * nnYLen;
  float policyProbsTmp[NNPos::MAX_NN_POLICY_SIZE];

  for(int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];
    assert(output->nnXLen == nnXLen);
    assert(output->nnYLen == nnYLen);

    float policyOptimism = (float)inputBufs[row]->policyOptimism;

    {
      const float* policyRowBase = policyData + row * numPolicyChannels * policyResultLen;
      float* policyProbs = output->policyProbs;

      if(numPolicyChannels >= 2) {
        const float* ch0 = policyRowBase;
        const float* ch1 = policyRowBase + policyResultLen;
        for(int i = 0; i < spatialPolicyLen; i++) {
          float p = ch0[i];
          float pOpt = ch1[i];
          policyProbsTmp[i] = p + (pOpt - p) * policyOptimism;
        }
        SymmetryHelpers::copyOutputsWithSymmetry(
          policyProbsTmp,
          policyProbs,
          1,
          nnYLen,
          nnXLen,
          inputBufs[row]->symmetry
        );
        policyProbs[spatialPolicyLen] = ch0[spatialPolicyLen] + (ch1[spatialPolicyLen] - ch0[spatialPolicyLen]) * policyOptimism;
      }
      else {
        assert(numPolicyChannels == 1);
        const float* ch0 = policyRowBase;
        SymmetryHelpers::copyOutputsWithSymmetry(ch0, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
        policyProbs[spatialPolicyLen] = ch0[spatialPolicyLen];
      }
    }

    {
      const int numVC = computeHandle->numValueChannels;
      assert(numVC == 3);
      output->whiteWinProb = valueData[row * numVC];
      output->whiteLossProb = valueData[row * numVC + 1];
      output->whiteNoResultProb = valueData[row * numVC + 2];
    }

    {
      const int numScoreValueChannels = computeHandle->numScoreValueChannels;
      if(computeHandle->modelVersion >= 9) {
        assert(numScoreValueChannels >= 6);
        output->whiteScoreMean = miscvalueData[row * numScoreValueChannels];
        output->whiteScoreMeanSq = miscvalueData[row * numScoreValueChannels + 1];
        output->whiteLead = miscvalueData[row * numScoreValueChannels + 2];
        output->varTimeLeft = miscvalueData[row * numScoreValueChannels + 3];
        output->shorttermWinlossError = miscvalueData[row * numScoreValueChannels + 4];
        output->shorttermScoreError = miscvalueData[row * numScoreValueChannels + 5];
      }
      else if(computeHandle->modelVersion >= 8) {
        assert(numScoreValueChannels >= 4);
        output->whiteScoreMean = miscvalueData[row * numScoreValueChannels];
        output->whiteScoreMeanSq = miscvalueData[row * numScoreValueChannels + 1];
        output->whiteLead = miscvalueData[row * numScoreValueChannels + 2];
        output->varTimeLeft = miscvalueData[row * numScoreValueChannels + 3];
        output->shorttermWinlossError = 0;
        output->shorttermScoreError = 0;
      }
      else if(computeHandle->modelVersion >= 4) {
        assert(numScoreValueChannels >= 2);
        output->whiteScoreMean = miscvalueData[row * numScoreValueChannels];
        output->whiteScoreMeanSq = miscvalueData[row * numScoreValueChannels + 1];
        output->whiteLead = output->whiteScoreMean;
        output->varTimeLeft = 0;
        output->shorttermWinlossError = 0;
        output->shorttermScoreError = 0;
      }
      else if(computeHandle->modelVersion >= 3) {
        assert(numScoreValueChannels >= 1);
        output->whiteScoreMean = miscvalueData[row * numScoreValueChannels];
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

    if(output->whiteOwnerMap != NULL) {
      assert(computeHandle->numOwnershipChannels == 1);
      const float* ownershipRowBuf = ownershipData + row * nnXLen * nnYLen;
      SymmetryHelpers::copyOutputsWithSymmetry(
        ownershipRowBuf,
        output->whiteOwnerMap,
        1,
        nnYLen,
        nnXLen,
        inputBufs[row]->symmetry
      );
    }
  }
}

void NeuralNet::printDevices() {
  try {
    ov::Core core;
    vector<string> devices = core.get_available_devices();
    cout << "OpenVINO backend available devices:" << endl;
    for(size_t i = 0; i < devices.size(); i++) {
      cout << "  [" << i << "] " << devices[i] << endl;
    }
  }
  catch(const std::exception& e) {
    cout << "OpenVINO backend: failed to list devices: " << e.what() << endl;
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
