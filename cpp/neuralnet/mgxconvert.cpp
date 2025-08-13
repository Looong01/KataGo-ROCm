#include "mgxconvert.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace MGXConvert {

  static const char* PYTHON_SCRIPT = R"PY(
import json
import struct
from argparse import ArgumentParser

import numpy as np
import torch
import torch.onnx
from packaging import version
import onnxmltools
from onnxmltools.utils.float16_converter import convert_float_to_float16


# -----------------------------------------------------------------------------
# Low-level operations and helpers
# -----------------------------------------------------------------------------


def norm(C_in, norm_name="FixUp", affine=True, fixup_use_gamma=False):
    if norm_name == "BN":
        return torch.nn.BatchNorm2d(C_in, affine=affine)
    if norm_name == "AN":
        return AttenNorm(C_in)
    if norm_name == "FixUp":
        return KataFixUp(C_in, fixup_use_gamma)
    raise NotImplementedError("Unknown feature norm name")


def act(activation, inplace=False):
    if activation == "ReLU":
        return torch.nn.ReLU(inplace=inplace)
    if activation == "Hardswish":
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            return torch.nn.Hardswish(inplace=inplace)
        else:
            return torch.nn.Hardswish()
    if activation == "Identity":
        return torch.nn.Identity()
    raise NotImplementedError("Unknown activation name")


def conv1x1(C_in, C_out):
    return torch.nn.Conv2d(C_in, C_out, 1, 1, 0, bias=False)


def conv3x3(C_in, C_out):
    return torch.nn.Conv2d(C_in, C_out, 3, 1, 1, bias=False)


class AttenNorm(torch.nn.BatchNorm2d):
    def __init__(
        self, num_features, K=5, eps=1e-5, momentum=0.1, track_running_stats=True
    ):
        super(AttenNorm, self).__init__(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=False,
            track_running_stats=track_running_stats,
        )
        self.gamma = torch.nn.Parameter(torch.Tensor(K, num_features))
        self.beta = torch.nn.Parameter(torch.Tensor(K, num_features))
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(num_features, K)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        output = super(AttenNorm, self).forward(x)
        size = output.size()
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y)
        y = self.sigmoid(y)
        gamma = y @ self.gamma
        beta = y @ self.beta
        gamma = gamma.unsqueeze(-1).unsqueeze(-1).expand(size)
        beta = beta.unsqueeze(-1).unsqueeze(-1).expand(size)
        return gamma * output + beta


class KataFixUp(torch.nn.Module):
    def __init__(self, C_in, use_gamma):
        super(KataFixUp, self).__init__()
        self.use_gamma = use_gamma
        if self.use_gamma:
            self.gamma = torch.nn.Parameter(torch.ones(1, C_in, 1, 1))
        self.beta = torch.nn.Parameter(torch.zeros(1, C_in, 1, 1))

    def forward(self, x):
        if self.use_gamma:
            return x * self.gamma + self.beta
        else:
            return x + self.beta


class NormMask(torch.nn.Module):
    def __init__(self, C_in, normalization="FixUp", affine=True, fixup_use_gamma=False):
        super(NormMask, self).__init__()
        self._norm = norm(C_in, normalization, affine, fixup_use_gamma)

    def forward(self, x, mask):
        return self._norm(x)


class NormMaskActConv(torch.nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        kernel_size,
        padding,
        affine=True,
        activation="ReLU",
        normalization="FixUp",
        fixup_use_gamma=False,
    ):
        super(NormMaskActConv, self).__init__()
        self.norm = NormMask(
            C_in, normalization, affine=affine, fixup_use_gamma=fixup_use_gamma
        )
        self.act = act(activation, inplace=False)
        self.conv = torch.nn.Conv2d(
            C_in, C_out, kernel_size, padding=padding, bias=False
        )

    def forward(self, x, mask):
        out = self.norm(x, mask)
        out = self.act(out)
        out = self.conv(out)
        return out


class NormMaskActConv3x3(torch.nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        affine=True,
        activation="ReLU",
        normalization="FixUp",
        fixup_use_gamma=False,
    ):
        super(NormMaskActConv3x3, self).__init__()
        self.op = NormMaskActConv(
            C_in,
            C_out,
            kernel_size=3,
            padding=1,
            affine=affine,
            activation=activation,
            normalization=normalization,
            fixup_use_gamma=fixup_use_gamma,
        )

    def forward(self, x, mask):
        return self.op(x, mask)


class NormMaskActConv1x1(torch.nn.Module):
    def __init__(
        self,
        C_in,
        C_out,
        affine=True,
        activation="ReLU",
        normalization="FixUp",
        fixup_use_gamma=False,
    ):
        super(NormMaskActConv1x1, self).__init__()
        self.op = NormMaskActConv(
            C_in,
            C_out,
            kernel_size=1,
            padding=0,
            affine=affine,
            activation=activation,
            normalization=normalization,
            fixup_use_gamma=fixup_use_gamma,
        )

    def forward(self, x, mask):
        return self.op(x, mask)


# -----------------------------------------------------------------------------
# Model basis building blocks
# -----------------------------------------------------------------------------


class KataGPool(torch.nn.Module):
    def __init__(self):
        super(KataGPool, self).__init__()

    def forward(self, x, mask):
        mask_sum_hw = mask.sum(dim=(1, 2, 3))
        mask_sum_hw_sqrt = mask_sum_hw.sqrt()
        div = mask_sum_hw.reshape((-1, 1, 1, 1))
        div_sqrt = mask_sum_hw_sqrt.reshape((-1, 1, 1, 1))

        layer_mean = x.sum(dim=(2, 3), keepdim=True) / div
        layer_max = x.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        out_pool1 = layer_mean
        out_pool2 = layer_mean * (div_sqrt - 14.0) / 10.0
        out_pool3 = layer_max

        out = torch.cat((out_pool1, out_pool2, out_pool3), 1)
        return out


class KataValueHeadGPool(torch.nn.Module):
    def __init__(self):
        super(KataValueHeadGPool, self).__init__()

    def forward(self, x, mask):
        mask_sum_hw = mask.sum(dim=(1, 2, 3))
        mask_sum_hw_sqrt = mask_sum_hw.sqrt()
        div = mask_sum_hw.reshape((-1, 1, 1, 1))
        div_sqrt = mask_sum_hw_sqrt.reshape((-1, 1, 1, 1))

        layer_mean = x.sum(dim=(2, 3), keepdim=True) / div

        out_pool1 = layer_mean
        out_pool2 = layer_mean * (div_sqrt - 14.0) / 10.0
        out_pool3 = layer_mean * ((div_sqrt - 14.0) * (div_sqrt - 14.0) / 100.0 - 0.1)

        out = torch.cat((out_pool1, out_pool2, out_pool3), 1)
        return out


class KataGPoolCell(torch.nn.Module):
    def __init__(self, C_in, C_gpool, C_regular, activation, normalization):
        super(KataGPoolCell, self).__init__()
        self.norm1 = NormMask(C_in, normalization)
        self.act1 = act(activation)
        self.conv1_3x3 = conv3x3(C_in, C_regular)
        self.conv2_3x3 = conv3x3(C_in, C_gpool)
        self.norm2 = NormMask(C_gpool, normalization)
        self.act2 = act(activation)
        self.gpool = KataGPool()
        self.linear = torch.nn.Linear(3 * C_gpool, C_regular, bias=False)

    def forward(self, x, mask):
        out0 = self.norm1(x, mask)
        out0 = self.act1(out0)
        out1 = self.conv1_3x3(out0)
        out2 = self.conv2_3x3(out0)
        out2 = self.norm2(out2, mask)
        out2 = self.act2(out2)
        out3 = self.gpool(out2, mask).squeeze()
        out3 = self.linear(out3).unsqueeze(-1).unsqueeze(-1)
        out = out1 + out3
        return out


class ResBlock(torch.nn.Module):
    def __init__(self, C_in, activation, normalization):
        super(ResBlock, self).__init__()
        self.conv1_3x3 = NormMaskActConv3x3(
            C_in, C_in, activation=activation, normalization=normalization
        )
        self.conv2_3x3 = NormMaskActConv3x3(
            C_in,
            C_in,
            activation=activation,
            normalization=normalization,
            fixup_use_gamma=True,
        )

    def forward(self, x, mask):
        residual = x
        out = self.conv1_3x3(x, mask)
        out = self.conv2_3x3(out, mask)
        out += residual
        return out


class GpoolResBlock(torch.nn.Module):
    def __init__(self, C_in, C_gpool, C_regular, activation, normalization):
        super(GpoolResBlock, self).__init__()
        self.pool = KataGPoolCell(
            C_in, C_gpool, C_regular, activation=activation, normalization=normalization
        )
        self.conv1_3x3 = NormMaskActConv3x3(
            C_regular,
            C_in,
            activation=activation,
            normalization=normalization,
            fixup_use_gamma=True,
        )

    def forward(self, x, mask):
        residual = x
        out = self.pool(x, mask)
        out = self.conv1_3x3(out, mask)
        out += residual
        return out


class PolicyHead(torch.nn.Module):
    def __init__(self, C_in, C_p, C_pg, activation, normalization):
        super(PolicyHead, self).__init__()
        self.conv1_1x1 = conv1x1(C_in, C_p)
        self.conv2_1x1 = conv1x1(C_in, C_pg)
        self.norm1 = NormMask(C_pg, normalization)
        self.act1 = act(activation)
        self.gpool = KataGPool()
        self.linear_pass = torch.nn.Linear(3 * C_pg, 1, bias=False)
        self.linear = torch.nn.Linear(3 * C_pg, C_p, bias=False)
        self.conv3_1x1 = NormMaskActConv1x1(
            C_p, 1, activation=activation, normalization=normalization
        )

    def forward(self, x, mask):
        out_p = self.conv1_1x1(x)
        out_g = self.conv2_1x1(x)
        out_g = self.norm1(out_g, mask)
        out_g = self.act1(out_g)
        out_pool = self.gpool(out_g, mask).squeeze()
        out_pass = self.linear_pass(out_pool)

        out_pool = self.linear(out_pool).unsqueeze(-1).unsqueeze(-1)
        out_p += out_pool
        out_policy = self.conv3_1x1(out_p, mask)
        out_policy = out_policy - (1.0 - mask) * 5000.0
        out_policy = out_policy.flatten(start_dim=1)
        out_pass = out_pass.reshape((-1, 1))
        return torch.cat((out_policy, out_pass), -1)


class ValueHead(torch.nn.Module):
    def __init__(self, C_in, C_v1, C_v2, activation, normalization):
        super(ValueHead, self).__init__()
        self.init_conv = conv1x1(C_in, C_v1)
        self.norm1 = NormMask(C_v1, normalization)
        self.act1 = act(activation)
        self.vh_gpool = KataValueHeadGPool()
        self.linear_after_pool = torch.nn.Linear(3 * C_v1, C_v2)
        self.act_after_pool = act(activation)
        self.linear_valuehead = torch.nn.Linear(C_v2, 3)
        self.linear_miscvaluehead = torch.nn.Linear(C_v2, 4)
        self.conv_ownership = conv1x1(C_v1, 1)

    def forward(self, x, mask):
        out_v1 = self.init_conv(x)
        out_v1 = self.norm1(out_v1, mask)
        out_v1 = self.act1(out_v1)
        out_pooled = self.vh_gpool(out_v1, mask).squeeze()
        out_pooled = self.linear_after_pool(out_pooled)
        out_v2 = self.act_after_pool(out_pooled)
        out_value = self.linear_valuehead(out_v2)
        out_miscvalue = self.linear_miscvaluehead(out_v2)
        out_ownership = self.conv_ownership(out_v1)
        return out_value, out_miscvalue, out_ownership


class KataGoInferenceModel(torch.nn.Module):
    def __init__(self, conf):
        super(KataGoInferenceModel, self).__init__()

        self.conf = conf
        self.block_kind = conf["config"]["block_kind"]
        self.C = conf["config"]["trunk_num_channels"]
        self.C_mid = conf["config"]["mid_num_channels"]
        self.C_gpool = conf["config"]["gpool_num_channels"]
        self.C_regular = conf["config"]["regular_num_channels"]
        self.C_p = conf["config"]["p1_num_channels"]
        self.C_pg = conf["config"]["g1_num_channels"]
        self.C_v1 = conf["config"]["v1_num_channels"]
        self.C_v2 = conf["config"]["v2_size"]
        self.activation = "ReLU"
        if conf["config"]["use_fixup"]:
            self.normalization = "FixUp"
        else:
            self.normalization = "BN"

        bin_in = conf["initial_conv"]["C_in"]
        diam_y = conf["initial_conv"]["diam_y"]
        diam_x = conf["initial_conv"]["diam_x"]
        dil_y = conf["initial_conv"]["dil_y"]
        dil_x = conf["initial_conv"]["dil_x"]
        pad_y = (diam_y // 2) * dil_y
        pad_x = (diam_x // 2) * dil_x
        glob_in = conf["initial_matmul"]["C_in"]

        self.linear_ginput = torch.nn.Linear(glob_in, self.C, bias=False)
        self.conv1 = torch.nn.Conv2d(
            bin_in,
            self.C,
            (diam_y, diam_x),
            1,
            (pad_y, pad_x),
            bias=False,
            dilation=(dil_y, dil_x),
        )

        self.blocks = torch.nn.ModuleList()
        for block_conf in self.block_kind:
            if block_conf[1] == "regular":
                self.blocks += [ResBlock(self.C, self.activation, self.normalization)]
            elif block_conf[1] == "gpool":
                self.blocks += [
                    GpoolResBlock(
                        self.C,
                        self.C_gpool,
                        self.C_regular,
                        self.activation,
                        self.normalization,
                    )
                ]
            else:
                assert False

        self.norm1 = NormMask(self.C, self.normalization)
        self.act1 = act(self.activation)
        self.policy_head = PolicyHead(
            self.C, self.C_p, self.C_pg, self.activation, self.normalization
        )
        self.value_head = ValueHead(
            self.C, self.C_v1, self.C_v2, self.activation, self.normalization
        )

    def forward(self, input_binary, input_global):
        mask = input_binary[:, 0:1, :, :]

        x_bin = self.conv1(input_binary)
        x_global = self.linear_ginput(input_global).unsqueeze(-1).unsqueeze(-1)
        out = x_bin + x_global

        for block in self.blocks:
            out = block(out, mask)

        out = self.norm1(out, mask)
        out = self.act1(out)

        out_policy = self.policy_head(out, mask)
        (out_value, out_miscvalue, out_ownership) = self.value_head(out, mask)

        return (
            out_policy,
            out_value,
            out_miscvalue,
            out_ownership,
        )

    def fill_misc_weights(self, conv1_weight, linear_ginput_weight, norm1_weight):
        self.conv1.weight.data = conv1_weight
        self.linear_ginput.weight.data = linear_ginput_weight
        self.norm1._norm.beta.data = norm1_weight

    def fill_regular_block(self, block, block_dict):
        block.conv1_3x3.op.norm._norm.beta.data = block_dict["norm1"]["beta"]
        block.conv1_3x3.op.conv.weight.data = block_dict["conv1"]["weights"]
        block.conv2_3x3.op.norm._norm.gamma.data = block_dict["norm2"]["gamma"]
        block.conv2_3x3.op.norm._norm.beta.data = block_dict["norm2"]["beta"]
        block.conv2_3x3.op.conv.weight.data = block_dict["conv2"]["weights"]

    def fill_gpool_block(self, block, block_dict):
        block.pool.norm1._norm.beta.data = block_dict["norm1"]["beta"]
        block.pool.conv1_3x3.weight.data = block_dict["conv1"]["weights"]
        block.pool.conv2_3x3.weight.data = block_dict["conv2"]["weights"]
        block.pool.norm2._norm.beta.data = block_dict["norm2"]["beta"]
        block.pool.linear.weight.data = block_dict["matmul1"]["weights"]
        block.conv1_3x3.op.norm._norm.gamma.data = block_dict["norm3"]["gamma"]
        block.conv1_3x3.op.norm._norm.beta.data = block_dict["norm3"]["beta"]
        block.conv1_3x3.op.conv.weight.data = block_dict["conv3"]["weights"]

    def fill_policy_head(self, policy_dict):
        self.policy_head.conv1_1x1.weight.data = policy_dict["convp"]["weights"]
        self.policy_head.conv2_1x1.weight.data = policy_dict["convg"]["weights"]
        self.policy_head.norm1._norm.beta.data = policy_dict["normg"]["beta"]
        self.policy_head.linear_pass.weight.data = policy_dict["matmulpass"]["weights"]
        self.policy_head.linear.weight.data = policy_dict["matmulg"]["weights"]
        self.policy_head.conv3_1x1.op.norm._norm.beta.data = policy_dict["norm2"]["beta"]
        self.policy_head.conv3_1x1.op.conv.weight.data = policy_dict["conv3"]["weights"]

    def fill_value_head(self, value_dict):
        self.value_head.init_conv.weight.data = value_dict["conv1"]["weights"]
        self.value_head.norm1._norm.beta.data = value_dict["norm1"]["beta"]
        self.value_head.linear_after_pool.weight.data = value_dict["matmul1"]["weights"]
        self.value_head.linear_after_pool.bias.data = value_dict["matbias1"]["weights"]
        self.value_head.linear_valuehead.weight.data = value_dict["matmul2"]["weights"]
        self.value_head.linear_valuehead.bias.data = value_dict["matbias2"]["weights"]
        self.value_head.linear_miscvaluehead.weight.data = value_dict["matmul3"]["weights"]
        self.value_head.linear_miscvaluehead.bias.data = value_dict["matbias3"]["weights"]
        self.value_head.conv_ownership.weight.data = value_dict["conv2"]["weights"]

    def fill_weights(self):
        print("Filling misc weights")
        self.fill_misc_weights(
            self.conf["initial_conv"]["weights"],
            self.conf["initial_matmul"]["weights"],
            self.conf["postprocess_norm"]["beta"],
        )
        for i, block_conf in enumerate(self.block_kind):
            print(f"Filling block {i} weights ({block_conf[1]})")
            if block_conf[1] == "regular":
                self.fill_regular_block(self.blocks[i], self.conf["blocks"][i])
            elif block_conf[1] == "gpool":
                self.fill_gpool_block(self.blocks[i], self.conf["blocks"][i])
            else:
                assert False
        print("Filling policy head weights")
        self.fill_policy_head(self.conf["policy_head"])
        print("Filling value head weights")
        self.fill_value_head(self.conf["value_head"])


# -----------------------------------------------------------------------------
# Netparser for KataGo binary model
# -----------------------------------------------------------------------------


def bin2str(binary_str):
    return binary_str.decode(encoding="ascii", errors="backslashreplace")


def read_header(lines, idx):
    header_dict = {
        "type": "header",
        "name": bin2str(lines[idx + 0]),
        "version": int(lines[idx + 1]),
        "num_bin_input_features": int(lines[idx + 2]),
        "num_global_input_features": int(lines[idx + 3]),
        "num_blocks": int(lines[idx + 5]),
        "num_channels": int(lines[idx + 6]),
        "num_mid_channels": int(lines[idx + 7]),
        "num_regular_channels": int(lines[idx + 8]),
        "num_dilated_channels": int(lines[idx + 9]),
        "num_gpool_channels": int(lines[idx + 10]),
    }

    return header_dict, idx + 11


def read_weights(lines, idx, shape):
    assert lines[idx][0:5] == b"@BIN@"
    buffer_size = struct.calcsize(f"<{np.prod(shape)}f")

    i_increment = 0
    buffer = lines[idx][5:]
    while len(buffer) < buffer_size:
        buffer += "\n".encode(encoding="ascii", errors="backslashreplace")
        if len(buffer) == buffer_size:
            break
        i_increment += 1
        buffer += lines[idx + i_increment]
    assert buffer_size == len(buffer)
    weights = np.array(struct.unpack(f"<{np.prod(shape)}f", buffer))
    weights = torch.tensor(weights.reshape(shape), dtype=torch.float)

    return weights, i_increment


def read_conv(lines, idx):
    diam_y = int(lines[idx + 1])
    diam_x = int(lines[idx + 2])
    C_in = int(lines[idx + 3])
    C_out = int(lines[idx + 4])
    weights, i_increment = read_weights(lines, idx + 7, (diam_y, diam_x, C_in, C_out))

    conv_dict = {
        "type": "conv",
        "name": bin2str(lines[idx]),
        "diam_y": diam_y,
        "diam_x": diam_x,
        "C_in": C_in,
        "C_out": C_out,
        "dil_y": int(lines[idx + 5]),
        "dil_x": int(lines[idx + 6]),
        "weights": weights.permute(3, 2, 0, 1),
    }

    return conv_dict, idx + 8 + i_increment


def read_norm(lines, idx):
    C_in = int(lines[idx + 1])
    norm_dict = {
        "type": "norm",
        "name": bin2str(lines[idx]),
        "C_in": C_in,
        "eps": float(lines[idx + 2]),
        "has_scale": bool(int(lines[idx + 3])),
        "has_bias": bool(int(lines[idx + 4])),
    }

    idx_increment = 0

    moving_mean, i_increment = read_weights(lines, idx + 5 + idx_increment, (C_in))
    norm_dict["moving_mean"] = moving_mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    idx_increment += 1 + i_increment

    moving_variance, i_increment = read_weights(lines, idx + 5 + idx_increment, (C_in))
    norm_dict["moving_variance"] = (
        moving_variance.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
    )
    idx_increment += 1 + i_increment

    if norm_dict["has_scale"]:
        gamma, i_increment = read_weights(lines, idx + 5 + idx_increment, (C_in))
        norm_dict["gamma"] = gamma.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        idx_increment += 1 + i_increment

    if norm_dict["has_bias"]:
        beta, i_increment = read_weights(lines, idx + 5 + idx_increment, (C_in))
        norm_dict["beta"] = beta.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        idx_increment += 1 + i_increment

    return norm_dict, idx + 5 + idx_increment


def read_act(lines, idx):
    act_dict = {"type": "act", "act": bin2str(lines[idx])}

    return act_dict, idx + 1


def read_matmul(lines, idx):
    C_in = int(lines[idx + 1])
    C_out = int(lines[idx + 2])
    weights, i_increment = read_weights(lines, idx + 3, (C_in, C_out))

    matmul_dict = {
        "type": "matmul",
        "name": bin2str(lines[idx]),
        "C_in": C_in,
        "C_out": C_out,
        "weights": weights.permute(1, 0),
    }

    return matmul_dict, idx + 4 + i_increment


def read_matbias(lines, idx):
    C_in = int(lines[idx + 1])
    weights, i_increment = read_weights(lines, idx + 2, (C_in))

    matbias_dict = {
        "type": "matbias",
        "name": bin2str(lines[idx]),
        "C_in": C_in,
        "weights": weights,
    }

    return matbias_dict, idx + 3 + i_increment


def read_block(lines, idx):
    block_type = bin2str(lines[idx])
    name = bin2str(lines[idx + 1])
    head_idx = idx + 2

    if block_type == "ordinary_block":
        norm1, head_idx = read_norm(lines, head_idx)
        act1, head_idx = read_act(lines, head_idx)
        conv1, head_idx = read_conv(lines, head_idx)
        norm2, head_idx = read_norm(lines, head_idx)
        act2, head_idx = read_act(lines, head_idx)
        conv2, head_idx = read_conv(lines, head_idx)

        block_dict = {
            "type": block_type,
            "name": name,
            "norm1": norm1,
            "act1": act1,
            "conv1": conv1,
            "norm2": norm2,
            "act2": act2,
            "conv2": conv2,
        }

        return block_dict, head_idx
    if block_type == "gpool_block":
        norm1, head_idx = read_norm(lines, head_idx)
        act1, head_idx = read_act(lines, head_idx)
        conv1, head_idx = read_conv(lines, head_idx)
        conv2, head_idx = read_conv(lines, head_idx)
        norm2, head_idx = read_norm(lines, head_idx)
        act2, head_idx = read_act(lines, head_idx)
        matmul1, head_idx = read_matmul(lines, head_idx)
        norm3, head_idx = read_norm(lines, head_idx)
        act3, head_idx = read_act(lines, head_idx)
        conv3, head_idx = read_conv(lines, head_idx)

        block_dict = {
            "type": block_type,
            "name": name,
            "norm1": norm1,
            "act1": act1,
            "conv1": conv1,
            "conv2": conv2,
            "norm2": norm2,
            "act2": act2,
            "matmul1": matmul1,
            "norm3": norm3,
            "act3": act3,
            "conv3": conv3,
        }

        return block_dict, head_idx
    else:
        assert False


def read_policy_head(lines, idx):
    name = bin2str(lines[idx])
    head_idx = idx + 1

    conv1, head_idx = read_conv(lines, head_idx)
    conv2, head_idx = read_conv(lines, head_idx)
    norm1, head_idx = read_norm(lines, head_idx)
    act1, head_idx = read_act(lines, head_idx)
    matmul1, head_idx = read_matmul(lines, head_idx)
    norm2, head_idx = read_norm(lines, head_idx)
    act2, head_idx = read_act(lines, head_idx)
    conv3, head_idx = read_conv(lines, head_idx)
    matmul2, head_idx = read_matmul(lines, head_idx)

    policy_head_dict = {
        "type": "policy head",
        "name": name,
        "convp": conv1,
        "convg": conv2,
        "normg": norm1,
        "actg": act1,
        "matmulg": matmul1,
        "norm2": norm2,
        "act2": act2,
        "conv3": conv3,
        "matmulpass": matmul2,
    }

    return policy_head_dict, head_idx


def read_value_head(lines, idx):
    name = bin2str(lines[idx])
    head_idx = idx + 1

    conv1, head_idx = read_conv(lines, head_idx)
    norm1, head_idx = read_norm(lines, head_idx)
    act1, head_idx = read_act(lines, head_idx)
    matmul1, head_idx = read_matmul(lines, head_idx)
    matbias1, head_idx = read_matbias(lines, head_idx)
    act2, head_idx = read_act(lines, head_idx)
    matmul2, head_idx = read_matmul(lines, head_idx)
    matbias2, head_idx = read_matbias(lines, head_idx)
    matmul3, head_idx = read_matmul(lines, head_idx)
    matbias3, head_idx = read_matbias(lines, head_idx)
    conv2, head_idx = read_conv(lines, head_idx)

    value_head_dict = {
        "type": "value head",
        "name": name,
        "conv1": conv1,
        "norm1": norm1,
        "act1": act1,
        "matmul1": matmul1,
        "matbias1": matbias1,
        "act2": act2,
        "matmul2": matmul2,
        "matbias2": matbias2,
        "matmul3": matmul3,
        "matbias3": matbias3,
        "conv2": conv2,
    }

    return value_head_dict, head_idx


def read_model(model_path):
    model_config = {
        "version": 16,
        "support_japanese_rules": True,
        "use_fixup": True,
        "use_scoremean_as_lead": False,
    }

    print(f"Model: {model_path}")
    with open(model_path, "rb") as f:
        contents = f.read()
    lines = contents.split("\n".encode(encoding="ascii", errors="backslashreplace"))
    print(f"Model file loading completed.")

    head_idx = 0
    header, head_idx = read_header(lines, head_idx)
    assert header["version"] == 8

    initial_conv, head_idx = read_conv(lines, head_idx)
    initial_matmul, head_idx = read_matmul(lines, head_idx)

    blocks = []
    for i in range(header["num_blocks"]):
        print(f"Reading block {i}")
        block, head_idx = read_block(lines, head_idx)
        blocks.append(block)

    postprocess_norm, head_idx = read_norm(lines, head_idx)
    postprocess_act, head_idx = read_act(lines, head_idx)

    print("Reading policy head")
    policy_head, head_idx = read_policy_head(lines, head_idx)
    print("Reading value head")
    value_head, head_idx = read_value_head(lines, head_idx)

    model_dict = {
        "config": model_config,
        "initial_conv": initial_conv,
        "initial_matmul": initial_matmul,
        "blocks": blocks,
        "postprocess_norm": postprocess_norm,
        "postprocess_act": postprocess_act,
        "policy_head": policy_head,
        "value_head": value_head,
    }

    return model_dict


# -----------------------------------------------------------------------------
# Conversion utilities
# -----------------------------------------------------------------------------


def convert(model, output):
    model_spec = read_model(model)
    model = KataGoInferenceModel(model_spec)
    print("Model building completed")
    model.fill_weights()

    bin_in = model_spec["initial_conv"]["C_in"]
    glob_in = model_spec["initial_matmul"]["C_in"]
    dummy_input_binary = torch.randn(10, bin_in, 19, 19)
    dummy_input_binary[:, 0, :, :] = 1.0
    dummy_input_global = torch.randn(10, glob_in)
    input_names = ["input_binary", "input_global"]
    output_names = [
        "output_policy",
        "output_value",
        "output_miscvalue",
        "output_ownership",
    ]

    torch.onnx.export(
        model,
        (dummy_input_binary, dummy_input_global),
        output,
        export_params=True,
        verbose=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            "input_binary": {0: "batch_size", 2: "y_size", 3: "x_size"},
            "input_global": {0: "batch_size"},
            "output_policy": {0: "batch_size", 1: "board_area + 1"},
            "output_value": {0: "batch_size"},
            "output_miscvalue": {0: "batch_size"},
            "output_ownership": {0: "batch_size", 2: "y_size", 3: "x_size"},
        },
    )

    print(f"ONNX model saved in {output}")


def quantize_fp16(input_path, output_path):
    onnx_model = onnxmltools.utils.load_model(input_path)
    fp16_model = convert_float_to_float16(onnx_model)
    onnxmltools.utils.save_model(fp16_model, output_path)
    print(f"{input_path} is quantized and saved as {output_path}.")


if __name__ == "__main__":
    description = """
    Convert KataGo .bin model to .onnx file.
    """
    parser = ArgumentParser(description)
    parser.add_argument(
        "--model", type=str, required=True, help="KataGo .bin network file location"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output .onnx network file location"
    )
    args = parser.parse_args()
    if args.output is None:
        args.output = args.model.replace(".bin", ".onnx")
    convert(args.model, args.output)
)PY";

  bool convertRawToOnnx(const std::string& modelPath, const std::string& outPath) {
    namespace fs = std::filesystem;
    std::string venvPath = "/tmp/katago_mgx_venv";
    std::string py = "python3";

    if(!fs::exists(venvPath)) {
      std::string cmd = py + " -m venv " + venvPath;
      if(std::system(cmd.c_str()) != 0)
        return false;
      cmd = venvPath + "/bin/pip install --quiet torch onnxmltools packaging numpy onnx";
      if(std::system(cmd.c_str()) != 0)
        return false;
    }

    std::string scriptPath = "/tmp/mgxconvert.py";
    std::ofstream ofs(scriptPath);
    ofs << PYTHON_SCRIPT;
    ofs.close();

    std::string cmd = venvPath + "/bin/python " + scriptPath + " --model " + modelPath + " --output " + outPath;
    int ret = std::system(cmd.c_str());
    return ret == 0;
  }

}  // namespace MGXConvert
