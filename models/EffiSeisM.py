import os

import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from ._factory import register_model
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath
from functools import partial

def _auto_pad_1d(
    x: torch.Tensor,
    kernel_size: int,
    stride: int = 1,
    dim: int = -1,
    padding_value: float = 0.0,
) -> torch.Tensor:
    """
    Auto pad for conv layer.

    The output of conv-layer has the shape as `ceil(x.size(dim)/stride)`.

    Use this function to replace `padding='same'` which `torch.jit` and `torch.onnx` do not support.

    Args:
        x (torch.Tensor): N-dimensional tensor.
                  input (Tensor): N-dimensional tensor
        kernel_size (int): Conv kernel size.
        stride (int): Conv stride.
        dim (int): Dimension to pad.
        padding_value (float): fill value.

    Raises:
        AssertionError: `kernel_size` is less than `stride`.

    Returns:
        torch.Tensor : padded tensor.
    """

    assert (
        kernel_size >= stride
    ), f"`kernel_size` must be greater than or equal to `stride`, got {kernel_size}, {stride}"
    pos_dim = dim if dim >= 0 else x.dim() + dim
    pds = (stride - (x.size(dim) % stride)) % stride + kernel_size - stride
    padding = (0, 0) * (x.dim() - pos_dim - 1) + (pds // 2, pds - pds // 2)
    padded_x = F.pad(x, padding, "constant", padding_value)
    return padded_x


class ScaledActivation(nn.Module):
    def __init__(self, act_layer: nn.Module, scale_factor: float):
        super().__init__()
        self.scale_factor = scale_factor
        self.act = act_layer()

    def forward(self, x):
        return self.act(x) * self.scale_factor

class RadialBasisFunction(nn.Module):
    def __init__(
            self,
            grid_min: float = -2.,
            grid_max: float = 2.,
            num_grids: int = 8,
            denominator: float = None,  # larger denominators lead to smoother basis
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = torch.nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x):
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)

class FastKANConvNDLayer(nn.Module):
    def __init__(self, conv_class, norm_class, input_dim, output_dim, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1,
                 ndim: int = 2, grid_size=8, base_activation=nn.SiLU, grid_range=[-2, 2], dropout=0.0, **norm_kwargs):
        super(FastKANConvNDLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.ndim = ndim
        self.grid_size = grid_size
        self.base_activation = base_activation()
        self.grid_range = grid_range
        self.norm_kwargs = norm_kwargs

        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if input_dim % groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if output_dim % groups != 0:
            raise ValueError('output_dim must be divisible by groups')

        self.base_conv = nn.ModuleList([conv_class(input_dim // groups,
                                                   output_dim // groups,
                                                   kernel_size,
                                                   stride,
                                                   padding,
                                                   dilation,
                                                   groups=1,
                                                   bias=False) for _ in range(groups)])

        self.spline_conv = nn.ModuleList([conv_class(grid_size * input_dim // groups,
                                                     output_dim // groups,
                                                     kernel_size,
                                                     stride,
                                                     padding,
                                                     dilation,
                                                     groups=1,
                                                     bias=False) for _ in range(groups)])

        self.layer_norm = nn.ModuleList([norm_class(input_dim // groups, **norm_kwargs) for _ in range(groups)])

        self.rbf = RadialBasisFunction(grid_range[0], grid_range[1], grid_size)

        self.dropout = None
        if dropout > 0:
            if ndim == 1:
                self.dropout = nn.Dropout1d(p=dropout)
            if ndim == 2:
                self.dropout = nn.Dropout2d(p=dropout)
            if ndim == 3:
                self.dropout = nn.Dropout3d(p=dropout)

        # Initialize weights using Kaiming uniform distribution for better training start
        for conv_layer in self.base_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

        for conv_layer in self.spline_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

    def forward_fast_kan(self, x, group_index):

        # Apply base activation to input and then linear transform with base weights
        base_output = self.base_conv[group_index](self.base_activation(x))
        if self.dropout is not None:
            x = self.dropout(x)
        spline_basis = self.rbf(self.layer_norm[group_index](x))
        spline_basis = spline_basis.moveaxis(-1, 2).flatten(1, 2)
        spline_output = self.spline_conv[group_index](spline_basis)
        x = base_output + spline_output

        return x

    def forward(self, x):
        x = _auto_pad_1d(x,kernel_size=self.kernel_size,stride= self.stride)
        split_x = torch.split(x, self.inputdim // self.groups, dim=1)
        output = []
        for group_ind, _x in enumerate(split_x):
            y = self.forward_fast_kan(_x, group_ind)
            output.append(y.clone())
        y = torch.cat(output, dim=1)
        return y

class FastKANConv1DLayer(FastKANConvNDLayer):
    def __init__(self, input_dim, output_dim, kernel_size, groups=1, padding=0, stride=1, dilation=1,
                 grid_size=8, base_activation=nn.SiLU, grid_range=[-2, 2], dropout=0.0,
                 norm_layer=nn.InstanceNorm1d, **norm_kwargs):
        super(FastKANConv1DLayer, self).__init__(nn.Conv1d, norm_layer,
                                                 input_dim, output_dim,
                                                 kernel_size,
                                                 groups=groups, padding=padding, stride=stride, dilation=dilation,
                                                 ndim=1,
                                                 grid_size=grid_size, base_activation=base_activation,
                                                 grid_range=grid_range,
                                                 dropout=dropout, **norm_kwargs)

class MultiScale_Feature_Extractor(nn.Module):
    """
    MultiScale Feature Extractor.
    """

    def __init__(
            self, in_dim, out_dim, kernel_size, stride, act_layer, norm_layer, npath=3
    ):
        super().__init__()

        self.convs = nn.ModuleList(
            [
                FastKANConv1DLayer(
                    in_dim,
                    out_dim,
                    kernel_size,
                    stride=stride,
                    norm_layer=norm_layer
                )
            ]
        )

        self.out_proj = nn.Conv1d(
            in_channels=out_dim, out_channels=out_dim, kernel_size=1, bias=False
        )
        self.norm = norm_layer(out_dim)

    def forward(self, x):
        outs = list()
        for conv in self.convs:
            xi = conv(x)
            outs.append(xi)
        x = torch.cat(outs, dim=1)
        x = self.out_proj(x)
        x = self.norm(x)
        return x

class LocalAwareAggregationBlock(nn.Module):
    """Local Aware Aggregation"""

    def __init__(self, in_dim, out_dim, kernel_size, norm_layer):
        super().__init__()

        if kernel_size > 1:
            self.avg_pool = nn.AvgPool1d(kernel_size, ceil_mode=True)
            self.max_pool = nn.MaxPool1d(kernel_size, ceil_mode=True)

        else:
            self.avg_pool = self.max_pool = None

        self.proj = nn.Conv1d(
            in_channels=in_dim, out_channels=out_dim, kernel_size=1, bias=False
        )
        self.norm = norm_layer(out_dim)

    def forward(self, x):
        if self.avg_pool is not None:
            x = self.avg_pool(x) + self.max_pool(x)
        x = self.proj(x)
        x = self.norm(x)
        return x

def _make_divisible(v: int, divisor: int) -> int:
    """
    Returns the closest integer to `v` that is divisible by `divisor`.

    Modified from: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class DSConvNormAct(nn.Module):
    """Depthwise separable convolution"""

    def __init__(self, in_dim, out_dim, kernel_size, stride, act_layer, norm_layer):
        super().__init__()

        self.in_proj = nn.Conv1d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1, bias=False
        )

        self.dconv = nn.Conv1d(
            in_channels=in_dim,
            out_channels=in_dim,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_dim,
            bias=False,
        )
        self.pconv = nn.Conv1d(
            in_channels=in_dim, out_channels=out_dim, kernel_size=1, bias=False
        )
        self.norm = norm_layer(out_dim)
        self.act = act_layer()

    def forward(self, x):
        x = self.in_proj(x)
        x = _auto_pad_1d(x, self.dconv.kernel_size[0], self.dconv.stride[0])
        x = self.dconv(x)
        x = self.pconv(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class FSC(nn.Module):
    def __init__(
        self,
        io_dim,
        groups,
        kernel_sizes,
        act_layer,
        norm_layer,
    ):
        super().__init__()

        group_size = io_dim // groups
        dims_ = []
        self.projs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.convs = nn.ModuleList()
        for kernel_size in kernel_sizes:
            dim = _make_divisible(
                (io_dim - sum(dims_)) // (len(kernel_sizes) - len(dims_)), group_size
            )

            assert dim > 0
            dims_.append(dim)

            proj = nn.Conv1d(
                in_channels=io_dim, out_channels=dim, kernel_size=1, bias=False
            )
            norm = norm_layer(dim)
            conv = DSConvNormAct(
                in_dim=dim,
                out_dim=dim,
                kernel_size=kernel_size,
                stride=1,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
            self.projs.append(proj)
            self.norms.append(norm)
            self.convs.append(conv)

        self.out_norm = norm_layer(io_dim)

    def forward(self, x):
        outs = list()
        for proj, norm, conv in zip(self.projs, self.norms, self.convs):
            xi = norm(proj(x))
            xi = xi + conv(xi)
            outs.append(xi)

        x = torch.cat(outs, dim=1)
        x = self.out_norm(x)

        return x

class SSM_Layer(nn.Module): #Mamba_Layer
    def __init__(self, mamba, d_model):
        super(SSM_Layer, self).__init__()
        self.mamba = mamba
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = x + self.mamba(x)
        x = self.norm(x)
        x = x.permute(0,2,1)

        return x

class HeadDetectionPicking(nn.Module):
    """Head of detection and phase-picking."""

    def __init__(
        self,
        feature_channels,
        layer_channels,
        layer_kernel_sizes,
        act_layer,
        norm_layer,
        out_act_layer=nn.Identity,
        out_channels=1,
        **kwargs,
    ):
        super().__init__()

        assert len(layer_channels) == len(layer_kernel_sizes)

        self.depth = len(layer_channels)

        self.up_layers = nn.ModuleList()

        for i, (inc, outc, kers) in enumerate(
            zip(
                [feature_channels] + layer_channels[:-1],
                layer_channels[:-1] + [out_channels * 2],
                layer_kernel_sizes,
            )
        ):
            conv = nn.Conv1d(in_channels=inc, out_channels=outc, kernel_size=kers)
            norm = norm_layer(outc)
            act = act_layer()

            self.up_layers.append(
                nn.Sequential(
                    OrderedDict([("conv", conv), ("norm", norm), ("act", act)])
                )
            )

        self.out_conv = nn.Conv1d(
            in_channels=out_channels * 2,
            out_channels=out_channels,
            kernel_size=7,
            padding=3,
        )
        self.out_act = out_act_layer()

    def _upsampling_sizes(self, in_size: int, out_size: int):
        sizes = [out_size] * self.depth
        factor = (out_size / in_size) ** (1 / self.depth)
        for i in range(self.depth - 2, -1, -1):
            sizes[i] = int(sizes[i + 1] / factor)
        return sizes

    def forward(self, x, x0):
        N, C, L = x.size()
        up_sizes = self._upsampling_sizes(in_size=L, out_size=x0.size(-1))
        for i, layer in enumerate(self.up_layers):
            upsize = up_sizes[i]
            x = F.interpolate(x, size=upsize, mode="linear")
            x = _auto_pad_1d(x, layer.conv.kernel_size[0], layer.conv.stride[0])
            x = layer(x)

        x = self.out_conv(x)
        x = self.out_act(x)
        return x

class HeadRegression(nn.Module):
    """Head of regression."""

    def __init__(self, feature_channels, out_act_layer, **kwargs):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(1, -1)
        self.lin = nn.Linear(feature_channels , 1)
        self.out_act = out_act_layer()

    def forward(self, x, _: torch.Tensor = None):
        x = self.pool(x)
        x = self.flatten(x)
        x = self.lin(x)
        x = self.out_act(x)
        return x


class EffiSeisM(nn.Module):
    """
    Seismic Mamba model
    """
    def __init__(
            self,
            output_head=HeadRegression,
            in_channels=3,
            SSC_channels=[16, 8, 16, 16],
            SSC_kernel_sizes=[11, 5, 5, 7],
            SSC_strides=[2, 1, 1, 2],
            layer_blocks=[2, 2, 2, 2],
            layer_channels=[16, 24, 32, 64],
            mamba_blocks=[1, 1, 1, 1],
            stage_aggr_ratios=[2, 2, 2,2],
            head_dims=[8, 8, 8, 16],
            adaptive_kernel_sizes=[3, 5],
            mlp_ratio=2,
            mlp_drop_rate=0.2,
            path_drop_rate=0.2,
            mlp_bias=True,
            act_layer=nn.GELU,
            norm_layer=nn.BatchNorm1d,
            use_checkpoint=False,
            pool_size=4,
            **kwargs,
    ):



        super().__init__()
        assert len(SSC_channels) == len(SSC_kernel_sizes) == len(SSC_strides)

        self.use_checkpoint = use_checkpoint
        self.pool_size = pool_size

        self.SSC = nn.Sequential(
            *[
                MultiScale_Feature_Extractor(
                    in_dim=inc,
                    out_dim=outc,
                    kernel_size=kers,
                    stride=strd,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                )
                for inc, outc, kers, strd in zip(
                    [in_channels] + SSC_channels[:-1],
                    SSC_channels,
                    SSC_kernel_sizes,
                    SSC_strides,
                )
            ]
        )

        pdprs = [x.item() for x in torch.linspace(0, path_drop_rate, sum(layer_blocks))]

        self.encoder_layers = nn.ModuleList()
        for i, (
            num_blocks,
            inc,
            lc,
            aggr_ratio,
            head_dim,
            num_mamba
        ) in enumerate(
            zip(
                layer_blocks,
                SSC_channels[-1:] + layer_channels,
                layer_channels,
                stage_aggr_ratios,
                head_dims,
                mamba_blocks
            )
        ):
            layer_modules = []
            stage_aggr = LocalAwareAggregationBlock(
                in_dim=inc,
                out_dim=lc,
                kernel_size=aggr_ratio,
                norm_layer=norm_layer,
            )
            layer_modules.append(stage_aggr)


            for j in range(num_blocks):
                pdpr = pdprs[sum(layer_blocks[:i] + [j])]
                if j >= num_blocks - num_mamba:
                    block = SSM_Layer(
                        Mamba(d_model=lc),
                        d_model=lc
                    )
                else:
                    #
                    block = FSC(
                        io_dim=lc,
                        groups=lc // head_dim,  # * 2**i,
                        kernel_sizes=adaptive_kernel_sizes,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                    )
                layer_modules.append(block)

            self.encoder_layers.append(nn.Sequential(*layer_modules))

        if (output_head in [HeadDetectionPicking]) or (
            isinstance(output_head, partial)
            and (output_head.func in [HeadDetectionPicking])
        ):
            out_layer_channels = []
            out_layer_kernel_sizes = []
            for channel, kernel, stride in zip(
                [in_channels] + SSC_channels + layer_channels[:-1],
                SSC_kernel_sizes
                + [max(adaptive_kernel_sizes)] * len(layer_channels),
                SSC_strides + stage_aggr_ratios,
            ):
                if stride > 1:
                    out_layer_channels.insert(0, channel)
                    out_layer_kernel_sizes.insert(0, kernel)

            self.out_head = output_head(
                in_channels=in_channels,
                feature_channels=layer_channels[-1],
                layer_channels=out_layer_channels,
                layer_kernel_sizes=out_layer_kernel_sizes,
                act_layer=act_layer,
                norm_layer=norm_layer,
                path_drop_rate=path_drop_rate,
                mlp_drop_rate=mlp_drop_rate,
                mlp_ratio=mlp_ratio,
                mlp_bias=mlp_bias,
            )

        else:
            self.out_head = output_head(
                feature_channels=layer_channels[-1],
                out_act_layer=partial(
                    ScaledActivation, act_layer=nn.Sigmoid, scale_factor=8
                ),
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(
            m, (nn.BatchNorm1d, nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm1d)
        ):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)


    def forward(self,x):
        x_input = x
        x = self.SSC(x)  # (B,C,L)

        # Basic layers

        for layer in self.encoder_layers:
            if self.use_checkpoint and not (
                torch.jit.is_tracing() or torch.jit.is_scripting()
            ):
                x = checkpoint.checkpoint(layer, x)
            else:
                x = layer(x)

        # Output head
        x = self.out_head(x,x_input)
        return x

@register_model
def EffiSeisM_emg(**kwargs):
    """
    Magnitude estimation using SeismicMamba model.
    """
    model = EffiSeisM(
        output_head=partial(
            HeadRegression,
            out_act_layer=partial(
                ScaledActivation, act_layer=nn.Sigmoid, scale_factor=8
            ),
        ),
        **kwargs,
    )
    return model


@register_model
def EffiSeisM_dpk(**kwargs):
    model = EffiSeisM(
        output_head=partial(
            HeadDetectionPicking, out_act_layer=nn.Sigmoid, out_channels=3
        ),
        **kwargs,
    )
    return model



