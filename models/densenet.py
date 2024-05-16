import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import densenet121

# class DenseNet(nn.Module):
#     def __init__(self, **kwargs):
#         super().__init__()
#         self.densenet = densenet121(**kwargs)
#         self.classification_activation = torch.sigmoid
#     def forward(self, x):
#         x = self.densenet(x)
#         return self.classification_activation(x)

import re
from collections import OrderedDict
from typing import Callable, Sequence, Type, Union

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

from monai.networks.layers.factories import Conv, Dropout, Pool
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils.module import look_up_option

__all__ = [
    "DenseNet",
    "Densenet",
    "DenseNet121",
    "densenet121",
    "Densenet121",
    "DenseNet169",
    "densenet169",
    "Densenet169",
    "DenseNet201",
    "densenet201",
    "Densenet201",
    "DenseNet264",
    "densenet264",
    "Densenet264",
]


class _DenseLayer(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        growth_rate: int,
        bn_size: int,
        dropout_prob: float,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            in_channels: number of the input channel.
            growth_rate: how many filters to add each layer (k in paper).
            bn_size: multiplicative factor for number of bottle neck layers.
                (i.e. bn_size * k features in the bottleneck layer)
            dropout_prob: dropout rate after each dense layer.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()

        out_channels = bn_size * growth_rate
        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        dropout_type: Callable = Dropout[Dropout.DROPOUT, spatial_dims]

        self.layers = nn.Sequential()

        self.layers.add_module("norm1", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels))
        self.layers.add_module("relu1", get_act_layer(name=act))
        self.layers.add_module("conv1", conv_type(in_channels, out_channels, kernel_size=1, bias=False))

        self.layers.add_module("norm2", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=out_channels))
        self.layers.add_module("relu2", get_act_layer(name=act))
        self.layers.add_module("conv2", conv_type(out_channels, growth_rate, kernel_size=3, padding=1, bias=False))

        if dropout_prob > 0:
            self.layers.add_module("dropout", dropout_type(dropout_prob))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.layers(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        layers: int,
        in_channels: int,
        bn_size: int,
        growth_rate: int,
        dropout_prob: float,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            layers: number of layers in the block.
            in_channels: number of the input channel.
            bn_size: multiplicative factor for number of bottle neck layers.
                (i.e. bn_size * k features in the bottleneck layer)
            growth_rate: how many filters to add each layer (k in paper).
            dropout_prob: dropout rate after each dense layer.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()
        for i in range(layers):
            layer = _DenseLayer(spatial_dims, in_channels, growth_rate, bn_size, dropout_prob, act=act, norm=norm)
            in_channels += growth_rate
            self.add_module("denselayer%d" % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            in_channels: number of the input channel.
            out_channels: number of the output classes.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        pool_type: Callable = Pool[Pool.AVG, spatial_dims]

        self.add_module("norm", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels))
        self.add_module("relu", get_act_layer(name=act))
        self.add_module("conv", conv_type(in_channels, out_channels, kernel_size=1, bias=False))
        self.add_module("pool", pool_type(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    """
    Densenet based on: `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993.pdf>`_.
    Adapted from PyTorch Hub 2D version: https://pytorch.org/vision/stable/models.html#id16.
    This network is non-deterministic When `spatial_dims` is 3 and CUDA is enabled. Please check the link below
    for more details:
    https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html#torch.use_deterministic_algorithms

    Args:
        spatial_dims: number of spatial dimensions of the input image.
        in_channels: number of the input channel.
        out_channels: number of the output classes.
        init_features: number of filters in the first convolution layer.
        growth_rate: how many filters to add each layer (k in paper).
        block_config: how many layers in each pooling block.
        bn_size: multiplicative factor for number of bottle neck layers.
            (i.e. bn_size * k features in the bottleneck layer)
        act: activation type and arguments. Defaults to relu.
        norm: feature normalization type and arguments. Defaults to batch norm.
        dropout_prob: dropout rate after each dense layer.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        feature_channels: int,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 24, 16),
        bn_size: int = 4,
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        norm: Union[str, tuple] = "batch",
        dropout_prob: float = 0.0,
    ) -> None:

        super().__init__()

        conv_type: Type[Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]] = Conv[Conv.CONV, spatial_dims]
        pool_type: Type[Union[nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d]] = Pool[Pool.MAX, spatial_dims]
        avg_pool_type: Type[Union[nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d]] = Pool[
            Pool.ADAPTIVEAVG, spatial_dims
        ]

        self.backbone = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", conv_type(in_channels, init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=init_features)),
                    ("relu0", get_act_layer(name=act)),
                    ("pool0", pool_type(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        in_channels = init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                spatial_dims=spatial_dims,
                layers=num_layers,
                in_channels=in_channels,
                bn_size=bn_size,
                growth_rate=growth_rate,
                dropout_prob=dropout_prob,
                act=act,
                norm=norm,
            )
            self.backbone.add_module(f"denseblock{i + 1}", block)
            in_channels += num_layers * growth_rate
            if i == len(block_config) - 1:
                self.backbone.add_module(
                    "norm5", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels)
                )
            else:
                _out_channels = in_channels // 2
                trans = _Transition(
                    spatial_dims, in_channels=in_channels, out_channels=_out_channels, act=act, norm=norm
                )
                self.backbone.add_module(f"transition{i + 1}", trans)
                in_channels = _out_channels

        ## MSK Custom additions
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("relu", get_act_layer(name=act)),
                    ("pool", avg_pool_type(1)),
                    ("flatten", nn.Flatten(1)),
                    ("feature_layer", nn.Linear(in_channels, feature_channels)),
                    # test addition
                    # ('norm_layer', nn.BatchNorm1d(feature_channels)),

                    ("dropout", nn.Dropout(dropout_prob))
                ]
            )
        )

        # pooling and classification
        self.class_layers = nn.Sequential(
            OrderedDict(
                [
                    ("out", nn.Linear(feature_channels, out_channels)),
                ]
            )
        )

        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight))
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(torch.as_tensor(m.bias), 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.features(x)
        x = self.class_layers(x)
        return x



def _load_state_dict(model: nn.Module, arch: str, progress: bool):
    """
    This function is used to load pretrained models.
    Adapted from PyTorch Hub 2D version: https://pytorch.org/vision/stable/models.html#id16.

    """
    model_urls = {
        "densenet121": "https://download.pytorch.org/models/densenet121-a639ec97.pth",
        "densenet169": "https://download.pytorch.org/models/densenet169-b2777c0a.pth",
        "densenet201": "https://download.pytorch.org/models/densenet201-c1103571.pth",
    }
    model_url = look_up_option(arch, model_urls, None)
    if model_url is None:
        raise ValueError(
            "only 'densenet121', 'densenet169' and 'densenet201' are supported to load pretrained weights."
        )

    pattern = re.compile(
        r"^(.*denselayer\d+)(\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + ".layers" + res.group(2) + res.group(3)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]

    model_dict = model.state_dict()
    state_dict = {
        k: v for k, v in state_dict.items() if (k in model_dict) and (model_dict[k].shape == state_dict[k].shape)
    }
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)


class DenseNet121(DenseNet):
    """DenseNet121 with optional pretrained support when `spatial_dims` is 2."""

    def __init__(
        self,
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 24, 16),
        pretrained: bool = False,
        progress: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(init_features=init_features, growth_rate=growth_rate, block_config=block_config, **kwargs)
        if pretrained:
            if kwargs["spatial_dims"] > 2:
                raise NotImplementedError(
                    "Parameter `spatial_dims` is > 2 ; currently PyTorch Hub does not"
                    "provide pretrained models for more than two spatial dimensions."
                )
            _load_state_dict(self, "densenet121", progress)

class TinyDensenet(DenseNet):
    """DenseNet121 with optional pretrained support when `spatial_dims` is 2."""

    def __init__(
        self,
        # init_features: int = 16,
        # growth_rate: int = 8,
        # block_config: Sequence[int] = (1,1),
        # block_config: Sequence[int] = (2, 2, 2),
        init_features: int = 64,
        growth_rate: int = 32,
        block_config: Sequence[int] = (6, 12, 4),
        pretrained: bool = False,
        progress: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(init_features=init_features, growth_rate=growth_rate, block_config=block_config, **kwargs)
        if pretrained:
            if kwargs["spatial_dims"] > 2:
                raise NotImplementedError(
                    "Parameter `spatial_dims` is > 2 ; currently PyTorch Hub does not"
                    "provide pretrained models for more than two spatial dimensions."
                )
            _load_state_dict(self, "densenet121", progress)

class TinyCNN(nn.Module):

    def __init__(self, in_channels=1, out_channels=3, feature_channels=12, dropout_prob=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_channels = feature_channels
        self.dropout_prob = dropout_prob

        self.conv1 = nn.Conv3d(self.in_channels, 64, 3)
        self.bn1 = nn.BatchNorm3d(64)
        self.drop1 = nn.Dropout3d(dropout_prob)

        self.conv2 = nn.Conv3d(64, 32, 3)
        self.bn2 = nn.BatchNorm3d(32)
        self.drop2 = nn.Dropout3d(dropout_prob)

        self.conv3 = nn.Conv3d(32, 32, 3)
        self.bn3 = nn.BatchNorm3d(32)
        self.drop3 = nn.Dropout3d(dropout_prob)

        self.conv4 = nn.Conv3d(32, 16, 3)
        self.bn4 = nn.BatchNorm3d(16)
        self.drop4 = nn.Dropout3d(dropout_prob)

        self.conv5 = nn.Conv3d(16, 16, 3)
        self.bn5 = nn.BatchNorm3d(16)
        self.drop5 = nn.Dropout3d(dropout_prob)

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.flatten = nn.Flatten(1)
        self.features = nn.Linear(16, self.feature_channels)
        self.drop_head = nn.Dropout(dropout_prob)
        self.out = nn.Linear(self.feature_channels, self.out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.drop3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.drop4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.drop5(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.features(x)
        x = self.dropout_head(x)
        x = self.out(x)

        return x 