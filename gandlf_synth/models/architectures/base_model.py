# -*- coding: utf-8 -*-
"""All Models in GANDLF are to be derived from this base class code."""
import warnings
from typing import Type

from torch import nn
from acsconv.converters import ACSConverter, Conv3dConverter, SoftACSConverter

from GANDLF.utils import get_linear_interpolation_mode
from GANDLF.utils.generic import checkPatchDimensions
from GANDLF.utils.modelbase import get_modelbase_final_layer
from GANDLF.models.seg_modules.average_pool import (
    GlobalAveragePooling3D,
    GlobalAveragePooling2D,
)
from gandlf_synth.models.configs.config_abc import AbstractModelConfig


class ModelBase(nn.Module):
    """
    This is the base model class that all other architectures will need to derive from
    """

    AVAILABLE_CONVERTERS = ["soft", "acs", "conv3d"]

    def __init__(self, model_config: Type[AbstractModelConfig]):
        """
        This defines all defaults that the model base uses

        Args:
            model_config (ModelConfig): The model configuration object.
        """
        super(ModelBase, self).__init__()
        self.model_name = model_config.model_name
        self.n_dimensions = model_config.n_dimensions
        self.n_channels = model_config.n_channels
        self.amp = model_config.amp
        self.norm_type = model_config.norm_type
        self.linear_interpolation_mode = get_linear_interpolation_mode(
            self.n_dimensions
        )
        if hasattr(model_config, "n_classes"):
            self.n_classes = model_config.n_classes
        # based on dimensionality, the following need to defined:
        # convolution, batch_norm, instancenorm, dropout
        assert self.n_dimensions in [
            2,
            3,
        ], f"Only 2D and 3D models are supported, but requested {self.n_dimensions}D."
        if self.n_dimensions == 2:
            self.Conv = nn.Conv2d
            self.ConvTranspose = nn.ConvTranspose2d
            self.InstanceNorm = nn.InstanceNorm2d
            self.Dropout = nn.Dropout2d
            self.BatchNorm = nn.BatchNorm2d
            self.MaxPool = nn.MaxPool2d
            self.AvgPool = nn.AvgPool2d
            self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d
            self.AdaptiveMaxPool = nn.AdaptiveMaxPool2d
            self.GlobalAvgPool = GlobalAveragePooling2D
            self.Norm = self.get_norm_type(self.norm_type.lower(), self.n_dimensions)
            self.converter = None

        elif self.n_dimensions == 3:
            self.Conv = nn.Conv3d
            self.ConvTranspose = nn.ConvTranspose3d
            self.InstanceNorm = nn.InstanceNorm3d
            self.Dropout = nn.Dropout3d
            self.BatchNorm = nn.BatchNorm3d
            self.MaxPool = nn.MaxPool3d
            self.AvgPool = nn.AvgPool3d
            self.AdaptiveAvgPool = nn.AdaptiveAvgPool3d
            self.AdaptiveMaxPool = nn.AdaptiveMaxPool3d
            self.GlobalAvgPool = GlobalAveragePooling3D
            self.Norm = self.get_norm_type(self.norm_type.lower(), self.n_dimensions)

            # define 2d to 3d model converters
            converter_type = model_config.converter_type.lower()

            self.converter = SoftACSConverter
            if converter_type == "acs":
                self.converter = ACSConverter
            elif converter_type == "conv3d":
                self.converter = Conv3dConverter
            else:
                warnings.warn(
                    f"ASC Converter type {converter_type} not found. Using `soft` converter."
                )

    def get_final_layer(self, final_convolution_layer: str) -> nn.Module:
        return get_modelbase_final_layer(final_convolution_layer)

    def get_norm_type(self, norm_type: str, dimensions: int) -> nn.Module:
        """
        This function gets the normalization type for the model.

        Args:
            norm_type (str): Normalization type as a string.
            dimensions (int): The dimensionality of the model.

        Returns:
            _InstanceNorm or _BatchNorm: The normalization type for the model.
        """
        if dimensions == 3:
            if norm_type == "batch":
                norm_type = nn.BatchNorm3d
            elif norm_type == "instance":
                norm_type = nn.InstanceNorm3d
            else:
                norm_type = None
        elif dimensions == 2:
            if norm_type == "batch":
                norm_type = nn.BatchNorm2d
            elif norm_type == "instance":
                norm_type = nn.InstanceNorm2d
            else:
                norm_type = None

        return norm_type

    def model_depth_check(self, parameters: dict) -> int:
        """
        This function checks if the patch size is large enough for the model.

        Args:
            parameters (dict): The entire set of parameters for the model.

        Returns:
            int: The model depth to use.
        """
        model_depth = checkPatchDimensions(
            parameters["patch_size"], numlay=parameters["model"]["depth"]
        )

        common_msg = "The patch size is not large enough for desired depth. It is expected that each dimension of the patch size is divisible by 2^i, where i is in a integer greater than or equal to 2."
        assert model_depth >= 2, common_msg

        if model_depth != parameters["model"]["depth"] and model_depth >= 2:
            print(common_msg + " Only the first %d layers will run." % model_depth)

        return model_depth
