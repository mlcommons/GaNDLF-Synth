"""Implementation of DCGAN model."""
from warnings import warn
from typing import Type, List

import torch
import torch.nn as nn

from gandlf_synth.models.architectures.base_model import ModelBase
from gandlf_synth.models.configs.config_abc import AbstractModelConfig


class _GeneratorDCGAN(nn.Module):
    """Generator for the DCGAN."""

    def __init__(
        self,
        output_size: List[int],
        n_dimensions: int,
        latent_vector_dim: int,
        num_output_channels: int,
        growth_rate: int,
        gen_init_channels: int,
        norm: nn.Module,
        conv: nn.Module,
    ) -> None:
        """
        Initializes a new instance of the _GneratorDCGAN class.

        Args:
            output_size (List[int]): The size of the generated output.
            n_dimensions (int): The dimensionality of the input and output.
            latent_vector_dim (int): The dimension of the latent vector
        to be used as input to the generator.
            num_output_channels (int): The number of output channels in
        the generated image.
            growth_rate (int): The growth rate of the number of hidden
        features in the consecutive layers of the generator. Note that
        in this case the growth will be in reverse order, i.e. the number
        of channels will DECREASE by this amount.
            gen_init_channels (int): Initial number of channels in the
        generator, which is scaled by the growth rate in the subsequent
        layers.
            norm (torch.nn.module): A normalization layer subclassing
        torch.nn.Module (i.e. nn.BatchNorm2d)
            conv (torch.nn.module): A convolutional layer subclassing
        torch.nn.Module. Note that in this case this
        should be a transposed convolution (i.e. nn.ConvTranspose2d).
        """
        super().__init__()
        self.feature_extractor = nn.Sequential()
        self.feature_extractor.add_module(
            "conv1t", conv(latent_vector_dim, gen_init_channels, 4, 1, 0, bias=False)
        )
        self.feature_extractor.add_module("norm1", norm(gen_init_channels))
        self.feature_extractor.add_module("relu1", nn.ReLU(inplace=True))
        self.feature_extractor.add_module(
            "conv2t",
            conv(
                gen_init_channels, gen_init_channels // growth_rate, 4, 2, 1, bias=False
            ),
        )
        self.feature_extractor.add_module(
            "norm2", norm(gen_init_channels // growth_rate)
        )
        self.feature_extractor.add_module("relu2", nn.ReLU(inplace=True))
        self.feature_extractor.add_module(
            "conv3t",
            conv(
                gen_init_channels // growth_rate,
                gen_init_channels // (growth_rate**2),
                4,
                2,
                1,
                bias=False,
            ),
        )
        self.feature_extractor.add_module(
            "norm3", norm(gen_init_channels // (growth_rate**2))
        )
        self.feature_extractor.add_module("relu3", nn.ReLU(inplace=True))
        self.feature_extractor.add_module(
            "conv4t",
            conv(
                gen_init_channels // (growth_rate**2),
                gen_init_channels // (growth_rate**3),
                4,
                2,
                1,
                bias=False,
            ),
        )
        self.feature_extractor.add_module(
            "norm4", norm(gen_init_channels // (growth_rate**3))
        )
        self.feature_extractor.add_module("relu4", nn.ReLU(inplace=True))

        self.feature_extractor.add_module(
            "conv5t",
            conv(
                gen_init_channels // (growth_rate**3),
                num_output_channels,
                4,
                2,
                1,
                bias=False,
            ),
        )
        feature_extractor_output_size = self._get_output_size_feature_extractor(
            self.feature_extractor, latent_vector_dim, n_dimensions
        )
        # if the output size of the feature extractor does not match
        # the output patch size, add an upsampling layer and a 1x1
        # convolution to match the output size and reparametrize the
        # interpoladed output
        if output_size != feature_extractor_output_size:
            self.feature_extractor.add_module(
                "upsample",
                nn.Upsample(
                    size=(output_size),
                    mode="bilinear" if n_dimensions == 2 else "trilinear",
                    align_corners=True,
                ),
            )
            # self.feature_extractor.add_module(
            #     "conv5",
            #     conv(num_output_channels, num_output_channels, 1, 1, bias=False),
            # )
            self.feature_extractor.add_module(
                "conv1_smooth",
                nn.Conv2d(num_output_channels, 128, 4, padding="same", bias=False),
            )
            self.feature_extractor.add_module("norm1_smooth", norm(128))
            self.feature_extractor.add_module("relu1_smooth", nn.ReLU(inplace=True))
            self.feature_extractor.add_module(
                "conv2_smooth",
                nn.Conv2d(128, num_output_channels, 3, padding="same", bias=False),
            )

        self.feature_extractor.add_module("tanh", nn.Tanh())

    @staticmethod
    def _get_output_size_feature_extractor(
        feature_extractor: nn.Module, latent_vector_dim: int, n_dimensions: int = 3
    ) -> int:
        """
        Determines the output size of the given module.

        Args:
            feature_extractor (nn.Module): The feature extractor module.
            latent_vector_dim (int): The dimension of the latent vector
        to be used as input to the generator.

        Returns:
            int: The output size of the feature extractor.
        """
        dummy_input_shape = [1, latent_vector_dim, 1, 1]
        if n_dimensions == 3:
            dummy_input_shape.append(1)
        dummy_input = torch.randn(dummy_input_shape)
        dummy_output = feature_extractor(dummy_input)
        return dummy_output.shape[2:]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator.

        Args:
            x (torch.Tensor): The latent vector to be used as input to
        the generator.

        Returns:
            torch.Tensor: The generated image.
        """
        out = self.feature_extractor(x)
        return out


class _DiscriminatorDCGAN(nn.Module):
    """Discriminator for the DCGAN."""

    def __init__(
        self,
        input_size: List[int],
        num_input_channels: int,
        growth_rate: int,
        disc_init_channels: int,
        slope: float,
        norm: nn.Module,
        conv: nn.Module,
    ) -> None:
        """
        Initializes a new instance of the _DiscriminatorDCGAN class.

        Args:
            input_size (List[int]): The size of the
        input patch.
            num_input_channels (int): The number of input channels in
        the image to be discriminated.
            growth_rate (int): The growth rate of the number of hidden
        features in the consecutive layers of the discriminator.
            disc_init_channels (int): Initial number of channels in the
        discriminator, which is scaled by the growth rate in the subsequent
        layers.
            drop_rate (float): The dropout rate in the classifier.
            slope (float): The slope of the LeakyReLU activation function.
            norm (torch.nn.module): A normalization layer subclassing
        torch.nn.Module (i.e. nn.BatchNorm2d)
            conv (torch.nn.module): A convolutional layer subclassing
        torch.nn.Module.
        """
        super().__init__()
        self.feature_extractor = nn.Sequential()
        self.classifier = nn.Sequential()
        self.feature_extractor.add_module(
            "conv1", conv(num_input_channels, disc_init_channels, 4, 2, 1, bias=False)
        )
        self.feature_extractor.add_module(
            "leaky_relu1", nn.LeakyReLU(slope, inplace=False)
        )
        self.feature_extractor.add_module(
            "conv2",
            conv(
                disc_init_channels,
                disc_init_channels * growth_rate,
                4,
                2,
                1,
                bias=False,
            ),
        )
        self.feature_extractor.add_module(
            "norm2", norm(disc_init_channels * growth_rate)
        )
        self.feature_extractor.add_module(
            "leaky_relu2", nn.LeakyReLU(slope, inplace=False)
        )
        self.feature_extractor.add_module(
            "conv3",
            conv(
                disc_init_channels * growth_rate,
                disc_init_channels * (growth_rate**2),
                4,
                2,
                1,
                bias=False,
            ),
        )
        self.feature_extractor.add_module(
            "norm3", norm(disc_init_channels * (growth_rate**2))
        )
        self.feature_extractor.add_module(
            "leaky_relu3", nn.LeakyReLU(slope, inplace=False)
        )
        self.feature_extractor.add_module(
            "conv4",
            conv(
                disc_init_channels * (growth_rate**2),
                disc_init_channels * (growth_rate**3),
                4,
                2,
                1,
                bias=False,
            ),
        )
        self.feature_extractor.add_module(
            "norm4", norm(disc_init_channels * (growth_rate**3))
        )
        self.feature_extractor.add_module(
            "leaky_relu4", nn.LeakyReLU(slope, inplace=False)
        )

        self.feature_extractor.add_module(
            "conv5",
            conv(disc_init_channels * (growth_rate**3), 1, 4, 1, 0, bias=False),
        )
        self.feature_extractor.add_module("flatten", nn.Flatten(start_dim=1))

        num_output_features = self._get_output_size_feature_extractor(
            self.feature_extractor, input_size, num_input_channels
        )
        self.classifier.add_module("linear1", nn.Linear(num_output_features, 1))
        self.classifier.add_module("sigmoid", nn.Sigmoid())

    @staticmethod
    def _get_output_size_feature_extractor(
        feature_extractor: nn.Module, input_size: List[int], n_channels: int = 1
    ) -> int:
        """
        Determines the output size of the feature extractor to
        initialize the classifier.

        Args:
            feature_extractor (nn.Module): The feature extractor module.
            input_size (List[int]): The size of the input size. This
        only includes (H, W, D) and not the number of channels (C).
            n_channels (int): The number of input channels in the image
        to be discriminated.

        Returns:
            int: The output size of the feature extractor.
        """
        dummy_input_shape = [1, n_channels, *input_size]
        dummy_input = torch.randn(dummy_input_shape)
        dummy_output = feature_extractor(dummy_input)
        return dummy_output.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator.

        Args:
            x (torch.Tensor): The image to be discriminated.

        Returns:
            torch.Tensor: The probability that the image is real.
        """
        features = self.feature_extractor(x)
        out = self.classifier(features)
        return out


class DCGAN(ModelBase):
    """
    DCGAN model class. This class implements the architecture and forward
    passes for the generator and discriminator subnetworks of the DCGAN.
    """

    def __init__(self, model_config: Type[AbstractModelConfig]) -> None:
        ModelBase.__init__(self, model_config)
        if self.Norm is None:
            warn("No normalization specified. Defaulting to BatchNorm", RuntimeWarning)
            self.Norm = self.BatchNorm
        self.generator = _GeneratorDCGAN(
<<<<<<< HEAD
            output_size=model_config.architecture["generator_output_size"],
=======
            output_size=model_config.input_shape,
>>>>>>> efa7a38 (Fixing assertion errors regardin input)
            latent_vector_dim=model_config.architecture["latent_vector_size"],
            growth_rate=model_config.architecture["growth_rate_generator"],
            gen_init_channels=model_config.architecture["init_channels_generator"],
            norm=self.Norm,
            conv=self.ConvTranspose,
            n_dimensions=self.n_dimensions,
            num_output_channels=self.n_channels,
        )
        self.discriminator = _DiscriminatorDCGAN(
            num_input_channels=self.n_channels,
            norm=self.Norm,
            conv=self.Conv,
<<<<<<< HEAD
            input_size=model_config.architecture["discriminator_input_size"],
=======
            input_size=model_config.input_shape,
>>>>>>> efa7a38 (Fixing assertion errors regardin input)
            growth_rate=model_config.architecture["growth_rate_discriminator"],
            disc_init_channels=model_config.architecture["init_channels_discriminator"],
            slope=model_config.architecture["leaky_relu_slope"],
        )
        # TODO this initialization is preventing the model from convergence
        # self._init_generator_weights(self.generator)
        # self._init_discriminator_weights(self.discriminator)

    def _init_generator_weights(self, generator: nn.Module) -> None:
        """
        Initializes the weights of the generator. This is mimicking the
        original implementation of the DCGAN.

        Args:
            generator (torch.nn.Module): The generator module.
        """
        for m in generator.modules():
            if isinstance(m, self.ConvTranspose):
                nn.init.normal_(m.weight, mean=0, std=0.02)
            elif isinstance(m, self.Norm):
                if m.weight is not None:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def _init_discriminator_weights(self, discriminator: nn.Module) -> None:
        """
        Initializes the weights of the discriminator. This is mimicking the
        original implementation of the DCGAN.

        Args:
            discriminator (torch.nn.Module): The discriminator module.
        """
        for m in discriminator.modules():
            if isinstance(m, self.ConvTranspose):
                nn.init.normal_(m.weight, mean=0, std=0.02)
            elif isinstance(m, self.Norm):
                if m.weight is not None:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def generator_forward(self, latent_vector: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the generator.

        Args:
            latent_vector (torch.Tensor): The latent vector to be used as
        input to the generator.

        Returns:
            torch.Tensor: The generated image.
        """
        return self.generator(latent_vector)

    def discriminator_forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the discriminator.

        Args:
            image (torch.Tensor): The image to be discriminated.

        Returns:
            torch.Tensor: The probability that the image is real.
        """
        return self.discriminator(image)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass, implemented simply as generator_forward.

        Args:
            x (torch.Tensor): The latent vector to be used as input to
        the generator.

        Returns:
            torch.Tensor: The generated image.
        """
        return self.generator_forward(x)
