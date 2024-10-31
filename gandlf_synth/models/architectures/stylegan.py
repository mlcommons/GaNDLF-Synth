import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class WeightScaledLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        """
        Weight scaled linear layer. The input is scaled proportionally to the number of
        input features before the linear operation.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scale = (2 / in_features) ** 0.5
        self.bias = self.linear.bias
        self.linear.bias = None

        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x * self.scale) + self.bias


class PixenNorm(nn.Module):
    def __init__(self):
        """
        Pixel-wise normalization layer. Normalizes each pixel in the input tensor across
        all channels.
        """
        super().__init__()
        self.epsilon = 1e-8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + self.epsilon)


class MappingNetwork(nn.Module):
    def __init__(self, z_dim: int, w_dim: int):
        """
        Mapping subnetwrok for StyleGAN. Maps the latent vector z to the intermediate
        latent space w.

        Args:
            z_dim (int): Dimensionality of the latent vector z.
            w_dim (int): Dimensionality of the intermediate latent space w.
        """

        super().__init__()
        self.mapping = nn.Sequential(
            PixenNorm(),
            WeightScaledLinear(z_dim, w_dim),
            nn.ReLU(),
            WeightScaledLinear(w_dim, w_dim),
            nn.ReLU(),
            WeightScaledLinear(w_dim, w_dim),
            nn.ReLU(),
            WeightScaledLinear(w_dim, w_dim),
            nn.ReLU(),
            WeightScaledLinear(w_dim, w_dim),
            nn.ReLU(),
            WeightScaledLinear(w_dim, w_dim),
            nn.ReLU(),
            WeightScaledLinear(w_dim, w_dim),
            nn.ReLU(),
            WeightScaledLinear(w_dim, w_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mapping(x)


class AdaptiveInstanceNormalization(nn.Module):
    def __init__(self, channels: int, w_dim: int):
        """
        Adaptive instance normalization layer. Applies instance normalization to the input
        tensor and then scales and biases it using the style vector w.
        """

        super().__init__()
        self.instance_norm = nn.InstanceNorm2d(channels)
        self.style_scale = WeightScaledLinear(w_dim, channels)
        self.style_bias = WeightScaledLinear(w_dim, channels)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x = self.instance_norm(x)
        style_scale = self.style_scale(w).unsqueeze(2).unsqueeze(3)
        style_bias = self.style_bias(w).unsqueeze(2).unsqueeze(3)
        return style_scale * x + style_bias


class LearnableNoiseInjector(nn.Module):
    def __init__(self, channels: int):
        """
        Learnable noise injector layer. Adds the noise tensor to the input tensor and,
        controlling the amount of noise added using a learnable weight parameter.

        Args:
            channels (int): Number of channels in the input tensor.
        """

        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)
        return x + self.weight + noise


class WeightScaledConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        """
        Weight scaled 2D convolutional layer. The input is scaled proportionally to the
        number of input channels before the convolution operation.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride of the convolution operation.
            padding (int): Padding of the input tensor.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (2 / (in_channels * (kernel_size**2))) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """
        Convolutional block for the discriminator network. Consists of two weight-scaled
        convolutional layers with leaky ReLU activation functions.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """

        super().__init__()
        self.conv1 = WeightScaledConv2d(in_channels, out_channels)
        self.conv2 = WeightScaledConv2d(out_channels, out_channels)
        self.leaky = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.leaky(self.conv1(x))
        x = self.leaky(self.conv2(x))
        return x


class GeneratorBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, w_dim: int):
        """
        Generator block for the generator network. Consists of two weight-scaled
        convolutional layers with adaptive instance normalization and leaky ReLU.

        Args:
            in_channel (int): Number of input channels.
            out_channel (int): Number of output channels.
            w_dim (int): Dimensionality of the intermediate latent space w.

        """
        super().__init__()
        self.conv1 = WeightScaledConv2d(in_channel, out_channel)
        self.conv2 = WeightScaledConv2d(out_channel, out_channel)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.inject_noise1 = LearnableNoiseInjector(out_channel)
        self.inject_noise2 = LearnableNoiseInjector(out_channel)
        self.adain1 = AdaptiveInstanceNormalization(out_channel, w_dim)
        self.adain2 = AdaptiveInstanceNormalization(out_channel, w_dim)

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x = self.adain1(self.leaky(self.inject_noise1(self.conv1(x))), w)
        x = self.adain2(self.leaky(self.inject_noise2(self.conv2(x))), w)
        return x


class StyleGanGenerator(nn.Module):
    def __init__(
        self,
        z_dim: int,
        w_dim: int,
        in_channels: int,
        img_channels: int,
        progressive_layers_scaling_factors: List[float],
    ):
        """
        StyleGAN generator network. Generates images from the latent vector z.

        Args:
            z_dim (int): Dimensionality of the latent vector z.
            w_dim (int): Dimensionality of the intermediate latent space w.
            in_channels (int): Number of input channels.
            img_channels (int): Number of output channels.
            progressive_layers_scaling_factors (List[float]): List of scaling factors for
        channels in consecutive convolutional layers.
        """
        super().__init__()
        self.starting_cte = nn.Parameter(torch.ones(1, in_channels, 4, 4))
        self.map = MappingNetwork(z_dim, w_dim)
        self.initial_adain1 = AdaptiveInstanceNormalization(in_channels, w_dim)
        self.initial_adain2 = AdaptiveInstanceNormalization(in_channels, w_dim)
        self.initial_noise1 = LearnableNoiseInjector(in_channels)
        self.initial_noise2 = LearnableNoiseInjector(in_channels)
        self.initial_conv = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )
        self.leaky = nn.LeakyReLU(0.2, inplace=True)

        self.initial_rgb = WeightScaledConv2d(
            in_channels, img_channels, kernel_size=1, stride=1, padding=0
        )
        self.prog_blocks, self.rgb_layers = (
            nn.ModuleList([]),
            nn.ModuleList([self.initial_rgb]),
        )

        for i in range(len(progressive_layers_scaling_factors) - 1):
            conv_in_c = int(in_channels * progressive_layers_scaling_factors[i])
            conv_out_c = int(in_channels * progressive_layers_scaling_factors[i + 1])
            self.prog_blocks.append(GeneratorBlock(conv_in_c, conv_out_c, w_dim))
            self.rgb_layers.append(
                WeightScaledConv2d(
                    conv_out_c, img_channels, kernel_size=1, stride=1, padding=0
                )
            )

    def fade_in(
        self, alpha: float, upscaled: torch.Tensor, generated: torch.Tensor
    ) -> torch.Tensor:
        """
        Fades in the upscaled image with the generated image using the alpha parameter.

        Args:
            alpha (float): Alpha parameter for fading in the images.
            upscaled (torch.Tensor): Upscaled image tensor.
            generated (torch.Tensor): Generated image tensor.
        """
        return torch.tanh(alpha * generated + (1 - alpha) * upscaled)

    def forward(self, noise: torch.Tensor, alpha: float, steps: int) -> torch.Tensor:
        w = self.map(noise)
        x = self.initial_adain1(self.initial_noise1(self.starting_cte), w)
        x = self.initial_conv(x)
        out = self.initial_adain2(self.leaky(self.initial_noise2(x)), w)

        if steps == 0:
            return self.initial_rgb(x)

        for step in range(steps):
            upscaled = F.interpolate(out, scale_factor=2, mode="bilinear")
            out = self.prog_blocks[step](upscaled, w)

        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        final_out = self.rgb_layers[steps](out)

        return self.fade_in(alpha, final_upscaled, final_out)


class StyleGanDiscriminator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        img_channels: int,
        progressive_layers_scaling_factors: List[float],
    ):
        """
        StyleGAN discriminator network.

        Args:
            in_channels (int): Number of input channels.
            img_channels (int): Number of output channels.
            progressive_layers_scaling_factors (List[float]): List of scaling factors for
        channels in consecutive convolutional layers.
        """
        super().__init__()
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)
        for i in range(len(progressive_layers_scaling_factors) - 1, 0, -1):
            conv_in = int(in_channels * progressive_layers_scaling_factors[i])
            conv_out = int(in_channels * progressive_layers_scaling_factors[i - 1])
            self.prog_blocks.append(ConvBlock(conv_in, conv_out))
            self.rgb_layers.append(
                WeightScaledConv2d(
                    img_channels, conv_in, kernel_size=1, stride=1, padding=0
                )
            )

        self.initial_conv = WeightScaledConv2d(
            img_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.rgb_layers.append(self.initial_rgb)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.final_block = nn.Sequential(
            # +1 to in_channels because we concatenate from MiniBatch std
            WeightScaledConv2d(in_channels + 1, in_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            WeightScaledConv2d(
                in_channels, in_channels, kernel_size=4, padding=0, stride=1
            ),
            nn.LeakyReLU(0.2),
            WeightScaledConv2d(in_channels, 1, kernel_size=1, padding=0, stride=1),
        )

    def fade_in(self, alpha: float, downscaled: torch.Tensor, out: torch.Tensor):
        """
        Fades in the downscaled image with the output image using the alpha parameter.

        Args:
            alpha (float): Alpha parameter for fading in the images.
            downscaled (torch.Tensor): Downscaled image tensor.
            out (torch.Tensor): Output image tensor.
        """

        return alpha * out + (1 - alpha) * downscaled

    def minibatch_std(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the minibatch standard deviation of the input tensor and concatenates it
        to the input tensor.
        """
        batch_statistics = (
            torch.std(x, dim=0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        )
        return torch.cat([x, batch_statistics], dim=1)

    def forward(self, x: torch.Tensor, alpha: float, steps: int) -> torch.Tensor:
        cur_step = len(self.prog_blocks) - steps
        out = self.leaky(self.rgb_layers[cur_step](x))

        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(out.shape[0], -1)

        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))
        out = self.fade_in(alpha, downscaled, out)

        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)

        out = self.minibatch_std(out)
        return self.final_block(out).view(out.shape[0], -1)
