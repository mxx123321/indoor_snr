import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

# def ConvMixer(input_dim, dim, depth, kernel_size=9, patch_size=7, n_classes=3):
#     return nn.Sequential(
#         nn.Conv2d(input_dim, dim, kernel_size=patch_size, stride=patch_size),
#         nn.GELU(),
#         nn.BatchNorm2d(dim),
#         *[nn.Sequential(
#                 Residual(nn.Sequential(
#                     nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
#                     nn.GELU(),
#                     nn.BatchNorm2d(dim)
#                 )),
#                 nn.Conv2d(dim, dim, kernel_size=1),
#                 nn.GELU(),
#                 nn.BatchNorm2d(dim)
#         ) for i in range(depth)],
#         nn.AdaptiveAvgPool2d((1,1)),
#         nn.Flatten(),
#         nn.Linear(dim, n_classes)
#     )

class ConvMixerBlock(nn.Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.conv_mix = nn.Sequential(
            Residual(nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size, groups=dim, padding='same'),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            )),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return self.conv_mix(x)

class ConvMixer(nn.Module):
    def __init__(self, input_dim, dim, depth, kernel_size=9, patch_size=7, n_classes=3):
        super().__init__()
        self.initial_conv = nn.Conv2d(input_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.gelu = nn.GELU()
        self.bn = nn.BatchNorm2d(dim)

        self.mixer_blocks = nn.Sequential(*[ConvMixerBlock(dim, kernel_size) for _ in range(depth)])

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
            )
        
        self.last = nn.Linear(dim, n_classes)

    def forward(self, x):
        x = self.bn(self.gelu(self.initial_conv(x)))
        x = self.mixer_blocks(x)
        x1 = self.head(x)
        x = self.last(x1)
        return x,x1