import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch



class ResNet_Block(nn.Module):
    def __init__(self, dim):

        super(ResNet_Block, self).__init__()
        self.conv_block = nn.Sequential(nn.ReflectionPad2d(padding=1),
                              nn.Conv2d(dim, dim, 3, padding=0, bias=True),
                              nn.InstanceNorm2d(num_features=dim),
                              nn.ReLU(),
                              nn.ReflectionPad2d(padding=1),
                              nn.Conv2d(dim, dim, 3, padding=0, bias=True))

        self.norm = nn.InstanceNorm2d(dim)

    def forward(self, input):

        #return nn.ReLU(self.norm(input + self.conv_block(input)))
        return input + self.conv_block(input)


class Cycle_Generator(nn.Module):

    def __init__(self, dim = 64, res_blocks = 9):
        super(Cycle_Generator, self).__init__()

        # First Layer
        layer = [nn.ReflectionPad2d(padding=3),
                 nn.Conv2d(3, dim, 7, 1, 0), nn.InstanceNorm2d(dim), nn.ReLU(inplace=True),
                 nn.Conv2d(dim, 2*dim, 3, 2, 1), nn.InstanceNorm2d(dim), nn.ReLU(inplace=True),
                 nn.Conv2d(2*dim, 4*dim, 3, 2, 1), nn.InstanceNorm2d(dim), nn.ReLU(inplace=True)]

        # Append ResNet Blocks
        for i in range(int(res_blocks)):
            layer.append(ResNet_Block(4*dim))

        # Last Layer, using Bilinear Upsampling
        #upsample1 = nn.Upsample(size=(4 * dim, 4 * dim),scale_factor=2, mode='bilinear', align_corners=True) # align Corners auch Ausschaltbar
        #upsample2 = nn.Upsample(size=(2 * dim, 2 * dim), scale_factor=2, mode='bilinear', align_corners=True)
        #upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        #upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        upsample1 = nn.ConvTranspose2d(4*dim, 8*dim, 3, 1, 1)
        upsample2 = nn.ConvTranspose2d(2*dim, 4*dim, 3, 1, 1) #nn.PixelShuffle(2)
        first_upsampling = upsample1
        second_upsampling = upsample2
        layer.extend([first_upsampling, nn.PixelShuffle(2), nn.InstanceNorm2d(2*dim), nn.ReLU(True),
                      second_upsampling, nn.PixelShuffle(2), nn.InstanceNorm2d(dim), nn.ReLU(True),
                      nn.ReflectionPad2d(3),
                      nn.Conv2d(dim, 3, 7, 1, 0),
                      ])    # Linear activation (no activation) used

        self.gen = nn.Sequential(*layer)

    def forward(self, input):

        return self.gen(input)

input_nc = 3
ndf = 64

class Cycle_Discriminator(nn.Module):
    def __init__(self):
        super(Cycle_Discriminator, self).__init__()

        self.disc = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            # 3 Layers
            nn.Conv2d(ndf, ndf * 2, kernel_size=4,stride=2,padding=1, bias=True),
            nn.InstanceNorm2d(ndf *2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, input):

        return  self.disc(input)



def init_weights(network, m):
    gain = 0.02
    def init_function(m):
        init.normal_(m.weight, 0.0, gain)

    network.apply(init_function())

