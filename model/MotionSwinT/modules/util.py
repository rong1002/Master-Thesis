from torch import nn

import torch.nn.functional as F
import torch

from sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from .swin_transformer import swin_t

def kp2gaussian(kp, spatial_size, kp_variance):
    """
    Transform a keypoint into gaussian like representation
    """
    #keypoint T(pk), shape [batch_size, 10, 2] 10個keypoint
    mean = kp['value']
    #z 本地座標關鍵點 shape [height, width, 2]
    coordinate_grid = make_coordinate_grid(spatial_size, mean.type()) 
    number_of_leading_dimensions = len(mean.shape) - 1 #2
    shape = (1,) * number_of_leading_dimensions + coordinate_grid.shape #(1, 1, height, width, 2)
    coordinate_grid = coordinate_grid.view(*shape) #(1, 1, height, width, 2)
    repeats = mean.shape[:number_of_leading_dimensions] + (1, 1, 1) #(1, 10, 1, 1, 1)
    coordinate_grid = coordinate_grid.repeat(*repeats) #[1, 10, 64, 64, 2]

    # Preprocess kp shape
    shape = mean.shape[:number_of_leading_dimensions] + (1, 1, 2)
    mean = mean.view(*shape)

    mean_sub = (coordinate_grid - mean) #T(pk)-z

    out = torch.exp(-0.5 * (mean_sub ** 2).sum(-1) / kp_variance) #之所以會有0.5出現是因為將兩個通道相加需要求均值 
    #out.shape = 1, 10, 64, 64
    return out


def make_coordinate_grid(spatial_size, type):
    """
    Create a meshgrid [-1,1] x [-1,1] of given spatial_size.
    """
    h, w = spatial_size
    x = torch.arange(w).type(type)
    y = torch.arange(h).type(type)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)

    return meshed


class ResBlock2d(nn.Module):
    """
    Res block, preserve spatial resolution.
    """

    def __init__(self, in_features, kernel_size, padding):
        super(ResBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=in_features, kernel_size=kernel_size,
                               padding=padding)
        self.norm1 = BatchNorm2d(in_features, affine=True)
        self.norm2 = BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out

class UpBlock2d(nn.Module):
    """
    Upsampling block for use in decoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(UpBlock2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        out = F.relu(out)
        return out

class DownBlock2d(nn.Module):
    """
    Downsampling block for use in encoder.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1, groups=1):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size,
                              padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        out = self.pool(out)
        return out

class SameBlock2d(nn.Module):
    """
    Simple block, preserve spatial resolution.
    """

    def __init__(self, in_features, out_features, groups=1, kernel_size=3, padding=1):
        super(SameBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features,
                              kernel_size=kernel_size, padding=padding, groups=groups)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = F.relu(out)
        return out

class Encoder(nn.Module):
    """
    Hourglass Encoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Encoder, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(DownBlock2d(in_features if i == 0 else min(max_features, block_expansion * (2 ** i)),
                                           min(max_features, block_expansion * (2 ** (i + 1))),
                                           kernel_size=3, padding=1))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        # print("123", x.shape)
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs

class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu

class residual(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(residual, self).__init__()

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
        self.bn1   = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
        self.bn2   = nn.BatchNorm2d(out_dim)
        
        self.skip  = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2   = self.bn2(conv2)

        skip  = self.skip(x)
        return self.relu(bn2 + skip)

class Auto_Swin(nn.Module):
    """
    Hourglass Auto_Swin
    """

    def __init__(self):
        super(Auto_Swin, self).__init__()
        # self.swin = swin_b(num_classes=6, window_size=2)
        # self.up3 = nn.ModuleList([nn.Sequential(
        #             residual(3, 1024, 1024),
        #             residual(3, 1024, 512),
        #             nn.Upsample(scale_factor=2)) for _ in range(1)
        #             ])
        # self.cn3 = nn.ModuleList([nn.Sequential(
        #             convolution(k=3, inp_dim=512, out_dim=512, stride=1),
        #             nn.BatchNorm2d(512, momentum=0.1),
        #             nn.ReLU(inplace=True))
        #             ])
        # self.up2 = nn.ModuleList([nn.Sequential(
        #             residual(3, 512, 512),
        #             residual(3, 512, 256),
        #             nn.Upsample(scale_factor=2)) for _ in range(1)
        #             ])
        # self.cn2 = nn.ModuleList([nn.Sequential(
        #             convolution(k=3, inp_dim=256, out_dim=256, stride=1),
        #             nn.BatchNorm2d(256, momentum=0.1),
        #             nn.ReLU(inplace=True))
        #             ])
        # self.up1 = nn.ModuleList([nn.Sequential(
        #             residual(3, 256, 256),
        #             residual(3, 256, 128),
        #             nn.Upsample(scale_factor=2)) for _ in range(1)
        #             ])
        # self.cn1 = nn.ModuleList([nn.Sequential(
        #             convolution(k=3, inp_dim=128, out_dim=128, stride=1),
        #             nn.BatchNorm2d(128, momentum=0.1),
        #             nn.ReLU(inplace=True))
        #             ])
        self.swin = swin_t(num_classes=6, window_size=2)
        self.up3 = nn.ModuleList([nn.Sequential(
                    residual(3, 768, 768),
                    residual(3, 768, 384),
                    nn.Upsample(scale_factor=2)) for _ in range(1)
                    ])
        self.cn3 = nn.ModuleList([nn.Sequential(
                    convolution(k=3, inp_dim=384, out_dim=384, stride=1),
                    nn.BatchNorm2d(384, momentum=0.1),
                    nn.ReLU(inplace=True))
                    ])
        self.up2 = nn.ModuleList([nn.Sequential(
                    residual(3, 384, 384),
                    residual(3, 384, 192),
                    nn.Upsample(scale_factor=2)) for _ in range(1)
                    ])
        self.cn2 = nn.ModuleList([nn.Sequential(
                    convolution(k=3, inp_dim=192, out_dim=192, stride=1),
                    nn.BatchNorm2d(192, momentum=0.1),
                    nn.ReLU(inplace=True))
                    ])
        self.up1 = nn.ModuleList([nn.Sequential(
                    residual(3, 192, 192),
                    residual(3, 192, 96),
                    nn.Upsample(scale_factor=2)) for _ in range(1)
                    ])
        self.cn1 = nn.ModuleList([nn.Sequential(
                    convolution(k=3, inp_dim=96, out_dim=96, stride=1),
                    nn.BatchNorm2d(96, momentum=0.1),
                    nn.ReLU(inplace=True))
                    ])

    def forward(self, x):
        s1, s2, s3, s4 = self.swin(x)
        # print(s1.size(), s2.size(),s3.size(),s4.size())
        up3 = self.up3[0](s4) #up3.size() = 16, 16, 384
        dcn_skip3 = self.cn3[0](s3)#dcn_skip3.size() = 16, 16, 384
        merge34 = up3 + dcn_skip3
        # merge34 = torch.cat((up3, dcn_skip3), 1)

        up2 = self.up2[0](merge34)
        dcn_skip2 = self.cn2[0](s2)
        merge23 = up2 + dcn_skip2
        # merge23 = torch.cat((up2, dcn_skip2), 1)

        
        up1 = self.up1[0](merge23)
        dcn_skip1 = self.cn1[0](s1)
        # merge12 = torch.cat((up1,dcn_skip1), 1)
        merge12 = up1 + dcn_skip1
        # merge12 = torch.cat((up1, dcn_skip1), 1)
        # merge12 = self.cn0(merge12)
        # print("merge12", merge12.size())
        # print(XXX)
        
        return merge12

class Decoder(nn.Module):
    """
    Hourglass Decoder
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Decoder, self).__init__()

        up_blocks = []

        for i in range(num_blocks)[::-1]:
            in_filters = (1 if i == num_blocks - 1 else 2) * min(max_features, block_expansion * (2 ** (i + 1)))
            out_filters = min(max_features, block_expansion * (2 ** i))
            up_blocks.append(UpBlock2d(in_filters, out_filters, kernel_size=3, padding=1))

        self.up_blocks = nn.ModuleList(up_blocks)
        self.out_filters = block_expansion + in_features

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            skip = x.pop()
            out = torch.cat([out, skip], dim=1)
        return out

class Hourglass(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass, self).__init__()
        self.encoder = Encoder(block_expansion, in_features, num_blocks, max_features)
        self.decoder = Decoder(block_expansion, in_features, num_blocks, max_features)
        self.out_filters = self.decoder.out_filters

    def forward(self, x):
        return self.decoder(self.encoder(x))

class Hourglass_Swin(nn.Module):
    """
    Hourglass architecture.
    """

    def __init__(self, block_expansion, in_features, num_blocks=3, max_features=256):
        super(Hourglass_Swin, self).__init__()
        self.auto_swin = Auto_Swin()
        self.out_filters = 96

    def forward(self, x):
        return self.auto_swin(x)

class AntiAliasInterpolation2d(nn.Module):
    """
    Band-limited downsampling, for better preservation of the input signal.
    """
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1
        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        kernel_size = [kernel_size, kernel_size]
        sigma = [sigma, sigma]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
                ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)
        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels
        self.scale = scale
        inv_scale = 1 / scale
        self.int_inv_scale = int(inv_scale)

    def forward(self, input):
        if self.scale == 1.0:
            return input

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]

        return out

# if __name__ == "__main__":
    # # model = Auto_Swin()
    # model = Hourglass((64, 44, 512, 5))
    # x = torch.ones(1, 44, 256, 256)
    # y = model(x)
    # print(y)
    # writer = SummaryWriter('D:/Project/Track/FOMM/exps')
    # writer.add_graph(model, input_to_model = x)
    # writer.close()
    # prediction = AntiAliasInterpolation2d(3, 0.25)
    # imgdata = imagedata = cv2.imread(r'C:\Users\MediaCore\Desktop\rec_copy.png')/255
    # imagedata = torch.unsqueeze(torch.tensor(imgdata, dtype=torch.float32), 0)
    # imagedata = imagedata.permute([0, 3, 1, 2])
    # x = outdata = prediction(imagedata)
    # figure, ax = plt.subplots(1, 2)
    # imgdata_rgb = imgdata[:,:,::-1]
    # outdata_rgb = outdata.permute([0, 2, 3, 1])[0].numpy()
    # outdata_rgb = outdata_rgb[:,:,::-1]

    # ax[0].imshow(imgdata_rgb)
    # ax[1].imshow(outdata_rgb)
    # # cv2.imwrite('output.png', imgdata_rgb*255)
    # plt.show()
