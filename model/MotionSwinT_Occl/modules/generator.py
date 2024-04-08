import torch
from torch import nn
import torch.nn.functional as F
from .util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d
from .dense_motion import DenseMotionNetwork

class OcclusionAwareGenerator(nn.Module):
    """
    Generator that given source image and and keypoints try to transform image according to movement trajectories
    induced by keypoints. Generator follows Johnson architecture.
    """

    def __init__(self, num_channels, num_kp, block_expansion, max_features, num_down_blocks, num_bottleneck_blocks, 
                 estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False):
        super(OcclusionAwareGenerator, self).__init__()

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, num_channels=num_channels, 
                                                           estimate_occlusion_map=estimate_occlusion_map, 
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None

        self.num_down_blocks = num_down_blocks
        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        up_blocks = []
        resblock = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
            decoder_in_feature = out_features * 2
            if i==num_down_blocks-1:
                decoder_in_feature = out_features
            up_blocks.append(UpBlock2d(decoder_in_feature, in_features, kernel_size=(3, 3), padding=(1, 1)))
            resblock.append(ResBlock2d(decoder_in_feature, kernel_size=(3, 3), padding=(1, 1)))
            resblock.append(ResBlock2d(decoder_in_feature, kernel_size=(3, 3), padding=(1, 1)))

        self.down_blocks = nn.ModuleList(down_blocks)
        self.up_blocks = nn.ModuleList(up_blocks[::-1])
        self.resblock = nn.ModuleList(resblock[::-1])

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.num_channels = num_channels

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear', align_corners=True)
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation, align_corners=True)

    def occlude_input(self, inp, occlusion_map):
        if not True:
            if inp.shape[2] != occlusion_map.shape[2] or inp.shape[3] != occlusion_map.shape[3]:
                occlusion_map = F.interpolate(occlusion_map, size=inp.shape[2:], mode='bilinear',align_corners=True)
        out = inp * occlusion_map
        return out

    def forward(self, source_image, kp_driving, kp_source):
        # Encoding (downsampling) part
        out = self.first(source_image)
        encoder_map = [out]
        for i in range(len(self.down_blocks)): #len(self.down_blocks) = 2
            out = self.down_blocks[i](out)
            encoder_map.append(out)
        # print(encoder_map[0].shape) #256, 256, 64
        # print(encoder_map[1].shape) #128, 128, 128
        # print(encoder_map[2].shape) #64, 64, 256
        # print(encoder_map[3].shape) #32, 32, 512
        # Transforming feature representation according to deformation and occlusion
        output_dict = {}
        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(source_image=source_image, kp_driving=kp_driving,
                                                     kp_source=kp_source)
            output_dict['mask'] = dense_motion['mask']
            output_dict['sparse_deformed'] = dense_motion['sparse_deformed']

            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
                output_dict['occlusion_map'] = occlusion_map
            else:
                occlusion_map = None

            deformation = dense_motion['deformation'] # 64, 64, 2
            output_dict['deformation'] = dense_motion['deformation']
            out = self.deform_input(out, deformation) # 512, 32, 32
            out = self.occlude_input(out, occlusion_map[0])# 512, 32, 32
            # print(out.shape)
            for i in range(self.num_down_blocks):
                out = self.resblock[2*i](out)
                out = self.resblock[2*i+1](out)
                out = self.up_blocks[i](out)

                encode_i = encoder_map[-(i+2)]
                encode_i = self.deform_input(encode_i, deformation)
                
                occlusion_ind = 0
                if True:
                    occlusion_ind = i+1
                encode_i = self.occlude_input(encode_i, occlusion_map[occlusion_ind])

                if(i==self.num_down_blocks-1):
                    break
                out = torch.cat([out, encode_i], 1)

            deformed_source = self.deform_input(source_image, deformation)
            output_dict["deformed"] = deformed_source
            occlusion_last = occlusion_map[-1]

            # Decoding part
            # print("out", out.shape) #256, 256, 64
            # print("occlusion_last", occlusion_last.shape) #256, 256, 1
            # print("encode_i", encode_i.shape) # 256, 256, 64
            out = out * (1 - occlusion_last) + encode_i
            out = self.final(out)
            out = torch.sigmoid(out)
            out = out * (1 - occlusion_last) + deformed_source * occlusion_last
            output_dict["prediction"] = out

        return output_dict