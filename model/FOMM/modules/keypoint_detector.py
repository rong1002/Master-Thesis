from torch import nn
import torch
import torch.nn.functional as F
from .util import Hourglass, make_coordinate_grid, AntiAliasInterpolation2d
# from torch.utils.tensorboard import SummaryWriter
# from thop import profile

class KPDetector(nn.Module):
    """
    Detecting a keypoints. Return keypoint position and jacobian near each keypoint.
    """

    def __init__(self, block_expansion, num_kp, num_channels, max_features,
                 num_blocks, temperature, estimate_jacobian=False, scale_factor=1,
                 single_jacobian_map=False, pad=0):
    # (32, 10, 3, 1024, 5, 0.1, True, 0.25)
        super(KPDetector, self).__init__()

        self.predictor = Hourglass(block_expansion, in_features=num_channels,
                                   max_features=max_features, num_blocks=num_blocks)
                        #(32, 3, 1024, 5)
        self.kp = nn.Conv2d(in_channels=self.predictor.out_filters, out_channels=num_kp, kernel_size=(7, 7),
                            padding=pad)

        if estimate_jacobian:
            self.num_jacobian_maps = 1 if single_jacobian_map else num_kp
            self.jacobian = nn.Conv2d(in_channels=self.predictor.out_filters,
                                      out_channels=4 * self.num_jacobian_maps, kernel_size=(7, 7), padding=pad)
            self.jacobian.weight.data.zero_()
            self.jacobian.bias.data.copy_(torch.tensor([1, 0, 0, 1] * self.num_jacobian_maps, dtype=torch.float))
        else:
            self.jacobian = None

        self.temperature = temperature
        self.scale_factor = scale_factor
        if self.scale_factor != 1:
            self.down = AntiAliasInterpolation2d(num_channels, self.scale_factor)

    def gaussian2kp(self, heatmap):
        """
        Extract the mean and from a heatmap
        """
        shape = heatmap.shape
        heatmap = heatmap.unsqueeze(-1)
        grid = make_coordinate_grid(shape[2:], heatmap.type()).unsqueeze_(0).unsqueeze_(0)
        value = (heatmap * grid).sum(dim=(2, 3))
        kp = {'value': value}
        return kp

    def forward(self, x):
        # print(x.size())
        # print(x.shape) = [5, 3, 256, 256]
        if self.scale_factor != 1:
            x = self.down(x)

        feature_map = self.predictor(x)
        prediction = self.kp(feature_map)

        final_shape = prediction.shape
        heatmap = prediction.view(final_shape[0], final_shape[1], -1)
        heatmap = F.softmax(heatmap / self.temperature, dim=2)
        heatmap = heatmap.view(*final_shape)

        out = self.gaussian2kp(heatmap)

        if self.jacobian is not None:
            jacobian_map = self.jacobian(feature_map)
            #final_shape = 1, 10, 58, 58
            #jacobian_map.shape = 1,40, 58, 58
            jacobian_map = jacobian_map.reshape(final_shape[0], self.num_jacobian_maps, 4, final_shape[2],
                                                final_shape[3])
            heatmap = heatmap.unsqueeze(2)
            #根據關鍵點heatmap的權重來設置jacobian行列式的重要性
            jacobian = heatmap * jacobian_map #jacobian.shape = 1, 10, 4, 58, 58
            jacobian = jacobian.view(final_shape[0], final_shape[1], 4, -1) #(1, 10, 4, -1)
            #jacobian.shape = 1, 10, 4, 3364
            jacobian = jacobian.sum(dim=-1)
            #jacobian.shape = 1, 10, 4
            jacobian = jacobian.view(jacobian.shape[0], jacobian.shape[1], 2, 2)
            #jacobian.shape = 1, 10, 2, 2
            out['jacobian'] = jacobian
            # print("***", out['value'])
            # print("*", out['jacobian'])
        return out

# if __name__ == "__main__":
#     model = KPDetector(32, 10, 3, 1024, 5, 0.1, True, 0.25).cuda()
#     x = torch.ones(1, 3, 256, 256).cuda()
#     # y = model(x)
#     # print("Finish")
#     flops, params = profile(model, inputs=(x))
#     print('FLOPs = ' + str(flops/1000**3) + 'G')
#     print('Params = ' + str(params/1000**2) + 'M')
#     # writer = SummaryWriter('D:/Project/Track/FOMM/exps')
#     # writer.add_graph(model, input_to_model = x)
#     # writer.close()