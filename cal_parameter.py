import torch
from thop import profile

def cal_parameter(generator, kp_detector, generator_full):
    #可以算 flops and paremeters, but paremeters會跟下面不太一樣
    x = torch.randn(1, 3, 256, 256).cuda()
    xx = {'value': torch.randn(1, 10, 2).cuda(), 
          'jacobian': torch.randn(1, 10, 2, 2).cuda()}
          # 'jacobian': torch.randn(1, 10, 2, 2).cuda(),
          # 'edgemap': torch.randn(1, 1, 256, 256).cuda()}
    flops_generator, params_generator = profile(generator, inputs=(x, xx, xx))

    flops_kp_detector, params_kp_detector = profile(kp_detector, inputs=([x]))

    flops = flops_generator + flops_kp_detector
    params = params_generator + params_kp_detector

    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')

    #只能算paremeters
    num_params = sum(param.numel() for param in generator_full.parameters())
    print('parameters = ', num_params)