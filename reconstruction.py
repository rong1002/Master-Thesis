import os
import numpy as np
import imageio
import torch

from torch.utils.data import DataLoader
from logger import Logger
from tqdm import tqdm
from sync_batchnorm import DataParallelWithCallback

def reconstruction(generator, kp_detector, checkpoint, png_dir, dataset):

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector)
    else:
        raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    loss_list = []
    if torch.cuda.is_available():
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()

    # for it, x in tqdm(enumerate(dataloader)):
    for x in tqdm(dataloader, ncols=100):
        # if config['reconstruction_params']['num_videos'] is not None:
        #     if it > config['reconstruction_params']['num_videos']:
        #         break
        with torch.no_grad():
            predictions = []
            visualizations = []
            if torch.cuda.is_available():
                x['video'] = x['video'].cuda()
            kp_source = kp_detector(x['video'][:, :, 0])
            for frame_idx in range(x['video'].shape[2]):
                source = x['video'][:, :, 0]
                driving = x['video'][:, :, frame_idx]
                kp_driving = kp_detector(driving)
                out = generator(source, kp_source=kp_source, kp_driving=kp_driving)
                out['kp_source'] = kp_source
                out['kp_driving'] = kp_driving
                del out['sparse_deformed']
                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

                # visualization = Visualizer(**config['visualizer_params']).visualize(source=source, driving=driving, out=out)
                # visualizations.append(visualization)

                loss_list.append(torch.abs(out['prediction'] - driving).mean().cpu().numpy())
            predictions = np.concatenate(predictions, axis=1)
            imageio.imsave(os.path.join(png_dir, x['name'][0] + '.png'), (255 * predictions).astype(np.uint8))

            # image_name = x['name'][0] + config['reconstruction_params']['format']
            # imageio.mimsave(os.path.join(log_dir, image_name), visualizations)

    print("Reconstruction loss: %s" % np.mean(loss_list))

# if __name__ == "__main__":
#     parser = ArgumentParser()
#     #reconstruction
#     parser.add_argument("--config", default="config/vox-256.yaml", help="path to config")
#     parser.add_argument("--log_dir", default='mod_Author/reconstruction', help="path to log into")
#     # parser.add_argument("--checkpoint", default=r'D:\Project\FOMM_rong\vox-cpk.pth.tar', help="path to checkpoint to restore")
#     # parser.add_argument("--checkpoint", default=r'D:\Project\FOMM_rong\mod_KpSwin\log\vox-256_02_28-03.45\099-checkpoint.pth.tar', help="path to checkpoint to restore")
#     parser.add_argument("--checkpoint", default=r'D:\Project\FOMM_rong\ckpt\mod_Author.pth.tar', help="path to checkpoint to restore")
#     parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))),help="Names of the devices comma separated.")
#     opt = parser.parse_args()

#     with open(opt.config) as f:
#         config = yaml.load(f, Loader=yaml.FullLoader)
        
#     log_dir = os.path.join(opt.log_dir, os.path.basename(opt.config).split('.')[0])
#     log_dir += '_' + strftime("%m_%d-%H.%M", localtime())
#     png_dir = log_dir + '/png'

#     generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
#                                         **config['model_params']['common_params'])
#     if torch.cuda.is_available():
#         generator.to(opt.device_ids[0])
#         torch.backends.cudnn.benchmark = True
#     # print(generator)
    
#     kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
#                              **config['model_params']['common_params'])
#     if torch.cuda.is_available():
#         kp_detector.to(opt.device_ids[0])
#         torch.backends.cudnn.benchmark = True
    
#     dataset = FramesDataset(is_train=False, **config['dataset_params'])

#     if not os.path.exists(log_dir):
#         os.makedirs(log_dir)
#     if not os.path.exists(png_dir):
#         os.makedirs(png_dir)

#     print("==>reconstruct...")
#     reconstruction(generator, kp_detector, opt.checkpoint, png_dir, dataset)

