import os
from frames_dataset import DatasetRepeater
from sync_batchnorm import DataParallelWithCallback
from logger import Logger
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import matplotlib
matplotlib.use('Agg')

def write_loss(i, perceptual_loss, equivariance_value_loss, equivariance_jacobian_loss, writer):
    # perceptual.loss
    writer.add_scalar('loss/perceptual', perceptual_loss.item(), i)
    # equivariance value loss
    writer.add_scalar('loss/equivariance value', equivariance_value_loss.item(), i)
    # equivariance jacobian loss
    writer.add_scalar('loss/equivariance jacobian', equivariance_jacobian_loss.item(), i)
    writer.flush()


def train(config, generator, discriminator, kp_detector, generator_full, discriminator_full, checkpoint, log_dir, dataset, device_ids):
    # 讀train參數:
    train_params = config['train_params']
    # 優化器
    optimizer_generator = torch.optim.Adam( generator.parameters(), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(checkpoint, generator, discriminator, kp_detector,
                                      optimizer_generator, optimizer_discriminator,
                                      None if train_params['lr_kp_detector'] == 0 else optimizer_kp_detector)
        start_epoch = start_epoch + 1
    else:
        start_epoch = 0

    # 調整Learn Rate
    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1, last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1, last_epoch=start_epoch - 1)
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1, last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])

    dataloader = DataLoader(
        dataset,
        batch_size=train_params['batch_size'],
        shuffle=True,
        num_workers=6,
        drop_last=True,
        pin_memory=True)

    # for 多GPU用的
    if torch.cuda.is_available():
        generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
        discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids)

    # tensorboard
    log_path = os.path.join(log_dir + '/tensorboard')
    writer = SummaryWriter(log_path)

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:

        for epoch in range(start_epoch, train_params['num_epochs']):
            perceptual_loss_avg = equivariance_value_loss_avg = equivariance_jacobian_loss_avg = keypoint_prior_loss_avg = 0
            with tqdm(total=len(dataloader), ncols=100) as _tqdm:
                _tqdm.set_description('Epoch: {}/{}'.format(epoch+1, train_params['num_epochs']))

                for out in dataloader:
                    losses_generator, generated = generator_full(out)
                    loss_values = [val.mean() for val in losses_generator.values()]
                    loss = sum(loss_values)

                    loss.backward()
                    optimizer_generator.step()
                    optimizer_generator.zero_grad()
                    optimizer_kp_detector.step()
                    optimizer_kp_detector.zero_grad()

                    if train_params['loss_weights']['generator_gan'] != 0:
                        optimizer_discriminator.zero_grad()
                        losses_discriminator = discriminator_full(out, generated)
                        loss_values = [val.mean() for val in losses_discriminator.values()]
                        loss = sum(loss_values)

                        loss.backward()
                        optimizer_discriminator.step()
                        optimizer_discriminator.zero_grad()
                    else:
                        losses_discriminator = {}

                    losses_generator.update(losses_discriminator)
                    losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                    logger.log_iter(losses=losses)
                    _tqdm.update(1)

                    perceptual_loss_avg = perceptual_loss_avg + losses['perceptual']
                    equivariance_value_loss_avg = equivariance_value_loss_avg + losses['equivariance_value']
                    equivariance_jacobian_loss_avg = equivariance_jacobian_loss_avg + losses['equivariance_jacobian']
                    # keypoint_prior_loss_avg = keypoint_prior_loss_avg + losses['keypoint_prior']

            # print("perceptual loss avg: %.2f, keypoint value loss avg: %.2f, keypoint jacobian loss avg: %.2f, keypoint prior loss avg: %.2f, lr: %.e"
            print("perceptual loss avg: %.2f, keypoint value loss avg: %.2f, keypoint jacobian loss avg: %.2f, lr: %.e"
                  % (perceptual_loss_avg/len(dataloader), 
                     equivariance_value_loss_avg/len(dataloader),
                     equivariance_jacobian_loss_avg/len(dataloader),
                     optimizer_generator.state_dict()['param_groups'][0]['lr']))
            write_loss(epoch, perceptual_loss_avg/len(dataloader), equivariance_value_loss_avg / len(dataloader), equivariance_jacobian_loss_avg/len(dataloader), writer)
            print("==> training finish")

            scheduler_generator.step()
            scheduler_discriminator.step()
            scheduler_kp_detector.step()
            logger.log_epoch(epoch, {'generator': generator,
                                     'discriminator': discriminator,
                                     'kp_detector': kp_detector,
                                     'optimizer_generator': optimizer_generator,
                                     'optimizer_discriminator': optimizer_discriminator,
                                     'optimizer_kp_detector': optimizer_kp_detector}, inp=out, out=generated)
            print("==> ckpt complete save ")


# if __name__ == "__main__":
#     parser = ArgumentParser()
#     parser.add_argument("--config", default="config/vox-256.yaml", help="path to config")
#     parser.add_argument("--log_dir", default='mod_KpSwin256/log', help="path to log into")
#     parser.add_argument("--checkpoint", default=None,help="path to checkpoint to restore")
#     parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))), help="Names of the devices comma separated.")

#     opt = parser.parse_args()

#     with open(opt.config) as f:
#         config = yaml.load(f, Loader=yaml.FullLoader)

#     if opt.checkpoint is not None:
#         log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
#     else:
#         log_dir = os.path.join(
#             opt.log_dir, os.path.basename(opt.config).split('.')[0])
#         log_dir += '_' + strftime("%m_%d-%H.%M", localtime())

#     generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
#                                         **config['model_params']['common_params'])
#     if torch.cuda.is_available():
#         generator.to(opt.device_ids[0])
#         torch.backends.cudnn.benchmark = True
#     # print(generator)

#     discriminator = MultiScaleDiscriminator(**config['model_params']['discriminator_params'],
#                                             **config['model_params']['common_params'])
#     if torch.cuda.is_available():
#         discriminator.to(opt.device_ids[0])
#         torch.backends.cudnn.benchmark = True

#     kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
#                              **config['model_params']['common_params'])
#     if torch.cuda.is_available():
#         kp_detector.to(opt.device_ids[0])
#         torch.backends.cudnn.benchmark = True

#     dataset = FramesDataset(is_train='train', **config['dataset_params'])
#     if not os.path.exists(log_dir):
#         os.makedirs(log_dir)
#     if not os.path.exists(os.path.join(log_dir, os.path.basename(opt.config))):
#         copy(opt.config, log_dir)

#     print("Training...")
#     train(config, generator, discriminator, kp_detector,opt.checkpoint, log_dir, dataset, opt.device_ids)
