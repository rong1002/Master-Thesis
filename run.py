import matplotlib
matplotlib.use('Agg')
import os
import yaml
from argparse import ArgumentParser
from time import localtime, strftime
from shutil import copy
from frames_dataset import FramesDataset
import torch
import warnings
warnings.simplefilter('ignore')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="config/vox-256.yaml", help="path to config")
    parser.add_argument("--mode", default="train", choices=["train", "reconstruction", "pose-evalution", "cal-parameter", "demo", "vis"])
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))), help="Names of the devices comma separated.")
    
    #train, vis
    parser.add_argument("--model", default="FOMM", choices=["FOMM", "MotionSwinT", "MotionSwinT_KP", "MotionSwinT_Occl", "MotionSwinT_Occl_KP"])
    # parser.add_argument("--log_dir", default='FOMM', choices=["FOMM", "MotionSwinT", "MotionSwinT_KP", "MotionSwinT_Occl", "MotionSwinT_Occl_KP"])
    parser.add_argument("--checkpoint", default=None,help="path to checkpoint to restore")

    #reconstruction
    # parser.add_argument("--final_ckpt", default='ckpt/mod_MotionSwin0120.tar', help="path to checkpoint to restore")
    parser.add_argument("--final_ckpt", default='mod_Author/log/vox-256_0212/mod_Author0212.tar', help="path to checkpoint to restore")


    # conda python Metrics
    #pose-evalution
    parser.add_argument("--out_file_face_pose",default="pose_evaluation/face_pose_AKD/mod_MotionSwinb0212.pkl", help="Extracted Test values")
    parser.add_argument("--out_file_face_id", default="pose_evaluation/face_id_AED/mod_MotionSwinb0212.pkl", help="Extracted Test values")

    #demo
    parser.add_argument("--source_image", default='test_data/source/yaw/yaw_004.png', help="path to source image")
    parser.add_argument("--source_file", default='test_data/source/yaw', help="path to source image")
    parser.add_argument("--driving_video", default='test_data/target/gt.mp4', help="path to driving video")
    parser.add_argument("--result_video", default='vox/test.mp4', help="path to output")

    opt = parser.parse_args()

    with open('model/' + opt.model + '/vox-256.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config_path = 'model/' + opt.model + '/vox-256.yaml'
    log_dir = 'model/' + opt.model
    # if (opt.checkpoint is not None) or opt.mode != 'train':
    #     with open('/model/' + opt.log_dir + '/vox-256.yaml') as f:
    #         config = yaml.load(f, Loader=yaml.FullLoader)
    # else:
    #     with open(opt.config) as f:
    #         config = yaml.load(f, Loader=yaml.FullLoader)

    if opt.model == 'FOMM':
        from model.FOMM.modules.generator import OcclusionAwareGenerator
        from model.FOMM.modules.discriminator import MultiScaleDiscriminator
        from model.FOMM.modules.keypoint_detector import KPDetector
        from model.FOMM.modules.model import GeneratorFullModel, DiscriminatorFullModel
    elif opt.model == 'MotionSwinT':
        from model.MotionSwinT.modules.generator import OcclusionAwareGenerator
        from model.MotionSwinT.modules.discriminator import MultiScaleDiscriminator
        from model.MotionSwinT.modules.keypoint_detector import KPDetector
        from model.MotionSwinT.modules.model import GeneratorFullModel, DiscriminatorFullModel
    elif opt.model == 'MotionSwinT_KP':
        from model.MotionSwinT_KP.modules.generator import OcclusionAwareGenerator
        from model.MotionSwinT_KP.modules.discriminator import MultiScaleDiscriminator
        from model.MotionSwinT_KP.modules.keypoint_detector import KPDetector
        from model.MotionSwinT_KP.modules.model import GeneratorFullModel, DiscriminatorFullModel
    elif opt.model == 'MotionSwinT_Occl':
        from model.MotionSwinT_Occl.modules.generator import OcclusionAwareGenerator
        from model.MotionSwinT_Occl.modules.discriminator import MultiScaleDiscriminator
        from model.MotionSwinT_Occl.modules.keypoint_detector import KPDetector
        from model.MotionSwinT_Occl.modules.model import GeneratorFullModel, DiscriminatorFullModel
    elif opt.model == 'MotionSwinT_Occl_KP':
        from model.MotionSwinT_Occl_KP.modules.generator import OcclusionAwareGenerator
        from model.MotionSwinT_Occl_KP.modules.discriminator import MultiScaleDiscriminator
        from model.MotionSwinT_Occl_KP.modules.keypoint_detector import KPDetector
        from model.MotionSwinT_Occl_KP.modules.model import GeneratorFullModel, DiscriminatorFullModel




    if opt.mode == 'pose-evalution':
        from pose_evaluation.extract import metrics

        print("==> Pose-evalution...")
        png_dir = log_dir + '/log/vox-256_0212/reconstruction'
        # png_dir = opt.log_dir + '/reconstruction'

        metrics(png_dir, opt.out_file_face_pose, opt.out_file_face_id)

    else: 
        generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])
        if torch.cuda.is_available():
            generator.to(opt.device_ids[0])
            torch.backends.cudnn.benchmark = True

        discriminator = MultiScaleDiscriminator(**config['model_params']['discriminator_params'],
                                                **config['model_params']['common_params'])
        if torch.cuda.is_available():
            discriminator.to(opt.device_ids[0])
            torch.backends.cudnn.benchmark = True

        kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                                **config['model_params']['common_params'])
        if torch.cuda.is_available():
            kp_detector.to(opt.device_ids[0])
            torch.backends.cudnn.benchmark = True

        generator_full = GeneratorFullModel(kp_detector, generator, discriminator, config['train_params'])
        discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, config['train_params'])

        dataset = FramesDataset(is_train=(opt.mode == 'train'), **config['dataset_params'])


        if opt.mode == 'train':
            from train import train
            print("==> Training...")
            if opt.checkpoint is not None:
                log_dir = os.path.join(*os.path.split(opt.checkpoint)[:-1])
            else:
                log_dir = os.path.join(log_dir + '/log', os.path.basename(config_path).split('.')[0])
                log_dir += '_' + strftime("%m_%d-%H.%M", localtime())

            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            if not os.path.exists(os.path.join(log_dir, os.path.basename(config_path))):
                copy(config_path, log_dir)

            train(config, generator, discriminator, kp_detector, generator_full, discriminator_full, opt.checkpoint, log_dir, dataset, opt.device_ids)

        elif opt.mode == 'reconstruction':
            from reconstruction import reconstruction
            print("==> Reconstruct...")
            png_dir = log_dir + '/reconstruction'
            if not os.path.exists(png_dir):
                os.makedirs(png_dir)
            reconstruction(generator, kp_detector, opt.final_ckpt, png_dir, dataset)

        elif opt.mode == 'cal-parameter':
            from cal_parameter import cal_parameter
            print("==> Calculating parameters and flops...")
            cal_parameter(generator, kp_detector, generator_full)
        
        elif opt.mode == 'demo':
            from demo import demo, demo_file

            print("==> Demo...")
            config = log_dir + '/vox-256.yaml'

            # demo(generator, kp_detector, config, opt.final_ckpt, opt.source_image, opt.driving_video, opt.result_video)
            demo_file(generator, kp_detector, config, opt.final_ckpt, opt.source_file, opt.driving_video, opt.result_video)

        elif opt.mode == 'vis':
            from visualization import vis
            checkpoint = 'result/' + log_dir + '_KP' + '/checkpoint.pth.tar'
            root_folder = 'result/' + log_dir + '_KP'
            # checkpoint = 'result/' + opt.log_dir + '/checkpoint.pth.tar'
            # root_folder = 'result/' + opt.log_dir
            print("==> Visualization...")
            vis(root_folder, generator, kp_detector, checkpoint)

