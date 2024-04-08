import yaml
from argparse import ArgumentParser
from tqdm.auto import tqdm

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
from sync_batchnorm import DataParallelWithCallback
from animate import normalize_kp
from logger import *
from visualization import *
from test import OpticalFlow3
from visualization import save_img_kp
from driving_kp import kp_video
import warnings
warnings.simplefilter('ignore')
import cv2
def load_checkpoints(generator, kp_detector, config_path, checkpoint_path, cpu=False):
    
    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)

    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])

    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)

    generator.eval()
    kp_detector.eval()
    # print(generator)
    return generator, kp_detector

def make_animation(config, source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False):
    with open(config) as f:
        config = yaml.full_load(f)
    with torch.no_grad():
        x = LoggerTest(log_dir=r'D:\Project\FOMM_rong', visualizer_params=config['visualizer_params'])
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)

        kp_driving_initial = kp_detector(driving[:, :, 0])
        count = 1
        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
            if (count == 494): 
                out.update({'kp_source': kp_source, 'kp_driving': kp_driving})
                # save_img_kp(count, source, out['kp_source']['value'], driving_frame, out['kp_driving']['value'], out['prediction'])
                # vis(out['edgemap'])
                # OpticalFlow3(out['deformation'])
                x.visualize_test(source, driving_frame, out, count)
            count = count + 1
            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
        # print(out['mask'].shape) [1, 11, 64, 64]
        # print(out['sparse_deformed'].shape) [1, 11, 3, 64, 64]
        # print(out['occlusion_map'].shape) [1, 1, 64, 64]
        # print(out['deformed'].shape) [1, 3, 256, 256]
        # print(out['prediction'].shape) [1, 3, 256, 256]
        # savespath = 'C:/Users/MediaCore/Desktop/result.png'
        # plt.imshow(savespath, )
        # print(out)
    return predictions

def find_best_frame(source, driving, cpu=False):
    import face_alignment  # type: ignore (local file)
    from scipy.spatial import ConvexHull

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num

def demo(generator, kp_detector, config, checkpoint, source_image, driving_video, result_video):

    source_image = imageio.imread(source_image)
    reader = imageio.get_reader(driving_video)
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()
    
    source_image = resize(source_image, (256, 256))[..., :3]
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
    generator, kp_detector = load_checkpoints(generator, kp_detector, config_path=config, checkpoint_path=checkpoint, cpu=False)
    predictions = make_animation(config, source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=False)
    # print(driving_kp[0])
    # print(predictions[0].shape)
    # predictions = make_animation(config, source_image, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=True, cpu=False)
    imageio.mimsave(result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)


def make_animation_test(config, source_file, driving_video, generator, kp_detector, relative, adapt_movement_scale):
    with open(config) as f:
        config = yaml.full_load(f)

    with torch.no_grad():
        kp_source_list = []
        source_list = []
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3).cuda()
        kp_driving_initial = kp_detector(driving[:, :, 0])
        kp_driving_value = kp_driving_initial['value']

        for source_file_name in os.listdir(source_file):
            source_image = source_file + '/' + source_file_name
            source_image = imageio.imread(source_image)
            source_image = resize(source_image, (256, 256))[..., :3]
            source_list.append(source_image)
            source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).cuda()
            kp_source = kp_detector(source)
            kp_source_list.append(kp_source['value'])

        distances = [torch.norm(kp_source_value - kp_driving_value) for kp_source_value in kp_source_list]
        min_distance_index = distances.index(min(distances))
        source_image = source_list[min_distance_index]
        print(f"The closest tensor is at index: {min_distance_index}")

    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3).cuda()
        kp_source = kp_detector(source)

        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx].cuda()
            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
            
            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])

    return predictions


def demo_file(generator, kp_detector, config, checkpoint, source_file, driving_video, result_video):


    reader = imageio.get_reader(driving_video)
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()
    
    # source_file = resize(source_file, (256, 256))[..., :3]
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
    generator, kp_detector = load_checkpoints(generator, kp_detector, config_path=config, checkpoint_path=checkpoint, cpu=False)
    predictions = make_animation_test(config, source_file, driving_video, generator, kp_detector, relative=True, adapt_movement_scale=False)
    
    imageio.mimsave(result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)