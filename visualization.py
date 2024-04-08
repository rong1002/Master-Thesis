import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torchvision
from matplotlib import colors
from PIL import Image
import os
import torch.nn.functional as F
import yaml
from sync_batchnorm import DataParallelWithCallback
from tqdm.auto import tqdm
from animate import normalize_kp
import imageio
from skimage.transform import resize
from skimage import img_as_ubyte
from logger import *
from argparse import ArgumentParser
import warnings
warnings.simplefilter('ignore')
from torchvision.utils import save_image
from FOMM.modules.generator import OcclusionAwareGenerator
from FOMM.modules.keypoint_detector import KPDetector


def get_part_color(n_parts):
    colormap = ('red', 'blue', 'yellow', 'magenta', 'green', 'indigo', 'darkorange', 'cyan', 'pink', 'yellowgreen',
                'rosybrown', 'coral', 'chocolate', 'bisque', 'gold', 'yellowgreen', 'aquamarine', 'deepskyblue', 'navy', 'orchid',
                'maroon', 'sienna', 'olive', 'lightgreen', 'teal', 'steelblue', 'slateblue', 'darkviolet', 'fuchsia', 'crimson',
                'honeydew', 'thistle',
                'red', 'blue', 'yellow', 'magenta', 'green', 'indigo', 'darkorange', 'cyan', 'pink', 'yellowgreen',
                'rosybrown', 'coral', 'chocolate', 'bisque', 'gold', 'yellowgreen', 'aquamarine', 'deepskyblue', 'navy', 'orchid',
                'maroon', 'sienna', 'olive', 'lightgreen', 'teal', 'steelblue', 'slateblue', 'darkviolet', 'fuchsia', 'crimson',
                'honeydew', 'thistle')[:n_parts]
    part_color = []
    for i in range(n_parts):
        part_color.append(colors.to_rgb(colormap[i]))
    part_color = np.array(part_color)

    return part_color

def denormalize(img):
    mean = torch.tensor((0.5, 0.5, 0.5), device=img.device).reshape(1, 3, 1, 1)
    std = torch.tensor((0.5, 0.5, 0.5), device=img.device).reshape(1, 3, 1, 1)
    img = img * std + mean
    img = torch.clamp(img, min=0, max=1)
    return img

def draw_matrix(mat):
    fig = plt.figure()
    sns.heatmap(mat, annot=True, fmt='.2f', cmap="YlGnBu")

    ncols, nrows = fig.canvas.get_width_height()
    fig.canvas.draw()
    plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(nrows, ncols, 3)
    plt.close(fig)
    return plot

def draw_kp_grid(img, kp):
    kp_color = get_part_color(kp.shape[1])
    img = img[:64].permute(0, 2, 3, 1).detach().cpu()
    kp = kp.detach().cpu()[:64]

    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0, hspace=0)

    for i, sample in enumerate(img):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.imshow(sample, vmin=0, vmax=1)
        ax.scatter(kp[i, :, 1], kp[i, :, 0], c=kp_color, s=20, marker='+')

    ncols, nrows = fig.canvas.get_width_height()
    fig.canvas.draw()
    plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(nrows, ncols, 3)
    plt.close(fig)
    return plot

def draw_kp_grid_unnorm(img, kp):
    kp_color = get_part_color(kp.shape[1])
    img = img[:64].permute(0, 2, 3, 1).detach().cpu()
    kp = kp.detach().cpu()[:64]

    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0, hspace=0)

    for i, sample in enumerate(img):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.imshow(sample)
        ax.scatter(kp[i, :, 1], kp[i, :, 0], c=kp_color, s=20, marker='+')

    ncols, nrows = fig.canvas.get_width_height()
    fig.canvas.draw()
    plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(nrows, ncols, 3)
    plt.close(fig)
    return plot

def draw_img_grid(img):
    img = img[:64].detach().cpu()
    nrow = min(8, img.shape[0])
    img = torchvision.utils.make_grid(img[:64], nrow=nrow).permute(1, 2, 0)
    return torch.clamp(img * 255, min=0, max=255).numpy().astype(np.uint8)

def save_img_kp(count, source, kp_source, driving, kp_driving, prediction):
    folder_name = 'D:/Project/FOMM_rong/test-vis/'
    source = source.squeeze(0).permute(1, 2, 0).cpu()
    driving = driving.squeeze(0).permute(1, 2, 0).cpu()
    prediction = prediction.squeeze(0).permute(1, 2, 0).cpu()
    kp_source = kp_source.data.cpu().numpy()
    kp_source = kp_source*128 + 128
    kp_driving = kp_driving.data.cpu().numpy()
    kp_driving = kp_driving*128 + 128

    kp_color = get_part_color(10)
    os.makedirs(os.path.join('det', folder_name, str(count)), exist_ok=True)
    
    # # draw image
    # Image.fromarray(np.uint8(source * 255)).save(os.path.join('det', folder_name, str(count), 'source.png'))
    # Image.fromarray(np.uint8(driving * 255)).save(os.path.join('det', folder_name, str(count), 'driving.png'))
    Image.fromarray(np.uint8(prediction * 255)).save(os.path.join('det', folder_name, str(count), 'prediction.png'))

    # draw kp source
    fig = plt.figure()
    fig.set_size_inches(1, 1, forward=False)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.tight_layout(pad=0)
    plt.axis('off')
    plt.imshow(source)
    plt.scatter(kp_source[:,:, 1], kp_source[:,:, 0], c=kp_color, s=5, marker='o')
    plt.savefig(os.path.join('det', folder_name, str(count), 'kp_source.png'), dpi=256)
    plt.close(fig)

    # draw kp driving
    fig = plt.figure()
    fig.set_size_inches(1, 1, forward=False)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.tight_layout(pad=0)
    plt.axis('off')
    plt.imshow(driving)
    plt.scatter(kp_driving[:,:, 1], kp_driving[:,:, 0], c=kp_color, s=5, marker='o')
    plt.savefig(os.path.join('det', folder_name, str(count), 'kp_driving.png'), dpi=256)
    plt.close(fig)


def load_checkpoints(generator, kp_detector, checkpoint_path, cpu=False):
    
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

def make_animation(dir_path, source_image, driving_video, generator, kp_detector, checkpoint, relative=True, adapt_movement_scale=True, cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).cuda()
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
            if (count == driving.shape[2]): 
            # if (count == 2): 
                save_image(driving_frame.squeeze(0), dir_path + '/groundtruth.png')
                save_image(out['prediction'].squeeze(0), dir_path + '/prediction.png')
                keypoint(dir_path + '/source_keypoint.png', dir_path + '/source.png', kp_detector)
                keypoint(dir_path + '/GT_keypoint.png', dir_path + '/groundtruth.png', kp_detector)
                keypoint(dir_path + '/pred_keypoint.png', dir_path + '/prediction.png', kp_detector)
            count = count + 1
            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions

def demo(dir_path, generator, kp_detector, checkpoint, source_image, driving_video, result_video):

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
    # print(dir_path)
    source_image = resize(source_image, (256, 256))[..., :3]
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
    generator, kp_detector = load_checkpoints(generator, kp_detector, checkpoint_path=checkpoint, cpu=False)
    predictions = make_animation(dir_path, source_image, driving_video, generator, kp_detector, checkpoint, relative=True, adapt_movement_scale=False)
    imageio.mimsave(result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)


def keypoint(dir_path, source, kp_detector):
    vis = Visualizer()
    images = []
    source = imageio.imread(source)
    source = resize(source, (256, 256))[..., :3]
    source = torch.tensor(source[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2).cuda()
    kp_source = kp_detector(source)
    source = source.data.cpu()
    source = np.transpose(source, [0, 2, 3, 1])
    kp_source = kp_source['value'].data.cpu().numpy()
    images.append((source, kp_source))#1
    image = vis.create_image_grid(*images)
    image = (255 * image).astype(np.uint8)
    imageio.imwrite(dir_path, image)



def vis(root_folder, generator, kp_detector, checkpoint):
    for main_dir in ['cross', 'same', 'keypoint']:  # The two main directories under 'test'
        path = os.path.join(root_folder, main_dir)
        if main_dir == 'cross':
            print(f"Listing files in '{main_dir}' directory:")
            for dir_name in os.listdir(path):
                dir_path = os.path.join(path, dir_name)
                source = dir_path + '/source.png'
                dirving = dir_path + '/driving.mp4'
                result = dir_path + '/result.mp4'
                demo(dir_path, generator, kp_detector, checkpoint, source, dirving, result)
                print("save" , result)
        elif main_dir == 'same':
            print(f"Listing files in '{main_dir}' directory:")
            for dir_name in os.listdir(path):
                dir_path = os.path.join(path, dir_name)
                dirving = dir_path + '/driving.mp4'
                source = imageio.get_reader(dirving).get_next_data()
                imageio.imwrite(dir_path + '/source.png', source) #save first frame
                source = dir_path + '/source.png'
                result = dir_path + '/result.mp4'
                demo(dir_path, generator, kp_detector, checkpoint, source, dirving, result)
                print("save" , result)
        elif main_dir == 'keypoint':
            print(f"Listing files in '{main_dir}' directory:")
            generator, kp_detector = load_checkpoints(generator, kp_detector, checkpoint_path=checkpoint, cpu=False)
            for dir_name in os.listdir(path):
                dir_path = os.path.join(path, dir_name)
                source = dir_path + '/source.png'
                result = dir_path + '/keypoint.png'
                keypoint(result, source, kp_detector)
                print("save" , result)




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="config/vox-256.yaml", help="path to config")
    parser.add_argument("--device_ids", default="0", type=lambda x: list(map(int, x.split(','))), help="Names of the devices comma separated.")
    parser.add_argument("--checkpoint", default='ckpt/mod_Author.tar', help="path to checkpoint to restore")
    # parser.add_argument("--checkpoint", default='D:/Project/FOMM_rong/mod_MotionSwin256/log/vox-256_0204/checkpoint.pth.tar', help="path to checkpoint to restore")
    parser.add_argument("--root_folder", default='C:/Users/MediaCore/Desktop/test', help="path to checkpoint to restore")
    opt = parser.parse_args()


    with open(opt.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])
    if torch.cuda.is_available():
        generator.to(opt.device_ids[0])
        torch.backends.cudnn.benchmark = True


    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                            **config['model_params']['common_params'])
    if torch.cuda.is_available():
        kp_detector.to(opt.device_ids[0])
        torch.backends.cudnn.benchmark = True

    vis(opt.root_folder, generator, kp_detector, opt.checkpoint)