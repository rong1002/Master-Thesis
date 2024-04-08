import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from skimage.transform import resize
from .metrics_util import frames2array
from .cmp import AED
from .cmp_kp import AKD

import warnings
warnings.simplefilter('ignore')


def extract_vgg(in_folder, is_video, image_shape, column):
    from torchvision.models import vgg
    from torchvision import transforms
    from torch import nn
    import torch

    class VggConv(nn.Module):
        def __init__(self):
            super(VggConv, self).__init__()
            self.original_model = vgg.vgg16(pretrained=True)

        def forward(self, x):
            x = self.original_model.features(x)
            return x

    net = VggConv().cuda()

    out_df = {'file_name': [], 'frame_number': [], 'value': []}

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    for file in tqdm(sorted(os.listdir(in_folder))):
        video = frames2array(os.path.join(in_folder, file),
                             is_video, image_shape, column)
        for i, frame in enumerate(video):
            with torch.no_grad():
                frame = frame.astype('float32') / 255.0
                frame = transform(frame)
                frame = frame.unsqueeze(0).cuda()
                feat = net(frame).data.cpu().numpy()
            out_df['file_name'].append(file)
            out_df['frame_number'].append(i)
            out_df['value'].append(feat)

    return pd.DataFrame(out_df)


def extract_face_pose(in_folder, out_file, is_video, column, image_shape):
    import face_alignment

    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D, flip_input=False)
    
    out_df = {'file_name': [], 'frame_number': [], 'value': []}
    for file in tqdm(os.listdir(in_folder), ncols=100):
        video = frames2array(os.path.join(in_folder, file),
                             is_video, image_shape, column)
        for i, frame in enumerate(video):
            kp = fa.get_landmarks(frame)
            if kp is not None:
                kp = kp[0]
            out_df['file_name'].append(file)
            out_df['frame_number'].append(i)
            out_df['value'].append(kp)

    pkl = pd.DataFrame(out_df).to_pickle(out_file)

    return pkl


def extract_face_id(in_folder, out_file, is_video, column, image_shape):
    from .OpenFacePytorch.loadOpenFace import prepareOpenFace
    from torch.autograd import Variable
    import torch
    from imageio import mimsave

    net = prepareOpenFace(useCuda=True, gpuDevice=0, useMultiGPU=False).eval()

    out_df = {'file_name': [], 'frame_number': [], 'value': []}

    for file in tqdm(os.listdir(in_folder), ncols=100):
        video = frames2array(os.path.join(in_folder, file),
                             is_video, image_shape, column)
        for i, frame in enumerate(video):
            frame = frame[..., ::-1]
            frame = resize(frame, (96, 96))
            frame = np.transpose(frame, (2, 0, 1))
            with torch.no_grad():
                frame = Variable(torch.Tensor(frame)).cuda()
                frame = frame.unsqueeze(0)
                id_vec = net(frame)[0].data.cpu().numpy()

            out_df['file_name'].append(file)
            out_df['frame_number'].append(i)
            out_df['value'].append(id_vec)

    pkl = pd.DataFrame(out_df).to_pickle(out_file)

    return pkl


def metrics(in_folder, out_file_face_pose, out_file_face_id):
    face_pose_gt = r"D:\Project\FOMM_rong\pose_evaluation\face_pose_AKD\300_gt.pkl"
    face_id_gt = r"D:\Project\FOMM_rong\pose_evaluation\face_id_AED\300_gt.pkl"


    print("==> Face pose (AKD)")
    extract_face_pose(in_folder, out_file_face_pose, is_video='true', column=0, image_shape=(256, 256))

    print("==> Face id (AED)")
    extract_face_id(in_folder, out_file_face_id, is_video='true', column=0, image_shape=(256, 256))
    
    print("==> Calculating AKD")
    AKD(face_pose_gt, out_file_face_pose)

    print("==> Calculating AED")
    AED(face_id_gt, out_file_face_id)

    
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--in_folder", default=r"D:\Project\FOMM_rong\mod_UnetAutoLinkSwin\reconstruction", help="Folder with images")

    # parser.add_argument("--in_folder_gt", default=r"E:\Dataset\VoxCeleb\try_video\test", help="Folder with images")

    parser.add_argument("--face_pose_gt", default=r"D:\Project\FOMM_rong\pose_evaluation\face_pose_AKD\300test_gt.pkl", help="Extracted GroundTruth values")
    parser.add_argument("--face_id_gt", default=r"D:\Project\FOMM_rong\pose_evaluation\face_id_AED\300test_gt.pkl", help="Extracted GroundTruth values")

    # test .pkl
    parser.add_argument("--out_file_face_pose",default=r"D:\Project\FOMM_rong\pose_evaluation\face_pose_AKD\mod_UnetAutoLinkSwin.pkl", help="Extracted Test values")
    parser.add_argument("--out_file_face_id", default=r"D:\Project\FOMM_rong\pose_evaluation\face_id_AED\mod_UnetAutoLinkSwin.pkl", help="Extracted Test values")

    parser.add_argument("--is_video", dest='is_video',action='store_true', help="If this is a video.")
    parser.add_argument("--column", default=0, type=int, help="Some generation tools stack multiple images together, the index of the comlumn with right images")
    parser.add_argument("--image_shape", default=(256, 256), type=lambda x: tuple([int(a) for a in x.split(',')]), help="Image shape")

    parser.set_defaults(is_video=True)
    args = parser.parse_args()

    # metrics(args.in_folder, args.out_file_face_pose, args.out_file_face_id)
    # print("==>face pose GroundTruth (AKD)")
    # extract_face_pose(args.in_folder_gt, args.face_pose_gt, args.is_video, args.column, args.image_shape)
    # print("==>face id GroundTruth (AED)")
    # extract_face_id(args.in_folder_gt, args.face_id_gt, args.is_video, args.column, args.image_shape)

    # print("==>face pose (AKD)")
    # extract_face_pose(args.in_folder, args.out_file_face_pose, args.is_video, args.column, args.image_shape)

    # print("==>face id (AED)")
    # extract_face_id(args.in_folder, args.out_file_face_id, args.is_video, args.column, args.image_shape)
    
    # print("==>calculating AKD")
    # AKD(args.face_pose_gt, args.out_file_face_pose)

    # print("==>calculating AED")
    # AED(args.face_id_gt, args.out_file_face_id)
