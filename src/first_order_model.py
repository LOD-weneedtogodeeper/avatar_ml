import matplotlib
matplotlib.use('Agg')
import os, sys
import yaml
import imageio
import numpy as np
import torch
torch.multiprocessing.set_start_method('forkserver')
from tqdm import tqdm
from skimage.transform import resize
from skimage import img_as_ubyte
from sync_batchnorm import DataParallelWithCallback
from argparse import ArgumentParser

from PIL import Image
from io import BytesIO
from modules.generator import OcclusionAwareGenerator
from modules.keypoint_detector import KPDetector
from utils.animate import normalize_kp
from scipy.spatial import ConvexHull

if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

class FirstOrderModel:

    def __init__(self, config_path, checkpoint_path, if_cpu):
        self.config_path = config_path
        self.cp_path = checkpoint_path
        self.cpu = if_cpu

        self.generator, self.kp_detector = self.load_checkpoints()

    def load_checkpoints(self):

        with open(self.config_path) as f:
            config = yaml.load(f)

        generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])
        if not self.cpu:
            generator.cuda()

        kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                                 **config['model_params']['common_params'])
        if not self.cpu:
            kp_detector.cuda()
    
        if self.cpu:
            checkpoint = torch.load(self.cp_path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(self.cp_path)
 
        generator.load_state_dict(checkpoint['generator'])
        kp_detector.load_state_dict(checkpoint['kp_detector'])
    
        if not self.cpu:
            generator = DataParallelWithCallback(generator)
            kp_detector = DataParallelWithCallback(kp_detector)

        generator.eval()
        kp_detector.eval()
    
        return generator, kp_detector


    def make_animation(self, source_image, driving_video,relative=True, adapt_movement_scale=True, cpu=False):
        with torch.no_grad():
            predictions = []
            source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
            if not cpu:
                source = source.cuda()
            driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
            kp_source = self.kp_detector(source)
            kp_driving_initial = self.kp_detector(driving[:, :, 0])

            for frame_idx in tqdm(range(driving.shape[2])):
                driving_frame = driving[:, :, frame_idx]
                if not cpu:
                    driving_frame = driving_frame.cuda()
                kp_driving = self.kp_detector(driving_frame)
                kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                       kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                       use_relative_jacobian=relative, adapt_movement_scale=adapt_movement_scale)
                out = self.generator(source, kp_source=kp_source, kp_driving=kp_norm)

                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
        return predictions


    def infer(self, source_image, video):
        source_image = imageio.imread(source_image, '.jpg')
        reader = imageio.get_reader(video,'.mp4')
        fps = reader.get_meta_data()['fps']
        driving_video = []
        for im in reader:
            driving_video.append(im)
        reader.close()

        source_image = resize(source_image, (256, 256))[..., :3]
        driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

        predictions = self.make_animation(source_image, driving_video)
        images = [img_as_ubyte(frame) for frame in predictions]
        
        
        video_seq = [Image.fromarray(v_frame) for v_frame in images]
        buffered = BytesIO()
        video_seq[0].save(buffered, save_all=True,format="GIF", append_images=video_seq[1:]\
                    ,loop=0,duration=int((1000)/fps))
        
        return buffered.getvalue()