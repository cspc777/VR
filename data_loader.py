import cv2
import numpy as np
from os import listdir
from os.path import isfile, join, isdir
import custom_transforms

import torch
from torch.utils.data import Dataset

def kitti_training_set(DATASET_PATH, scenes=['city', 'residential', 'road'], is_rgb = False):
    """
    Returns
    -------
    list
        A list of training sequences
    """
    KITTI_scenes = scenes

    if is_rgb == True:
        data_path = 'image_02/data'
    else:
        data_path = 'image_00/data'

    clips = []
    for scene in KITTI_scenes:
        scene_path = join(DATASET_PATH, scene)
        for s in sorted(listdir(scene_path)):
            if isdir(join(scene_path, s)):
                scene_date_path = join(scene_path, s)
                for d in sorted(listdir(scene_date_path)):
                    if isdir(join(scene_date_path, d)):
                        img_folder = join(join(scene_date_path, d), data_path)
                        all_frames = []
                        # loop over all the images in the folder (0.png,1.png,..,199.png)
                        for i in sorted(listdir(img_folder)):
                            if str(join(img_folder, i))[-3:] == "png":
                                img_path = join(img_folder, i)
                                all_frames.append(img_path)
                        # get the 10-frames sequences from the list of images after applying data augmentation
                        for stride in range(1, 2):
                            clips.extend(get_clips_by_stride(stride=stride, frames_list=all_frames, sequence_size=11))
    return clips

def my_training_set(DATASET_PATH, scenes=['level_1', 'level_2', 'level_3'], is_rgb = False):
    """
    Returns
    -------
    list
        A list of training sequences
    """
    my_scenes = scenes

#     if is_rgb == True:
#         data_path = 'image_02/data'
#     else:
#         data_path = 'image_00/data'

    clips = []
    for scene in my_scenes:
        scene_path = join(DATASET_PATH, scene)
        for s in sorted(listdir(scene_path)):
            if isdir(join(scene_path, s)):
                img_folder = join(scene_path, s)
                all_frames = []
                # loop over all the images in the folder (0.png,1.png,..,199.png)
                dir_path = listdir(img_folder)
                dir_path = sorted(dir_path, key=lambda name: int(name[0:-4]))
                for i in dir_path:
                    if str(join(img_folder, i))[-3:] == "png":
                        img_path = join(img_folder, i)
                        all_frames.append(img_path)
                # get the 10-frames sequences from the list of images after applying data augmentation
                for stride in range(1, 2):
                    clips.extend(get_clips_by_stride(stride=stride, frames_list=all_frames, sequence_size=11))
    return clips

def test_video(video_path):
    """
    Returns
    -------
    list
        A list of training sequences
    """
    def get_clips(frames_list, sequence_size=11):
        clips = []
        clip = []
        cnt = 0
        sz = len(frames_list)
        for i in range(0, sz-sequence_size):
            for idx in range(i, i+sequence_size):
                clip.append(frames_list[idx])
            clips.append(clip)
            clip = []
        return clips
    
    all_frames = []
    # loop over all the images in the folder (0.png,1.png,..,199.png)
    dir_path = listdir(video_path)
    dir_path = sorted(dir_path, key=lambda name: int(name[0:-4]))
    for i in dir_path:
        if str(join(video_path, i))[-3:] == "png":
            img_path = join(video_path, i)
            all_frames.append(img_path)
    clips = get_clips(frames_list=all_frames, sequence_size=11)
#     clips = get_clips_by_stride(stride=1, frames_list=all_frames, sequence_size=11)
    return clips

def get_clips_by_stride(stride, frames_list, sequence_size):
    """ For data augmenting purposes.
    Parameters
    ----------
    stride : int
        The desired distance between two consecutive frames
    frames_list : list
        A list of image path
    sequence_size: int
        The size of sequence
    Returns
    -------
    list
        A list of clips , 10 frames each ex. [0, 1, 2, 3, 4, 5]
                                             [6, 7, 8, 9, 10, 11]
    """
    clips = []
    sz = len(frames_list)
    clip = []
    cnt = 0
    for start in range(0, stride):
        for i in range(start, sz, stride):
            clip.append(frames_list[i])
            cnt = cnt + 1
            if cnt == sequence_size:
                clips.append(clip)
                clip = []
                cnt = 0
    return clips

def get_clips_by_stride2(stride, frames_list, sequence_size):
    """ For data augmenting purposes.
    Parameters
    ----------
    stride : int
        The desired distance between two consecutive frames
    frames_list : list
        A list of image path
    sequence_size: int
        The size of sequence
    Returns
    -------
    list
        A list of clips , 10 frames each  [0, 1, 2, 3, 4, 5]
                                          [1, 2, 3, 4, 5, 6]
    """
    clips = []
    sz = len(frames_list)
    clip = []
    cnt = 0
    for start in range(0, sz-sequence_size):
        for i in range(start, start+sequence_size):
            clip.append(frames_list[i])
        clips.append(clip)
        clip = []
    return clips

class training_data_kitti(Dataset):
    def __init__(self, list_datasets, is_rgb=False, transform=None):
        self.list_datasets = list_datasets
        self.is_rgb = is_rgb
        self.transform = transform
        self.h = 224
        self.w = 224
        self.c = 3 if self.is_rgb == True else 1

    def __getitem__(self, index):
        data_path = self.list_datasets[index]
        X = self.read_data(data_path)
        return X

    def __len__(self):
        return len(self.list_datasets)
    
    def read_data(self, path_list):
        X = np.empty((len(path_list), self.c, self.h, self.w))
        X = torch.from_numpy(X).float()
        
        for idx in range(len(path_list)):
            if self.is_rgb:
                img = cv2.imread(path_list[idx])
                img = cv2.resize(img,(self.w, self.h)).astype(np.float32)
            else:
                img = cv2.imread(path_list[idx], cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img,(self.w, self.h))
                img = img.reshape(img.shape[0], img.shape[1], 1).astype(np.float32)
            if self.transform is not None:
                X[idx] = self.transform(img)
        
        return X
'''    
if __name__ == "__main__":
    class Config:
        DATASET_PATH = '/media/kuo/124C0E504C0E2ED3/KITTI'
        SINGLE_TEST_PATH = ""
        IS_RGB=False
        BATCH_SIZE = 4
        EPOCHS = 4

    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                std=[0.5, 0.5, 0.5])

    train_transform = custom_transforms.Compose([
            custom_transforms.ArrayToTensor()])

    # kitti_training_set: scenes = ['campus', 'city', 'residential', 'road']
    kitti_training_set = kitti_training_set(DATASET_PATH = Config.DATASET_PATH, scenes=['city', 'residential', 'road'],
                                            is_rgb = Config.IS_RGB)

    training_data_kitti = training_data_kitti(kitti_training_set, is_rgb = Config.IS_RGB, transform=train_transform)
'''