import torch
import random
import numpy as np
import cv2

'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        for t, m, s in zip(image, self.mean, self.std):
            t.sub_(m).div_(s)
        return image

class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) to torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, image):
        # put it from HWC to CHW format
        image = np.transpose(image, (2, 0, 1))
        # handle numpy array
        image = torch.from_numpy(image).float()/255
        return image


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy array with a probability of 0.5"""

    def __call__(self, image):
        if random.random() < 0.5:
            output_image = np.copy(np.fliplr(image))
        else:
            output_image = image
        return output_image