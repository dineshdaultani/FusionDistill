import torch
import numpy as np

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Implementation based on the implementation from:
    https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes =  1, length = 8):
        self.n_holes = n_holes
        self.length = length
        
    def __call__(self, img, mask):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
            mask (Tensor): Mask with n_holes of dimension length x length cut out of it.
        Returns:
            Tensor: Image with applied mask that contains n_holes of dimension length 
                    x length cut out of it.
        """
        img = img * mask
        return img
    
    def get_mask(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Mask with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        return mask
