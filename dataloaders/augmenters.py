import cv2
from PIL import Image
import numpy as np

import torchvision.transforms as transforms
import imgaug as ia

from imgaug import augmenters as iaa
from imgaug.augmenters import Augmenter


sometimes = lambda aug: iaa.Sometimes(0.4, aug)
often = lambda aug: iaa.Sometimes(0.8, aug)


class ImgAugTransform(object):
    def __init__(self, shared_transform=None, geom_transform=None, color_transform=None):
        self.shared_transform = shared_transform if shared_transform is not None else Identity
        self.geom_transform = geom_transform if geom_transform is not None else Identity
        self.color_transform = color_transform if color_transform is not None else Identity
        
    def __call__(self, imgs, masks=[]):
        if not isinstance(imgs, list):
            assert False

        pil = False
        if isinstance(imgs[0], Image.Image):
            pil = True
            imgs = [np.array(img) for img in imgs]
            masks = [np.array(img) for img in masks]


        shared_transform_det  = self.shared_transform.to_deterministic()
        imgs  = [shared_transform_det.augment_image(x) for x in imgs]
        masks = [shared_transform_det.augment_image(x) for x in masks]


        geom_transform_det  = self.geom_transform.to_deterministic()
        imgs  = [geom_transform_det.augment_image(x) for x in imgs]
        masks = [geom_transform_det.augment_image(x) for x in masks]

        color_transform_det  = self.color_transform.to_deterministic()
        imgs  = [color_transform_det.augment_image(x) for x in imgs]


        if pil:
            imgs =  [Image.fromarray(x) for x in imgs]
            masks = [Image.fromarray(x) for x in masks]

        return imgs, masks 

        
# -------- -----------------------------------
# -------- GaussianBlurCV2 -------------------
# --------------------------------------------

class GaussianBlurCV2(iaa.GaussianBlur): # pylint: disable=locally-disabled, unused-variable, line-too-long

    def _augment_images(self, images, random_state, parents, hooks):
        result = images
        nb_images = len(images)
        samples = self.sigma.draw_samples((nb_images,), random_state=random_state)
        for i in range(nb_images):
            nb_channels = images[i].shape[2]
            sig = samples[i]
            if sig > 0 + self.eps:
                kernel_size = int(4 * sig) | 1
                kernel = cv2.getGaussianKernel(kernel_size, sig, cv2.CV_32F)
                result[i] = cv2.sepFilter2D(images[i], -1, kernel, kernel)

        return result

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images
    


# -------- -----------------------------------
# -------- LambdaKW --------------------------
# --------------------------------------------


class LambdaKW(Augmenter):
    def __init__(self, func_images, func_keypoints, name=None, deterministic=False, random_state=None, **kwargs):
        super(LambdaKW, self).__init__(name=name, deterministic=deterministic, random_state=random_state)
        self.func_images = func_images
        self.func_keypoints = func_keypoints
        self.kwargs = kwargs


    def _augment_images(self, images, random_state, parents, hooks):
        return self.func_images(images, random_state, parents=parents, hooks=hooks, **self.kwargs)

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        result = self.func_keypoints(keypoints_on_images, random_state, parents=parents, hooks=hooks, **self.kwargs)
        ia.do_assert(isinstance(result, list))
        ia.do_assert(all([isinstance(el, ia.KeypointsOnImage) for el in result]))
        return result

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        assert False


    def get_parameters(self):
        return []


# -------------------------------------------------
# ------------- Random Crop  ----------------------
# -------------------------------------------------


def get_crop_xy(img, crop_size, random_state):
    h, w = img.shape[0], img.shape[1]
        
    x = random_state.randint(0, w - crop_size + 1)
    y = random_state.randint(0, h - crop_size + 1)
    
    return x, y


def crop_images(images, random_state, parents, hooks, **kwargs):

    if kwargs['shared_crop']:
        assert len(set([x.shape[0] for x in images])) == 1
        assert len(set([x.shape[1] for x in images])) == 1
        
        x, y = get_crop_xy(images[0], kwargs['crop_size'], random_state)        
    
    out = []
    for img in images:
        
        if not kwargs['shared_crop']:
            x, y = get_crop_xy(img, kwargs['crop_size'], random_state)
    
        out.append(img[y: y + kwargs['crop_size'], x: x + kwargs['crop_size']])

    return out


def crop_keypoints(keypoints_on_images, random_state, parents, hooks):
    print('Not implemented')
    return keypoints_on_images


def RandomCrop(crop_size, shared_crop):
    return LambdaKW(
                    func_images=crop_images,
                    func_keypoints=crop_keypoints,
                    crop_size=crop_size,
                    shared_crop=shared_crop
    )


# -------------------------------------------------
# ------------- Identity  -------------------------
# -------------------------------------------------

def identity(x):
    return x
Identity = transforms.Lambda(identity)
def lambda_to_deterministic(self):
    return self
def lambda_augment_image(self, img):
    return self(img)
transforms.Lambda.to_deterministic = lambda_to_deterministic
transforms.Lambda.augment_image = lambda_augment_image

# -------------------------------------------------
# ------------- ResizeCV2  ------------------------
# -------------------------------------------------


def resize_cv2_(images, random_state, parents, hooks, resolution, interpolation):
    return [cv2.resize(img, (resolution['width'], resolution['height']), interpolation=interpolation) for img in images]

def resize_cv2_kp(keypoints_on_images, random_state, parents, hooks):
    print('Not implemented')
    return keypoints_on_images

def ResizeCV2(resolution, interpolation):
    return LambdaKW(
                    func_images=resize_cv2_,
                    func_keypoints=resize_cv2_kp,
                    resolution=resolution,
                    interpolation=interpolation
    )


# -------------------------------------------------
# ------------- PadOrCrop  ------------------------
# -------------------------------------------------

def pad_or_crop(img, target_height, target_width, border_mode=cv2.BORDER_REFLECT_101):
    height, width = img.shape[:2]

    if height < target_height:
        h_pad_top = int((min_height - height) / 2.0)
        h_pad_bottom = min_height - height - h_pad_top
    else:
        h_pad_top = 0
        h_pad_bottom = 0

    if width < target_width:
        w_pad_left = int((min_width - width) / 2.0)
        w_pad_right = min_width - width - w_pad_left
    else:
        w_pad_left = 0
        w_pad_right = 0

    img = cv2.copyMakeBorder(img, h_pad_top, h_pad_bottom, w_pad_left, w_pad_right, border_mode)
    
    print(img.shape)
    img = img[:target_height, :target_width]
    
    assert img.shape[0] == target_height
    assert img.shape[1] == target_width

    return img


    