import cv2
from PIL import Image
import numpy as np

import torchvision.transforms as transforms
import imgaug as ia

from imgaug import augmenters as iaa
from imgaug.augmenters import Augmenter
import imgaug.parameters as iap
from functools import partial
import torch

sometimes = lambda aug: iaa.Sometimes(0.4, aug)
often = lambda aug: iaa.Sometimes(0.8, aug)





class ImgAugTransform(object):
    def __init__(self, shared_transform=None, geom_transform=None, color_transform=None):
        self.shared_transform = shared_transform if shared_transform is not None else Identity
        self.geom_transform = geom_transform if geom_transform is not None else Identity
        self.color_transform = color_transform if color_transform is not None else Identity
        
    def __call__(self, imgs, masks=[]):
        
        one = False
        if not isinstance(imgs, list):
            one = True
            imgs = [imgs]

        pil = False
        if isinstance(imgs[0], Image.Image):
            pil   = True
            imgs  = [np.array(img) for img in imgs]
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

        if len(masks) == 0:
            if one:
                return imgs[0]
            
            return imgs

        return imgs, masks 


class ImgAugTransformNew(object):
    def __init__(self, shared_transform_pre=None, mask_transform=None, color_transform=None, shared_transform_post=None):
        self.shared_transform_pre = shared_transform_pre if shared_transform_pre is not None else Identity
        self.shared_transform_post = shared_transform_post if shared_transform_post is not None else Identity
        self.mask_transform = mask_transform if mask_transform is not None else Identity
        self.color_transform = color_transform if color_transform is not None else Identity
        
    def to_deterministic(self):
        self.shared_transform_pre_det  = self.shared_transform_pre.to_deterministic()
        self.mask_transform_det  = self.mask_transform.to_deterministic()
        
        self.color_transform_det  = self.color_transform.to_deterministic()
        self.shared_transform_post_det  = self.shared_transform_post.to_deterministic()


    def __call__(self, color, masks=[], call_to_deterministic=True):
        if not isinstance(color, list):
            assert False

        if call_to_deterministic:
            self.to_deterministic()

        pil = False
        if isinstance(color[0], Image.Image):
            pil = True
            color = [np.array(img) for img in color]
            masks = [np.array(img) for img in masks]


        
        
        color  = [self.shared_transform_pre_det.augment_image(x) for x in color]
        masks = [self.shared_transform_pre_det.augment_image(x) for x in masks]


        masks = [self.mask_transform_det.augment_image(x) for x in masks]


        color  = [self.color_transform_det.augment_image(x) for x in color]


        color  = [self.shared_transform_post_det.augment_image(x) for x in color]
        masks = [self.shared_transform_post_det.augment_image(x) for x in masks]

        if pil:
            color =  [Image.fromarray(x) for x in color]
            masks = [Image.fromarray(x) for x in masks]

        return color, masks 



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
        
    # print(h - crop_size + 1)
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
# ------------- ShiftHSV  -------------------------
# -------------------------------------------------

# def clip(img, dtype, maxval):
#     return np.clip(img, 0, maxval).astype(dtype)

# def shift_hsv_(img, hue_shift, sat_shift, val_shift):
#     dtype = img.dtype
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
#     if dtype == np.uint8:
#         img = img.astype(np.int32)
#     hue, sat, val = cv2.split(img)
#     hue = cv2.add(hue, hue_shift)
#     hue = np.where(hue < 0, hue + 180, hue)
#     hue = np.where(hue > 180, hue - 180, hue)
#     hue = hue.astype(dtype)
#     sat = clip(cv2.add(sat, sat_shift), dtype, 255 if dtype == np.uint8 else 1.0)
#     val = clip(cv2.add(val, val_shift), dtype, 255 if dtype == np.uint8 else 1.0)
#     img = cv2.merge((hue, sat, val)).astype(dtype)
#     img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
#     return img

# def get_crop_xy(hue_shift, sat_shift_range, val_shift, random_state):
#     h, w = img.shape[0], img.shape[1]
        
#     hue_shift = random_state.randint(hue_shift_range[0], hue_shift_range[1])
#     sat_shift = random_state.randint(sat_shift_range[0], sat_shift_range[1])
#     val_shift = random_state.randint(val_shift_range[0], val_shift_range[1])

#     return hue_shift, sat_shift, val_shift


# def shift_hsv(images, random_state, parents, hooks, **kwargs):

        
#     hue_shift, sat_shift, val_shift = get_shift_params(kwargs['hue_shift'], kwargs['sat_shift'], kwargs['val_shift'], random_state)        
    
#     out = []
#     for img in images:
#         out.append(shift_hsv_(img, hue_shift, sat_shift, val_shift))

#     return out


# def ShiftHSVRandomCrop(crop_size, shared_crop):
#     return LambdaKW(
#                     func_images=shift_hsv,
#                     func_keypoints=crop_keypoints,
#                     crop_size=crop_size,
#                     shared_crop=shared_crop
#     )

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
    return [cv2.resize(img, (resolution['width'], resolution['height']), interpolation=interpolation) if (img.shape[0] != resolution['height']) or (img.shape[1] != resolution['width']) else img  for img in images]

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


    
class Rot90(Augmenter):
    """
    Augmenter to rotate images by multiples of 90 degrees.
    This could also be achieved using ``Affine``, but Rot90 is significantly more efficient.
    Parameters
    ----------
    k : int or list of int or tuple of int or imaug.ALL or imgaug.parameters.StochasticParameter, optional
        How often to rotate by 90 degrees.
            * If a single int, then that value will be used for all images.
            * If a tuple ``(a, b)``, then a random value from the discrete
              range ``a <= x <= b`` is picked per image.
            * If a list, then for each image a random value will be sampled
              from that list.
            * If imgaug.ALL, then equivalant to list ``[0, 1, 2, 3]``.
            * If StochasticParameter, then that parameter is queried per image
              to sample the value to use.
    keep_size : bool, optional
        After rotation by an odd-valued `k` (e.g. 1 or 3), the resulting image
        may have a different height/width than the original image.
        If this parameter is set to True, then the rotated
        image will be resized to the input image's size. Note that this might also
        cause the augmented image to look distorted.
    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.
    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.
    random_state : None or int or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.
    Examples
    --------
    >>> aug = iaa.Rot90(1)
    Rotates all images by 90 degrees.
    Resizes all images afterwards to keep the size that they had before augmentation.
    This may cause the images to look distorted.
    >>> aug = iaa.Rot90([1, 3])
    Rotates all images by 90 or 270 degrees.
    Resizes all images afterwards to keep the size that they had before augmentation.
    This may cause the images to look distorted.
    >>> aug = iaa.Rot90((1, 3))
    Rotates all images by 90, 180 or 270 degrees.
    Resizes all images afterwards to keep the size that they had before augmentation.
    This may cause the images to look distorted.
    >>> aug = iaa.Rot90((1, 3), keep_size=False)
    Rotates all images by 90, 180 or 270 degrees.
    Does not resize to the original image size afterwards, i.e. each image's size may change.
    """

    def __init__(self, k, keep_size=True, name=None, deterministic=False, random_state=None):
        super(Rot90, self).__init__(name=name, deterministic=deterministic, random_state=random_state)
        self.k = iap.handle_discrete_param(k, "k", value_range=None, tuple_to_uniform=True, list_to_choice=True,
                                           allow_floats=False)
        self.keep_size = keep_size

    def _draw_samples(self, nb_images, random_state):
        return self.k.draw_samples((nb_images,), random_state=random_state)

    def _augment_arrays(self, arrs, random_state, resize_func):
        ks = self._draw_samples(len(arrs), random_state)
        return self._augment_arrays_by_samples(arrs, ks, self.keep_size, resize_func), ks

    @classmethod
    def _augment_arrays_by_samples(cls, arrs, ks, keep_size, resize_func):
        input_was_array = ia.is_np_array(arrs)
        input_dtype = arrs.dtype if input_was_array else None
        arrs_aug = []
        for arr, k_i in zip(arrs, ks):
            arr_aug = np.rot90(arr, k_i)
            if keep_size and arr.shape != arr_aug.shape and resize_func is not None:
                arr_aug = resize_func(arr_aug, arr.shape[0:2])
            arrs_aug.append(arr_aug)
        if keep_size and input_was_array:
            n_shapes = len(set([arr.shape for arr in arrs_aug]))
            if n_shapes == 1:
                arrs_aug = np.array(arrs_aug, dtype=input_dtype)
        return arrs_aug

    def _augment_images(self, images, random_state, parents, hooks):
        resize_func = partial(ia.imresize_single_image, interpolation="cubic")
        images_aug, _ = self._augment_arrays(images, random_state, resize_func)
        return images_aug

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        arrs = [heatmaps_i.arr_0to1 for heatmaps_i in heatmaps]
        arrs_aug, ks = self._augment_arrays(arrs, random_state, None)
        heatmaps_aug = []
        for heatmaps_i, arr_aug, k_i in zip(heatmaps, arrs_aug, ks):
            shape_orig = heatmaps_i.arr_0to1.shape
            heatmaps_i.arr_0to1 = arr_aug
            if self.keep_size:
                heatmaps_i = heatmaps_i.scale(shape_orig[0:2])
            elif k_i % 2 == 1:
                h, w = heatmaps_i.shape[0:2]
                heatmaps_i.shape = tuple([w, h] + list(heatmaps_i.shape[2:]))
            else:
                # keep_size was False, but rotated by a multiple of 2, hence height and width do not change
                pass
            heatmaps_aug.append(heatmaps_i)
        return heatmaps_aug

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        nb_images = len(keypoints_on_images)
        ks = self._draw_samples(nb_images, random_state)
        result = []
        for kpsoi_i, k_i in zip(keypoints_on_images, ks):
            if (k_i % 4) == 0:
                result.append(kpsoi_i)
            else:
                k_i = k_i % 4  # this is also correct when k_i is negative
                kps_aug = []
                h, w = kpsoi_i.shape[0:2]
                h_aug, w_aug = (h, w) if (k_i % 2) == 0 else (w, h)
                for kp in kpsoi_i.keypoints:
                    y, x = kp.y, kp.x
                    y_diff = abs(h - y)
                    x_diff = abs(w - x)
                    if k_i == 1:
                        # (W-yd, xd)
                        x_aug = w_aug - y_diff
                        y_aug = x_diff
                    elif k_i == 2:
                        # (xd, yd)
                        x_aug = x_diff
                        y_aug = y_diff
                    else:  # k_i == 3
                        # (yd, H-xd)
                        x_aug = y_diff
                        y_aug = h_aug - x_diff
                    kps_aug.append(ia.Keypoint(x=x_aug, y=y_aug))

                shape_aug = tuple([h_aug, w_aug] + list(kpsoi_i.shape[2:]))
                kpsoi_i_aug = ia.KeypointsOnImage(kps_aug, shape=shape_aug)
                if self.keep_size and (h, w) != (h_aug, w_aug):
                    kpsoi_i_aug = kpsoi_i_aug.on(kpsoi_i.shape)
                    kpsoi_i_aug.shape = kpsoi_i.shape

                result.append(kpsoi_i_aug)
        return result

    def get_parameters(self):
        return [self.k, self.keep_size]