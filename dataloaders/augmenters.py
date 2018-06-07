from imgaug import augmenters as iaa
from imgaug.augmenters import Augmenter
import imgaug as ia
import cv2

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

    def get_parameters(self):
        return []


# -------------------------------------------------
# ------------- LambdaKW --------------------------
# -------------------------------------------------


def get_crop_xy(img, crop_size, random_state):
    h, w = img.shape[0], img.shape[1]
        
    x = random_state.randint(0, w - crop_size - 1)
    y = random_state.randint(0, h - crop_size - 1)
    
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

