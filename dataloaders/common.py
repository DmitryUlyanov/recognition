from PIL import Image
import cv2

def is_image_file(filename):
    return any(filename.lower().endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def get_image_cv2(path):
    img = cv2.imread(path, -1)
    img = img[:, :, :3]
    img = img[:, :, ::-1]
    return img

def get_image_pil(path):
    img = Image.open(path).convert('RGB')
    return img



