import os
import cv2
import tqdm
import torch
import numpy as np
from typing import Callable


def apply(root_dir: str, name: str, function: Callable):
    data = torch.load(os.path.join(root_dir, 'partition_db.data'))
    imgs = [d[0] for d in data]
    os.makedirs(os.path.join(root_dir, name), exist_ok=True)
    for img_name in tqdm.tqdm(os.listdir(os.path.join(root_dir, 'images_resized')), desc=name):
        # if img_name != "000001.jpg": continue
        if img_name not in imgs: continue
        img = cv2.imread(os.path.join(root_dir, 'images_resized', img_name))
        img = function(img)
        cv2.imwrite(os.path.join(root_dir, name, img_name), img)


def _add_gaussian_noise(image, sigma):
    h, w, c = image.shape
    noise = np.random.normal(0, sigma, (h, w, c))
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image


def _calculate_psnr(original_image, noisy_image):
    mse = np.mean((original_image - noisy_image) ** 2)
    if mse == 0: return float('inf')
    max_pixel = 255.0
    psnr = 10 * np.log10(max_pixel ** 2 / mse)
    return psnr


def apply_noise(root_dir: str, psnr_min: int, psnr_max: int):
    def _function(img):
        psnr_mean = (psnr_max + psnr_min) / 2
        sigma = np.sqrt(255 ** 2 / (10 ** (psnr_mean / 10)))
        noisy_image = _add_gaussian_noise(img, sigma)
        calculated_psnr = _calculate_psnr(img, noisy_image)
        # print(calculated_psnr)
        return noisy_image
    apply(root_dir, f'noise_{psnr_min}dB_{psnr_max}dB', _function)



def apply_limunesence_square(root_dir: str):
    def _function(img):
        hls_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        hls_image[:,:,1] = np.clip(hls_image[:,:,1] ** 2, 0, 255)
        lower_luminance_image = cv2.cvtColor(hls_image, cv2.COLOR_HLS2BGR)
        return lower_luminance_image
    apply(root_dir, f'lum_square', _function)


def apply_limunesence_linear(root_dir: str, c: float):
    def _function(img):
        hls_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        hls_image[:,:,1] = np.clip(hls_image[:,:,1] * c, 0, 255)
        lower_luminance_image = cv2.cvtColor(hls_image, cv2.COLOR_HLS2BGR)
        return lower_luminance_image
    apply(root_dir, f'lum_linear_{c:.2f}', _function)


def apply_limunesence_const(root_dir: str, c: float):
    def _function(img):
        hls_image = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        hls_image[:,:,1] = np.clip(hls_image[:,:,1] + c, 0, 255)
        lower_luminance_image = cv2.cvtColor(hls_image, cv2.COLOR_HLS2BGR)
        return lower_luminance_image
    apply(root_dir, f'lum_const_{c}', _function)


if __name__ == '__main__':
    # apply_noise('data/inputs', 50, 80)
    # apply_noise('data/inputs', 40, 50)
    # apply_noise('data/inputs', 30, 40)
    # apply_noise('data/inputs', 20, 30)
    # apply_noise('data/inputs', 10, 20)
    # apply_limunesence_square('data/inputs')
    # apply_limunesence_linear('data/inputs', 0.5)
    # apply_limunesence_linear('data/inputs', 0.6)
    # apply_limunesence_linear('data/inputs', 0.75)
    # apply_limunesence_linear('data/inputs', 1.33)
    # apply_limunesence_linear('data/inputs', 1.5)
    # apply_limunesence_const('data/inputs', -100)
    # apply_limunesence_const('data/inputs', -20)
    # apply_limunesence_const('data/inputs', -10)
    # apply_limunesence_const('data/inputs', 30)
    pass
