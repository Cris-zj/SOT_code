import os
import cv2
import numpy as np
import glob
import math
import pdb


def gaussC(xx, yy, sigma, center):
    cx = center[0]
    cy = center[1]
    exponent = ((xx - cx)**2 + (yy - cy)**2) / (2 * (sigma**2))
    return np.exp(-exponent)

def preprocess(img):
    h, w = img.shape
    win = window2(w, h)
    eps = 1e-5
    img = np.log(img + 1.)
    img = (img - np.mean(img)) / (np.std(img) + eps)
    img = img * win
    return img

def window2(width, height):
    maskr, maskc = np.meshgrid(np.hanning(width), np.hanning(height))
    return maskr * maskc

def dense_gauss_kernel(sigma, x, y=None):
    xf = np.fft.fft2(x)
    xx = np.dot(x.flatten().T, x.flatten())

    if y is None:
        yf = xf
        yy = xx        
    else:
        yf = np.fft.fft2(y)
        yy = np.dot(y.flatten().T, y.flatten())
    
    xyf = xf * np.conj(yf)
    xy = np.real(np.roll(np.fft.ifft2(xyf), (math.floor(x.shape[1] / 2.), math.floor(x.shape[0] / 2.))))
    numel = x.shape[0] * x.shape[1]   
    exponent = np.maximum((xx + yy - 2 * xy) / numel, 0) / (sigma**2)
    k = np.exp(-exponent)

    return k

def main():
    padding = 1
    output_sigma_factor = 1/16
    sigma = 0.2
    lambda_   = 1e-2
    interp_factor = 0.075

    image_dir = os.path.join('./Surfer/img')
    image_list = glob.glob(os.path.join(image_dir, '*.jpg'))
    image_list.sort()
    im = cv2.imread(image_list[0])

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter('./result_video.avi', fourcc, 5, (im.shape[1], im.shape[0]))

    gt_file = os.path.join('./Surfer/groundtruth_rect.txt')
    gt = []
    with open(gt_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.split('\t')
        gt.append([int(float(line[0])), int(float(line[1])), int(float(line[2])), int(float(line[3]))])

    init_rect = gt[0]
    cx = init_rect[0] + init_rect[2] / 2
    cy = init_rect[1] + init_rect[3] / 2
    w = init_rect[2]
    h = init_rect[3]


    sz_w = w * 2
    sz_h = h * 2
    output_sigma = np.sqrt(sz_w * sz_h) * output_sigma_factor
    xx, yy = np.meshgrid(np.arange(sz_w), np.arange(sz_h))
    y = gaussC(xx, yy, output_sigma, [sz_w/2, sz_h/2])
    yf = np.fft.fft2(y)

    positions = []

    for i in range(len(image_list)):
        # print(i)
        im_BGR = cv2.imread(image_list[i])
        im = cv2.cvtColor(im_BGR, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        y1 = max(0, int(cy - sz_h/2))
        y2 = min(int(cy + sz_h/2), im.shape[0])
        x1 = max(0, int(cx - sz_w / 2))
        x2 = min(int(cx + sz_w / 2), im.shape[1])
        im_patch = im[y1: y2, x1: x2]
        x = preprocess(cv2.resize(im_patch, (sz_w, sz_h)))

        if i > 0:
            k = dense_gauss_kernel(sigma, x, z)
            response = np.real(np.fft.ifft2(alphaf * np.fft.fft2(k)))

            maxpos = np.where(response == np.max(response))

            dy = int(np.mean(maxpos[0]) - int(sz_h / 2))
            dx = int(np.mean(maxpos[1]) - int(sz_w / 2))

            cy += dy
            cx += dx
        y1 = max(0, int(cy - sz_h/2))
        y2 = min(int(cy + sz_h/2), im.shape[0])
        x1 = max(0, int(cx - sz_w / 2))
        x2 = min(int(cx + sz_w / 2), im.shape[1])
        im_patch = im[y1: y2, x1: x2]
        x = preprocess(cv2.resize(im_patch, (sz_w, sz_h)))
        k = dense_gauss_kernel(sigma, x)
        new_alphaf = yf / (np.fft.fft2(k) + lambda_)
        new_z = x

        if i == 0:
            alphaf = new_alphaf
            z = x
        else:
            alphaf = (1 - interp_factor) * alphaf + interp_factor * new_alphaf
            z = (1 - interp_factor) * z + interp_factor * new_z
        
        positions.append([cx - w / 2, cy - h/2, w, h])
        cv2.rectangle(im_BGR, (int(cx - w / 2), int(cy - h/2)), (int(cx + w / 2), int(cy + h/2)), (255, 0, 0), 2)
        video.write(im_BGR)

    video.release()

if __name__ == '__main__':
    main()
