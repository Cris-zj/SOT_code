import os
import glob
import cv2
import numpy as np
import pdb

def linear_mapping(images):
    max_value = images.max()
    min_value = images.min()

    parameter_a = 1 / (max_value - min_value)
    parameter_b = 1 - max_value * parameter_a

    image_after_mapping = parameter_a * images + parameter_b

    return image_after_mapping

def gaussC(xx, yy, sigma, center):
    cx = center[0]
    cy = center[1]
    exponent = ((xx - cx)**2 + (yy - cy)**2) / (2 * sigma)
    return np.exp(-exponent)

def preprocess(img):
    h, w = img.shape
    win = window2(h, w)
    eps = 1e-5
    img = np.log(img + 1.)
    img = (img - np.mean(img)) / (np.std(img) + eps)
    img = img * win
    return img

def window2(height, width):
    maskr, maskc = np.meshgrid(np.hanning(width), np.hanning(height))
    return maskr * maskc

def rand_warp(img):
    a = -180 / 16
    b = 180 / 16
    r = a + (b - a) * np.random.uniform()
    matrix_rot = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), r, 1)
    img_rot = cv2.warpAffine(np.uint8(img * 255), matrix_rot, (img.shape[1], img.shape[0]))
    img_rot = img_rot.astype(np.float32) / 255
    return img_rot

def main():
    image_dir = os.path.join('./Surfer/img')
    image_list = glob.glob(os.path.join(image_dir, '*.jpg'))
    image_list.sort()
    im = cv2.imread(image_list[0])

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter('result_video.avi', fourcc, 5, (im.shape[1], im.shape[0]))

    gt_file = os.path.join('./Surfer/groundtruth_rect.txt')
    gt = []
    with open(gt_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.split('\t')
        gt.append([int(float(line[0])), int(float(line[1])), int(float(line[2])), int(float(line[3]))])

    init_rect = gt[0]
    cx = init_rect[0] + init_rect[2] /2
    cy = init_rect[1] + init_rect[3] /2

    sigma = 100
    im_w = im.shape[1]
    im_h = im.shape[0]
    xx, yy = np.meshgrid(np.arange(im_w), np.arange(im_h))
    g = gaussC(xx, yy, sigma, [cx, cy])
    g = linear_mapping(g)

    img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


    x1 = init_rect[0]
    y1 = init_rect[1]
    w = init_rect[2]
    h = init_rect[3]
    img = img[y1:y1+h, x1:x1+w]
    g = g[y1:y1+h, x1:x1+w]
    G = np.fft.fft2(g)

    fi = preprocess(cv2.resize(img, (w, h)))
    Ai = G * np.conj(np.fft.fft2(fi))
    Bi = np.fft.fft2(fi) * np.conj(np.fft.fft2(fi))

    N = 128
    for i in range(N):
        fi = preprocess(img)# preprocess(rand_warp(img))
        Ai = Ai + (G * np.conj(np.fft.fft2(fi)))
        Bi = Bi + (np.fft.fft2(fi) * np.conj(np.fft.fft2(fi)))

    eta = 0.125
    for i in range(len(image_list)):
        im = cv2.imread(image_list[i])
        img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if i == 0:
            Ai *= eta
            Bi *= eta
        else:
            Hi = Ai / Bi
            fi = img[y1:y1+h, x1:x1+w]
            fi = preprocess(cv2.resize(fi, (int(w), int(h))))
            Gi = np.fft.ifft2(Hi * np.fft.fft2(fi))
            gi = np.real(linear_mapping(Gi))
            maxval = np.max(gi)
            maxpos = np.where(gi == maxval)

            dy = int(np.mean(maxpos[0]) - int(h / 2))
            dx = int(np.mean(maxpos[1]) - int(w / 2))
            
            x1 += dx
            y1 += dy
            
            fi = img[y1:y1+h, x1:x1+w]    
            fi = preprocess(cv2.resize(fi, (int(w), int(h))))

            Ai = eta * (G * np.conj(np.fft.fft2(fi))) + (1 - eta) * Ai
            Bi = eta * (np.fft.fft2(fi) * np.conj(np.fft.fft2(fi))) + (1 - eta) * Bi
        
        cv2.rectangle(im, (x1, y1), (x1+w, y1+h), (255, 0, 0), 2)
        video.write(im)

    video.release()

if __name__ == '__main__':
    main()
