import numpy as np
import cv2

def dark_channel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark

def transmission_estimate(im, sz):
    A = im.max()
    omega = 0.95
    dark = dark_channel(im, sz)
    transmission = 1 - omega * dark / A
    return transmission

def atmospheric_light(im, transmission, A_min=220):
    A = im.max()
    if A < A_min:
        A = A_min
    return A

def dehaze(im, t, A, t0=0.1):
    im = im.astype('float64') / 255.0
    J = np.zeros(im.shape)
    for i in range(3):
        J[:, :, i] = (im[:, :, i] - A) / np.maximum(t, t0) + A
    J = np.clip(J, 0, 1)
    J = (J * 255).astype('uint8')
    return J

if __name__ == '__main__':
    input_image_path = 'input_image.jpg'
    output_image_path = 'dehazed_image.jpg'
    
    image = cv2.imread(input_image_path)
    window_size = 15
    transmission = transmission_estimate(image, window_size)
    atmospheric = atmospheric_light(image, transmission)
    dehazed_image = dehaze(image, transmission, atmospheric)
    
    cv2.imwrite(output_image_path, dehazed_image)
