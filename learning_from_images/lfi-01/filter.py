import numpy as np
import cv2


def im2double(im):
    """
    Converts uint image (0-255) to double image (0.0-1.0) and generalizes
    this concept to any range.

    :param im:
    :return: normalized image
    """
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out


def make_gaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    k = np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)
    return k / np.sum(k)

def extract_pixel_area(img, x, y, offset):
    return img[x-offset:x+offset+1,y-offset:y+offset+1]

def convolution_2d(img, kernel):
    """
    Computes the convolution between kernel and image

    :param img: grayscale image
    :param kernel: convolution matrix - 3x3, or 5x5 matrix
    :return: result of the convolution
    """
    # TODO write convolution of arbritrary sized convolution here
    # Hint: you need the kernelsize

    offset = int(kernel.shape[0]/2)
    x_new = img.shape[0]
    y_new = img.shape[1]

    newimg = np.zeros([x_new,y_new],dtype=img.dtype)
    norm_gray_img_padded = np.pad(img, offset, mode='edge')
    # Do convolution
    for x in range(offset,norm_gray_img_padded.shape[0] - offset * 2):
        for y in range(offset, norm_gray_img_padded.shape[0] - offset * 2):
            pixel_area = extract_pixel_area(norm_gray_img_padded,x,y,offset)
            value = np.sum(np.multiply(pixel_area,kernel))
            newimg[x, y] = value
    return newimg


if __name__ == "__main__":

    # 1. load image in grayscale
    # 2. convert image to 0-1 image (see im2double)
    gray_img = cv2.imread("./data/Lenna.png", cv2.IMREAD_GRAYSCALE)
    norm_gray_img = im2double(gray_img)
    
    # image kernels
    sobelmask_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobelmask_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    gk = make_gaussian(11)

    # 3 .use image kernels on normalized image
    sobel_x = convolution_2d(norm_gray_img, sobelmask_x)
    sobel_y = convolution_2d(norm_gray_img, sobelmask_y)
    mog = convolution_2d(norm_gray_img, gk)
    # 4. compute magnitude of gradients
    sqrtd = np.sqrt((sobel_x**2)+(sobel_y**2))
    # Show resulting images
    cv2.imshow("sobel_x", sobel_x)
    cv2.imshow("sobel_y", sobel_y)
    cv2.imshow("mog", mog)
    cv2.imshow("mag", sqrtd)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
