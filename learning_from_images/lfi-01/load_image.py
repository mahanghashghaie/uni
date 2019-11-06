import cv2
import numpy as np


def load_images(img_location):
    img_raw = cv2.imread(img_location)
    img_gray = cv2.cvtColor(img_raw, cv2.COLOR_RGB2GRAY)
    img_gray_3channel = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    return img_raw, img_gray_3channel
if __name__ == '__main__':
    img_raw, img_gray = load_images('./data/Lenna.png')
    both_images = np.hstack((img_raw, img_gray))
    cv2.imshow("3 Channel Gray Scale image and BGR Image", both_images)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
