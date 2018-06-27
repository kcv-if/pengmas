from skimage.color import rgb2gray
from scipy.misc import imresize
from scipy.signal import convolve2d
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import cv2, numpy

def imgradient(gx, gy):
    sobelx = cv2.Sobel(gx, cv2.CV_64F,1,0)
    sobely = cv2.Sobel(gy, cv2.CV_64F,0,1)

    magnitude = numpy.sqrt(sobelx**2.0 + sobely**2.0)
    angle = numpy.arctan2(sobely, sobelx) * (180 / numpy.pi)
    return magnitude, angle

def blendImage(image1, image2):
    newimage = image1
    for x in range(len(image1)):
        for y in range(len(image1[x])):
            newimage[x][y] = (image1[x][y]+image2[x][y])/2
    return newimage

def conv2(x, y, mode='same'):
    return numpy.rot90(convolve2d(numpy.rot90(x, 2), numpy.rot90(y, 2), mode=mode), 2)

if __name__ == '__main__':
    image = cv2.imread('test.png')
    row, col, channel = image.shape
    if row > col:
        if row > 1000:
            m = 1000.0/row
        else:
            m = 1
    else:
        if col > 1000:
            m = 1000.0/col
        else:
            m = 1
    image_resized = imresize(image, m)
    image = rgb2gray(image_resized)
    image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    krnl_x = [[1, 0, -1]]
    krnl_y = [[1], [0], [-1]]
    Gx = conv2(image, krnl_x)
    Gy = conv2(image, krnl_y)
    gausfitX = gaussian_filter(Gx, 0.5)
    gausfitY = gaussian_filter(Gy, 0.5)
    G_X = numpy.array(Gx) + (numpy.array(Gx) - numpy.array(gausfitX)) * 0.1
    G_Y = numpy.array(Gy) + (numpy.array(Gy) - numpy.array(gausfitY)) * 0.1
    Gmag, Gdir = imgradient(G_X, G_Y)
    for x in range(len(Gmag)):
        for y in range(len(Gmag[x])):
            if Gmag[x][y] > 0:
                Gmag[x][y] = float(1)
            else:
                Gmag[x][y] = float(0)
    Amg = gaussian_filter(Gmag, 2)
    fuse = blendImage(G_X, G_Y)
    Am = fuse-(fuse-gaussian_filter(fuse, 2))

    fig = plt.figure()
    fig.add_subplot(1,2,1)
    plt.imshow(Am)
    fig.add_subplot(1, 2, 2)
    plt.imshow(image_resized)
    plt.show()