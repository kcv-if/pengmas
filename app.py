from skimage.color import rgb2gray
from scipy.misc import imresize
from scipy.signal import convolve2d
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import cv2, numpy
from tqdm import tqdm

def imgradient(gx, gy):
    sobelx = cv2.Sobel(gx, cv2.CV_64F,1,0)
    sobely = cv2.Sobel(gy, cv2.CV_64F,0,1)

    magnitude = numpy.sqrt(sobelx**2.0 + sobely**2.0)
    angle = numpy.arctan2(sobely, sobelx) * (180 / numpy.pi)
    return magnitude, angle

def blendImage(image1, image2):
    newImage = (image1 + image2)/2.0
    return newImage

def conv2(x, y, mode='same'):
    return numpy.rot90(convolve2d(numpy.rot90(x, 2), numpy.rot90(y, 2), mode=mode), 2)

def savecontoh(hasil, asli, flname='hasil1.jpg'):
    hh = asli*255
    jj = hasil*255
    jj = 255 - jj
    img_hasil = numpy.vstack((jj, hh))
    cv2.imwrite(flname, img_hasil)
    
def trianglemesh(Am, Amg, fuse):
    scl = 30
    fid = open('result.obj', 'w')
    towrite = ''
    med = numpy.median(fuse)
    minim = numpy.min(fuse)
    rows, columns = fuse.shape
    for a in range(rows):
        for b in range(columns):
            if Amg[a][b] < 0.99:
                t = Amg[a][b]
                f = t*med/255*scl
            elif Am[a][b] > med:
                t = med - (Am[a][b] - med)
                f = t/255*scl
            else:
                t = Am[a][b]
                f = t/255*scl
            towrite += 'v {} {} {}\r\n'.format(a+1, b+1, f)
    towrite += 's 1\r\n'
    
    x = rows*columns
    for d in tqdm(range(1, x+1)):
        if d/columns <= rows-1:
            if (d+columns) % 2 != 0 and d % columns != 0:
                towrite += 'f {} {} {}\r\n'.format(d, d+1, columns + d)
            if (d+columns) % 2 != 0 and (d+columns) % columns != 1:
                towrite += 'f {} {} {}\r\n'.format(d, d-1, columns+d)
            if (d+columns) % 2 == 0 and d % columns != 1:
                towrite += 'f {} {} {}\r\n'.format(d, columns+d, columns+d-1)
            if (d+columns) % 2 == 0 and d % columns != 0:
                towrite += 'f {} {} {}\r\n'.format(d, columns+d, columns+d+1)
    fid.write(towrite)
    fid.close()
    del towrite

if __name__ == '__main__':
    image = cv2.imread('ship.jpg')
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
#    image = cv2.normalize(image.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    h = numpy.array([[1], [2], [1]])
    v = numpy.array([1, 0, -1])
    krnl_x = h*v
    krnl_y = krnl_x.T
    Gx = convolve2d(image, krnl_x, mode='same')
    Gy = convolve2d(image, krnl_y, mode='same')
    gausfitX = gaussian_filter(Gx, 0.5)
    gausfitY = gaussian_filter(Gy, 0.5)
    G_X = numpy.array(Gx) + (numpy.array(Gx) - numpy.array(gausfitX)) * 0.1
    G_Y = numpy.array(Gy) + (numpy.array(Gy) - numpy.array(gausfitY)) * 0.1
    Gmag, Gdir = imgradient(G_X, G_Y)
    Gmag[Gmag > 0] = 1.0
    Gmag[Gmag <= 0] = 0.0
    Amg = gaussian_filter(Gmag, 1)
    fuse = blendImage(G_X, G_Y)
    Am = fuse-(fuse-gaussian_filter(fuse, 2))

    fig = plt.figure()
    fig.add_subplot(1,2,1)
    plt.imshow(Am)
    fig.add_subplot(1, 2, 2)
    plt.imshow(image_resized)
    plt.show()
    
    