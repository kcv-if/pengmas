import cv2, sys, os
import numpy as np
from scipy.misc import imresize
from tqdm import trange

def create3D(greyed, image_name):
    fid = open('result_{}.obj'.format(image_name), 'w')
    scl = 20 #ngatur ketinggian
    med = np.max(greyed) #np.max(greyed) kalau mau timbul, np.median(greyed) kalau cekung
    rows, columns = greyed.shape
    for a in trange(rows, desc='create vertex'):
        for b in range(columns):
            if greyed[a][b] == 0:
                t = greyed[a][b]
                f = t*med/255.0*scl
            elif greyed[a][b] > med:
                t = med - (greyed[a][b] - med)
                f = t/255.0*scl
            else:
                t = greyed[a][b]
                f = t/255.0*scl
            fid.write('v {} {} {}\n'.format(a+1, b+1, f))
    fid.write('s 1\n')
    x = rows * columns
    for d in trange(1, x+1, desc='create triangle mesh'):
        if d+columns+1 <= x:
            if d%2==1 and d%columns!=0:
                fid.write('f {} {} {}\n'.format(d, d+1, d + columns))
            if d%2==1 and (d+columns)%columns != 1:
                fid.write('f {} {} {}\n'.format(d, d-1, d+columns))
            if d%2==0 and (d+columns)%columns !=1:
                fid.write('f {} {} {}\n'.format(d, d+columns-1, d+columns))
            if d%2==0 and d%columns!=0:
                fid.write('f {} {} {}\n'.format(d, d+columns, d+columns+1))
    fid.close()
    print('result_{}.obj created'.format(image_name))

def process_image(image_object, image_name):
    copied = image_object.copy()
    edged = cv2.Canny(copied, 10, 250)
    # cv2.imshow('edged', edged)
    kernel = np.ones((5, 5), np.uint8)
    dilation = cv2.dilate(edged, kernel, iterations=1)
    # cv2.imshow('dilation', dilation)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow('closing', closing)
    (image, cnts, hiers) = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont = cv2.drawContours(copied, cnts, -1, (0, 0, 0), 1, cv2.LINE_AA)
    # cv2.imshow('contour', cont)
    mask = np.zeros(cont.shape[:2], dtype="uint8") * 255
    cv2.drawContours(mask, cnts, -1, (255, 255, 255), -1)
    img = cv2.bitwise_and(cont, cont, mask=mask)
    # cv2.imshow('masked', img)
    # cv2.waitKey(0)
    new_img = None
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w > 50 and h > 130:
            new_img = img[y:y + h, x:x + w]
    if new_img is not None:
        cv2.imwrite('cropped_{}.png'.format(image_name), new_img)
        print('cropped_{}.png created'.format(image_name))
    return new_img

if __name__ == '__main__':
    image_path = sys.argv[1:]
    if len(image_path) == 0:
        print('please insert image file name, ex: python app.py [image1] [image2] [image3]....')
        exit()
    for individual in image_path:
        image_name = os.path.basename(individual)
        image_object = cv2.imread(individual,0)
        if image_object is None:
            print('{} is not an image file'.format(image_name))
            exit()
        row, col = image_object.shape
        if row > col:
            if row > 1000:
                m = 1000.0/row
            else:
                m = 1.0
        else:
            if col > 1000:
                m = 1000.0/col
            else:
                m = 1.0
        resized_image = imresize(image_object, m)
        new_image = process_image(resized_image, image_name.split('.')[0])
        if new_image is not None:
            create3D(new_image, image_name.split('.')[0])
        else:
            print('image processing failed, create 3d from original image')
            create3D(resized_image, image_name.split('.')[0])
        print('\n')