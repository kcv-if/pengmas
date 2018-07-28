import cv2
import numpy as np

def create3D(image):
    fid = open('result4.obj', 'w')
    scl = 20
    greyed = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    med = np.median(greyed) #np.max(greyed) kalau mau timbul
    rows, columns = greyed.shape
    counter = 0
    for a in range(rows):
        for b in range(columns):
            if greyed[a][b] < 0.99:
                t = greyed[a][b]
                f = t*med/255.0*scl
            elif greyed[a][b] > med:
                t = med - (greyed[a][b] - med)
                f = t/255.0*scl
            else:
                t = greyed[a][b]
                f = t/255.0*scl
            fid.write('v {} {} {}\n'.format(a+1, b+1, f))
            counter = counter + 1
    fid.write('s 1\n')

    x = rows * columns
    print(x, counter)
    for d in range(1, x+1):
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

image = cv2.imread("test4.jpg")
copy = image.copy()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Gray', gray)
edged = cv2.Canny(gray, 10, 250)
# cv2.imshow('Edged', edged)
kernel = np.ones((5, 5), np.uint8)
dilation = cv2.dilate(edged, kernel, iterations=1)
# cv2.imshow('Dilation', dilation)
closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
# cv2.imshow('Closing', closing)
(image, cnts, hiers) = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cont = cv2.drawContours(copy, cnts, -1, (0, 0, 0), 1, cv2.LINE_AA)
# cv2.imshow('Contours', cont)
mask = np.zeros(cont.shape[:2], dtype="uint8") * 255
# Draw the contours on the mask
cv2.drawContours(mask, cnts, -1, (255, 255, 255), -1)
# remove the contours from the image and show the resulting images
img = cv2.bitwise_and(cont, cont, mask=mask)
# cv2.imshow("Mask", img)
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    if w > 50 and h > 130:
        new_img = img[y:y + h, x:x + w]
        cv2.imwrite('Cropped4.png', new_img)
        create3D(new_img)