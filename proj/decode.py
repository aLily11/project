import cv2
import numpy as np
from myqrcode import demask, getContours, find
import x
import struct
from CRC import CRC_Decoding

first = 0
end = 0


def JReduce(image, m, n):
    H = int(image.shape[0] * m)
    W = int(image.shape[1] * n)
    size = (W, H, 3)
    iJReduce = np.zeros(size, np.float32)
    for i in range(H):
        for j in range(W):
            x1 = int(i / m)
            x2 = int((i + 1) / m)
            y1 = int(j / n)
            y2 = int((j + 1) / n)
            sum = [0, 0, 0]
            # sum = image[x1+1][y1+1]
            for k in range(x1, x2):
                for l in range(y1, y2):
                    if (k == x1 + 1 and l == y1 + 1):
                        sum = sum + image[k, l] * 0.52
                    else:
                        sum = sum + image[k, l] * 0.06
            # num = 9
            iJReduce[i][j] = sum
            # iJReduce[i][j] = sum
    return iJReduce


def checkStart(img):
    # cv2.imshow("ss",img)
    # cv2.waitKey(0)
    global first
    contours, hierachy = getContours(img)
    img = find(img, contours, np.squeeze(hierachy))  # 对空图片进行检测。以确定解码开始
    binstring = ""
    decode(img, binstring)
    return


def decode(image, binstring):
    width = x.width - 8
    # cv2.imshow("ss",image)
    # cv2.waitKey(0)
    mat = np.full((width, width, 3), 0, dtype=np.float32)
    pwidth = 10
    i = 0
    # last1 = 0
    # last2 = 0
    # sumtemp = [0,0,0]
    # for i in range(width * 10):
    #     normali = i // pwidth
    #     for j in range(width * 10):
    #         normalj = j // pwidth
    #         if(last1!=normali or last2!=normalj):
    #             mat[last1][last2]=sumtemp
    #             last1=0
    #             last2=0
    #             sumtemp=[0,0,0]
    #         if (normali < len(mat) and normalj < len(mat)):
    #             # 加权
    #             sumtemp+=image[i][j]*0.01
    #             # if i % 10 < 3 or i % 10 > 6 or j % 10 < 3 or j % 10 > 6:
    #             #     mat[normali][normalj][0] += image[i][j][0] * 0.2 / 84
    #             #     mat[normali][normalj][1] += image[i][j][1] * 0.2 / 84
    #             #     mat[normali][normalj][2] += image[i][j][2] * 0.2 / 84
    #             # else:
    #             #     mat[normali][normalj][0] += image[i][j][0] * 0.05
    #             #     mat[normali][normalj][1] += image[i][j][1] * 0.05
    #             #     mat[normali][normalj][2] += image[i][j][2] * 0.05
    mat = JReduce(image, 0.33333333333333333333, 0.33333333333333333333)
    # cv2.imshow("sss",mat)
    # cv2.waitKeyEx(0)
    row = 0
    thre = 110.0
    col = x.locWidth  #
    count = 0;
    while row < width:
        if row < x.locWidth:
            if (row + col) % 2 == 0:
                demask(mat, row, col, count, thre)
            if mat[row][col][count] > thre:
                binstring += "0"
            else:
                binstring += "1"
            count += 1;
            if count >= 3:
                count -= 3
                col += 1
                if col > width - x.locWidth - 1:
                    if row != x.locWidth - 1:
                        col = x.locWidth
                    else:
                        col = 0
                    row += 1
        elif row < width - x.locWidth:
            if (row + col) % 2 == 0:
                demask(mat, row, col, count, thre)
            if mat[row][col][count] > thre:
                binstring += "0"
            else:
                binstring += "1"
            count += 1;
            if count >= 3:
                count -= 3
                col += 1
                if (col > width - 1):
                    if row != width - x.locWidth - 1:
                        col = 0
                    else:
                        col = x.locWidth
                    row += 1
        elif row < width - x.sLocWidth:
            if (row + col) % 2 == 0:
                demask(mat, row, col, count, thre)
            if mat[row][col][count] > thre:
                binstring += "0"
            else:
                binstring += "1"
            count += 1;
            if count >= 3:
                count -= 3
                col += 1
                if col > width - 1:
                    col = x.locWidth
                    row += 1
        else:
            if (row + col) % 2 == 0:
                demask(mat, row, col, count, thre)
            if mat[row][col][count] > thre:
                binstring += "0"
            else:
                binstring += "1"
            count += 1;
            if count >= 3:
                count -= 3
                col += 1
                if col > width - x.sLocWidth - 1:
                    col = x.locWidth
                    row += 1
    # print(binstring)
    startOrEnd = 170
    startOrEndStr = ""
    for i in range(8):
        startOrEndStr += '{:08b}'.format(startOrEnd)
    global first
    global end

    if first == 0 and binstring[:64] != startOrEndStr:
        return
    elif first == 0 and binstring[:64] == startOrEndStr:
        binstring = binstring[64:]
        first = 1
    elif first == 1 and binstring[:64] == startOrEndStr:
        binstring = binstring[64:]
    elif first == 1 and binstring.find(startOrEndStr) != -1:
        if binstring[:64] != startOrEndStr:
            binstring = binstring[:binstring.find(startOrEndStr)]
            end = 1

    return binstring


def decodeFromVideo(filename):
    global end
    global first
    binstring = ""
    vc = cv2.VideoCapture(filename)
    if vc.isOpened():
        rval = True
    else:
        rval = False
    fps = 3
    k = 1
    count = 1
    record = 0
    while rval:
        rval, frame = vc.read()
        if frame is None:
            break
        if first == 0:
            print(k)
            checkStart(frame)
            if (first == 1):
                record = k + 1
        elif (k - record) % fps == 0:
            print("Processing ", count, " frame")
            contours, hierachy = getContours(frame)
            img = find(frame, contours, np.squeeze(hierachy))
            
            binstring = decode(img, binstring)
            
            binstring = wirteResult(binstring)
            print(binstring)
            count += 1
        # cv2.imwrite("./output/"+str(k)+".png",frame)
        k += 1
        if (end == 1):
            break;
    # cv2.waitKey(1)
    vc.release()
    return


def wirteResult(binstring):
    writer = open("./output/output.bin", 'ab+')
    writerCheck = open("./output/valid.bin", 'ab+')
    # count = 0
    # print(len(binstring))
    while len(binstring) >= 11:
        if CRC_Decoding(binstring[:11], x.key) == True:
            writerCheck.write(struct.pack('B', 255))
        else:
            writerCheck.write(struct.pack('B', 0))
        # count+=1
        # print("throw ",count)
        t = (int(binstring[:8], 2))
        res = struct.pack('B', t)
        # if str(res) > 'Z' or str(res) < 'a':
        #     print("wrong!!!")
        writer.write(res)
        binstring = binstring[11:]
    writer.close()
    return binstring


if __name__ == '__main__':
    decodeFromVideo("video/in.mp4")
    #frame = cv2.imread("video/2.png");
    #contours, hierachy = getContours(frame)
    #binstring = ""
    #img = find(frame, contours, np.squeeze(hierachy))
    #cv2.imshow("ss",img)
    #cv2.waitKey(0)
    #binstring = decode(img, binstring)
    #print(binstring)
    #wirteResult(binstring)
