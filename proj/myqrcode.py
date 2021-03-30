import cv2
import numpy as np
import x


def getContours(image):  # 寻找二维码边框轮廓
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 重载为灰度图，单通道比较省事而且比较方便操作，更有利于分辨边框
    gray = cv2.blur(gray, (5, 5), 0)  # 将图片进行滤波处理，目的是排除一些椒盐噪音（除去雪花）。参数为（源图片，（分块像素规格），滤波阈值参数）
    ret, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # 进行黑白二值化处理，以255的一半（127）为界限，进行二值化
    contours, hierachy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 第二个参数表示轮廓检索模式，第三个表示轮廓的存储方法（本次实验只需要确定一个矩形，因此SIMPLE参数就够用了）
    # contours返回轮廓本身的四个顶点信息，hierachy在本次项目中由于只存在一个边框，所以返回值用处不大。
    return contours, hierachy

def detectContours(vec):#判断定位点位置（三个大的定位码需要确定）
    distance1 = np.sqrt((vec[0] - vec[2]) ** 2 + (vec[1] - vec[3]) ** 2)
    distance2 = np.sqrt((vec[0] - vec[4]) ** 2 + (vec[1] - vec[5]) ** 2)
    distance3 = np.sqrt((vec[2] - vec[4]) ** 2 + (vec[3] - vec[5]) ** 2)
    if sum((distance1, distance2, distance3)) / 3 < 3:
        # print(sum((distance1,distance2,distance3))/3)
        return True
    return False


def getCenter(contours, i):  # 获取中心点
    M = cv2.moments(contours[i])  # 计算中心矩
    x = (M['m10'] / M['m00'])
    y = (M['m01'] / M['m00'])
    # x，y为中心点像素的坐标
    return x, y


def mask(mat, row, col, count):  # 掩膜  000型掩膜  (i+j)%2==0
    if (mat[row][col][count] == 255):
        mat[row][col][count] = 0
    else:
        mat[row][col][count] = 255
    return


def demask(mat, row, col, count, thre):
    if (mat[row][col][count] > thre):
        mat[row][col][count] = 0
    else:
        mat[row][col][count] = 255


def drawLocPoint(mat):
    # 创建定位点
    for i in range(x.border, x.border + x.blackWhite):
        # 左上角
        mat[i][x.border] = 0  # 竖
        mat[i][x.border + x.blackWhite - 1] = 0
        mat[i][x.border + 1] = 0
        mat[i][x.border + x.blackWhite - 1 - 1] = 0
        mat[x.border][i] = 0  # 横
        mat[x.border + x.blackWhite - 1][i] = 0
        mat[x.border + 1][i] = 0  # 横
        mat[x.border + x.blackWhite - 1 - 1][i] = 0
        # 左下角
        mat[x.width - x.border - x.blackWhite][i] = 0
        mat[x.width - x.border - 1][i] = 0
        mat[x.width - x.border - x.blackWhite + 1][i] = 0
        mat[x.width - x.border - 1 - 1][i] = 0
        mat[x.width - i - 1][x.border] = 0
        mat[x.width - i - 1][x.border + x.blackWhite - 1] = 0
        mat[x.width - i - 1][x.border + 1] = 0
        mat[x.width - i - 1][x.border + x.blackWhite - 1 - 1] = 0
        # 右上角和左上角对称
        mat[i][x.width - x.border - x.blackWhite] = 0
        mat[i][x.width - x.border - 1] = 0
        mat[i][x.width - x.border - x.blackWhite + 1] = 0
        mat[i][x.width - x.border - 1 - 1] = 0
        mat[x.border][x.width - i - 1] = 0
        mat[x.border + x.blackWhite - 1][x.width - i - 1] = 0
        mat[x.border + 1][x.width - i - 1] = 0
        mat[x.border + x.blackWhite - 1 - 1][x.width - i - 1] = 0
    for i in range(x.border, x.border + x.sblackWhite+1):
        # 右下角
        mat[x.width - i - 1][x.width - x.border - 1] = 0  # 竖
        mat[x.width - i - 1][x.width - (x.border + x.sblackWhite)] = 0
        mat[x.width - x.border - 1][x.width - i - 1] = 0  # 横
        mat[x.width - (x.border + x.sblackWhite)][x.width - i - 1] = 0
        mat[x.width - i - 1][x.width - x.border - 1-1] = 0  # 竖
        mat[x.width - i - 1][x.width - (x.border + x.sblackWhite)-1] = 0
        mat[x.width - x.border - 1-1][x.width - i - 1] = 0  # 横
        mat[x.width - (x.border + x.sblackWhite)-1][x.width - i - 1] = 0


    for i in range(x.border + 5, x.border + 5 + 13):
        for j in range(x.border + 5, x.border + 5 + 13):
            mat[i][j] = 0
            mat[i][x.width - j - 1] = 0
            mat[x.width - j - 1][i] = 0

    for i in range(x.border + 5, x.border + 5 + 7):
        for j in range(x.border + 5, x.border + 5 + 7):
            mat[x.width - i - 1][x.width - j - 1] = 0
    return


def encode(mat, binstring):
    row = x.border  # 记录绘制到第几行
    col = x.border + x.locWidth  # 记录绘制到第几列
    count = 0  # rgb通道变换
    while binstring and row < x.width - x.border:
        if row < x.border + x.locWidth:  # 上端两个定位码之间
            if int(binstring[0]) == 1:
                mat[row][col][count] = 0
            else:
                mat[row][col][count] = 255
            if (col + row) % 2 == 0:
                mask(mat, row, col, count)
            count += 1
            if count >= 3:
                count -= 3
                col += 1
                if col > x.width - x.border - x.locWidth - 1:
                    if row != x.border + x.locWidth - 1:
                        col = x.border + x.locWidth
                    else:
                        col = x.border
                    row += 1
            binstring = binstring[1:]
        elif row < x.width - x.border - x.locWidth:  # 二维码的肚子部份
            if int(binstring[0]) == 1:
                mat[row][col][count] = 0
            else:
                mat[row][col][count] = 255
            if (col + row) % 2 == 0:
                mask(mat, row, col, count)
            count += 1
            if count >= 3:
                count -= 3
                col += 1
                if col > x.width - x.border - 1:
                    if row != x.width - x.border - x.locWidth - 1:
                        col = x.border
                    else:
                        col = x.border + x.locWidth
                    row += 1
            binstring = binstring[1:]
        elif row < x.width - x.border - x.sLocWidth:  # 下端定位码部分
            if int(binstring[0]) == 1:
                mat[row][col][count] = 0
            else:
                mat[row][col][count] = 255
            if (col + row) % 2 == 0:
                mask(mat, row, col, count)
            count += 1
            if count >= 3:
                count -= 3
                col += 1
                if col > x.width - x.border - 1:
                    col = x.border + x.locWidth
                    row += 1
            binstring = binstring[1:]
        else:
            if int(binstring[0]) == 1:
                mat[row][col][count] = 0
            else:
                mat[row][col][count] = 255
            if (col + row) % 2 == 0:
                mask(mat, row, col, count)
            count += 1
            if count >= 3:
                count -= 3
                col += 1
                if col > x.width - x.sLocWidth - x.border - 1:
                    col = x.border + x.locWidth
                    row += 1
            binstring = binstring[1:]
    return binstring



def computeRate1(contours, i, j): #判断是否存在大回型
    area1 = cv2.contourArea(contours[i])
    area2 = cv2.contourArea(contours[j])
    if area2 == 0:
        return False
    ratio = area1 * 1.0 / area2
    # if abs(ratio - 49.0 / 25) < 1:
    if abs(ratio - 529.0 / 441) < 1:
        # print(abs(ratio-49.0/25))
        return True
    return False

def computeRate2(contours, i, j):#判断是否存在小回型
    area1 = cv2.contourArea(contours[i])
    area2 = cv2.contourArea(contours[j])
    if area2 == 0:
        return False
    ratio = area1 * 1.0 / area2
    # if abs(ratio - 25.0 / 9) < 1:
    if abs(ratio - 441.0/169) < 1:
        # print(abs(ratio-25.0/9))
        return True
    return False

def computeRate3(contours, i, j):#判断是否存在小回型
    area1 = cv2.contourArea(contours[i])
    area2 = cv2.contourArea(contours[j])
    if area2 == 0:
        return False
    ratio = area1 * 1.0 / area2
    # if abs(ratio - 25.0 / 9) < 1:
    if abs(ratio-324.0/196) < 1:
        # print(abs(ratio-25.0/9))
        return True
    return False

def computeRate4(contours, i, j):  # 判断是否存在小回型
    area1 = cv2.contourArea(contours[i])
    area2 = cv2.contourArea(contours[j])
    if area2 == 0:
        return False
    ratio = area1 * 1.0 / area2
    # if abs(ratio - 25.0 / 9) < 1:
    if abs(ratio - 196.0 / 49) < 1:
        # print(abs(ratio-25.0/9))
        return True
    return False

def genImage(mat, width, filename):  # 放大图片，由于cv2.resize放大图片一定会出现模糊情况，使用直接对nparray放大。
    img = np.zeros((width, width, 3), dtype=np.uint8)
    pwidth = 3
    #确定每一个像素的rgb三色值，放大倍数为10倍
    for i in range(width):
        normali = i // pwidth
        for j in range(width):
            normalj = j // pwidth
            if (normali < len(mat) and normalj < len(mat)):
                img[i][j][0] = (mat[normali][normalj][0])
                img[i][j][1] = (mat[normali][normalj][1])
                img[i][j][2] = (mat[normali][normalj][2])
            # img[i][j][0]=int(mat[int(normali)][int(normalj)][0])
            # img[i][j][1]=int(mat[int(normali)][int(normalj)][1])
            # img[i][j][2]=int(mat[int(normali)][int(normalj)][2])
    # img = cv2.resize(mat, (800,800), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(filename, img)
    return


def genBlankFrame(): #绘制起始标志图，避免采集无用图的情况发生
    mat = np.full((x.width, x.width, 3), 255, dtype=np.uint8)
    drawLocPoint(mat)
    genImage(mat, x.width * 3, "./video/" + str(0) + ".png")


def judgeOrder(rec): #由于视频可能翻转，按照顺时针确定四个定位点的位置，然后将图片校正
    if len(rec) < 4:
        return -1, -1, -1, -1
    max = 0
    index = 0
    for i in range(len(rec)):#取出四个定位点中最右端的点index
        if (rec[i][0] > max):
            max = rec[i][0]
            index = i
    for i in range(len(rec)):
        if i == index:
            continue
        for j in range(i + 1, len(rec)):
            if j == index:
                continue
            for k in range(j + 1, len(rec)):
                if k == index:
                    continue
                #取三个点两两的距离
                distance1 = np.sqrt((rec[i][0] - rec[j][0]) ** 2 + (rec[i][1] - rec[j][1]) ** 2)
                distance2 = np.sqrt((rec[i][0] - rec[k][0]) ** 2 + (rec[i][1] - rec[k][1]) ** 2)
                distance3 = np.sqrt((rec[j][0] - rec[k][0]) ** 2 + (rec[j][1] - rec[k][1]) ** 2)
                #利用余弦定理计算 余弦值为0的∠为直角，从而确定四个点的方位顺序index为最小点的位置。
                if abs(np.square(distance1) + np.square(distance2) - np.square(distance3)) / (
                        2 * distance1 * distance2) < 0.1:
                    if rec[j][0] < rec[k][0]:
                        return i, j, index, k
                    else:
                        return i, k, index, j
                elif abs(np.square(distance1) + np.square(distance3) - np.square(distance2)) / (
                        2 * distance1 * distance3) < 0.1:
                    if rec[i][0] < rec[k][0]:
                        return j, i, index, k
                    else:
                        return j, k, index, i
                elif abs(np.square(distance2) + np.square(distance3) - np.square(distance1)) / (
                        2 * distance2 * distance3) < 0.1:
                    if rec[i][0] < rec[j][0]:
                        return k, i, index, j
                    else:
                        return k, j, index, i
    return -1, -1, -1, -1


def find(image, contours, hierachy, root=0):
    width = x.width - 8
    rec = []
    #确定四个定位点，定位点为回字型，因此轮廓应该是有嵌套的孩子和孙子的
    for i in range(len(hierachy)):
        child = hierachy[i][2]
        grandchild = hierachy[child][2]
        if child != -1 and grandchild != -1:
            # x1, y1 = getCenter(contours, i)
            # print(x1)
            # print(y1)
            if (computeRate1(contours, i, child) and computeRate2(contours, child, grandchild)) or (computeRate3(contours, i, child) and computeRate4(contours, child, grandchild)): #查找回字形是否存在
                x1, y1 = getCenter(contours, i)
                x2, y2 = getCenter(contours, child)
                x3, y3 = getCenter(contours, grandchild)
                if detectContours([x1, y1, x2, y2, x3, y3, i, child, grandchild]):  #根据三个大的定位点查找轮廓
                    rec.append([x1, y1, x2, y2, x3, y3, i, child, grandchild])

    # print((rec))
    if len(rec) < 4:
        cv2.imwrite("wrong.png", image)
        print("二维码定位点数量不足！")
    i, j, k, t = judgeOrder(rec)
    if i == -1 or j == -1 or k == -1 or t == -1:
        print("定位点有问题，不符合二维码格式！")
        return

    vertexSrc = np.array(
        [[rec[i][0], rec[i][1]], [rec[j][0], rec[j][1]], [rec[k][0], rec[k][1]], [rec[t][0], rec[t][1]]],
        dtype="float32")
    # print(vertexSrc)
    # vertexWarp = np.array([[69.5, 69.5], [69.5, 659.5], [694.5, 694.5], [659.5, 69.5]], dtype="float32")
    vertexWarp = np.array([[ 34, 34.], [ 35,750.], [759,759.], [750,35.]], dtype="float32")
    M = cv2.getPerspectiveTransform(vertexSrc, vertexWarp)
    out = cv2.warpPerspective(image, M, (width * 3, width * 3))
    cv2.imwrite("tem.png", out)
    return out



