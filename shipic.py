import numpy as np
import argparse
import cv2
#ap=argparse.ArgumentParser()
#ap.add_argument("-i", "--image", required = True, help = "path to the image file")
#args = vars(ap.parse_args())
def zhaopian(file_path):
    image = cv2.imread(file_path)#读入图片
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#转成灰度图

    #接下来使图片二值化，为了减小误差
    gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
    gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)
    #图片锐利化对比度什么的
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    #cv2.imshow("gradient",gradient)
    #原本没有过滤颜色通道的时候，这个高斯模糊有效，但是如果进行了颜色过滤，不用高斯模糊效果更好
    #blurred = cv2.blur(gradient, (9, 9))
    #二值化保存到thresh
    (_, thresh) = cv2.threshold(gradient, 225, 255, cv2.THRESH_BINARY)
    #cv2.imshow("thresh",thresh)
    #cv2.imwrite('thresh.jpg',thresh)

    #寻找最小外接矩形
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    #cv2.imshow("closed",closed)
    #cv2.imwrite('closed.jpg',closed)
    #膨胀与腐蚀，这个作用是提高对比度  另外，此处方法不唯一。。。。这个方法是我在网上搜的，博主倾力推荐。。
    closed = cv2.erode(closed, None, iterations = 4)
    closed = cv2.dilate(closed, None, iterations = 4)
    #cv2.imwrite('closed1.jpg',closed)
    #确定包含二维码的矩形位置
    cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))

    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
    #cv2.imwrite("final.jpg",image)
    #cv2.imshow("Image", image)
    #剪裁矩形，用类似切片的操作只能减矩形
    left_point_x = np.min(box[:, 0])
    right_point_x = np.max(box[:, 0])
    top_point_y = np.min(box[:, 1])
    bottom_point_y = np.max(box[:, 1])
    ans = image[top_point_y:bottom_point_y,left_point_x:right_point_x]
    #cv2.imshow("ans",ans)
    #cv2.imwrite("ans.jpg",ans)
    return ans

def imgToSize(img):
    ''' imgToSize()
    # ----------------------------------------
    # Function:   将图像等比例缩放到 512x512 大小
    #             根据图像长宽不同分为两种缩放方式
    # Param img:  图像 Mat
    # Return img: 返回缩放后的图片
    # Example:    img = imgToSize(img)
    # ----------------------------------------
    '''
    # 测试点
    # cv2.imshow('metaImg.jpg', img)

    imgHeight, imgWidth = img.shape[:2]

    # cv.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])
    # src 原图像，dsize 输出图像的大小，
    # img = cv2.resize(img, (512,512))
    zoomHeight = 512
    zoomWidth = int(imgWidth*512/imgHeight)
    img = cv2.resize(img, (zoomWidth,zoomHeight))

    # 测试点
    # cv2.imshow('resizeImg', img)

    # 如果图片属于 Width<Height，那么宽度将达不到 512
    if imgWidth >= imgHeight:
        # 正常截取图像
        w1 = (zoomWidth-512)//2
        # 图像坐标为先 Height，后 Width
        img = img[0:512, w1:w1+512]
    else:
        # 如果宽度小于 512，那么对两侧边界填充为全黑色
        # 根据图像的边界的像素值，向外扩充图片，每个方向扩充50个像素，常数填充：
        # dst = cv2.copyMakeBorder(src, top, bottom, left, right, borderType[, dst[, value]])
        # dst = cv2.copyMakeBorder(img,50,50,50,50, cv2.BORDER_CONSTANT,value=[0,255,0])
        # 需要填充的宽度为 512-zoomWidth
        left = (512-zoomWidth)//2
        # 避免余数取不到
        right = left+1
        img = cv2.copyMakeBorder(img, 0,0,left,right, cv2.BORDER_CONSTANT, value=[0,0,0])
        img = img[0:512, 0:512]
    # 测试点
    # cv2.imshow('size512', img)
    return img