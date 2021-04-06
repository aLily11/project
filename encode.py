import sys

import cv2
import numpy as np
import myqrcode
import x
from CRC import CRC_Encoding


def imgToVideo(outputFileName, num):
    fps = 20  # 视频帧数
    size = (x.width * 9, x.width * 9)  # 需要转为视频的图片的尺寸
    video = cv2.VideoWriter(outputFileName, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
    img = cv2.imread("./video/" + str(0) + ".png")
    img = cv2.imread("./video/" + str(0) + ".png")
    img = cv2.imread("./video/" + str(0) + ".png")
    for i in range(num):
        img = cv2.imread("./video/" + str(i) + ".png")
        video.write(img)

    img = cv2.imread("./video/" + str(0) + ".png")
    video.write(img)


def main(argv):
    inputFileName = "e1.bin"
    outputFileName = "./video/shiyan.avi"
    if len(argv) > 1:
        inputFileName = argv[1]
        outputFileName = argv[2]
    with open(inputFileName, 'rb') as reader:
        data = reader.read()
    binstring = ""
    for ch in data:
        binstring += CRC_Encoding('{:08b}'.format(ch), x.key)
        if(len(binstring)>908000):
            break
    # end_time = time()
    # run_time = end_time-begin_time
    # print ('该循环程序运行时间：',run_time)
    startOrEnd = 170
    startOrEndStr = ""
    for i in range(8):
        startOrEndStr += '{:08b}'.format(startOrEnd)
    binstring = startOrEndStr + binstring
    binstring = binstring + startOrEndStr

    myqrcode.genBlankFrame()
    num = 1
    x.datalen=len(binstring)
    print(len(binstring))
    while (x.dataindex<len(binstring)):
        # mat = np.zeros([width, width, 3], np.uint8)
        mat = np.full((x.width, x.width, 3), 255, dtype=np.uint8)
        # mat = [[255 for i in range(width)]for j in range(width)]
        myqrcode.drawLocPoint(mat)
        # begin_time = time()
        binstring = myqrcode.encode(mat, binstring)
        myqrcode.genImage(mat, x.width * 9, "./video/" + str(num) + ".png")
        # end_time = time()
        # run_time = end_time-begin_time
        # print ('该循环程序运行时间：',run_time)
        num += 1
        print(x.dataindex)
    imgToVideo("./video/in.avi", num)


if __name__ == '__main__':
    main(sys.argv)
