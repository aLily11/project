import cv2
#from cv2 import VideoWriter, VideoWriter_fourcc, imread, resize
import os
import shipic
import math
from PIL import Image
from pyzbar import pyzbar
from pyzbar.pyzbar import decode
import pyzbar.pyzbar as pyzbar
import qrcode
import struct
import sys
'''二维码大小设置'''
fps = 25
mesmax = 7089
MAX = 177
k = 0 #表示生成二维码个数

def duru(infliename , savepagename):
    inn = open(infliename,'rb')
    messagee = inn.read()
    i = 0
    while i < len(messagee) :  #截断二进制文件生成二维码
        temp = messagee[i:i+mesmax]
        i = i+mesmax
        global k
        k = k+1
        #print(i,len(temp))
        flag = qrcode.QRCode(
             # 二维码矩阵尺寸
            version=40,
            # 二维码容错率
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            # 二维码中每个Box的像素值
            box_size=9,
            # 二维码与图片边界的距离,默认为4
            border=4,
        )
        flag.add_data(temp)
        flag.make(fit=True)
        img1 = flag.make_image()
        img1.save(savepagename + str(k).zfill(4) + '.png')



def PictoVideo():
    imgPath = "imgtemp\\"  # 读取图片路径
    videoPath = "imgtemp\\shi.mp4"  # 保存视频路径
    '''注意设置时间与帧数！！！！'''
    images = os.listdir(imgPath)
    fps = 25
    #fps = math.ceil(int(k)/(limtime/1000))
    # 正常情况下每秒25帧数，为了充分利用时长，应重新设帧率！！
    '''fps=int(k)/(time/1000) '''

    # VideoWriter_fourcc为视频编解码器 ('I', '4', '2', '0') —>(.avi) 、('P', 'I', 'M', 'I')—>(.avi)、('X', 'V', 'I', 'D')—>(.avi)、('T', 'H', 'E', 'O')—>.ogv、('F', 'L', 'V', '1')—>.flv、('m', 'p', '4', 'v')—>.mp4
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    image = Image.open(imgPath + images[0])
    videoWriter = cv2.VideoWriter(videoPath, fourcc, fps, image.size)
    for im_name in range(len(images)):
        frame = cv2.imread(imgPath + images[im_name])  # 这里的路径只能是英文路径
        # frame = cv2.imdecode(np.fromfile((imgPath + images[im_name]), dtype=np.uint8), 1)  # 此句话的路径可以为中文路径
        print(im_name)
        videoWriter.write(frame)
    print("图片转视频结束！")
    videoWriter.release()
    cv2.destroyAllWindows()

def check(pic):
    texts = pyzbar.decode(pic)
    if not texts :
        return False
    else :
        return True

def VideotoPic():
    videoPath = "work.mp4"  # 读取视频路径
    imgPath = "new\\"  # 保存图片路径

    cap = cv2.VideoCapture(videoPath)
    # fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧率
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取宽度
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取高度
    suc = cap.isOpened()  # 是否成功打开
    frame_count = 0
    flag = 0
    print (suc)
    while suc:
        suc, frame = cap.read()
        if check(frame) == True:
            flag = 1
        if flag == 0 :
            continue
        frame_count += 1
        cv2.imwrite(imgPath + str(frame_count).zfill(4) + ".jpg", frame)
        cv2.waitKey(1)
    cap.release()
    print("视频转图片结束！")

#def decode_qr_code(imagee):  #二维码解码
    # Here, set only recognize QR Code and ignore other type of code
    #return pyzbar.decode(imagee, symbols=[pyzbar.ZBarSymbol.QRCODE])

def save_bin(file_path):
    images = os.listdir(file_path)
    f = open("ans.in", "wb")
    for im_name in range(len(images)):
        frame = cv2.imread(file_path+images[im_name])
        frame = shipic.pictu(frame)
        #cv2.imshow("s",frame)
        #cv2.waitKey(0)
        barcodes = decode(frame)
        for barcode in barcodes:
            url = barcode.data.decode("utf-8")
            f.write(str(url).encode())

    f.close()

if __name__ == '__main__':
    #time_length = int (sys.argv[2])
    #vediosave = str(sys.argv[1])
    #duru(str(sys.argv[0]),'imgtemp\\')
    duru('in.bin','imgtemp\\')
    #VideotoPic()
    #save_bin("imgtemp\\")
    #PictoVideo()