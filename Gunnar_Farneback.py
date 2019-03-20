#coding=utf-8
import cv2
import numpy as np
from numpy import *
import cmath
from skimage import transform
import matplotlib.pyplot as plt

cap = cv2.VideoCapture("./F/8/shuibo_8.avi")

i=0
num=0
number = 0

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[...,1] = 255

def draw_flow(im, flow, step=1):
    h, w = im.shape[:2]
    y, x = mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1)
    x = x.astype('int64')
    y = y.astype('int64')
    fx, fy = flow[y, x].T
    for index in range(h*w):
        f.writelines([str(x[index]),' ',str(y[index]),' ',str(fx[index]),' ',str(fy[index]),'\n'])

    # create line endpoints
    lines = vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = int32(lines)

    vis=np.ones((h,w),)

    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
        cv2.arrowedLine(vis,(x1,y1),(x2,y2),color=(0,0,255),tipLength=1)
    return vis

while(1):
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    h, w = next.shape[:2]
    # 返回一个两通道的光流向量，实际上是每个点的像素位移值
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # x方向：flow[...,0]
    # y方向：flow[...,1]
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1]) #笛卡尔坐标转换为极坐标
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # plt=draw_flow(next, flow)
    # plt.show()
    f=open('./F/8/shuibo_8_Farneback(x,y,fx,fy).txt','w')
    vis = np.ones((h,w), )
    vis=draw_flow(next, flow)
    f.close()
    cv2.imshow('Optical flow', vis) #划线显示光流
    # cv2.imshow('frame2', rgb) #hsv坐标显示光流
    prvs = next
    cv2.waitKey()
    cv2.destroyAllWindows()
