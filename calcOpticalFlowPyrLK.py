#coding=utf-8
import numpy as np
import cv2
# from common import anorm2, draw_str
from time import clock
import cmath

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# maxCorners : 设置最多返回的关键点数量。
# qualityLevel : 反应一个像素点强度有多强才能成为关键点。
# minDistance : 关键点之间的最少像素点。
# blockSize : 计算一个像素点是否为关键点时所取的区域大小。
# useHarrisDetector :使用原声的 Harris 角侦测器或最小特征值标准。
# k : 一个用在Harris侦测器中的自由变量。
feature_params = dict(maxCorners=5000000,
                      qualityLevel=0.1,
                      minDistance=7,
                      blockSize=7)

class App:
    def __init__(self, video_src):  # 构造方法，初始化一些参数和视频路径
        self.track_len = 10
        self.detect_interval = 1
        self.tracks = []
        self.cam = cv2.VideoCapture(video_src)
        self.frame_idx = 0
        self.num = 0
        self.i = 0
        self.all_distance = 0
        self.count = 0

    def run(self):  # 光流运行方法
        while True:
            ret, frame = self.cam.read()  # 读取视频帧
            if ret == True:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转化为灰度虚图像
                # vis = frame.copy()
                h, w = frame.shape[:2]
                vis = np.ones((h, w), )
                f = open('./F/8/shuibo_8_LK(x1,y1,x2,y2).txt','w')

                if len(self.tracks) > 0:  # 检测到角点后进行光流跟踪
                    img0, img1 = self.prev_gray, frame_gray
                    p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                    """
                    nextPts, status, err = calcOpticalFlowPyrLK(prevImg, nextImg, prevPts[, nextPts[, status[, 
                    err[, winSize[, maxLevel[, criteria[, flags[, minEigThreshold]]]]]]]])
                    参数说明：
                      prevImage 前一帧8-bit图像
                      nextImage 当前帧8-bit图像
                      prevPts 待跟踪的特征点向量
                      nextPts 输出跟踪特征点向量
                      status 特征点是否找到，找到的状态为1，未找到的状态为0
                      err 输出错误向量，（不太理解用途...）
                      winSize 搜索窗口的大小
                      maxLevel 最大的金字塔层数
                      flags 可选标识：OPTFLOW_USE_INITIAL_FLOW   OPTFLOW_LK_GET_MIN_EIGENVALS
                    """
                    p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None,
                                                           **lk_params)  # 前一帧的角点和当前帧的图像作为输入来得到角点在当前帧的位置
                    p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None,
                                                            **lk_params)  # 当前帧跟踪到的角点及图像和前一帧的图像作为输入来找到前一帧的角点位置
                    d = abs(p0 - p0r).reshape(-1, 2).max(-1)  # 得到角点回溯与前一帧实际角点的位置变化关系

                    # good = d < 1  # 判断d内的值是否小于1，大于1跟踪被认为是错误的跟踪点
                    good=d
                    new_tracks = []
                    for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):  # 将跟踪正确的点列入成功跟踪点
                        if not good_flag:
                            continue
                        tr.append((x, y))#tr是前一帧的角点，与当前帧的角点(x,y)合并。标志为good_flag
                        if len(tr) > self.track_len:
                            del tr[0]
                        new_tracks.append(tr)
                        cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)#当前帧角点画圆
                    self.tracks = new_tracks #self.tracks中的值的格式是：(前一帧角点)(当前帧角点)
                    # print(self.tracks[0])
                    # print(self.tracks[1])

                    distance = 0

                    for tr in self.tracks:
                        # tr[0]=list(tr[0])
                        # tr[1]=list(tr[1])
                        x1=tr[0][0]
                        y1=tr[0][1]
                        x2 = tr[1][0]
                        y2 = tr[1][1]

                        f.writelines([ str(x1), ' ', str(y1), ' ', str(x2), ' ', str(y2),'\n'])
                        dis=cmath.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))
                        #正确追踪的点的个数
                        print(len(self.tracks))
                        #每一个正确追踪的点的像素点的位移
                        print(dis.real)
                        distance=distance+dis
                    distance=distance/len(self.tracks)
                    self.all_distance=self.all_distance+distance
                    self.count=self.count+1
                    print("每一帧像素点平均位移：",distance,"第几帧：",self.count)
                    print("所有帧平均位移：",(self.all_distance/self.count).real)
                f.close()

                if self.frame_idx % self.detect_interval == 0:  #每1帧检测一次特征点
                    mask = np.zeros_like(frame_gray)  # 初始化和视频大小相同的图像
                    mask[:] = 255  # 将mask赋值255也就是算全部图像的角点
                    for x, y in [np.int32(tr[-1]) for tr in self.tracks]:  #跟踪的角点画圆
                        cv2.circle(mask, (x, y), 5, 0, -1)
                    p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params)  # 像素级别角点检测
                    if p is not None:
                        for x, y in np.float32(p).reshape(-1, 2):
                            self.tracks.append([(x, y)])  # 将检测到的角点放在待跟踪序列中

                self.frame_idx += 1
                self.prev_gray = frame_gray
                cv2.imshow('lk_track', vis)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                # cv2.imwrite("./mashiti-result4.png", vis)
                break


def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = "./F/8/shuibo_8.avi"

    print
    __doc__
    App(video_src).run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


