#python实现opencv中的几个光流函数
##1）calcOpticalFlowPyrLK
通过金字塔Lucas-Kanade 光流方法计算某些点集的光流（稀疏光流）。
相关论文：”Pyramidal Implementation of the Lucas Kanade Feature TrackerDescription of the algorithm”

代码实现:calcOpticalFlowPyrLK.py
环境：python3+opencv3

##2）calcOpticalFlowFarneback
用Gunnar Farneback 的算法计算稠密光流（即图像上全部像素点的光流都计算出来）。
相关论文："Two-Frame Motion Estimation Based on PolynomialExpansion"
代码实现:Gunnar_Farneback.py
环境：python3+opencv3

##3）CalcOpticalFlowBM
通过块匹配的方法来计算光流。
代码实现:CalcOpticalFlowBM.py
环境：python2+opencv2

##4）CalcOpticalFlowHS
用Horn-Schunck 的算法计算稠密光流。
代码实现:CalcOpticalFlowHS.py
环境：python2+opencv2
