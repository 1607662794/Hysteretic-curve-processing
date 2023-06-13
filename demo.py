# '''草稿纸'''
# from time import sleep
#
# from matplotlib import pyplot as plt
# import turtle
#
# import numpy as np
# from matplotlib import pyplot as plt
# from tqdm import tqdm
#
# # 手动选择自己想要执行哪部分代码，总共有两部分内容，第一部分用于进行文件的转化，转化为所需数据的txt格式，第二部分为将数据进行退化刚度特征处理
# generate_txt = False
# calculate_degraded_stiffness = True
#
# # 加载数据
# InputName = r"E:\Code\Hysteretic curve processing\stiffness.csv"
# force = np.loadtxt(InputName, delimiter=',', skiprows=1, usecols=0)
# displace = np.loadtxt(InputName, delimiter=',', skiprows=1, usecols=1)
# time_index = np.loadtxt(InputName, delimiter=',', skiprows=1, usecols=2)
# x = []
# y = []
# plt.ion()  #打开交互模式
# # plt.yticks([t for t in range(-50, 50, 2)])
# # plt.xticks([t for t in range(-6, 6, 1)])
# for i in range(len(force)):
#     # turtle.screensize(1, 1, "green")
#     # turtle.goto(displace[i]*10,force[i]*10)
#     x.append(displace[i])
#     y.append(force[i])
#
#     # plt.figure()  # 默认画布大小
#     # plt.figure(figsize=(5, 1))  # 自定义画布大小(width,height)
#     # turtle.speed(1)
#     plt.plot(x, y)
#
#     plt.show()
#     plt.pause(1)
#     plt.clf()  # 清除图像
a = [1,2,3,4]
print(a[1::2])