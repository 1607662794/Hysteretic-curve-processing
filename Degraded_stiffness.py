'''该文件用于计算滞回曲线的退化刚度'''

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

# 手动选择自己想要执行哪部分代码，总共有两部分内容，第一部分用于进行文件的转化，转化为所需数据的txt格式，第二部分为将数据进行退化刚度特征处理
generate_txt = False
calculate_degraded_stiffness = True

# 加载文件
InputName = r"E:\Code\Hysteretic curve processing\stiffness.csv"
displace = np.loadtxt(InputName, delimiter=',', skiprows=1, usecols=0)
force = np.loadtxt(InputName, delimiter=',', skiprows=1, usecols=1)
time_index = np.loadtxt(InputName, delimiter=',', skiprows=1, usecols=2)

if generate_txt:
    ''''''

# 因为自己手动将数据合在一块儿了，所以不用编写上一部分代码来进行数据预处理
if calculate_degraded_stiffness:
    # print(displace)  # 用于检验自己的数据是否加载成功
    # print(force)  # 用于检验自己的数据是否加载成功
    # print(time_index)  # 用于检验自己的数据是否加载成功
    # print(type(displace))
    initial_stiff = np.abs((force[1]-force[0])/(displace[1]-displace[0]))
    stiff = []
    degraded_stiff = []
    for i in tqdm(range(1, len(displace))):
        stiff.append(np.abs((force[i]-force[i-1])/(displace[i]-displace[i-1])))
        degraded_stiff.append(initial_stiff-stiff[i-1])
    plt.plot(stiff)
    plt.show()
    plt.plot(degraded_stiff)
    plt.show()

