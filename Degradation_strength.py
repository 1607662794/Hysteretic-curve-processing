'''该文件用于计算滞回曲线的退化强度'''
import pandas as pd

'''其实更好的方式是定义一个数据结构，用于存放四种数据，序号，力，位移以及时间'''

import numpy as np
from matplotlib import pyplot as plt

# 加载数据
InputName = r"E:\Code\Hysteretic curve processing\data_new\RS3.csv"
displace = np.loadtxt(InputName, delimiter=',', skiprows=1, usecols=0)
force = np.loadtxt(InputName, delimiter=',', skiprows=1, usecols=1)
time_index = np.loadtxt(InputName, delimiter=',', skiprows=1, usecols=2)

# 结果保存设置
save_dir = True
target_dir = r"E:\Code\Image regression\data\degraded_strength.csv"

# 因为自己手动将数据合在一块儿了，所以不用编写generate_txt部分代码来进行数据预处理
if True:
    initial_strength = np.abs((force[1] - force[0]) / (displace[1] - displace[0]))
    degraded_strength = [None] * len(force)  # 定义一个空列表，用于后边存放退化强度

    '''计算翻转点'''
    '''当一个点的横坐标绝对值比其上一个点和其下一个点都小时，这个点即为翻转点，根据每两个翻转点计算每一圈的退化强度，退化强度是针对一个滞回环而言的，从横坐标出发到横坐标'''
    reverse_number = []
    reverse_disp = []
    reverse_force = []
    reverse_time = []

    # 记录翻转点信息函数
    def reversal_point(point):
        """将对应的点添加到翻转点列表中"""
        reverse_number.append(point)
        reverse_disp.append(displace[point])
        reverse_force.append(force[point])
        reverse_time.append(time_index[point])

    # 计算翻转点
    for i in range(1, len(displace) - 1):
        if np.abs(displace[i]) > np.abs(displace[i + 1]) and np.abs(displace[i]) > np.abs(displace[i - 1]):
            reversal_point(i)
    # print("翻转点的序号为{}".format(reverse_number))
    # print("翻转点的位移值为{}".format(reverse_disp))
    # print("翻转点的力值为{}".format(reverse_force))
    # print("翻转点的时刻为{}".format(reverse_time))
    print("不包括起始点与最终点的翻转点{}".format(reverse_number))
    print("不包括起始点与最终点的翻转点数量有{}个".format(len(reverse_number)))


    '''计算每个滞回圈的终点'''
    zero_number = []
    zero_disp = []
    zero_force = []
    zero_time = []

    def zero_point(point):
        """将对应的点添加到零点列表中"""
        zero_number.append(point)
        zero_disp.append(displace[point])
        zero_force.append(force[point])
        zero_time.append(time_index[point])

    for i in range(1, len(displace) - 1):  # 坐标原点不需要对其进行判断
        if force[i] * force[i + 1] <= 0:
            zero_point(i)
    print("滞回曲线各零点的序号为{}".format(zero_number))
    print("滞回曲线的零点总数为{}".format(len(zero_number)))


    '''刚开始没有发生强度退化，因此编写一个计算列表中绝对值最大的函数，该函数返回绝对值最大的序号'''
    def abs_value(object):
        value = object[0]
        for i in range(len(object)):
            if abs(value) <= abs(object[i]):
                j = i
                value = object[i]
        return j

    '''计算每一圈退化强度，还是有点儿问题，退化强度并不是递减的,是实验数据的问题'''
    Periodic_cycle_point = reverse_number[1::2]  # 每两个翻转点提取出第一个点
    Periodic_cycle_degraded_strength = []  # 用于存储每一圈的退化强度
    print("存在滞回环的数量为{}个".format(len(Periodic_cycle_point)))


    for i in range(len(Periodic_cycle_point)):
        Periodic_cycle_degraded_strength.append(
            (np.abs(force[reverse_number[2 * i + 1]]) + np.abs(force[reverse_number[2 * i]])) / 2)
    print("退化强度:{}".format(Periodic_cycle_degraded_strength))

    '''计算初始强度以及初始强度零点序号'''
    for i in zero_number:
        if i >= Periodic_cycle_point[abs_value(Periodic_cycle_degraded_strength)]:
            init_strength_number = i
            break
    init_strength = Periodic_cycle_degraded_strength[abs_value(Periodic_cycle_degraded_strength)]
    print("初始强度为：{}，初始强度零点序号：{}".format(init_strength, init_strength_number))

    '''计算累计位移'''
    cumulative_deformation = [0]
    for i in range(1, len(force)):
        cumulative_deformation.append(cumulative_deformation[i - 1] + np.abs(displace[i] - displace[i - 1]))

    # 线性插值得到每个点处的退化强度
    tag = abs_value(Periodic_cycle_degraded_strength)
    for i in range(len(force)):
        if i <= init_strength_number:
            degraded_strength[i] = init_strength
        elif i < zero_number[1]:  # 刚开始一截不完整滞回环的线性插值，外插
            degraded_strength[i] = (Periodic_cycle_degraded_strength[0] +
                                 (Periodic_cycle_degraded_strength[0] - Periodic_cycle_degraded_strength[1]) /
                                 (cumulative_deformation[Periodic_cycle_point[0]] - cumulative_deformation[
                                     Periodic_cycle_point[1]]) *
                                 (cumulative_deformation[i] - cumulative_deformation[Periodic_cycle_point[0]]))
        elif i > zero_number[-1]:  # 最后一截不完整滞回环的线性插值，外插
            degraded_strength[i] = (Periodic_cycle_degraded_strength[-2] +
                                 (Periodic_cycle_degraded_strength[-2] - Periodic_cycle_degraded_strength[-1]) /
                                 (cumulative_deformation[Periodic_cycle_point[-2]] - cumulative_deformation[
                                     Periodic_cycle_point[-1]]) *
                                 (cumulative_deformation[i] - cumulative_deformation[Periodic_cycle_point[-2]]))
        elif i in zero_number[1::2]:  # 完整滞回环的线性插值，内插，每两个零点处的强度退化值不需要内插
            degraded_strength[i] = Periodic_cycle_degraded_strength[tag + 1]
            tag += 1
        else:  # 完整滞回环的线性插值，内插，每两个零点处的强度退化值不需要内插
            degraded_strength[i] = (Periodic_cycle_degraded_strength[tag] +
                                 (Periodic_cycle_degraded_strength[tag + 1] - Periodic_cycle_degraded_strength[tag]) /
                                 (cumulative_deformation[Periodic_cycle_point[tag + 1]] - cumulative_deformation[
                                     Periodic_cycle_point[tag]]) *
                                 (cumulative_deformation[i] - cumulative_deformation[Periodic_cycle_point[tag]]))

    print("退化强度的数量是散点的数量".format(len(degraded_strength)))
    print("degraded_strength:{}".format(degraded_strength))
    print("initial_strength:{}".format(initial_strength))

    # for i in range(len(force)):
    #     print("退化强度：{}，时间戳：{},累计位移：{}".format(degraded_strength[i], time_index[i], cumulative_deformation[i]))
    '''数据可视化'''
    # 设置坐标轴名称
    plt.xlabel('cumulative deformation(mm)')
    # plt.xlabel('time_index')
    plt.ylabel('degraded strength(KN/mm)')
    parameter = np.polyfit(cumulative_deformation, degraded_strength, 4)  # 用8次函数进行拟合
    p = np.poly1d(parameter)
    plt.scatter(cumulative_deformation, degraded_strength)
    plt.plot(cumulative_deformation, p(cumulative_deformation), color='g')
    plt.show()

    # 结果保存
    if save_dir:
        label = pd.DataFrame(
            {'cumulative deformation(mm)': cumulative_deformation, 'degraded strength(KN)': degraded_strength})
        label.to_csv(target_dir, index=None)