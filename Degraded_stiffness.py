'''该文件用于计算滞回曲线的退化刚度'''
import pandas as pd
from scipy.interpolate import CubicSpline
import numpy as np
from matplotlib import pyplot as plt

from angle import vector_angle
from find_max import find_max_abs_force_indices

# 结果保存设置
save_dir = False  # 是否保存累计位移与刚度退化
target_dir = r"sampling_data/degraded_stiff_all.csv"
pre = "single_task"  # 设置预测点事单任务训练还是多任务训练的single_task/multi_task
show_predict = True  # 是否展示测试集预测点
show_point_predict = False  # 是否展示抽取的一个预测点位置，注意，这个最好和上一逻辑值相反

# 插值方式选取
interpolation_method = "linear interpolation"  # 插值方式spline interpolation（三次插值）/linear interpolation（线性插值）

# 翻转点的寻找方式
reverse_method = "force"  # 按照夹角的方式找angle/displace(前后三个点中，中间点的位移绝对值值最大）/force（每个滞回角中力值最大的点）

# 是否打印周期点数据
period_print = False

# 加载数据
# 使用genfromtxt函数加载CSV文件
Input_dir = r"E:\Code\Hysteretic curve processing\sampling_data\RS3.csv"  # 原数据
test_dir = r"E:\Code\Image regression\data\data_test.csv"  # 测试集数据
pic_index_dir = r"E:\Code\Image regression\data\data.csv" # 所有图片的指标
data = np.genfromtxt(Input_dir, delimiter=',', skip_header=1,
                     dtype=[('image_names', 'U50'), ('u [mm]', float), ('Fh [kN]', float)])

# 获取加载后的数据
image_names = data['image_names']
displace = data['u_mm']
force = data['Fh_kN']

# 获取测试集数据
test_data = np.genfromtxt(test_dir, delimiter=',', skip_header=1,
                          dtype=[('image_dir', 'U50'), ('cumulative_deformation', float), ('degraded_stiff', float),
                                 ('degraded_strength', float), ('damage_index', float)])
test_cumulative_deformation = test_data['cumulative_deformation']
test_degraded_stiff = test_data['degraded_stiff']

#获取所有图片的三大指标数据
pic_index_data = np.genfromtxt(pic_index_dir, delimiter=',', skip_header=1,
                          dtype=[('image_dir', 'U50'), ('cumulative_deformation', float), ('degraded_stiff', float),
                                 ('degraded_strength', float), ('damage_index', float)])
pic_index_cumulative_deformation = pic_index_data['cumulative_deformation']
pic_index_degraded_stiff = pic_index_data['degraded_stiff']

# 预测文件加载
if pre == 'multi_task':
    dir_predict = r'E:\Code\Image regression\data\data_multi_task_predict.csv'  # 多任务预测数据
    pre_x = np.loadtxt(dir_predict, delimiter=',', skiprows=1, usecols=1)
    pre_y = np.loadtxt(dir_predict, delimiter=',', skiprows=1, usecols=2)
else:
    dir_predict = r'E:\Code\Image regression\data\data_stiff_predict.csv'  # 单任务预测数据
    pre_x = np.loadtxt(dir_predict, delimiter=',', skiprows=1, usecols=1)
    pre_y = np.loadtxt(dir_predict, delimiter=',', skiprows=1, usecols=2)

if __name__ == '__main__':
    degraded_stiff = [None] * len(force)  # 定义一个空列表，用于后边存放退化刚度

    '''计算每个滞回圈的终点'''
    zero_number = []
    zero_disp = []
    zero_force = []


    def zero_point(point):
        """将对应的点添加到翻转点列表中"""
        zero_number.append(point)
        zero_disp.append(displace[point])
        zero_force.append(force[point])


    for i in range(1, len(displace) - 1):  # 坐标原点不需要对其进行判断
        if force[i] * force[i + 1] <= 0:
            zero_point(i)
    print("滞回曲线各零点的序号为{}".format(zero_number))
    print("滞回曲线的零点总数为{}".format(len(zero_number)))

    '''计算翻转点'''
    '''当一个点的横坐标绝对值比其上一个点和其下一个点都小时，这个点即为翻转点，根据每两个翻转点计算每一圈的退化刚度，退化刚度是针对一个滞回环而言的，从横坐标出发到横坐标'''
    reverse_number = []
    reverse_disp = []
    reverse_force = []


    # 记录翻转点信息函数
    def reversal_point(point):
        """将对应的点添加到翻转点列表中"""
        reverse_number.append(point)
        reverse_disp.append(displace[point])
        reverse_force.append(force[point])


    # 如果仅仅根据位移绝对值大小判断翻转点的时候，会发现，在滞回曲线一角存在多个翻转点的情况，
    # 因此我根据每一次一侧只能有一个翻转点的条件将已记录的翻转点信息进行删选
    def delete_elements(lst, lst1, lst2):
        i = 0
        while i < len(lst) - 1:
            if lst[i] * lst[i + 1] > 0:
                lst.pop(i + 1)  # 删除乘积中的第二个元素
                lst1.pop(i + 1)
                lst2.pop(i + 1)
            else:
                i += 1  # 乘积为负或零，继续检查下一组相邻元素

        # 翻转点寻找


    if reverse_method == 'angle':  # 按照夹角的方式进行寻找
        # 在每一个滞回角找一个夹角最小的点
        minimum = 180
        for j in range(1, zero_number[0]):  # 在刚开始的半圈里找到一个夹角最小的值
            v1 = [displace[j] - displace[j - 1], force[j] - force[j - 1]]
            v2 = [displace[j] - displace[j + 1], force[j] - force[j + 1]]
            if vector_angle(v1, v2) < minimum:
                minimum = vector_angle(v1, v2)
                minimum_num = j
        reversal_point(minimum_num)
        for i in range(len(zero_number) - 1):  # 以零点为索引，找到每一角，夹角最小的点
            minimum = 180
            for j in range(zero_number[i] + 1, zero_number[i + 1]):
                v1 = [displace[j] - displace[j - 1], force[j] - force[j - 1]]
                v2 = [displace[j] - displace[j + 1], force[j] - force[j + 1]]
                if vector_angle(v1, v2) < minimum:
                    # if vector_angle(v1, v2) < minimum and abs(force[j]) > 0.9 * abs(force[j - 1]):
                    minimum = vector_angle(v1, v2)
                    minimum_num = j
            reversal_point(minimum_num)
    elif reverse_method == 'displace':
        for i in range(1, len(displace) - 1):
            if np.abs(displace[i]) > np.abs(displace[i + 1]) and np.abs(displace[i]) > np.abs(displace[i - 1]):
                reversal_point(i)  # 存在42个翻转点,与零点数量一致
        delete_elements(reverse_disp, reverse_number, reverse_force)
    else:
        segmentation_index = [0] + zero_number
        max_force_points = find_max_abs_force_indices(force.tolist(), segmentation_index)
        for i in max_force_points:
            reversal_point(i)
    print("翻转点的位移值为{}".format(reverse_disp))
    print("翻转点的力值为{}".format(reverse_force))
    print("翻转点的序号为{}".format(reverse_number))
    print("翻转点数量为{}".format(len(reverse_number)))


    def abs_value(object):
            # 找到一个序列中最大值的序号
            value = object[0]
            for i in range(len(object)):
                if abs(value) <= abs(object[i]):
                    j = i
                    value = object[i]
            return j, value


    '''计算每一圈退化刚度，还是有点儿问题，退化刚度并不是递减的,是实验数据的问题'''
    Periodic_cycle_point = zero_number[1::2]  # 每两个翻转点提取出第一个点
    Periodic_cycle_degraded_stiffness = []  # 用于存储每一圈的退化刚度
    print("存在滞回环的数量为{}个".format(len(Periodic_cycle_point)))

    for i in range(len(Periodic_cycle_point)):
        Periodic_cycle_degraded_stiffness.append(
            (np.abs(force[reverse_number[2 * i + 1]]) + np.abs(force[reverse_number[2 * i]])) / (
                    np.abs(displace[reverse_number[2 * i + 1]]) + np.abs(displace[reverse_number[2 * i]])))
    print("退化刚度数量:{}(主要用于验证）".format(len(Periodic_cycle_degraded_stiffness)))
    print("退化刚度:{}".format(Periodic_cycle_degraded_stiffness))

    '''计算初始强度以及初始强度零点序号'''
    tag, init_stiff = abs_value(Periodic_cycle_degraded_stiffness)
    init_stiff_number = Periodic_cycle_point[tag]
    print("初始强度为：{}，初始强度零点序号：{}".format(init_stiff, init_stiff_number))

    '''计算累计位移'''
    cumulative_deformation = [0]
    for i in range(1, len(force)):
        cumulative_deformation.append(cumulative_deformation[i - 1] + np.abs(displace[i] - displace[i - 1]))

    if interpolation_method == "linear interpolation":
        # 线性插值得到每个点处的退化刚度
        for i in range(len(force)):
            if i <= init_stiff_number:
                degraded_stiff[i] = init_stiff
            elif i < zero_number[1]:  # 刚开始一截不完整滞回环的线性插值，外插
                degraded_stiff[i] = (Periodic_cycle_degraded_stiffness[0] +
                                     (Periodic_cycle_degraded_stiffness[0] - Periodic_cycle_degraded_stiffness[1]) /
                                     (cumulative_deformation[Periodic_cycle_point[0]] - cumulative_deformation[
                                         Periodic_cycle_point[1]]) *
                                     (cumulative_deformation[i] - cumulative_deformation[Periodic_cycle_point[0]]))
            elif i > zero_number[-1]:  # 最后一截不完整滞回环的线性插值，外插，该试件零点总数为42个刚好是偶数
                degraded_stiff[i] = (Periodic_cycle_degraded_stiffness[-2] +
                                     (Periodic_cycle_degraded_stiffness[-2] - Periodic_cycle_degraded_stiffness[-1]) /
                                     (cumulative_deformation[Periodic_cycle_point[-2]] - cumulative_deformation[
                                         Periodic_cycle_point[-1]]) *
                                     (cumulative_deformation[i] - cumulative_deformation[Periodic_cycle_point[-2]]))
            elif i in zero_number[1::2]:  # 完整滞回环的线性插值，内插，每两个零点处的刚度退化值不需要内插
                degraded_stiff[i] = Periodic_cycle_degraded_stiffness[tag + 1]
                tag += 1
            else:  # 完整滞回环的线性插值，内插，每两个零点处的刚度退化值不需要内插
                degraded_stiff[i] = (Periodic_cycle_degraded_stiffness[tag] +
                                     (Periodic_cycle_degraded_stiffness[tag + 1] - Periodic_cycle_degraded_stiffness[
                                         tag]) /
                                     (cumulative_deformation[Periodic_cycle_point[tag + 1]] - cumulative_deformation[
                                         Periodic_cycle_point[tag]]) *
                                     (cumulative_deformation[i] - cumulative_deformation[Periodic_cycle_point[tag]]))
    else:
        # 三次样条插值
        # 首先确定已知点
        known_x = []
        known_y = []
        for i in range(len(force)):
            if i <= init_stiff_number:
                known_x.append(cumulative_deformation[i])
                known_y.append(init_stiff)
            elif i > zero_number[-1]:  # 最后一截不完整滞回环的线性插值，外插
                known_x.append(cumulative_deformation[i])
                known_y.append(Periodic_cycle_degraded_stiffness[-2] +
                               (Periodic_cycle_degraded_stiffness[-2] - Periodic_cycle_degraded_stiffness[-1]) /
                               (cumulative_deformation[Periodic_cycle_point[-2]] - cumulative_deformation[
                                   Periodic_cycle_point[-1]]) *
                               (cumulative_deformation[i] - cumulative_deformation[Periodic_cycle_point[-2]]))
            elif i in zero_number[1::2]:  # 完整滞回环的线性插值，内插，每两个零点处的强度退化值不需要内插
                known_x.append(cumulative_deformation[i])
                known_y.append(Periodic_cycle_degraded_stiffness[tag + 1])
                tag += 1

        # 横坐标待拟合点
        x_to_fit = cumulative_deformation

        # 样条插值
        f = CubicSpline(known_x, known_y)  # k=3表示使用三次样条插值
        degraded_stiff = f(x_to_fit)

    print(len(degraded_stiff))
    print("degraded_stiff:{}".format(degraded_stiff))

    # for i in range(len(force)):
    #     print("退化刚度：{}，时间戳：{},累计位移：{}".format(degraded_stiff[i],time_index[i],cumulative_deformation[i]))

    '''数据可视化'''
    # 添加标题
    # plt.title('degraded stiff index')
    plt.title('刚度退化指标')
    # 设置坐标轴名称
    # plt.xlabel('cumulative deformation(mm)')
    plt.xlabel('累计位移(mm)')
    # plt.xlabel('time_index')
    # plt.ylabel('degraded stiff(KN/mm)')
    plt.ylabel('刚度退化(KN/mm)')
    # parameter = np.polyfit(cumulative_deformation, degraded_stiff, 8)  # 用8次函数进行拟合
    # p = np.poly1d(parameter)
    if show_predict == False and show_point_predict==False:
        s1 = plt.scatter(cumulative_deformation, degraded_stiff, marker='s', s=10, edgecolors=['dimgray'])
    if period_print:
        cumulative_deformation2 = []
        for i in range(len(Periodic_cycle_point)):
            cumulative_deformation2.append(cumulative_deformation[Periodic_cycle_point[i]])
        s2 = plt.scatter(cumulative_deformation2, Periodic_cycle_degraded_stiffness, marker='s', s=10,
                         edgecolors=['dimgray'])
        plt.legend((s1, s2), ('全体值', '周期值'), loc='best')
    if show_predict:
        # 展示预测点
        s3 = plt.plot(cumulative_deformation, degraded_stiff, linewidth=2.5, label='真实标签')
        s4 = plt.scatter(test_cumulative_deformation, test_degraded_stiff, marker='s', s=50, edgecolors=['dimgray'],
                         label='样本真实标签')
        s5 = plt.scatter(pre_x, pre_y, c='orange', marker='^', s=50, edgecolors=['dimgray'], label='样本预测标签')
        # plt.legend((s1, s2), ('true label', 'predict label'), loc='best')
        plt.legend(loc='best')

    if show_point_predict:  # 是否展示其中的某一个点
        s7 = plt.scatter(pic_index_cumulative_deformation, pic_index_degraded_stiff, marker='s',
                         s=30, edgecolors=['dimgray'])
        s6 = plt.scatter(pre_x[0], pre_y[0], c='orange', marker='^', s=80, edgecolors=['dimgray'])
        # plt.annotate('predict_point', xy=(pre_x[29], pre_y[29]), xytext=(pre_x[29] + 50, pre_y[29] + 10),
        #              arrowprops=dict(facecolor='black', shrink=0.05, headwidth=10, width=3), )
        plt.annotate('预测点', xy=(pre_x[0], pre_y[0]), xytext=(pre_x[0] + 30, pre_y[0] + 5),
                     arrowprops=dict(facecolor='black', shrink=0.05, headwidth=10, width=3), )

        # plt.legend((s7, s6), ('true label', 'predict label'), loc='best')
        plt.legend((s7, s6), ('真实标签', '预测标签'), loc='best')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False
    plt.show()

    # 结果保存
    if save_dir:
        label = pd.DataFrame(
            {'image_names': image_names, 'cumulative deformation(mm)': cumulative_deformation,
             'degraded stiff(KN/mm)': degraded_stiff})
        label.to_csv(target_dir, index=None)
