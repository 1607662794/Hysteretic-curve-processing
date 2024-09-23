import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from angle import vector_angle
from find_max import find_max_abs_force_indices
from hysteresis_loop_feng import plot_hysteresis_loop

'''该文件用于计算滞回曲线中每个点的损伤指标，计算模型为park-ang'''
'''最后可以生成五种图表，分别对应
    graph_1 ： 绘制滞回曲线以及其上的翻转点；
    graph_2 ：绘制累计滞回耗能通过计算每一圈的耗能，然后进行线性插值和三次样条插值，得到每个点上的耗能；
    graph_3 ： 绘制骨架曲线，该骨架曲线上标出利用park——ang损伤模型公式所用到的参数，等效屈服点；
    graph_4 ： 绘制损伤指标曲线，通过park——ang损伤模型计算损伤指标，并输出
    graph_5 ： 绘制滞回圈，通过park——ang损伤模型计算损伤指标，并输出'''

# 结果保存设置
save_dir = False  # 是否保存累计位移与强度退化
target_dir = r"sampling_data/damage_index_all.csv"
pre = "multi_task"  # 设置预测点使用单任务训练还是多任务训练的single_task/multi_task
show_predict = False  # 是否展示测试集预测点
show_point_predict = True # 是否展示抽取的一个预测点位置，注意，这个最好和上一逻辑值相反

# 绘制可视化图表
graph_1 = False  # 绘制滞回曲线
graph_2 = False  # 绘制能量曲线
graph_3 = False  # 绘制骨架曲线
graph_4 = False  # 绘制损伤指标曲线
graph_5 = False  # 绘制滞回圈

# 插值方式选取
interpolation_method = "spline interpolation"  # 插值方式spline interpolation（三次插值）/linear interpolation（线性插值）

# 翻转点的寻找方式
reverse_method = "force"  # 按照夹角的方式找angle/displace(前后三个点中，中间点的位移绝对值值最大）/force（每个滞回角中力值最大的点）

# 是否打印周期点数据
period_print = False

# 加载数据
# 使用genfromtxt函数加载CSV文件
Input_dir = r"E:\Code\Hysteretic curve processing\sampling_data\RS3_time_appended.csv"  # 原数据
test_dir = r"E:\Code\Image regression\data\data_test.csv"  # 测试集数据
pic_index_dir = r"E:\Code\Image regression\data\data.csv"  # 所有图片的指标
Input_pic_dir = r"E:\Code\Hysteretic curve processing\data_new\RS3.csv"  # 经过处理后的拥有图片部分的数据
data = np.genfromtxt(Input_dir, delimiter=',', skip_header=1,
                     dtype=[('image_names', 'U50'), ('u [mm]', float), ('Fh [kN]', float), ('times [s]', int)])

# 获取加载后的数据
image_names = data['image_names']
displace = data['u_mm']
force = data['Fh_kN']
times = data['times_s']
print(displace.shape, force.shape)
print(type(displace), type(force))

# 获取测试集数据
test_data = np.genfromtxt(test_dir, delimiter=',', skip_header=1,
                          dtype=[('image_dir', 'U50'), ('cumulative_deformation', float), ('degraded_stiff', float),
                                 ('degraded_strength', float), ('damage_index', float), ('times', float)])
test_times = test_data['times']
test_damage_index = test_data['damage_index']

# 获取所有图片的三大指标数据
pic_index_data = np.genfromtxt(pic_index_dir, delimiter=',', skip_header=1,
                               dtype=[('image_dir', 'U50'), ('cumulative_deformation', float),
                                      ('degraded_stiff', float),
                                      ('degraded_strength', float), ('damage_index', float), ('times', float)])
pic_index_times = pic_index_data['times']
pic_index_damage_index = pic_index_data['damage_index']

# 预测文件加载
if pre == 'multi_task':
    dir_predict = r'E:\Code\Image regression\data\data_multi_task_predict.csv'  # 多任务预测数据
    pre_x = np.loadtxt(dir_predict, delimiter=',', skiprows=1, usecols=5)
    pre_y = np.loadtxt(dir_predict, delimiter=',', skiprows=1, usecols=6)

else:
    dir_predict = r'E:\Code\Image regression\data\data_damage_predict.csv'  # 单任务预测数据
    pre_x = np.loadtxt(dir_predict, delimiter=',', skiprows=1, usecols=1)
    pre_y = np.loadtxt(dir_predict, delimiter=',', skiprows=1, usecols=2)

if __name__ == '__main__':

    '''计算每个滞回圈的终点'''
    zero_number = []
    zero_disp = []
    zero_force = []


    def zero_point(point):
        """将对应的点添加到零点列表中"""
        zero_number.append(point)
        zero_disp.append(displace[point])
        zero_force.append(force[point])


    for i in range(1, len(displace) - 1):  # 坐标原点不需要对其进行判断
        if force[i] * force[i + 1] <= 0:
            zero_point(i)
    print("滞回曲线各零点的序号为{}".format(zero_number))
    print("滞回曲线的零点总数为{}".format(len(zero_number)))

    # 计算翻转点：当一个点的横坐标绝对值比其上一个点和其下一个点都小时，这个点即为翻转点
    reverse_disp = []
    reverse_force = []
    reverse_number = []


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
    # 注意，此时的翻转点是包含了原点以及最后一个点的
    if reverse_method == 'angle':  # 按照夹角的方式进行寻找
        reversal_point(0)
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
        reversal_point(len(displace) - 1)
    elif reverse_method == 'displace':
        reversal_point(0)
        for i in range(1, len(displace) - 1):
            if np.abs(displace[i]) > np.abs(displace[i + 1]) and np.abs(displace[i]) > np.abs(displace[i - 1]):
                reversal_point(i)  # 存在42个翻转点,与零点数量一致
        delete_elements(reverse_disp, reverse_number, reverse_force)
        reversal_point(len(displace) - 1)
    else:
        reversal_point(0)
        segmentation_index = [0] + zero_number
        max_force_points = find_max_abs_force_indices(force.tolist(), segmentation_index)
        for i in max_force_points:
            reversal_point(i)
        reversal_point(len(displace) - 1)
    print("翻转点的位移值为{}".format(reverse_disp))
    print("翻转点的力值为{}".format(reverse_force))
    print("翻转点的序号为{}".format(reverse_number))
    print("翻转点数量为{}".format(len(reverse_number)))

    # 计算累计位移
    cumulative_deformation = [0]
    for i in range(1, len(force)):
        cumulative_deformation.append(cumulative_deformation[i - 1] + np.abs(displace[i] - displace[i - 1]))


    # 计算指定初始点与结束点之间曲线与横坐标围成的面积，在这儿我用的是周期零点到周期零点的指标
    def consumption_energy(start_point, end_point):
        cumulative_energy = 0
        for i in range(start_point, end_point):
            increment_deformation = displace[i + 1] - displace[i]
            # 对于两点之间存在零点的情况，分别进行处理
            if force[i] * force[i + 1] < 0:
                # i点与i-1点之间的存在零点，part-deformation为前半段长度
                part_deformation = np.abs(force[i]) / (np.abs(force[i]) + np.abs(force[i + 1])) * increment_deformation
                increment_energy = 0.5 * part_deformation * force[i] + 0.5 * (
                        increment_deformation - part_deformation) * force[i + 1]
                cumulative_energy += increment_energy
            else:
                increment_energy = 0.5 * increment_deformation * (force[i] + force[i + 1])
                cumulative_energy += increment_energy
        return cumulative_energy


    # 将计算所得每一圈的耗能分配给每一圈的零点
    periodic_cycle_number = zero_number[1::2]  # 每两个零点提取出第一个点
    periodic_cycle_consumption_energy = []  # 用于存储每一圈的退化刚度
    print("存在滞回环的数量为{}个".format(len(periodic_cycle_number)))

    periodic_cycle_consumption_energy.append(
        consumption_energy(0, periodic_cycle_number[0]))  # 第一个圈比较特殊，是从0点开始的，所以单独列出来
    for i in range(len(periodic_cycle_number) - 1):
        periodic_cycle_consumption_energy.append(
            consumption_energy(periodic_cycle_number[i], periodic_cycle_number[i + 1]))
    print("耗能数量:{}(主要用于验证）".format(len(periodic_cycle_consumption_energy)))
    print("每圈耗能:{}".format(periodic_cycle_consumption_energy))

    # 将每一圈的耗能进行叠加
    for i in range(1, len(periodic_cycle_consumption_energy)):
        periodic_cycle_consumption_energy[i] = periodic_cycle_consumption_energy[i] + periodic_cycle_consumption_energy[
            i - 1]
    print("进行叠加后的每圈耗能:{}".format(periodic_cycle_consumption_energy))

    tag = 0
    cumulative_hysteretic_energy_consumption = [None] * len(force)  # 定义一个空列表，用于后边存放退化刚度
    if interpolation_method == "linear interpolation":
        # 根据零点将耗能分配给每一个点：线性插值
        for i in range(len(force)):
            if i < zero_number[1]:  # 刚开始一截不完整滞回环的线性插值，外插
                cumulative_hysteretic_energy_consumption[i] = (periodic_cycle_consumption_energy[0] +
                                                               (periodic_cycle_consumption_energy[0] -
                                                                periodic_cycle_consumption_energy[1]) /
                                                               (cumulative_deformation[periodic_cycle_number[0]] -
                                                                cumulative_deformation[
                                                                    periodic_cycle_number[1]]) *
                                                               (cumulative_deformation[i] - cumulative_deformation[
                                                                   periodic_cycle_number[0]]))
            elif i > zero_number[-1]:  # 最后一截不完整滞回环的线性插值，外插，该试件零点总数为42个刚好是偶数
                cumulative_hysteretic_energy_consumption[i] = (periodic_cycle_consumption_energy[-2] +
                                                               (periodic_cycle_consumption_energy[-2] -
                                                                periodic_cycle_consumption_energy[-1]) /
                                                               (cumulative_deformation[periodic_cycle_number[-2]] -
                                                                cumulative_deformation[
                                                                    periodic_cycle_number[-1]]) *
                                                               (cumulative_deformation[i] - cumulative_deformation[
                                                                   periodic_cycle_number[-2]]))
            elif i in zero_number[1::2]:  # 完整滞回环的线性插值，内插，每两个零点处的刚度退化值不需要内插
                cumulative_hysteretic_energy_consumption[i] = periodic_cycle_consumption_energy[tag]
                tag += 1
            else:  # 完整滞回环的线性插值，内插，每两个零点处的刚度退化值不需要内插
                cumulative_hysteretic_energy_consumption[i] = (periodic_cycle_consumption_energy[tag - 1] +
                                                               (periodic_cycle_consumption_energy[tag - 1] -
                                                                periodic_cycle_consumption_energy[tag]) /
                                                               (cumulative_deformation[periodic_cycle_number[tag - 1]] -
                                                                cumulative_deformation[
                                                                    periodic_cycle_number[tag]]) *
                                                               (cumulative_deformation[i] - cumulative_deformation[
                                                                   periodic_cycle_number[tag - 1]]))
    else:
        # 三次样条插值
        # 首先确定已知点
        known_x = []
        known_y = []
        for i in range(len(force)):
            if i in zero_number[1::2]:  # 完整滞回环的线性插值，内插，每两个零点处的强度退化值不需要内插
                known_x.append(cumulative_deformation[i])
                known_y.append(periodic_cycle_consumption_energy[tag])
                tag += 1
        # 横坐标待拟合点
        x_to_fit = cumulative_deformation
        # 样条插值
        f = CubicSpline(known_x, known_y)  # k=3表示使用三次样条插值
        cumulative_hysteretic_energy_consumption = f(x_to_fit)
    print("累计滞回耗能数量：{}".format(len(cumulative_hysteretic_energy_consumption) == len(force)))
    print("cumulative_hysteretic_energy_consumption:{}".format(cumulative_hysteretic_energy_consumption))

    # 计算骨架曲线：连接各翻转点即为骨架曲线（正规的翻转点应为最大力值与最大位移值，因为数据稀疏的原因，在我的数据库中，这两个值对应的点并不一致，
    # 因此我只能选其一，即选择最大位移值），唯一需要更改的地方在于翻转点的顺序
    bone_curve = {}  # 首先创建一个字典用于存放位移和对应的力值
    for i in range(len(reverse_number)):
        bone_curve[reverse_disp[i]] = reverse_force[i]
        if reverse_force[i] == max(reverse_force):  # 求出刚开始的峰值点
            peak_force = reverse_force[i]
            peak_disp = reverse_disp[i]
    bone_disp = []  # 用于存放骨架曲线数据的列表
    bone_force = []
    j = 0
    for i in sorted(bone_curve.keys()):
        bone_disp.append(i)
        bone_force.append(bone_curve[i])
        j += 1
        if i == 0:
            bone_zero_number = j - 1  # 骨架曲线中的原点序号
        if i == peak_disp:
            bone_max_number = j - 1  # 骨架曲线正向中的最大值序号

    # 根据能量原理求出等效屈服点和屈服位移
    # 计算出能量总量
    energy = 0
    for i in range(bone_zero_number + 1, bone_max_number + 1):
        increment_deformation = np.abs(bone_disp[i] - bone_disp[i - 1])
        increment_energy = 0.5 * increment_deformation * (np.abs(bone_force[i - 1]) + np.abs(bone_force[i]))
        energy += increment_energy

    # 反推屈服点
    equivalent_disp = (peak_disp - energy / peak_force) * 2  # 点Y
    for i in range(len(bone_curve) - 1):
        if equivalent_disp >= bone_disp[i] and equivalent_disp <= bone_disp[i + 1]:
            yield_disp = ((bone_disp[i + 1] - bone_disp[i]) / (bone_force[i + 1] - bone_force[i]) * bone_force[i] -
                          bone_disp[i]) / \
                         ((bone_disp[i + 1] - bone_disp[i]) / (bone_force[i + 1] - bone_force[i]) * (
                                 peak_force / equivalent_disp) - 1)
            yield_force = peak_force / equivalent_disp * yield_disp
            break
    print("屈服位移{}".format(yield_disp))  # B点
    print("屈服荷载{}".format(yield_force))

    # 计算损伤因子
    damage_index = []
    for i in range(len(force)):
        damage_index.append(1 + 0.15 * cumulative_hysteretic_energy_consumption[i] / (yield_force * yield_disp))

    if graph_1:
        # 绘制滞回曲线
        # 设置灰色底色
        # plt.rcParams['axes.facecolor'] = 'lightgray'
        # 创建图形和坐标轴对象
        fig, ax = plt.subplots()
        ax.plot(displace, force, color='k', linewidth=0.8)
        # ax.scatter(reverse_disp, reverse_force, c='k', marker='s', s=15, edgecolors=['k'])
        # 添加分割线背景
        xiloc = plt.MultipleLocator(2.5)  # Set a tick on each integer multiple of the *base* within the view interval.
        yiloc = plt.MultipleLocator(10)  # Set a tick on each integer multiple of the *base* within the view interval.
        ax.xaxis.set_minor_locator(xiloc)
        ax.yaxis.set_minor_locator(yiloc)
        ax.grid(color='lightgray', linestyle='-', linewidth=0.5, axis='both', which='both')
        # ax.set_title('Hysteretic curve')
        # ax.set_xlabel('displace(mm)')
        # ax.set_ylabel('force(KN)')
        # ax.set_title('滞回曲线')

        # 刻度值字体大小设置（x轴和y轴同时设置）
        plt.tick_params(labelsize=10)

        ax.set_xlabel('位移(mm)', font={'family': 'SimSun', 'size': 22})
        ax.set_ylabel('荷载 (kN)', font={'family': 'SimSun', 'size': 22})
        # ax = plt.gca()  # 获取当前坐标的位置
        # ax.spines['right'].set_color('None')  # 去掉坐标图的上和右 spine翻译成脊梁
        # ax.spines['top'].set_color('None')
        # ax.xaxis.set_ticks_position('bottom')  # 设置bottom为x轴
        # ax.yaxis.set_ticks_position('left')  # 设置left为x轴
        # ax.spines['bottom'].set_position(('data', 0))  # 这个位置的括号要注意
        # ax.spines['left'].set_position(('data', 0))
        plt.tight_layout()
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.rcParams['axes.unicode_minus'] = False
        plt.savefig(f'E:\\研究生\\科研生活\\横向课题\\4.12-4.26冯老师教材\\绘图\\\滞回曲线_用于系列预测图片绘图.png', dpi=600, bbox_inches='tight')

    if graph_2:
        # 绘制能量曲线，这个图不用
        # plt.xlabel("cumulative_deformation(mm)")
        # plt.ylabel("cumulative_energy(J)")
        # plt.xlabel("累计位移(mm)")
        plt.xlabel('时间(s)')
        plt.ylabel("耗能(J)")
        # plt.plot(cumulative_deformation, cumulative_hysteretic_energy_consumption)
        plt.scatter(times, cumulative_hysteretic_energy_consumption, marker='s', s=5, edgecolors=['dimgray'])
        # plt.scatter(cumulative_deformation, cumulative_hysteretic_energy_consumption, marker='s', s=5,edgecolors=['dimgray'])
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.rcParams['axes.unicode_minus'] = False
        plt.title("累计耗能曲线")

    if graph_3:
        # 绘制骨架曲线
        # plt.xlabel("bone_disp(mm)")
        # plt.ylabel("bone_force(KN)")
        fig, ax = plt.subplots()
        ax.set_xlabel('位移 (mm)', font={'family': 'SimSun', 'size': 22}, labelpad=6.5, x=0.8)
        ax.set_ylabel('荷载 (kN)', font={'family': 'SimSun', 'size': 22}, y=0.9)
        ax.plot(bone_disp, bone_force, zorder=1, color='red', linestyle='--')
        ax.scatter(bone_disp, bone_force, marker='s', color='darkblue', s=15)
        ax.scatter(yield_disp, yield_force, s=80, c='mediumturquoise', marker='^')  # 等效屈服点
        # plt.annotate('yield_point', xy=(yield_disp, yield_force), xytext=(yield_disp - 10, yield_force + 10),
        #              arrowprops=dict(facecolor='black', shrink=0.05, headwidth=10, width=3), )
        ax.annotate('等效屈服点', xy=(yield_disp + 1, yield_force - 1),
                    xytext=(yield_disp + 9, yield_force - 35),
                    arrowprops=dict(facecolor='0.2', shrink=0.1, headwidth=5, width=2),
                    font={'family': 'SimSun', 'size': 22}, ha='center')
        # plt.scatter(bone_disp[bone_zero_number], bone_force[bone_zero_number], c='red', marker='o')
        # plt.scatter(bone_disp[bone_max_number], bone_force[bone_max_number], c='red', marker='o')
        ax = plt.gca()  # 获取当前坐标的位置
        ax.spines['right'].set_color('None')  # 去掉坐标图的上和右 spine翻译成脊梁
        ax.spines['top'].set_color('None')
        ax.xaxis.set_ticks_position('bottom')  # 设置bottom为x轴
        ax.yaxis.set_ticks_position('left')  # 设置left为x轴
        ax.spines['bottom'].set_position(('data', 0))  # 这个位置的括号要注意
        ax.spines['left'].set_position(('data', 0))
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.rcParams['axes.unicode_minus'] = False
        # plt.title("Skeleton curve")
        # plt.title("Skeleton curve", font={'family': 'Times New Roman', 'size': 18})
        # 刻度值字体大小设置（x轴和y轴同时设置）
        plt.tick_params(labelsize=14)
        plt.savefig(f'E:\\研究生\\科研生活\\横向课题\\4.12-4.26冯老师教材\\绘图\\骨架曲线.png', dpi=600, bbox_inches='tight')

    if graph_4:
        # 绘制损伤指标曲线
        # 添加标题
        # plt.title('Park-Ange damage index')
        # plt.xlabel("cumulative_deformation(mm)")
        # plt.ylabel("damage_index")
        plt.title('Park-Ange 损伤指标')
        # plt.xlabel("累计位移(mm)")
        plt.ylabel("损伤指标")
        plt.xlabel('时间(s)')
        # plt.plot(cumulative_deformation, damage_index)
        if show_predict == False and show_point_predict == False:
            s1 = plt.scatter(times, damage_index, marker='s', s=5, edgecolors=['dimgray'])
            # s1 = plt.scatter(times, damage_index, marker='s', s=5, edgecolors=['dimgray'])#原图
            # s1 = plt.scatter(cumulative_deformation, damage_index, marker='s',s=5, edgecolors=['dimgray'])
        # parameter = np.polyfit(cumulative_deformation, damage_index, 8)  # 用8次函数进行拟合
        # p = np.poly1d(parameter)
        # plt.plot(cumulative_deformation, p(cumulative_deformation), color='g')

    if graph_5:
        # 绘制滞回圈
        periodic_cycle_number.insert(0, 0)
        periodic_cycle_number.append(len(force))
        plot_hysteresis_loop(force, displace, periodic_cycle_number, reverse_disp, reverse_force,
                             switch=False)  # switch控制是否开启多图显示

    if show_predict:
        # 展示预测点
        # plt.title('Park-Ange 损伤指标')
        fig, ax = plt.subplots()
        ax.set_xlabel('时间(s)', font={'family': 'SimSun', 'size': 28})
        ax.set_ylabel('损伤比率', font={'family': 'SimSun', 'size': 28})
        # plt.xlabel("时间(s)")
        # plt.ylabel("损伤指标")
        damage_index_array = np.array(damage_index)
        s3 = plt.plot(times, damage_index_array / 18.53644085626638, color='red',  label='真实值',
                      zorder=1, linestyle='--')
        s4 = plt.scatter(test_times, test_damage_index / 18.53644085626638, marker='s', s=50,
                         color=(122 / 255, 27 / 255, 109 / 255),
                         label='采样的真实值')
        s5 = plt.scatter(pre_x, pre_y / 18.53644085626638, marker='^', s=50, color=(237 / 255, 104 / 255, 37 / 255),
                         label='采样的预测值')
        # s3 = plt.plot(times, damage_index, linewidth=2.5, label='真实标签', zorder=1)
        # s4 = plt.scatter(test_times, test_damage_index, marker='s', s=50, edgecolors=['dimgray'],
        #                  label='样本真实标签')
        # s5 = plt.scatter(pre_x, pre_y, c='orange', marker='^', s=50, edgecolors=['dimgray'], label='样本预测标签')

        plt.legend(loc='best', prop={'family': 'SimSun', 'size': 25})

        # 刻度值字体大小设置（x轴和y轴同时设置）
        plt.tick_params(labelsize=18)

        # 移动轴的边缘来为刻度标注腾出空间
        plt.tight_layout()
        plt.savefig(f'E:\\研究生\\科研生活\横向课题\\4.12-4.26冯老师教材\\绘图\\Python绘图\\测试集损伤指标预测.png', dpi=600,
                    bbox_inches='tight')

    if show_point_predict:  # 是否展示其中的某一个点
        fig, ax = plt.subplots()
        # plt.title('Park-Ang damage index', font={'family': 'Times New Roman', 'size': 20})
        ax.set_xlabel('时间(s)', font={'family': 'SimSun', 'size': 28})
        ax.set_ylabel('损伤比率', font={'family': 'SimSun', 'size': 28})
        damage_index_array = np.array(damage_index)
        s7 = plt.scatter(pic_index_times, pic_index_damage_index / 18.53644085626638, color=(122/255, 27/255, 109/255), marker='s')
        # s6 = plt.scatter(pre_x[0], pre_y[0] / 18.53644085626638, c='mediumturquoise', marker='^', s=80)
        s6 = plt.scatter(pre_x[5], pre_y[5] / 18.53644085626638, c='mediumturquoise', marker='^', s=80)
        # s6 = plt.scatter(pre_x[18], pre_y[18] / 18.53644085626638, c='mediumturquoise', marker='^', s=80)
        s8 = plt.plot(times, damage_index_array / 18.53644085626638, color='red', label='Real labels',
                      zorder=0, linestyle='--')
        # plt.annotate('预测点', xy=(pre_x[0], pre_y[0] / 18.53644085626638),
        #              font={'family': 'SimSun', 'size': 25},
        #              xytext=(pre_x[18] - 1500, pre_y[18] / 18.53644085626638+0.001),
        #              arrowprops=dict(facecolor='0.2', shrink=0.05, headwidth=8, width=3))
        plt.annotate('预测点', xy=(pre_x[5], pre_y[5] / 18.53644085626638),
                     font={'family': 'SimSun', 'size': 25},
                     xytext=(pre_x[18] - 2100, pre_y[18] / 18.53644085626638-0.1),
                     arrowprops=dict(facecolor='0.2', shrink=0.05, headwidth=8, width=3))
        # plt.annotate('预测点', xy=(pre_x[18], pre_y[18] / 18.53644085626638),
        #              font={'family': 'SimSun', 'size': 25},
        #              xytext=(pre_x[18] - 1000, pre_y[18] / 18.53644085626638+0.1),
        #              arrowprops=dict(facecolor='0.2', shrink=0.05, headwidth=8, width=3))

        # plt.legend((s7, s6), ('true label', 'predict label'), loc='best')
        plt.legend((s7, s6), ('真实值', '预测值'), loc='best', prop={'family': 'SimSun', 'size': 24})
        # 刻度值字体大小设置（x轴和y轴同时设置）
        plt.tick_params(labelsize=20)
        # plt.savefig(f'E:\\研究生\\科研生活\\横向课题\\4.12-4.26冯老师教材\\绘图\\Python绘图\\point2_多任务学习单点预测Damage.png', dpi=600, bbox_inches='tight')
        plt.savefig(f'E:\\研究生\\科研生活\\横向课题\\4.12-4.26冯老师教材\\绘图\\Python绘图\\point1_多任务学习单点预测Damage.png', dpi=600, bbox_inches='tight')
        # plt.savefig(f'E:\\研究生\\科研生活\\横向课题\\4.12-4.26冯老师教材\\绘图\\Python绘图\\point3_多任务学习单点预测Damage.png', dpi=600, bbox_inches='tight')
    # 移动轴的边缘来为刻度标注腾出空间
    plt.tight_layout()
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False
    plt.show()

    # 结果保存
    if save_dir:
        label = pd.DataFrame(
            {'image_names': image_names, 'cumulative deformation(mm)': cumulative_deformation,
             'damage_index': damage_index, 'times(s)': times})
        label.to_csv(target_dir, index=None)
