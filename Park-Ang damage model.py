import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''该文件用于计算滞回曲线中每个点的损伤指标，计算模型为park-ang'''
'''最后可以生成五种图表，分别对应
    graph_1 ： 绘制滞回曲线以及其上的翻转点；
    graph_2 ：绘制能量曲线（没有实际意义，只是将横纵坐标值相乘，这样可以实现递增的效果）
    graph_3 ： 绘制骨架曲线，该骨架曲线上标出利用park——ang损伤模型公式所用到的参数，等效屈服点；
    graph_4 ：绘制累计滞回耗能通过计算每一圈的耗能，然后进行线性插值，得到每个点上的耗能；
    graph_5 ： 绘制损伤指标曲线，通过park——ang损伤模型计算损伤指标，并输出'''

# 先导入文件，提取力、位移数据
InputName = r"E:\Code\Hysteretic curve processing\data_new\RS3.csv"
displace = np.loadtxt(InputName, delimiter=',', skiprows=1, usecols=0)
force = np.loadtxt(InputName, delimiter=',', skiprows=1, usecols=1)
print(displace.shape, force.shape)
print(type(displace), type(force))

# 结果保存设置
save_dir = True
target_dir = r"E:\Code\Image regression\data\damage_index.csv"

if __name__ == '__main__':
    # 计算零点：因其在横坐标附近有波动，不能用前一个坐标与当前坐标的乘积为负值来计算
    zero_disp = []
    zero_number = []
    for i in range(1, len(displace) - 1):
        if np.abs(force[i]) < 1:
            zero_disp.append(displace[i])
            zero_number.append(i)
    print("滞回曲线的零点序号为{}".format(zero_number))
    print("滞回曲线的零点数量为{}".format(len(zero_disp)))

    # 计算翻转点：当一个点的横坐标绝对值比其上一个点和其下一个点都小时，这个点即为翻转点
    reverse_disp = []
    reverse_force = []
    reverse_number = []


    def reversal_point(point):
        """将对应的点添加到翻转点列表中"""
        reverse_number.append(point)
        reverse_disp.append(displace[point])
        reverse_force.append(force[point])


    for i in range(0, len(displace)):  # 注意，此时的翻转点是包含了原点以及最后一个点的
        if i == 0:
            reversal_point(i)
        elif i == len(displace) - 1:
            reversal_point(i)
        else:
            if np.abs(displace[i]) > np.abs(displace[i + 1]) and np.abs(displace[i]) > np.abs(displace[i - 1]):
                reversal_point(i)

    print("翻转点的位移值为{}".format(reverse_disp))
    print("翻转点的力值为{}".format(reverse_force))
    print("翻转点的序号为{}".format(reverse_number))

    # 计算累计位移
    cumulative_deformation = [0]
    for i in range(1, len(force)):
        cumulative_deformation.append(cumulative_deformation[i - 1] + np.abs(displace[i] - displace[i - 1]))


    # 计算指定初始点与结束点之间曲线与横坐标围成的面积
    def consumption_energy(start_point, end_point):
        cumulative_energy = 0
        for i in range(start_point, end_point):
            increment_deformation = displace[i+1] - displace[i]
            # 对于两点之间存在零点的情况，分别进行处理
            if i in zero_number:
                # i点与i-1点之间的存在零点，part-deformation为前半段长度
                part_deformation = np.abs(force[i]) / (np.abs(force[i]) + np.abs(force[i + 1])) * increment_deformation
                increment_energy = 0.5 * part_deformation * force[i] + 0.5 * (
                        increment_deformation - part_deformation) * force[i + 1]
                cumulative_energy += increment_energy
            else:
                increment_energy = 0.5 * increment_deformation * (force[i] + force[i+1])
                cumulative_energy += increment_energy
        return cumulative_energy


    # 将计算所得每一圈的耗能分配给每一圈的零点
    periodic_cycle_number = zero_number[1::2]  # 每两个翻转点提取出第一个点
    periodic_cycle_consumption_energy = []  # 用于存储每一圈的退化刚度
    print("存在滞回环的数量为{}个".format(len(periodic_cycle_number)))

    periodic_cycle_consumption_energy.append(consumption_energy(0, periodic_cycle_number[0]))  # 第一个圈比较特殊，是从0点开始的，所以单独列出来
    for i in range(len(periodic_cycle_number) - 1):
        periodic_cycle_consumption_energy.append(consumption_energy(periodic_cycle_number[i], periodic_cycle_number[i + 1]))
    print("耗能数量:{}(主要用于验证）".format(len(periodic_cycle_consumption_energy)))
    print("每圈耗能:{}".format(periodic_cycle_consumption_energy))

    # 根据零点将耗能分配给每一个点：线性插值
    tag = 0
    cumulative_hysteretic_energy_consumption = [None] * len(force)  # 定义一个空列表，用于后边存放退化刚度
    for i in range(len(force)):
        if i < zero_number[1]:  # 刚开始一截不完整滞回环的线性插值，外插
            cumulative_hysteretic_energy_consumption[i] = (periodic_cycle_consumption_energy[0] +
                                 (periodic_cycle_consumption_energy[0] - periodic_cycle_consumption_energy[1]) /
                                 (cumulative_deformation[periodic_cycle_number[0]] - cumulative_deformation[
                                     periodic_cycle_number[1]]) *
                                 (cumulative_deformation[i] - cumulative_deformation[periodic_cycle_number[0]]))
        elif i > zero_number[-1]:  # 最后一截不完整滞回环的线性插值，外插，该试件零点总数为42个刚好是偶数
            cumulative_hysteretic_energy_consumption[i] = (periodic_cycle_consumption_energy[-2] +
                                 (periodic_cycle_consumption_energy[-2] - periodic_cycle_consumption_energy[-1]) /
                                 (cumulative_deformation[periodic_cycle_number[-2]] - cumulative_deformation[
                                     periodic_cycle_number[-1]]) *
                                 (cumulative_deformation[i] - cumulative_deformation[periodic_cycle_number[-2]]))
        elif i in zero_number[1::2]:  # 完整滞回环的线性插值，内插，每两个零点处的刚度退化值不需要内插
            cumulative_hysteretic_energy_consumption[i] = periodic_cycle_consumption_energy[tag]
            tag += 1
        else:  # 完整滞回环的线性插值，内插，每两个零点处的刚度退化值不需要内插
            cumulative_hysteretic_energy_consumption[i] = (periodic_cycle_consumption_energy[tag - 1] +
                                 (periodic_cycle_consumption_energy[tag - 1] - periodic_cycle_consumption_energy[tag]) /
                                 (cumulative_deformation[periodic_cycle_number[tag - 1]] - cumulative_deformation[
                                     periodic_cycle_number[tag]]) *
                                 (cumulative_deformation[i] - cumulative_deformation[periodic_cycle_number[tag - 1]]))
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
    equivalent_disp = (peak_disp - energy / peak_force) * 2
    # print(equivalent_disp)
    # print(len(bone_curve))
    for i in range(len(bone_curve) - 1):
        if equivalent_disp >= bone_disp[i] and equivalent_disp <= bone_disp[i + 1]:
            yield_disp = ((bone_disp[i + 1] - bone_disp[i]) / (bone_force[i + 1] - bone_force[i]) * bone_force[i] -
                          bone_disp[i]) / \
                         ((bone_disp[i + 1] - bone_disp[i]) / (bone_force[i + 1] - bone_force[i]) * (
                                 peak_force / equivalent_disp) - 1)
            yield_force = peak_force / equivalent_disp * yield_disp
            break
    print("屈服位移{}".format(yield_disp))
    print("屈服荷载{}".format(yield_force))

    # 计算损伤因子
    damage_index = []
    for i in range(len(force)):
        damage_index.append(1 + 0.15*cumulative_hysteretic_energy_consumption[i]/(yield_force*yield_disp))


    # 绘制可视化图表
    graph_1 = False  # 绘制滞回曲线
    graph_2 = False  # 绘制能量曲线
    graph_3 = False  # 绘制骨架曲线
    graph_4 = False  # 绘制累计滞回耗能
    graph_5 = True  # 绘制损伤指标曲线

    if graph_1:
        # 绘制滞回曲线
        plt.xlabel("displace(mm)")
        plt.ylabel("force(KN)")
        plt.plot(displace, force)
        plt.scatter(reverse_disp, reverse_force, s=80, c='orange', marker='x')
        ax = plt.gca()  # 获取当前坐标的位置
        ax.spines['right'].set_color('None')  # 去掉坐标图的上和右 spine翻译成脊梁
        ax.spines['top'].set_color('None')
        ax.xaxis.set_ticks_position('bottom')  # 设置bottom为x轴
        ax.yaxis.set_ticks_position('left')  # 设置left为x轴
        ax.spines['bottom'].set_position(('data', 0))  # 这个位置的括号要注意
        ax.spines['left'].set_position(('data', 0))
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.rcParams['axes.unicode_minus'] = False
        plt.title("plot：滞回曲线")

        # plt.savefig(r'E:\研究生\科研生活\滞回曲线实操\数据库-new\Shear_compression_tests\RS1\滞回曲线.jpg')

    if graph_2:
        # 绘制能量曲线
        plt.xlabel("cumulative_deformation(mm)")
        plt.ylabel("cumulative_energy(J)")
        plt.plot(cumulative_deformation, cumulative_energy)
        plt.scatter(cumulative_deformation, cumulative_energy, s=20, c='orange', marker='x')
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.rcParams['axes.unicode_minus'] = False
        plt.title("累计耗能曲线")
        plt.show()

    if graph_3:
        # 绘制骨架曲线
        plt.xlabel("bone_disp(mm)")
        plt.ylabel("bone_force(KN)")
        plt.plot(bone_disp, bone_force)
        plt.scatter(bone_disp, bone_force, s=20, c='orange', marker='x')
        plt.scatter(yield_disp, yield_force, s=20, c='green', marker='o')
        plt.scatter(bone_disp[bone_zero_number], bone_force[bone_zero_number], s=20, c='red', marker='o')
        plt.scatter(bone_disp[bone_max_number], bone_force[bone_max_number], s=20, c='red', marker='o')
        ax = plt.gca()  # 获取当前坐标的位置
        ax.spines['right'].set_color('None')  # 去掉坐标图的上和右 spine翻译成脊梁
        ax.spines['top'].set_color('None')
        ax.xaxis.set_ticks_position('bottom')  # 设置bottom为x轴
        ax.yaxis.set_ticks_position('left')  # 设置left为x轴
        ax.spines['bottom'].set_position(('data', 0))  # 这个位置的括号要注意
        ax.spines['left'].set_position(('data', 0))
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.rcParams['axes.unicode_minus'] = False
        plt.title("骨架曲线")
        # plt.savefig(r'E:\研究生\科研生活\滞回曲线实操\数据库-new\Shear_compression_tests\RS1\特征曲线.jpg')
        plt.show()

    if graph_4:
        # 绘制累计滞回耗能曲线，观察滞回曲线图你会发现，确实是每两个圈作为一组进行递增的，符合预期
        plt.xlabel("cumulative_deformation(mm)")
        plt.ylabel("cumulative_hysteretic_energy_consumption(J)")
        plt.plot(cumulative_deformation, cumulative_hysteretic_energy_consumption)
        plt.scatter(cumulative_deformation, cumulative_hysteretic_energy_consumption, s=20, c='orange', marker='x')
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.rcParams['axes.unicode_minus'] = False
        plt.title("累计滞回耗能曲线")
        plt.show()

    if graph_5:
        # 绘制损伤指标曲线
        plt.xlabel("cumulative_deformation(mm)")
        plt.ylabel("damage_index")
        plt.plot(cumulative_deformation, damage_index)
        plt.scatter(cumulative_deformation, damage_index, s=20, c='orange', marker='x')
        # parameter = np.polyfit(cumulative_deformation, damage_index, 8)  # 用8次函数进行拟合
        # p = np.poly1d(parameter)
        # plt.plot(cumulative_deformation, p(cumulative_deformation), color='g')
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.rcParams['axes.unicode_minus'] = False
        plt.title("损伤指标曲线")
        plt.show()

    # 结果保存
    if save_dir:
        label = pd.DataFrame(
            {'cumulative deformation(mm)': cumulative_deformation, 'damage_index': damage_index})
        label.to_csv(target_dir, index=None)
