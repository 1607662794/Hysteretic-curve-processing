import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from demo import vector_angle

'''该文件用于计算滞回曲线中每个点的损伤指标，计算模型为park-ang'''
'''最后可以生成五种图表，分别对应
    graph_1 ： 绘制滞回曲线以及其上的翻转点；
    graph_2 ：绘制能量曲线（没有实际意义，只是将横纵坐标值相乘，这样可以实现递增的效果）
    graph_3 ： 绘制骨架曲线，该骨架曲线上标出利用park——ang损伤模型公式所用到的参数，等效屈服点；
    graph_4 ：绘制累计滞回耗能通过计算每一圈的耗能，然后进行线性插值，得到每个点上的耗能；
    graph_5 ： 绘制损伤指标曲线，通过park——ang损伤模型计算损伤指标，并输出'''

# 结果保存设置
save_dir = True  # 是否保存累计位移与强度退化
target_dir = r"sampling_data/damage_index_all.csv"
pre = "multi_task"  # 设置预测点事单任务训练还是多任务训练的single_task/multi_task
show_predict = False  # 是否展示测试集预测点
show_point_predict = False  # 是否展示抽取的一个预测点位置，注意，这个最好和上一逻辑值相反

# 绘制可视化图表
graph_1 = True  # 绘制滞回曲线
graph_2 = False  # 绘制能量曲线
graph_3 = False  # 绘制骨架曲线
graph_4 = False  # 绘制累计滞回耗能
graph_5 = False  # 绘制损伤指标曲线

# 插值方式选取
interpolation_method = "linear interpolation"  # 插值方式spline interpolation（三次插值）/linear interpolation（线性插值）

# 是否打印周期点数据
period_print = True

# 加载数据
# 使用genfromtxt函数加载CSV文件
Input_dir = r"E:\Code\Hysteretic curve processing\sampling_data\RS3.csv"  # 原数据
Input_pic_dir = r"E:\Code\Hysteretic curve processing\data_new\RS3.csv"  # 经过处理后的拥有图片部分的数据
data = np.genfromtxt(Input_dir, delimiter=',', skip_header=1,
                     dtype=[('image_names', 'U50'), ('u [mm]', float), ('Fh [kN]', float)])

# 获取加载后的数据
image_names = data['image_names']
displace = data['u_mm']
force = data['Fh_kN']
print(displace.shape, force.shape)
print(type(displace), type(force))

# 预测文件加载
if pre == 'multi_task':
    dir_predict = r'E:\Code\Image regression\data\data_multi_task_predict.csv'  # 多任务预测数据
    pre_x = np.loadtxt(dir_predict, delimiter=',', skiprows=1, usecols=5)
    pre_y = np.loadtxt(dir_predict, delimiter=',', skiprows=1, usecols=6)

else:
    dir_predict = r'E:\Code\Image regression\data\data_stiff_predict.csv'  # 单任务预测数据
    pre_x = np.loadtxt(dir_predict, delimiter=',', skiprows=1, usecols=1)
    pre_y = np.loadtxt(dir_predict, delimiter=',', skiprows=1, usecols=2)

if __name__ == '__main__':

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


    for i in range(0, len(displace)):  # 注意，此时的翻转点是包含了原点以及最后一个点的
        if i == 0:
            reversal_point(i)
        elif i == len(displace) - 1:
            reversal_point(i)
        else:#在每一个滞回角找一个夹角最小的点
            # if np.abs(displace[i]) > np.abs(displace[i + 1]) and np.abs(displace[i]) > np.abs(displace[i - 1]):
            # if np.abs(force[i]) > np.abs(force[i + 1]) and np.abs(force[i]) > np.abs(force[i - 1]):
            v1 = [force[i] - force[i - 1], displace[i] - displace[i - 1]]
            v2 = [force[i] - force[i + 1], displace[i] - displace[i + 1]]
            if vector_angle(v1,v2) < 100 :
                reversal_point(i)
    # delete_elements(reverse_disp, reverse_number, reverse_force)
    print("翻转点的位移值为{}".format(reverse_disp))
    print("翻转点的力值为{}".format(reverse_force))
    print("翻转点的序号为{}".format(reverse_number))
    print("翻转点数量为{}".format(len(reverse_number)))

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
        ax.plot(displace, force)
        ax.scatter(reverse_disp, reverse_force, c='orange', marker='s', edgecolors=['dimgray'])
        # 添加分割线背景
        ax.grid(color='lightgray', linestyle='-', linewidth=0.5)
        # ax.set_title('Hysteretic curve')
        # ax.set_xlabel('displace(mm)')
        # ax.set_ylabel('force(KN)')
        ax.set_title('滞回曲线')
        ax.set_xlabel('位移(mm)')
        ax.set_ylabel('力(KN)')
        # ax = plt.gca()  # 获取当前坐标的位置
        # ax.spines['right'].set_color('None')  # 去掉坐标图的上和右 spine翻译成脊梁
        # ax.spines['top'].set_color('None')
        # ax.xaxis.set_ticks_position('bottom')  # 设置bottom为x轴
        # ax.yaxis.set_ticks_position('left')  # 设置left为x轴
        # ax.spines['bottom'].set_position(('data', 0))  # 这个位置的括号要注意
        # ax.spines['left'].set_position(('data', 0))
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.rcParams['axes.unicode_minus'] = False

        # plt.savefig(r'E:\研究生\科研生活\滞回曲线实操\数据库-new\Shear_compression_tests\RS1\滞回曲线.jpg')

    if graph_2:
        # 绘制能量曲线，这个图不用
        # plt.xlabel("cumulative_deformation(mm)")
        # plt.ylabel("cumulative_energy(J)")
        plt.xlabel("累计位移(mm)")
        plt.ylabel("耗能(J)")
        # plt.plot(cumulative_deformation, cumulative_hysteretic_energy_consumption)
        plt.scatter(cumulative_deformation, cumulative_hysteretic_energy_consumption, marker='s',
                    edgecolors=['dimgray'])

        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.rcParams['axes.unicode_minus'] = False
        plt.title("累计耗能曲线")

    if graph_3:
        # 绘制骨架曲线
        # plt.xlabel("bone_disp(mm)")
        # plt.ylabel("bone_force(KN)")
        plt.xlabel("位移(mm)")
        plt.ylabel("力(KN)")
        plt.plot(bone_disp, bone_force)
        plt.scatter(bone_disp, bone_force, marker='s')
        plt.scatter(yield_disp, yield_force, s=60, c='green', marker='^', )  # 等效屈服点
        # plt.annotate('yield_point', xy=(yield_disp, yield_force), xytext=(yield_disp - 10, yield_force + 10),
        #              arrowprops=dict(facecolor='black', shrink=0.05, headwidth=10, width=3), )
        plt.annotate('预测点', xy=(yield_disp, yield_force), xytext=(yield_disp - 10, yield_force + 10),
                     arrowprops=dict(facecolor='black', shrink=0.05, headwidth=10, width=3), )
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
        plt.title("骨架曲线")
        # plt.savefig(r'E:\研究生\科研生活\滞回曲线实操\数据库-new\Shear_compression_tests\RS1\特征曲线.jpg')

    if graph_4:
        # 绘制累计滞回耗能曲线，观察滞回曲线图你会发现，确实是每两个圈作为一组进行递增的（这时候并没有进行累加），符合预期，这个图不用
        # plt.xlabel("cumulative_deformation(mm)")
        # plt.ylabel("cumulative_hysteretic_energy_consumption(J)")
        plt.xlabel("累计位移(mm)")
        plt.ylabel("累计耗能(J)")
        plt.plot(cumulative_deformation, cumulative_hysteretic_energy_consumption)
        plt.scatter(cumulative_deformation, cumulative_hysteretic_energy_consumption, s=20, c='orange', marker='x')
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.rcParams['axes.unicode_minus'] = False
        plt.title("累计滞回耗能曲线")

    if graph_5:
        # 绘制损伤指标曲线
        # 添加标题
        # plt.title('Park-Ange damage index')
        # plt.xlabel("cumulative_deformation(mm)")
        # plt.ylabel("damage_index")
        plt.title('Park-Ange 损伤指标')
        plt.xlabel("累计位移(mm)")
        plt.ylabel("损伤指标")
        # plt.plot(cumulative_deformation, damage_index)
        s1 = plt.scatter(cumulative_deformation, damage_index, marker='s', edgecolors=['dimgray'])
        # parameter = np.polyfit(cumulative_deformation, damage_index, 8)  # 用8次函数进行拟合
        # p = np.poly1d(parameter)
        # plt.plot(cumulative_deformation, p(cumulative_deformation), color='g')
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.rcParams['axes.unicode_minus'] = False

    if show_predict:
        # 展示预测点
        s2 = plt.scatter(pre_x, pre_y, c='orange', marker='^', s=50, edgecolors=['dimgray'])
        # plt.legend((s1, s2), ('true label', 'predict label'), loc='best')
        plt.legend((s1, s2), ('真实标签', '预测标签'), loc='best')

    if show_point_predict:  # 是否展示其中的某一个点
        s3 = plt.scatter(pre_x[29], pre_y[29], c='orange', marker='^', s=80, edgecolors=['dimgray'],
                         label="predict label")
        # plt.annotate('predict_point', xy=(pre_x[29], pre_y[29]), xytext=(pre_x[29] - 150, pre_y[29] + 0.5),
        #              arrowprops=dict(facecolor='black', shrink=0.05, headwidth=10, width=3), )
        plt.annotate('预测点', xy=(pre_x[29], pre_y[29]), xytext=(pre_x[29] - 50, pre_y[29] + 0.3),
                     arrowprops=dict(facecolor='black', shrink=0.05, headwidth=10, width=3), )

        # plt.legend((s1, s3), ('true label', 'predict label'), loc='best')
        plt.legend((s1, s3), ('真实标签', '预测标签'), loc='best')
    plt.show()

    # 结果保存
    if save_dir:
        label = pd.DataFrame(
            {'image_names': image_names, 'cumulative deformation(mm)': cumulative_deformation,
             'damage_index': damage_index})
        label.to_csv(target_dir, index=None)
