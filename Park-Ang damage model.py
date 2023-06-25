import matplotlib.pyplot as plt
import numpy as np

'''该文件用于计算滞回曲线中每个点的损伤指标，计算模型为park-ang'''

# 先导入文件，提取力、位移数据
InputName = r"E:\Code\Hysteretic curve processing\data_new\RS3.csv"
displace = np.loadtxt(InputName, delimiter=',', skiprows=1, usecols=0)
force = np.loadtxt(InputName, delimiter=',', skiprows=1, usecols=1)
print(displace.shape, force.shape)
print(type(displace), type(force))

# 计算零点：该点与上一个点的乘积为负，将滞回曲线分解成上下部分
zero_point = []
zero_number = []
for i in range(1, len(displace) - 1):  # 坐标原点不需要对其进行判断
    if force[i] * force[i + 1] <= 0:
        zero_point.append(displace[i + 1])
        zero_number.append(i + 1)
print("滞回曲线的零点横坐标为{}".format(zero_point))
print("滞回曲线的零点序号为{}".format(zero_number))

# 计算翻转点：当一个点的横坐标绝对值比其上一个点和其下一个点都小时，这个点即为翻转点，通过翻转点分解滞回环，对每个分段上的点进行近似积分求和
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

# 并根据翻转点（甚至都不需要用到翻转点，卧槽, 因为翻转点处总会存在一个点，而我是用点来进行离散积分的），求累计变形与累计能量图
cumulative_energy = [0]
cumulative_deformation = [0]


def deformation_energy(point):
    for i in range(1, point):  # i为第几个点
        increment_deformation = np.abs(displace[i] - displace[i - 1])
        cumulative_deformation.append(cumulative_deformation[i - 1] + increment_deformation)
        # 对于两点之间存在零点的情况，分别进行处理
        if i in zero_number:
            # i点与i-1点之间的存在零点，part-deformation为前半段长度
            part_deformation = np.abs(force[i - 1]) / (np.abs(force[i - 1]) + np.abs(force[i])) * increment_deformation
            increment_energy = 0.5 * part_deformation * np.abs(force[i - 1]) + 0.5 * (
                    increment_deformation - part_deformation) * np.abs(force[i])
            cumulative_energy.append(cumulative_energy[i - 1] + increment_energy)
        else:
            increment_energy = 0.5 * increment_deformation * (np.abs(force[i - 1]) + np.abs(force[i]))
            cumulative_energy.append(cumulative_energy[i - 1] + increment_energy)


deformation_energy(len(displace))
print("累计位移{}".format(cumulative_deformation))
print("累计能量{}".format(cumulative_energy))

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


# 绘制可视化图表
graph_1 = False  # 绘制滞回曲线
graph_2 = False  # 绘制累计耗能曲线
graph_3 = True  # 绘制骨架曲线

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
    # 绘制累计耗能曲线
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
