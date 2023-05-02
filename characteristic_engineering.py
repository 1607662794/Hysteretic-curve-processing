import matplotlib.pyplot as plt
import numpy as np

# 先导入文件，提取力、位移数据
InputName = r"E:\研究生\科研生活\滞回曲线实操\数据库-new\Shear_compression_tests\RS1\RS1.txt"
displace = np.loadtxt(InputName, delimiter='\t', skiprows=1, usecols=0)
force = np.loadtxt(InputName, delimiter='\t', skiprows=1, usecols=1)
print(displace.shape, force.shape)
# print(type(displace), type(force))
print(displace)
print(force)


'''下面注释掉的这部分，对于不大规则的滞回曲线处理效果极差，因为有些滞回圈数据极少，甚至只有一个，取每个圈中的最值并不科学'''
# 分解滞回环，先根据零点(该点与上一个点的乘积为负），将滞回曲线分解成上下部分
zero_point = []
zero_number = []
for i in range(1, len(displace) - 1):#坐标原点不需要对其进行判断
    if force[i] * force[i + 1] <= 0:
        zero_point.append(displace[i + 1])
        zero_number.append(i + 1)
print("滞回曲线的零点横坐标为{}".format(zero_point))
print("滞回曲线的零点序号为{}".format(zero_number))

# # 求出每个滞回环的位移最大值点，即骨架曲线上的点
# bone_point = []
# bone_number = []
# bone_force = []
# for i in range(0, len(zero_number) - 1):
#     x_max = np.max(np.abs(displace[zero_number[i]:zero_number[i + 1]]))
#     for j in range(zero_number[i], zero_number[i + 1]):
#         if x_max == displace[j] or x_max == -displace[j]:
#             bone_number.append(j)
#             bone_point.append(displace[j])
# # 考虑最后一个不完整圈
# if zero_number[-1] < len(displace) - 1:
#     x_max = np.max(np.abs(displace[zero_number[-1]:len(displace)]))
#     for j in range(zero_number[-1], len(displace)):
#         if x_max == displace[j] or x_max == -displace[j]:
#             bone_number.append(j)
#             bone_point.append(displace[j])
#
# print("每个滞回环位移最大值的横坐标为{}".format(bone_point))
# print("每个滞回环位移最大值的序号为{}".format(bone_number))

# for i in bone_number:
#     bone_force.append(force[i])


'''当一个点的横坐标绝对值比其上一个点和其下一个点都小时，这个点即为翻转点，求取了该值，才能对每个分段上的点进行近似积分求和'''

bone_point = []
bone_number = []
bone_force = []

def reversal_point(point):
    """将对应的点添加到翻转点列表中"""
    bone_number.append(point)
    bone_point.append(displace[point])
    bone_force.append(force[point])

for i in range(0, len(displace)):
    if i == 0:
        reversal_point(i)
    elif i == len(displace) - 1:
        reversal_point(i)
    else:
        if np.abs(displace[i]) > np.abs(displace[i + 1]) and np.abs(displace[i]) > np.abs(displace[i - 1]):
            reversal_point(i)

print("翻转点的横坐标为{}".format(bone_point))
print("翻转点的序号为{}".format(bone_number))
print("翻转点的力值为{}".format(bone_force))

'''并根据翻转点（甚至都不需要用到翻转点，卧槽, 因为翻转点处总会存在一个点，而我是用点来进行离散积分的），求累计变形与累计能量图'''
# 这儿解释一下为什么不用到每个点的构件耗能作为指标，因为没发生刚度退化时，每个点上对应的耗能都为零，这样的话，最后的图形将没有区分度

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


# 保存文件
result = np.array(list(zip(cumulative_deformation, cumulative_energy)))
# print(np.concatenate((cumulative_deformation,cumulative_energy),axis=0))
np.savetxt(r"E:\研究生\科研生活\滞回曲线实操\数据库-new\Shear_compression_tests\RS1\characteristic_curve.csv", result, delimiter=",", fmt='%.4f',
           header='cumulative_deformation,cumulative_energy', comments='')
#csv会自动识别“，”，所以虽然开头没有分开，但最终仍然会被分开

# 绘制可视化图表
# plt.subplot(1, 2, 1)#两张图放在一起会扭曲，所以我就把两张图直接保存在文件夹了
plt.xlabel("displace")
plt.ylabel("force")
plt.plot(displace, force)
plt.scatter(bone_point, bone_force, s=80, c='orange', marker='x')
ax = plt.gca()  # 获取当前坐标的位置
# 去掉坐标图的上和右 spine翻译成脊梁
ax.spines['right'].set_color('None')
ax.spines['top'].set_color('None')
ax.xaxis.set_ticks_position('bottom')  # 设置bottom为x轴
ax.yaxis.set_ticks_position('left')  # 设置left为x轴
ax.spines['bottom'].set_position(('data', 0))  # 这个位置的括号要注意
ax.spines['left'].set_position(('data', 0))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False
plt.title("plot：滞回曲线")
plt.savefig(r'E:\研究生\科研生活\滞回曲线实操\数据库-new\Shear_compression_tests\RS1\滞回曲线.jpg')
plt.clf()  # 清空画布

# plt.subplot(1, 2, 2)
plt.plot(cumulative_deformation, cumulative_energy)
plt.scatter(cumulative_deformation, cumulative_energy, s=20, c='orange', marker='x')
plt.title("plot:特征曲线")
plt.savefig(r'E:\研究生\科研生活\滞回曲线实操\数据库-new\Shear_compression_tests\RS1\特征曲线.jpg')
plt.show()
