import pandas as pd
import matplotlib.pyplot as plt

'''该文件用于根据图片名提取对应数据，并绘制相应的提取图'''
# 加载txt文件，获取图片名列表
txt_file_path = 'sampling_data/LIST.txt'
with open(txt_file_path, 'r') as txt_file:
    image_names_to_filter = [line.strip() for line in txt_file]

# 加载CSV文件
csv_file_path = 'sampling_data/degraded_stiff_all.csv'
df = pd.read_csv(csv_file_path)

# 根据图片名筛选CSV文件内容
filtered_df = df[df['image_names'].isin(image_names_to_filter)]

# 将筛选后的数据保存到新的CSV文件中
output_file_path = r'E:\Code\Image regression\data\degraded_stiffness.csv'
# filtered_df.to_csv(output_file_path, index=False)

# 绘制散点图
plt.figure(figsize=(10, 6))

fig, ax = plt.subplots()

# 原图
ax.plot(df['times(s)'], df['degraded stiff(KN/mm)'] / 61.27174006657513, label='源数据',color=(58/255, 9/255, 100/255))
# plt.scatter(df['cumulative deformation(mm)'], df['degraded stiff(KN/mm)'], label='Original Data', marker='o', s=20)

# 筛选后的图
# plt.scatter(filtered_df['times(s)'], filtered_df['degraded stiff(KN/mm)'], label='过滤数据',  c='orange', marker='s', edgecolors=['dimgray'])#原图
ax.scatter(filtered_df['times(s)'], filtered_df['degraded stiff(KN/mm)'] / 61.27174006657513, label='采样数据',  color=(58/255, 9/255, 100/255),s=30, marker='v',zorder=2)
# plt.scatter(filtered_df['cumulative deformation(mm)'], filtered_df['degraded stiff(KN/mm)'], label='Filtered Data', color='red', marker='x', s=50)

# 设置标题和标签
# plt.title('Degraded Stiffness')
# plt.xlabel('cumulative deformation(mm)')
# plt.ylabel('degraded stiffness(KN/mm)')

# plt.xlabel('累计位移(mm)')
ax.set_xlabel('时间(s)', font={'family': 'SimSun', 'size': 22})
ax.set_ylabel('刚度退化比率', font={'family': 'SimSun', 'size': 22})

# 刻度值字体大小设置（x轴和y轴同时设置）
plt.tick_params(labelsize=20)

# 添加网格线
xiloc = plt.MultipleLocator(250)  # Set a tick on each integer multiple of the *base* within the view interval.
yiloc = plt.MultipleLocator(0.1)  # Set a tick on each integer multiple of the *base* within the view interval.
ax.xaxis.set_minor_locator(xiloc)
ax.yaxis.set_minor_locator(yiloc)
ax.grid(color='lightgray', linestyle='-', linewidth=0.5, axis='both', which='both')

# 添加图例
plt.legend(prop={'family': 'SimSun', 'size': 22})

# 移动轴的边缘来为刻度标注腾出空间
plt.tight_layout()

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False
plt.savefig(f'E:\\研究生\\科研生活\\横向课题\\4.12-4.26冯老师教材\\绘图\\python绘图\\stiff抽样图.png', dpi=600, bbox_inches='tight')


# 显示图形
plt.show()
