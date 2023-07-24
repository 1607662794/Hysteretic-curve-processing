import pandas as pd
import matplotlib.pyplot as plt
'''该文件用于根据图片名提取对应数据，并绘制相应的提取图'''
# 加载txt文件，获取图片名列表
txt_file_path = 'sampling_data/LIST.txt'
with open(txt_file_path, 'r') as txt_file:
    image_names_to_filter = [line.strip() for line in txt_file]

# 加载CSV文件
csv_file_path = 'sampling_data/degraded_strength_all.csv'
df = pd.read_csv(csv_file_path)

# 根据图片名筛选CSV文件内容
filtered_df = df[df['image_names'].isin(image_names_to_filter)]

# 将筛选后的数据保存到新的CSV文件中
output_file_path = r'E:\Code\Image regression\data\degraded_strength.csv'
filtered_df.to_csv(output_file_path, index=False)

# 绘制散点图
plt.figure(figsize=(10, 6))

# 原图
plt.scatter(df['cumulative deformation(mm)'], df['degraded strength(KN)'], label='Original Data', marker='o', s=20)

# 筛选后的图
plt.scatter(filtered_df['cumulative deformation(mm)'], filtered_df['degraded strength(KN)'], label='Filtered Data', color='red', marker='x', s=50)

# 设置标题和标签
plt.title('Degraded Strength')
plt.xlabel('cumulative deformation(mm)')
plt.ylabel('degraded strength(KN)')

# 添加网格线
plt.grid(True, linestyle='--', alpha=0.5)

# 添加图例
plt.legend()

# 显示图形
plt.show()