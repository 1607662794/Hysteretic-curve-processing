import pandas as pd

'''该文件用于为RS3.csv文件增加时间维度'''


# 读取CSV文件
df = pd.read_csv(r'E:\Code\Hysteretic curve processing\sampling_data\RS3.csv')

# 初始化times列为0
df['times(s)'] = 0

# 遍历每一行数据，计算times列的值
last_time_value = 0
for index, row in df.iterrows():
    image_name = row['image_names']
    stage_number = int(image_name.split('_')[0][2:])
    image_number = int(image_name.split('_')[-1])
    current_time_value = last_time_value + image_number
    df.at[index, 'times(s)'] = current_time_value

    # 更新last_time_value，以便在下一行计算时使用
    if index < len(df) - 1:
        next_stage_number = int(df.at[index + 1, 'image_names'].split('_')[0][2:])
        if next_stage_number > stage_number:
            last_time_value = current_time_value

# 将结果保存回CSV文件
df.to_csv(r'E:\Code\Hysteretic curve processing\sampling_data\RS3_time_appended.csv', index=False)