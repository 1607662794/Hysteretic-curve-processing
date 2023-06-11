#!/usr/bin/env python
# coding: utf-8

# # 提取原始数据中的力与位移列数据

# In[1]:


import pandas as pd
from tqdm import *

str_1 = 'E:\\研究生\\科研生活\\滞回曲线实操\\数据库-new\\Shear_compression_tests'
str_2 = '_cyclic_data_disposed.csv'

for i in tqdm(range(1, 7)):

    '''构造地址'''
    open_address = [str_1, '\\RS', '%s' % i, '\\RS', '%s' % i, str_2]
    open_address = ''.join(open_address)
    # print(address)

    '''载入的是dataframe格式数据'''
    temp = pd.read_csv(open_address, encoding="gbk")
    # print(temp)
    # print(type(temp))
    # print(temp.columns)

    temp = temp[['u [mm]', 'Fh [kN]']]
    # print(temp[['Fh [kN]', 'u [mm]']])

    '''将dataframe保存为txt'''
    out_address = [str_1, '\\RS', '%s' % i, '\\RS', '%s' % i, ".txt"]
    out_address = ''.join(out_address)
    # print(out_address)
    temp.to_csv(out_address, sep='\t', index=False, header = True)


# # 将滞回曲线数据与图片文件名数据合并，用于后边划分数据集

# In[62]:


import numpy as np
import pandas as pd
import os

src_label_folder = r"E:\研究生\科研生活\滞回曲线实操\滞回曲线特征工程\Hysteresis curve_ in action\characteristic_curve.csv"
src_data_folder = r'E:\研究生\科研生活\滞回曲线实操\图像回归\data_example\Plaster_side'
target_data_folder = r"E:\研究生\科研生活\滞回曲线实操\图像回归\data_example"

# 先加载np数据
deformation = np.loadtxt(
    src_label_folder, delimiter=',', skiprows=1, usecols=0)
energy = np.loadtxt(src_label_folder, delimiter=',', skiprows=1, usecols=1)
print(type(deformation))

# 将两个数组先放进DataFrame
label = pd.DataFrame({'deformation': deformation, 'energy': energy})

# 载入图片文件名并转化为Series类
img_names = os.listdir(
    r'E:\研究生\科研生活\滞回曲线实操\图像回归\data_example\Plaster_side')  # 将图片文件夹下所有的文件名进行提取
image_dir = src_data_folder + pd.Series(img_names)
data = pd.DataFrame({'image_dir': image_dir})

# 将两组数据合并
data = pd.concat([data, label], 1)
target_data_file = os.path.join(target_data_folder, 'data.csv')


data.to_csv(target_data_file)
print(data)


# # 载入合并后的文件，提取向量

# In[33]:


import numpy as np
import pandas as pd
import os
# 读入的是DataFrame格式
data = pd.read_csv(r'E:\研究生\科研生活\滞回曲线实操\图像回归\data_example\data_train.csv')
imgs = data.loc[:,["deformation", "energy"]]  # df.loc[ ]:主要是通过列名和行名来抽取数据
# imgs = data.iloc[:, 1:3]  # df.iloc[ ]:主要是通过行索引和列索引来抽取数据
# imgs.tolist()# Series 中的 tolist() 方法将一列转换为一个列表
imgs = imgs.values
print(imgs[0])
print(type(imgs))


# In[67]:


src_data_folder = r"data_example/Plaster_side"
src_label_file = r"data_example/characteristic_curve.csv"
target_data_folder = r"data_example"
target_data_file = "data.csv"
target_data_file = os.path.join(target_data_folder, target_data_file)
print(target_data_file)
print(target_data_folder)


# In[81]:


data = pd.read_csv(r"E:\研究生\科研生活\滞回曲线实操\图像回归\data_example\data.csv")
'data_example/Plaster_side'+data['image_dir']
# os.path.join('data_example/Plaster_side', k) for k in data['image_dir'].tolist()
# data['energy'].tolist()
# data_train = pd.DataFrame(data=None, columns=['image_dir', 'deformation', 'energy'])
# print(data)
# data_train = data_train.append(data.iloc[0], ignore_index=True)
# data_train.append(data.iloc[1], ignore_index=True)

