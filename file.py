#!/usr/bin/env python
# coding: utf-8

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


# In[ ]:





# In[35]:


123

