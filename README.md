# Hysteretic curve processing

 [Degraded_stiffness.py](Degraded_stiffness.py)：退化刚度计算，其基本思想可参照下边两张图片，计算出每一圈滞回曲线的退化刚度，然后根据累计位移值求解出每个点上的名义上的退化刚度值，输出 [degraded_stiff_all.csv](sampling_data\degraded_stiff_all.csv)

 [Degradation_strength.py](Degradation_strength.py) ：退化强度计算，计算依理论放置于Notion中https://www.notion.so/0d58a52d011445369d7d0ab8c1cf045b?pvs=4，输出 [degraded_strength_all.csv](sampling_data\degraded_strength_all.csv) 

 [Park-Ang damage model.py](Park-Ang damage model.py) ：损伤指标计算，依据Park-Ang损伤模型公式计算

 [filter_stiff.py](filter_stiff.py) ：从计算出的所有数据 [degraded_stiff_all.csv](sampling_data\degraded_stiff_all.csv) 的刚度值提取有图片对应的部分，输出 [degraded_stiffness.csv](..\Image regression\data\degraded_stiffness.csv) 

 [filter_strength.py](filter_strength.py) ：从计算出的所有数据 [degraded_strength_all.csv](sampling_data\degraded_strength_all.csv) 的强度值提取有图片对应的部分，输出 [degraded_strength.csv](..\Image regression\data\degraded_strength.csv) （ [filter_strength.py](filter_strength.py)  ，因为 [Degradation_strength.py](Degradation_strength.py) 等三个指标计算脚本会生成 [degraded_strength_all.csv](sampling_data\degraded_strength_all.csv) 包含所有数据的三大指标计算数据，根据提取出的 [LIST.TXT](sampling_data\LIST.TXT) 图片名文件对其进行筛选，最后生成 [degraded_strength.csv](..\Image regression\data\degraded_strength.csv) 文件）

 [filter_damage.py](filter_damage.py) ：从计算出的所有数据 [damage_index_all.csv](sampling_data\damage_index_all.csv) 的强度值提取有图片对应的部分，输出  [damage_index.csv](..\Image regression\data\damage_index.csv) 

------

 utils.[angle.py](angle.py) ：计算两个向量的角度

 utils.[find_max.py](find_max.py) ：找到每个滞回角中力的绝对值最大点的序号

 utils.[hysteresis_loop.py](hysteresis_loop.py) ：滞回圈绘制函数，可以单图绘制，也可以多图绘制

------

 **[characteristic_engineering.py](characteristic_engineering.py)** ：文件用于实现特征工程，其输入文件为提取出力-位移数据的txt文件，该文件用于绘制滞回曲线与累计耗能曲线，并没有什么实际意义。

------

[file.py](file.py) ：文件为提取力与位移，并且将图片路径提取出来的文件，没啥用

> ------
>
> 算出来的翻转点数为48个，刚好和文件的命名LS48_to_LS49_RS1_0067中48,94对应，**一共有60多个加载（那个文章介绍文献中介绍的加载机制），但是后边的数据网站并没有展示**

 **[demo.py**](angle.py) ：该文件用于为RS3.csv文件增加时间维度

------

 [damage_index_all.csv](sampling_data\damage_index_all.csv) ：所有数据点的损伤指标

 [degraded_stiff_all.csv](sampling_data\degraded_stiff_all.csv) ：所有数据点的刚度

 [degraded_strength_all.csv](sampling_data\degraded_strength_all.csv) ：所有数据点的承载力

 [LIST.txt](sampling_data\LIST.txt) ：所有图片的名称

------

 [characteristic_curve.csv](characteristic_curve.csv) 文件，里面包含着累计位移与累计耗能的数据值

 [1.csv](1.csv) RS1的力值与位移值

 [stiffness.csv](stiffness.csv) ：RS1的力位移时间数据，时间由excel计算而来

 [characteristic_curve.csv](characteristic_curve.csv) 文件，是特征函数上对应的离散点

 [RS1.txt](RS1.txt) 为移动过来的提取出力和位移数据的文件

**“![滞回曲线](E:\Code\Hysteretic curve processing\滞回曲线.jpg)，![特征曲线](E:\Code\Hysteretic curve processing\特征曲线.jpg)”**为生成的文件：我将处理后的照片移动了过来，存在两张照片，其中一张为滞回曲线原图，其中x为曲线上的翻转点，另一张，为进行特征映射后的照片

------

 [data_new](data_new) ：因为RS1的数据离散性太大，我更换了RS3的数据作为操作对象
