# Hysteretic curve processing

**characteristic_engineering.py**文件用于实现特征工程，其输入文件为提取出力-位移数据的txt文件，输出数据为**characteristic_curve.csv**文件，里面包含着累计位移与累计耗能的数据值下面为其生成的文件

**file.py**文件为提取力与位移，并且将图片路径提取出来的文件

> ------
>
> 算出来的翻转点数为48个，刚好和文件的命名LS48_to_LS49_RS1_0067中48,94对应，**一共有60多个加载（那个文章介绍文献中介绍的加载机制），但是后边的数据网站并没有展示**

**demo.py：**草稿纸编程文件

**Degraded_stiffness.py**：退化刚度计算，其基本思想可参照下边两张图片，计算出每一圈滞回曲线的退化刚度，然后根据累计位移值求解出每个点上的名义上的退化刚度值

![	](C:\Users\admin\AppData\Local\Temp\WeChat Files\7c02f6ff15a63e7c1afbaad68027ab1.png)![d8a97f38b2cb8f244d9eb497e9fbfcf](C:\Users\admin\AppData\Local\Temp\WeChat Files\d8a97f38b2cb8f244d9eb497e9fbfcf.png)

------

**stiffness.csv：**RS1的力位移时间数据，时间由excel计算而来

**characteristic_curve.csv**文件，是特征函数上对应的离散点

**RS1.txt**为移动过来的提取出力和位移数据的文件

**“特征曲线，特征工程”**为生成的文件：我将处理后的照片移动了过来，存在两张照片，其中一张为滞回曲线原图，其中x为曲线上的翻转点，另一张，为进行特征映射后的照片

------

data_new：因为RS1的数据离散性太大，我更换了RS3的数据作为操作对象
