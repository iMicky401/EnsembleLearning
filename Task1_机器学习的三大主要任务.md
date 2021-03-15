# 01 机器学习基础

1. 机器学习的任务：利用数学模型理解数据，分析预测数据。

2. 机器学习的任务可分为：**有监督学习**、**无监督学习**
   1. 有监督学习——因变量存在。
      - 根据因变量是否连续，有监督学习又分为**回归**、**分类**。
        1. 回归——因变量为连续型变量
        2. 分类——因变量为离散型变量
   2. 无监督学习——因变量不存在。建模的目的是学习数据本身的结构和关系。
   
3. 数据的表达形式：
   
   - 第i个样本：$x_i$
   - 因变量:y
   - 第k个特征：$x^{(k)}$
   - 特征矩阵：X
   
4. **回归**

   例：使用sklearn内置数据集Boston房价数据集。进行可视化

   ```python
   # 引入相关科学计算包
   import numpy as np
   import pandas as pd #做表格
   import matplotlib.pyplot as plt #作图
   %matplotlib inline #设置插图
   plt.style.use("ggplot")      
   import seaborn as sns
   
   #sklearn中所有内置数据集都封装在datasets对象内
   #返回的对象有：
   #-data:特征X的矩阵(ndarray)
   #-target:因变量的向量(ndarray)
   #-feature_names:特征名称(ndarray)
   from sklearn import datasets
   boston = datasets.load_boston()     # 返回一个类似于字典的类
   X = boston.data
   y = boston.target
   features = boston.feature_names
   boston_data = pd.DataFrame(X,columns=features)
   boston_data["Price"] = y
   boston_data.head()
   #结果：表格1
   
   #可视化
   sns.scatterplot(boston_data['NOX'],boston_data['Price'],color="r",alpha=0.6)
   plt.title("Price~NOX")
   plt.show()
   #结果：图片1
   ```

   表格1：		

   <img src="C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20210315161207646.png" alt="image-20210315161207646" style="zoom: 80%;" />

   图片1：

   <img src="C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20210315161258546.png" alt="image-20210315161258546" style="zoom:50%;" />

5. **分类**

   例：iris数据集

   ```python
   # 引入相关科学计算包
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   %matplotlib inline 
   plt.style.use("ggplot")      
   import seaborn as sns
   
   from sklearn import datasets
   iris = datasets.load_iris()
   X = iris.data
   y = iris.target
   features = iris.feature_names
   iris_data = pd.DataFrame(X,columns=features)  #iris是个CSV文件，通过Pandas进行调用读取
   iris_data['target'] = y
   iris_data.head()
   
   # 可视化特征
   marker = ['s','x','o']
   for index,c in enumerate(np.unique(y)):
       plt.scatter(x=iris_data.loc[y==c,"sepal length (cm)"],y=iris_data.loc[y==c,"sepal width (cm)"],alpha=0.8,label=c,marker=marker[c])
   plt.xlabel("sepal length (cm)")
   plt.ylabel("sepal width (cm)")
   plt.legend()
   plt.show()
   ```

   表格2：

   ![image-20210315162321081](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20210315162321081.png)

   图片2：

   <img src="C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20210315162359091.png" alt="image-20210315162359091" style="zoom: 50%;" />