1. 一个完整的机器学习项目分为以下步骤：

   - 明确项目任务：回归/分类
   - 收集数据集并选择合适的特征。
   - 选择度量模型性能的指标。
   - 选择具体的模型并进行训练以优化模型。
   - 评估模型的性能并调参。

2. 度量模型性能的指标：

   ![image-20210319071341203](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20210319071341203.png)

   ![image-20210319071414350](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20210319071414350.png)

3. **线性回归模型**

   *子代的身高有向族群平均身高"平均"的趋势*，这就是统计学上"回归"的最初含义。回归分析是一种预测性的建模技术，它研究的是因变量（目标）和自变量（特征）之间的关系。这种技术通常用于预测分析，时间序列模型以及发现变量之间的因果关系。通常使用曲线/线来拟合数据点，目标是使曲线到数据点的距离差异最小。而线性回归就是回归问题中的一种，线性回归假设目标值与特征之间线性相关，即满足一个多元一次方程。通过构建损失函数，来求解损失函数最小时的参数w ：

   假设：数据集$D = \{(x_1,y_1),...,(x_N,y_N) \}$，$x_i \in R^p,y_i \in R,i = 1,2,...,N$，$X = (x_1,x_2,...,x_N)^T,Y=(y_1,y_2,...,y_N)^T$                       

   假设X和Y之间存在线性关系，模型的具体形式为$\hat{y}=f(w) =w^Tx$          

   ![image-20210319071729922](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20210319071729922.png)

    (a) 最小二乘估计：                 

   我们需要衡量真实值$y_i$与线性回归模型的预测值$w^Tx_i$之间的差距，在这里我们和使用二范数的平方和L(w)来描述这种差距，即：                      
   $$
   L(w) = \sum\limits_{i=1}^{N}||w^Tx_i-y_i||_2^2=\sum\limits_{i=1}^{N}(w^Tx_i-y_i)^2 = (w^TX^T-Y^T)(w^TX^T-Y^T)^T = w^TX^TXw - 2w^TX^TY+YY^T\\
   因此，我们需要找到使得L(w)最小时对应的参数w，即：\\
   \hat{w} = argmin\;L(w)\\
   为了达到求解最小化L(w)问题，我们应用高等数学的知识，使用求导来解决这个问题： \\
      \frac{\partial L(w)}{\partial w} = 2X^TXw-2X^TY = 0,因此： \\
      \hat{w} = (X^TX)^{-1}X^TY
   $$
      (b) 几何解释：                
      在线性代数中，我们知道两个向量a和b相互垂直可以得出：$<a,b> = a.b = a^Tb = 0$,而平面X的法向量为Y-Xw，与平面X互相垂直，因此：$X^T(Y-Xw) = 0$，即：$w = (X^TX)^{-1}X^TY$                             
    ![image-20210319071941447](C:\Users\Lenovo\AppData\Roaming\Typora\typora-user-images\image-20210319071941447.png)              (c) 概率视角：       

   假设噪声$\epsilon \backsim N(0,\sigma^2),y=f(w)+\epsilon=w^Tx+\epsilon$，因此：$y|x_i,w ~ N(w^Tx,\sigma^2)$          

    我们使用极大似然估计MLE对参数w进行估计：       
   $$
   L(w) = log\;P(Y|X;w) = log\;\prod_{i=1}^N P(y_i|x_i;w) = \sum\limits_{i=1}^{N} log\; P(y_i|x_i;w)\\
       = \sum\limits_{i=1}^{N}log(\frac{1}{\sqrt{2\pi \sigma}}exp(-\frac{(y_i-w^Tx_i)^2}{2\sigma^2})) = \sum\limits_{i=1}^{N}[log(\frac{1}{\sqrt{2\pi}\sigma})-\frac{1}{2\sigma^2}(y_i-w^Tx_i)^2] \\
       argmax_w L(w) = argmin_w[l(w) = \sum\limits_{i = 1}^{N}(y_i-w^Tx_i)^2]\\
       因此：线性回归的最小二乘估计<==>噪声\epsilon\backsim N(0,\sigma^2)的极大似然估计
   $$