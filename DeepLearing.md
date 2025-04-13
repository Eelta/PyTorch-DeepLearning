# 1.线性模型

核心思想：

X是n维，Y是m维，则W一定是n×m维。

X：左横右竖；Y：朝向与X相同。

情况1：X(1,n) × W(n,m) = Y(1,m)

情况2：W^T(m,n) × X(n,1) = Y(m,1)

（以上均为针对单个样本下的计算）

设每次计算单个W对n个样本的的取值结果，则每次n个样本将输出n个预测值构成n维向量，将这个预测向量与实际向量（即标签）进行误差对比，对应的n个样本元素做减法，求平方，再除以n（即取n个样本的均值），可得出MSE，也就是损失函数。

每次对W进行测试都将输出一个损失值（针对n个样本的），则k次测试将生成k个损失值。最终目标是确定k次测试中的某个W，使得对应损失值最小。

**总结：**

1。使用单个权重，去乘以多个样本，获取结果作为预测值集合。

2。将预测值集合与真实值集合计算获取单个权重下的误差。

3。对权重更新，使得对应误差最小。

# 2.梯度下降

这里，如何选择最优的权重向量W？可以尝试多次，这一般在机器学习中也称为epoch。

可以采用visdom，在迭代时同时展示损失值，方便查看模型是否收敛。



这里还有更好的方式：梯度下降（正规方程一般不使用）

梯度下降原理：设损失函数为L，对L求W（权重）的导数，通过导数大于或者小于0，判断L基于W下降的方向，不断靠近L的最低点（贪心算法）。 

W更新：W=W-αL‘，这里α是学习率，即每次向最低点移动的步长，过大会导致模型无法收敛。

局部最优：由于点是连续进行移动的，当点移动至某个位置时，导数可能接近0，但是此时只是局部最优点，而不是全局最优点（但实际上深度学习中，这样的局部最优点很少）。

鞍点：一段曲线内，导数值均为0，将导致点无法移动，模型无法继续迭代，解决方式：随机梯度下降（SGD）：在计算误差时，每次只使用一个样本。如每次选用一小组样本，则称为mini-Batch方法。

# 3.反向传播

在图上进行梯度传播，创建更具有弹性的模型结构。

<img src="pic\fb67ec5e2611c675e44696d20c730dc3.png" alt="fb67ec5e2611c675e44696d20c730dc3" style="zoom: 33%;" />

输入X是5维向量，第1层h(1)输出的Y是6维向量，则W只能是5行6列，即权重矩阵需要30个元素。

这里的点看作是X中的每个特征，连接线看作是W与X进行线性组合的匹配计算过程，例如这里：

<img src="pic\78c83073c16ff4f5b488d4445d6a41db.png" alt="78c83073c16ff4f5b488d4445d6a41db" style="zoom: 50%;" />

这里的神经元：针对变量指的是向量维数，而针对矩阵则是通道数量。

<img src="pic\0d7c23ddef101e97c105c132ac5b90b0.png" alt="0d7c23ddef101e97c105c132ac5b90b0" style="zoom:50%;" />

在一个神经网络中，权重*输入+偏置量=第1层

<img src="pic\2cd40aba1e9f50bbb868d767a113cb41.png" alt="2cd40aba1e9f50bbb868d767a113cb41" style="zoom: 33%;" />

问题：如果每次都类似上面进行计算，函数展开后形式将始终不变化，变换失去意义。为此，在每次结果输出时需要对结果向量做一个非线性函数的变换（即激活函数）。

<img src="pic\fa7ae0ef402295b7136744f5146b4dd5.png" alt="fa7ae0ef402295b7136744f5146b4dd5" style="zoom: 33%;" />

回忆链式法则（chain rule）：(f(g(x)))' = f'(g(x)) * g'(x)，它解决了什么问题？其实就是嵌套过多层函数时，多层求导难度较大的问题。也就是在多层网络中，L对W如何得出求导结果的问题。

<img src="C:\Users\MSI-NB\AppData\Roaming\Typora\typora-user-images\image-20241026170053916.png" alt="image-20241026170053916" style="zoom: 33%;" />

这里，Z表示X和W的运算规则，知道Z函数表达式即可轻松获取Z对X导数、Z对W导数。

之后从Loss处返回损失函数L对于Z的导数，

目标是计算L对X的偏导数（这里需要计算的原因是，X不一定就代表数据输入，它可能是一个中间层输出，因此需要有对它进行计算的能力）、以及L对W的偏导数，此时再采用链式法则，由于先前Z对X导数、Z对W导数、L对于Z导数都已经获取，可直接计算出结果。

整体上，先前馈传播，再反向传播，最终得出L对W的梯度（导数）、L对X的梯度。

<img src="pic\bd4544408b1713c020fb50ddb2f0b4c6.png" alt="bd4544408b1713c020fb50ddb2f0b4c6" style="zoom: 33%;" />

PyTorch中最基础的数据结构是Tensor，它可以存储一维向量、矩阵、多维矩阵等，它是一个类，属性包括W的数值和损失函数对W的导数（这里data和grad均为Tensor（张量））。

```python
import torch

#这里以列表形式设置了初始数据
x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

#设置初始权重值
#可以看到这里设置的是一维权重，因为x_data中每列是一个样本，且单个样本的特征维度是1，Y的维度也是1。
w = torch.Tensor([1.0])
#此处设置了需要计算梯度。即设定之后会需要对这个参数进行更新。
w.requires_grad = True

#forward 前向传播（前面提及的Z函数的表达式）
def forward(x):
    return x * w
#这里w是Tensor，所以x * w自动重载为两个Tensor间的数乘运算，x直接被自动转换为Tensor类型以参与计算。
#针对此处，因为输入时w是需要计算梯度的，所以输出结果z=x * w这个Tensor也被设置为需要计算梯度（回忆之前内容中，存在着L对z求梯度的过程）
 
#定义损失函数，这里每运行一次，就构建一个“计算图”
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2
 
#输出预测数值
print("predict (before training)", 4, forward(4).item())

#设置epoch=100，进行100次完整的样本训练
for epoch in range(100):
    #在每个epoch中，每次是针对单个样本计算，这里的计算方式是随机梯度下降SGD
    for x, y in zip(x_data, y_data):
        #构建“计算图”，获取损失函数的Tensor
        l = loss(x, y)
        #调用backward函数后，将“计算图”中所有设置的-需要梯度的地方计算对应梯度，并保存在对应Tensor中。之后释放“计算图”。
        #这里释放“计算图”，是因为每次构建的“计算图”可能是不同的，因此选择不保留，这种方式也使得Pytorch较为灵活。
        #之后如何获取梯度值：w.grad.data，注意不是直接w.grad，因为w.grad是一个Tensor，不能直接参与计算。
        #区别：w.grad.data允许在张量层面进行运算并对梯度值进行更新，而w.grad.item()则只能在标量层面进行操作。基于w.grad.data这种方式可对梯度进行更新，同时不会修改“计算图”。
        l.backward()
        #展示梯度值和权重数值
        print('\tgrad:', x, y, w.grad.item(), w.data)
        #W更新：W=W-αL'
        w.data = w.data - 0.01 * w.grad.data
        print(w.data)
        #PyTorch 会自动将梯度累积到.grad属性中。也就是说，如果不清除梯度，它们会在每次迭代时叠加，导致梯度越来越大。w.grad.data.zero_()用于将梯度清空为0。
        w.grad.data.zero_()
        #展示更新后的权重数值和更新后的梯度（这里是0）
        print(w.data, w.grad.item())
	#展示每个epoch中第3个样本对应的损失值
    print('progress:', epoch, l.item())
 
print('predict (after taining)', 4, forward(4).item())

#总结：总计3个步骤：
#1.前向：算损失
#2.反向：算梯度
#3.更新：更新W权重
```

# 4.用PyTorch实现线性回归

广播机制：不同形状的矩阵计算中，对矩阵进行自动扩充。即将空的维度部分补充至最高维。

<img src="pic\4c3e3ae5d09a602b788a84e05967f9c8.png" alt="4c3e3ae5d09a602b788a84e05967f9c8" style="zoom:50%;" />

```python
import torch

# 和先前不同的地方在于这里将初始数据直接设置为张量，数据集：样本量为3，X和Y的特征维度均为1。
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])


# 自己想要实现的方式Pytorch难以实现：
# 1.如果可以使用Python基础方法实现，则将其封装为模型类（继承torch.nn.Module）。
# 2.如果求导效率不高，需要自己手动实现反向传播的计算块，可以继承Functions类，如继承Functions类则需要手动额外实现反向传播（因为反向传播过程涉及求导）。

# 这里将模型定义为继承自Module的类（因Module中具有较多合适方法）
class LinearModel(torch.nn.Module):
    # 这里模型至少需要设置2个函数：构造函数和forward函数
    # 1）设置构造函数，初始化对象时可进行属性定义
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)
        # torch.nn.Linear是一个类，此处在构造对象，torch.nn.Linear(in features, out features, bias=True)
        # 这里，in features=每一个输入样本（X）的特征维数，out features=每一个输出样本（Y）的维数，bias=是否需要偏置项，这里通过in features和out features可确定每一个权重矩阵W的形状。

    # 2）必须定义并实现forward函数，用于定义前馈计算Z
    # 此处实际上是重载了父类方法，类似_call_方法，当调用对象时会直接使用该函数中的内容。也称为magic method
    # 之后要使用该类时，直接：model=LinearModel(), model(x)即可，其内部做的就是：wx+b，返回的是y_pred
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


# _call_方法示例：
class Shit:
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        print(args)
        print(kwargs)


# 当实例化对象时，隐式调用的是__init__
shit = Shit()
# 当直接调用对象时，调用的是__call__
shit(1, 2, 3, x=6, y=7)
'''
输出结果：
(1, 2, 3)
{'x': 6, 'y': 7}
'''

# 定义线性模型对象
model = LinearModel()

# 定义损失函数，此处设置为MSE，在 PyTorch 的新版本中，size_average 已经被弃用，
# 取而代之的是 reduction 参数。这个参数有三个选项：
# reduction='mean'：类似于 size_average=True，计算损失的平均值。
# reduction='sum'：类似于 size_average=False，计算损失的总和。
# reduction='none'：不进行任何平均或求和，返回每个样本的损失值。
criterion = torch.nn.MSELoss(reduction='mean')

# 构建对W进行更新的优化器（W=W-αL'），此处设置SGD随机梯度下降
# 但是实际上，SGD是一个方法类，确定是使用单个样本，还是全样本，还是mini-batch，是根据输入数据确定。类似此处使用的就是全样本进行统一计算，对所有样本计算损失均值，并更新W权重。Batch-SGD。
# 实例化了一个SGD类，model是LinearModel，LinearModel里有linear，而linear有parameters。
# 这里使用model.parameters()，目的是找出所有LinearModel中需要进行更新的权重参数。
# 设定固定学习率为0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练过程，这里对全样本进行计算更新
for epoch in range(1000):
    # 1.前馈计算，算出y_pred
    y_pred = model(x_data)
    # 2.计算损失，传入y_pred（预测）和y_data（真实），返回损失的Tensor
    loss = criterion(y_pred, y_data)
    # 展示迭代次数和对应Tensor的损失数值
    print(epoch, loss.item())
    # 3.先将优化器中的梯度值清零（防止在上一次迭代中遗留下W的梯度）
    optimizer.zero_grad()
    # 4.利用损失的Tensor进行反向传播
    loss.backward()
    # 5.利用优化器进行权值W和偏置b的迭代
    optimizer.step()

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred = ', y_test.data)
```

不同优化器的效果展示：

torch.optim.Adagrad
torch.optim.Adam
torch.optim.Adamax
torch.optim.ASGD
torch.optim.LBFGS
torch.optim.RMsprop
torch.optim.Rprop
torch.optim.SGD

<img src="pic\50606a723f3b467257ce560d69a616d8.png" alt="50606a723f3b467257ce560d69a616d8" style="zoom: 33%;" />

# 5.Logistic（逻辑）回归

n分类问题：和回归问题不同，分类模型的输出应该是和为1的n个概率值，并最终判别结果属于哪个类（概率最大项）。

P(x∈Z) = n

Logistic回归是针对二分类问题的。

经典激活函数Sigmod：用于将线性回归的结果映射至[0, 1]区间内。其导函数类似正态分布形状。

考虑：当线性回归的预测值非常大（正负）时，Sigmoid函数的输出会接近0或1吗？

实际上是会的，解决方案：

1. 特征标准化（缩放特征为0至1范围内）。

2. L1、L2正则化，防止W矩阵调节程度过大。
3. 使用其它激活函数。

<img src="pic\7ed7aefe0d4cd592c66aea4fc7bc0bf0.png" alt="7ed7aefe0d4cd592c66aea4fc7bc0bf0" style="zoom: 33%;" />

常见的Simgmod函数：

<img src="C:\Users\MSI-NB\AppData\Roaming\Typora\typora-user-images\image-20241119115054473.png" alt="image-20241119115054473" style="zoom:33%;" />

使用激活函数之后，仍只能获取范围在[0 ,1]区间的预测值集合。

此时损失函数不再是计算数值之间的距离，而是用于计算两个分布概率值之间的差异。可用KL散度（又称相对熵）、交叉熵等作为损失函数。

例如BCE损失（Binary Cross-Entropy Loss，二元交叉熵损失）的计算公式：

<img src="pic\4a07772035b7687da9bafad4d2cc8d9c.png" alt="4a07772035b7687da9bafad4d2cc8d9c" style="zoom:33%;" />

当Y_ac（实际值）=1时，Y_pre（线性回归结果经Sigmod映射后的数值）越接近1，将使得整体函数数值越小（类似先前内容中，预测值和真实值越接近，会导致损失函数越小）；Y_pre越接近0，整体函数数值越大。

当Y_ac=0时，Y_pre越接近0，整体函数数值越小；Y_pre越接近1，整体函数数值越大。

<img src="pic\4a0084535328759b85fce27478daba17.png" alt="4a0084535328759b85fce27478daba17" style="zoom:33%;" />

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = '18'
matplotlib.use('TkAgg')

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
# 真实数据Y变为类别0, 1，而不是数值类型
y_data = torch.Tensor([[0], [0], [1]])


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    # 区别1，前向传播对预测值进行计算时，添加了sigmoid函数映射
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred


model = LogisticRegressionModel()
# 区别2，定义BCE为损失函数
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 针对不同的x生成预测值并展示
x = np.linspace(0, 10, 200)
# reshape操作
x_t = torch.Tensor(x).view((200, 1))
y_t = model(x_t)
y = y_t.data.numpy()
plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], c='r')
plt.xlabel('Hours')
plt.ylabel('Probability of Pass')
plt.grid()
plt.show()
```

# 6.处理多维特征的输入

假设对8维特征的N个样本进行Mini-Batch计算：

注意此处Sigmod函数进行的是向量化运算（即对一个列表中的每个元素，进行逐个元素的计算）。偏置项b直接通过广播机制进行扩充。

这里，Z代表函数值。X矩阵中，每行代表一个样本。

<img src="pic\3abb0df78cc8b50d3386dc0803db1937.png" alt="3abb0df78cc8b50d3386dc0803db1937" style="zoom:33%;" />

假设需要按照上面的方式进行调整，只需修改此处模型的输入维数即可。 

为什么有时会调整输出维度：

可以看作，这里只是调整中间步骤输出的维度，需要在进行多次变换后，最后再次转换为一维输出结果。

思想：通过多个层的线性变换，去拟合非线性变换。神经网络目的：寻找一种非线性变换的空间函数。

<img src="C:\Users\MSI-NB\AppData\Roaming\Typora\typora-user-images\image-20241120172248203.png" alt="image-20241120172248203" style="zoom:33%;" />

```python
import numpy as np
import torch
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
x = diabetes.data
y = diabetes.target

# 这里使用32位浮点数，而不使用64位Double，是因为N卡一般支持32位浮点。
xy = np.array(np.concatenate([x, np.array([y]).T], axis=1), dtype=np.float32)
# 取numpy数据构造tensor
x_data = torch.from_numpy(xy[:, :-1])
# 这里[-1]表示取出的是矩阵，否则是向量
y_data = torch.from_numpy(xy[:, [-1]])


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(10, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        # 这里定义Sigmoid模块，注意和之前定义的函数torch.sigmod()是不同的，这里定义一个模块用于多次调用，构建计算图。
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # 这里每层都可以设置不同的激活函数
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        # 这里如果设置激活函数为Relu，所有小于0的数值将映射为0，因此最后一层一般设置为sigmod
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

不同激活函数的对比：

[torch.nn — PyTorch 2.5 documentation](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity)

<img src="pic\ce09fa1d32a28e0e735be501e744f704.png" alt="ce09fa1d32a28e0e735be501e744f704" style="zoom: 50%;" />

# 7.加载数据集

**对比 SGD 和 Mini-batch GD**

- **纯 SGD（单样本）**：每次迭代使用一个样本，随机性最强，有助于逃离局部极小值。速度慢。
- **Mini-batch SGD**：每次迭代使用一个小批量样本（例如 Batch Size=32 或 64）。速度适中。
- **Batch GD（全数据）**：没有随机性，每次迭代使用全数据，存在鞍点问题。速度快。

因此，**Mini-batch SGD** 是实际应用中最常用的形式，它在效率和梯度波动之间找到了一个平衡点。



常用词汇解释：

**Epoch**：整个数据集被完整训练一次。

**Batch Size**：每次迭代时所处理的数据量。

**Iteration**：一次权重更新，处理一个Batch。

1个Epoch中，Iteration = 数据样本总数 / Batch Size

<img src="pic\f5d22739127664a70c8221a9b623b76d.png" alt="f5d22739127664a70c8221a9b623b76d" style="zoom: 33%;" />

Dataset：用于数据索引

DataLoader：用于数据加载mini-batch

<img src="pic\108f832436b8c37f1534360adfc7a0d7.png" alt="108f832436b8c37f1534360adfc7a0d7" style="zoom:33%;" />

```python
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_diabetes

# 这里，Dataset是一个抽象类（不能被实例化，只能被继承）
from torch.utils.data import Dataset, DataLoader

# 读取并创建数据集
diabetes = load_diabetes()
x = diabetes.data
y = diabetes.target
data = np.array(np.concatenate([x, np.array([y]).T], axis=1), dtype=np.float32)
pd.DataFrame(data).to_csv('diabetes.csv', index=False)


class DiabetesDataset(Dataset):
    # 初始化读取数据：一般2种方式
    # 1.如果数据集不大，可直接全部加载进入内存。
    # 2.如果数据集较大，仅保存数据样本的索引，在getitem被调用时再实时读取数据。
    def __init__(self, filepath):
        xy = pd.read_csv(filepath)
        self.len = xy.shape[0]
        self.x_data = xy.iloc[:, :-1].values.astype('float32')
        self.y_data = xy.iloc[:, xy.shape[1] - 1:xy.shape[1]].values.astype('float32')

    # 这里__getitem__是一个magic method，当对象使用下标时自动调用该方法
    # 此处返回元组
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # 这里__len__也是一个magic method，当对象调用len()时自动调用该方法
    def __len__(self):
        return self.len


dataset = DiabetesDataset('diabetes.csv')
# 这里，num_workers代表设置去读取batch_size数据的并行线程数。当数据集较小时，使用多线程反而可能降低运行速度。
train_loader = DataLoader(dataset=dataset,
                          batch_size=32, 
                          shuffle=True,
                          num_workers=2)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(10, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 此处设置if __name__ == '__main__'，是为了防止进程并发时发生 运行时异常 错误
if __name__ == '__main__':
    # 设置epoch为100
    for epoch in range(100):
        # 这里采用enumerate获取iteration次数i
        # 此处，train_loader将X和Y数据进行封装，形成各个Batch下的数据矩阵（X和Y分别形成矩阵）
        for i, data in enumerate(train_loader, 1):
            inputs, labels = data
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

```

<img src="pic\6d409f6ee94cddcfec9afeb85f5ee46e.png" alt="6d409f6ee94cddcfec9afeb85f5ee46e" style="zoom:33%;" />

总结流程顺序：1.准备数据集 2.设置模型 3.构造损失和优化器 4.训练周期

# 8.多分类问题

第1种方式：将多分类看作多个二分类问题，将某种类别标签记作1，其它类别标签记作0，针对每种类别构造并训练分类器。

“一对多”方法的最大缺点是它将每个类别独立看待，忽视了类别之间的相似性、关系和层次结构。它通过多个独立的二分类器来处理多分类问题，但这可能导致以下问题：

1. 类别之间的相似性被忽视。
2. 决策边界可能过于复杂，不自然。
3. 类别之间的决策可能不一致。
4. 无法捕捉类别的层次结构。
5. 无法处理“新类别”的问题。

所希望的针对各类别的输出：

1.输出均大于零，

2.输出的和等于1，

解决方案：神经网络保持先前的sigmod层不变，输出层使用softmax。

这里，直接控制输出的预测值y_pre维数为样本类别数：

例如，设样本类别为k（k分类问题），输入x为n维（即特征维数），则进行变换的权重矩阵w的形状（n，k）。

<img src="pic\4285a03e0b7f7cf41970e730c3e4d3c8.png" alt="4285a03e0b7f7cf41970e730c3e4d3c8" style="zoom: 33%;" />

<img src="C:\Users\MSI-NB\AppData\Roaming\Typora\typora-user-images\image-20241204202943771.png" alt="image-20241204202943771" style="zoom:33%;" />

对数似然损失：

将标签构造成为One-hot编码形式，例如有5种类别，该样本真实样本标签属于第4类，则对应标签表示为：[0，0，0，1，0]

将Softmax层输出的预测值结果（例如：[0.3，0.2，0.1，0.3，0.1]）和该列表（[0，0，0，1，0]）计算对数似然损失。因为计算过程中的真实标签有很多0项所以相乘为零，只看真实项（此处实际只计算第4类）。

为什么会设计这样的函数：目标是希望第4类预测值结果输出地尽可能接近1，当Y=1时，Y_pre越接近1，整体损失越接近0；Y_pre越接近0，损失越接近无穷大。

<img src="pic\904a9232babf7377f47b88a0c32de121.png" alt="904a9232babf7377f47b88a0c32de121" style="zoom:33%;" />

交叉熵：Softmax + Negative Log Likelihood Loss

<img src="pic\981e6b664c88e509a8a0fe35f254f44f.png" alt="981e6b664c88e509a8a0fe35f254f44f" style="zoom:33%;" />

对数据损失计算过程进行测试验证：

```python
import torch

'''
对数据损失计算过程进行测试验证
'''

# 定义损失函数，此处设置reduction='none'，返回每个样本的损失值而不是批量计算均值
criterion = torch.nn.CrossEntropyLoss(reduction='none')

# 设置真实样本类别
Y = torch.LongTensor([2, 0, 1, 0])

# 这里有4个样本，3个类别，在输入softmax层之前已控制每个样本的输出维度为3。
# 针对4个样本，依次大概率预测：2,0,1,0
Y_pred1 = torch.Tensor([[0.1, 0.2, 0.9], 
                        [1.1, 0.1, 0.2],
                        [0.2, 2.1, 0.1],
                        [1.2, 0.1, 0.3]])
# 大概率预测：0,2,2,1
Y_pred2 = torch.Tensor([[0.8, 0.2, 0.3],
                        [0.2, 0.3, 0.5],
                        [0.2, 0.2, 0.5],
                        [0.1, 0.7, 0.4]])

l1 = criterion(Y_pred1, Y)
l2 = criterion(Y_pred2, Y)
print("Loss1 = ", l1.data, "\nLoss2 = ", l2.data)

```

每张图像都可以看作矩阵形式，使用8位（1字节）来表示每个像素的灰度值，8位可以表示的整数范围是0到255（2^8=256）。

将（0-255）进行标准化，则可以构成数据集矩阵。例如下图是（1，28，28）（C(channel)，W(width)，H(height)）。

channel指颜色通道，RGB通道数为3，此处黑白则为1。

<img src="pic\dc115f30-be5c-4c96-a370-dbbc2695a18d.png" alt="dc115f30-be5c-4c96-a370-dbbc2695a18d" style="zoom:33%;" />

```python
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F

# 设置批量处理的样本数=64
batch_size = 64
# 构造对图像进行处理的转换器transform，对每个元素进行进行pipline处理：
# 首先将每张图片转换为（C，W，H）的Tensor形式，
# 然后将每个Tensor进行标准化缩放，此处缩放至均值0.1307，标准差0.3081的正态分布（MNIST数据集图像的均值和标准差）。
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ])

# 加载训练和测试数据集，并构造Dataset和DataLoader
train_dataset = datasets.MNIST(root='../dataset/mnist/',
                               train=True,
                               download=True,
                               transform=transform)
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size, num_workers=0)
test_dataset = datasets.MNIST(root='../dataset/mnist/',
                              train=False,
                              download=True,
                              transform=transform)
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size, num_workers=0)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 这里为何输入是784？因为将28*28的矩阵按行展开后再拼接，构造了一个（1，784）的向量。
        # 全连接网络通常要求输入是一维向量。
        # 因此，如果将一个 28x28 的图像输入到一个全连接网络（例如MLP）时，
        # 通常需要将其展平（flatten）为一个一维向量。
        # 这是因为全连接层将每个输入像素与一个神经元连接，不能直接处理多维结构。
        # 此处以全连接网络为例
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        # 为何输出10？因为MNIST数据集的图像有10种类别（0-9）
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        # 通过view()函数改变Tensor形状。-1代表设置列数后，行数将进行自动计算。
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        # 返回时，最后一层无需再做Relu激活，因为交叉熵函数自身将使用Softmax进行激活，并计算对数似然损失
        return self.l5(x)


# 设置模型在GPU上运行
# 2步：1将模型迁移至GPU；2将数据迁移至GPU
model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 设置损失函数和优化器
# 定义交叉熵损失函数
criterion = torch.nn.CrossEntropyLoss(reduction='mean')
# momentum=0.5：动量方法通过考虑过去梯度的累积来调整当前梯度更新的方向和幅度。缩短训练时间。
# 通常在0到1之间，越接近1，历史梯度的影响越大。
# 简单来说，动量方法在每一步的梯度更新中不仅考虑当前的梯度，还会加上一部分之前更新方向的“记忆”，
# 从而在优化过程中起到“惯性”作用。
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# 分别将每个epoch的训练和测试过程封装为函数
def train(epoch):
    # 定义累计损失
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):

        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        # 每300轮iteration输出1次平均损失（对累计损失求300次的平均），而不是输出所有iteration的损失
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


# 定义测试过程的函数
def test():
    # 定义正确数量和总数量
    correct = 0
    total = 0
    # 设置不需要计算梯度，因为此处没有任何训练过程，只验证模型结果的准确性
    with torch.no_grad():
        # 多个Batch，这里的循环次数为：测试集样本总数/Mini-Batch(64)
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # 针对单个Batch的样本，每次输出的是64个10维向量，10维表示其对10个类别的预测概率。
            # 目标是在对应类别的向量位置上，数据尽可能地相比其它类别大。
            # 注意此处还未经过Softmax层和对数似然损失，这里是Relu层输出，目标就是在计算损失前，让该层输出结果尽可能大。
            # （Softmax层不会改变原始数据的大小分布情况）
            outputs = model(images)
            # 针对64个样本按行拼接构成的矩阵，每次按列找出最大值及其对应索引。
            # 其中detach()表示获取数值。dim=1表示按列进行操作并计算最大值。返回=（最大值，最大值索引）。
            _, predicted = torch.max(outputs.detach(), dim=1)
            # 计算真实标签矩阵形状的第1个维度（这里是64）并进行累加
            total += labels.size(0)
            # 计算每64个样本里分类正确的样本数，并将其累加
            correct += (predicted == labels).sum().item()
    # 计算分类的准确率：即分类正确的样本数/总样本数
    print('Accuracy on test set: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    # 在每个epoch中，每次训练完就进行测试，展示模型性能变化
    for epoch in range(10):
        train(epoch)
        test()

```

# 9.卷积神经网络

相比先前全连接神经网络中，直接将图像矩阵展开并构造为一维向量形式。卷积神经网络选择保留图像空间信息，并将输入设置为原始图像矩阵，且首先需确定输入数据通道信息和输出的通道信息，步骤：

1 输入

2 特征提取（做卷积变换）

3 分类（将卷积最后层按照一维展开，并输入全连接层，再通过softmax层映射，对数似然损失）

<img src="pic\2f8e3e96-40ee-4d81-8b16-01c31c8d4149.png" alt="2f8e3e96-40ee-4d81-8b16-01c31c8d4149" style="zoom:33%;" />

卷积的过程：对每一小块图像进行遍历，遍历时对小块图像进行卷积操作。

<img src="pic\4c5a652c-c14b-42f0-a0f2-0f90c026461c.png" alt="4c5a652c-c14b-42f0-a0f2-0f90c026461c" style="zoom:33%;" />

单通道卷积操作：按卷积核对矩阵元素进行数乘，遍历整个图像，将结果数值放入输出矩阵。

<img src="pic\b22b8342-4a10-4dd2-af06-6e413729a2a7.png" alt="b22b8342-4a10-4dd2-af06-6e413729a2a7" style="zoom:33%;" />

多通道卷积操作：分别使用3个卷积核对每个通道进行卷积操作，并将获取的输出矩阵求和。

<img src="pic\80ee78aa-1a13-4ace-b925-c741dcf829e3.png" alt="80ee78aa-1a13-4ace-b925-c741dcf829e3" style="zoom:33%;" />

输入为n个通道时，采用的卷积核通道数也应为n，并输出通道数为1的卷积结果。

<img src="pic\efbfeb98-8d2f-47b7-a39d-74a09405c673.png" alt="efbfeb98-8d2f-47b7-a39d-74a09405c673" style="zoom:33%;" />

当进一步将结果输出通道数设定为m，则将以上处理的n通道卷积组数量增多至m个（在m次运算中，n通道卷积组都相同-称为共享权重），进行m次运算后将卷积结果堆叠可形成m个通道：

<img src="pic\0500c0a3-9d2e-40fb-8db9-ceee67de90a5.png" alt="0500c0a3-9d2e-40fb-8db9-ceee67de90a5" style="zoom:33%;" />

当输入通道数为n，输出通道数为m时，将卷积层设定为4维张量（m，n，w，h）：

类似先前线性回归时的矩阵乘法，通过输入和输出的通道维度，可确定权重矩阵的通道维度。

<img src="pic\f8d04610-b055-4d6c-a38d-6da561a4d79b.png" alt="f8d04610-b055-4d6c-a38d-6da561a4d79b" style="zoom:33%;" />

演示计算的进行过程：

```python
import torch

in_channels, out_channels = 5, 10
width, height = 100, 100
kernel_size = (3, 3)
batch_size = 1

# 随机生成一个批量大小为1，形状（5，100，100）（C，W，H）的输入张量
input = torch.randn(batch_size,
                    in_channels,
                    width,
                    height)

# 设置卷积层，必须的参数：输入通道、输出通道、卷积核形状
# 卷积层与图像输入的W和H没有关系
conv_layer = torch.nn.Conv2d(in_channels,
                             out_channels,
                             kernel_size=kernel_size)

output = conv_layer(input)

print(input.shape)
# 此处输出的W和H为什么是98：
# 因为卷积核是3，减去中心=2列，即在输入张量中，移动至极限时的左边距和右边距相加等于2，
# 使用100-2=98=卷积核在输入张量中部的可移动距离。
# 只要确定了输入张量的W和H，输出张量的W和H可自动确定。
print(output.shape)
print(conv_layer.weight.shape)

```

原始输入矩阵为5×5，卷积核3×3，因此结果=3×3（5-3+1），但如果想让输出结果=5×5（即保持输出尺寸的W、H=输入尺寸的W、H）？

对原始矩阵填充1圈均为0的padding，使结果输出矩阵变为5×5。

填充圈数：

x=填充后输入维度，input=i（填充前输入维度）, kernel=k（卷积核维度）, out=o（填充后输出维度）,

x-(k-1)=o, 则x=o-1+k,

圈数=(填充后输入维度-填充前输入维度)/2 = (x-i)/2 = (o-1+k-i)/2

因为这里o=i（填充后输出维度=填充前输入维度）, 所以圈数= **(k-1)/2** =3-1/2=1

<img src="pic\04e7569e-59bb-4458-a09a-4086aa5f51b2.png" alt="04e7569e-59bb-4458-a09a-4086aa5f51b2" style="zoom:33%;" />

填充padding的代码实现：

```python
import torch

in_channels, out_channels = 1, 1

# 手动设置输入的数据
input = [3, 4, 6, 5, 7,
         2, 4, 6, 8, 2,
         1, 6, 7, 8, 4,
         9, 7, 4, 6, 2,
         3, 7, 5, 4, 1]

input = torch.Tensor(input).view(1, in_channels, 5, 5)

# 和先前的不同在于对卷积层设置padding=1
conv_layer = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                             kernel_size=(3, 3), padding=1, bias=False)

# 手动设置卷积核数据
# 注意此处，先out_channels, 再in_channels（不像先前Conv2d中，固定先设置输入通道数，再输出通道数）
kernel = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).view(out_channels, in_channels, 3, 3)
# kernel.data是 kernel 的一个属性，
# 它返回一个与 kernel 共享相同数据的张量（tensor），
# 但不会跟踪计算历史（即不会参与自动求导）。
# 其存在是为了在不影响计算图的情况下访问张量的数据。
# 它通常用于在不需要梯度的情况下操作张量。
conv_layer.weight.data = kernel.data

output = conv_layer(input)
print(output)

```

设置stride：即设置卷积核移动的步长

填充圈数：

x=填充后输入维度，input=i（填充前输入维度）, kernel=k（卷积核维度）, out=o（填充后输出维度）, 移动步长为s

卷积核中部可移动距离 = x-(k-1) = move

o = 1+(move-1)/s = 1+[x-(k-1)-1]/s = **1+(x-k)/s**, （此处可根据该公式，将x看作输入并计算获取输出维度）则 x = s(o−1)+k

圈数=(填充后输入维度-填充前输入维度)/2 = (x-i)/2 = [s(o−1)+k-i]/2

因为这里o=i（填充后输出维度=填充前输入维度）, 所以圈数= **(si−s-i+k)/2**

<img src="pic\7ec22b15-e1f8-45f1-8659-00b3c849c661.png" alt="7ec22b15-e1f8-45f1-8659-00b3c849c661" style="zoom:33%;" />

设置stride的代码实现：

```python
conv_layer = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                             stride=2, # 只需修改卷积层中设置参数即可
                             kernel_size=(3, 3), bias=False)
```

Max Pooling最大池化：2×2的Max Pooling先将输入矩阵划分为2*2的子块集合，在每个划分的子块集里寻找最大值后进行组合（这里默认步长为2）。

因此，Max Pooling不会改变通道数，只会改变图像大小（经过Max Pooling后的图像大小将变为原先的1/2）

<img src="pic\253b7ad9-4fe8-4212-a8e3-9338820b80bb.png" alt="253b7ad9-4fe8-4212-a8e3-9338820b80bb" style="zoom:33%;" />

Max Pooling层代码实现：

```python
input = [3, 4, 6, 5,
         2, 4, 6, 8,
         1, 6, 7, 8,
         9, 7, 4, 6]
input = torch.Tensor(input).view(1, 1, 4, 4)

# 设置Max Pooling层
maxpooling_layer = torch.nn.MaxPool2d(kernel_size=2)
 
output = maxpooling_layer(input)
```

接下来，使用以下架构为例的CNN进行实现：

<img src="pic\836ba80e-c19b-4c05-9355-f709284c2e94.png" alt="836ba80e-c19b-4c05-9355-f709284c2e94" style="zoom:33%;" />

<img src="pic\a0fe1a32-af3d-46fd-ad0c-0f1dac5f4fe3.png" alt="a0fe1a32-af3d-46fd-ad0c-0f1dac5f4fe3" style="zoom:33%;" />

基础神经网络实现代码（这里直接在先前 多分类全连接神经网络的基础上，修改模型定义部分即可）：

```python
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ])

train_dataset = datasets.MNIST(root='../dataset/mnist/',
                               train=True,
                               download=True,
                               transform=transform)
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size, num_workers=0)
test_dataset = datasets.MNIST(root='../dataset/mnist/',
                              train=False,
                              download=True,
                              transform=transform)
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size, num_workers=0)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 修改此处，设置卷积层、池化层、全连接层
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=(5, 5))
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=(5, 5))
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        batch_sizes = x.size(0)
        # 这里由于卷积操作是线性的，通过激活函数引入非线性操作
        # 先池化降低W和H，再Relu，减少计算复杂度和运算量
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        # 将输出数据（1个batch-size）展平为一维后输入全连接层：64×20×4×4 -> 64×320
        x = x.view(batch_sizes, -1)
        # 此处无需激活，后面函数中包含了Softmax操作
        x = self.fc(x)
        return x


model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.CrossEntropyLoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):

        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.detach(), dim=1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    print('Accuracy on test set: %d %%' % (100 * correct / total))


if __name__ == '__main__':

    for epoch in range(10):
        train(epoch)
        test()

```

# 10.卷积神经网络（高级）

之前所学习的多层感知机（即全连接神经网络）、卷积神经网络，在网络架构上都是穿行结构，此处介绍几类更复杂架构。

## GoogleNet

可发现其中存在大量重复卷积模块，称为Inception。

<img src="pic\6c5281e5-3f8a-4eee-bbc9-25ee2f84bb7b.png" alt="6c5281e5-3f8a-4eee-bbc9-25ee2f84bb7b" style="zoom:33%;" />

<img src="pic\585e561f-9b37-4520-845c-1cd6ab1d14d3.png" alt="585e561f-9b37-4520-845c-1cd6ab1d14d3" style="zoom:33%;" />

Inception模块中提供了多类卷积核组合预选，通过训练计算权重后，将自动选择其中较优的组合方式。

其中Concatenate：将卷积结果按照通道的方向进行拼接（因此要求W和H必须相同，而C可以不同）。

Average Pooling：均值池化（Max Pool是取区域内最大保留，Average Pooling则是对区域求均值并保留）。

其中，针对3×3、5×5卷积核下的卷积操作，做padding即可保证图像W和H不变。针对AveragePooling也可做padding实现类似效果（类似卷积操作，但并没有卷积核进行运算，而是求卷积核范围内的均值）。

1×1 卷积的主要目标是调整特征图的通道数（同时还有“信息融合”的作用，将多通道信息汇聚在一起），而不改变W和H。

对于（C，W，H）的图像，进行1×1卷积后均变为（1，W，H），如需要输出通道数为m，则设置卷积层为m个即可（m，C，1，1）。

1×1 卷积可以起到类似降维的作用，相较于直接通过卷积操作改变通道C数量，基于1×1卷积能大幅降低运算量。

针对Inception模块的实现代码如下：

<img src="pic\edc59757-2002-40b7-b6d0-d593a7a1d9a5.png" alt="edc59757-2002-40b7-b6d0-d593a7a1d9a5" style="zoom: 33%;" />

```python
import torch

from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ])

train_dataset = datasets.MNIST(root='../dataset/mnist',
                               train=True,
                               download=True,
                               transform=transform)
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)

test_dataset = datasets.MNIST(root='../dataset/mnist',
                              train=False,
                              download=True,
                              transform=transform)
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)


# 定义Inception模型
class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        # 均值池化，有函数方法可直接调用
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        # 连接4个branch的输出张量，dim=1表示按照通道拼接
        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)


# 定义GoogleNet模型，其中包含Inception模块
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)

        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)

        self.mp = nn.MaxPool2d(2)
        # 此处，1408=[[28-(5-1)]/2 - (5-1)]/2 * (16 + 24*3)
        self.fc = nn.Linear(1408, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        # 在这步操作之后，经过inception模块的张量通道数均为24*3+16=88，所以先前设置conv2的输入通道数为88
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)

        return x


model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 2000))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy on test set: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()

```

当全连接层的输入维度难以确定时，可尝试使用一个张量作为输入，并直接输出结果形状，不用自己手动计算：

```python
import torch
import torch.nn.functional as F
import torch.nn as nn


class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)

        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)

        self.mp = nn.MaxPool2d(2)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size, -1)

        return x


net = Net()
print(net(torch.randn(64, 1, 28, 28)).data.shape[1])

```

## Residual Network (ResNet)

梯度消失问题：在链式法则中，当每次乘以的梯度都小于1，梯度将会越来越小，最终W将无法有效更新。（如果每次相乘的梯度都大于1则称为梯度爆炸问题，梯度过大将导致模型无法有效收敛）

老方法：划分输入层、多个隐藏层和输出层。逐层训练隐藏层，在对应隐藏层训练完毕后将其锁住，继续训练下一层，以解决梯度消失问题。该方法难以实现，因为神经网络中层数过多。

引入残差连接：输入x经过2个卷积层后得到F（x），先不激活，将F（x）与x相加后再通过Relu激活。

F（x）与x做加法，意味着其C，W，H均要一致。

<img src="pic\d3b88e18-dae9-4d86-ae5f-0fa79f7d23c5.png" alt="d3b88e18-dae9-4d86-ae5f-0fa79f7d23c5" style="zoom:33%;" />

使用以下网络结构为例进行ResNet实现：

<img src="pic\eb0081f6-7d11-484b-9367-817bfe2ef048.png" alt="eb0081f6-7d11-484b-9367-817bfe2ef048" style="zoom:33%;" />

<img src="pic\1675bb24-b189-41b7-9802-5c6e36ecb1cf.png" alt="1675bb24-b189-41b7-9802-5c6e36ecb1cf" style="zoom:33%;" />

```python
import torch

from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                                ])

train_dataset = datasets.MNIST(root='../dataset/mnist',
                               train=True,
                               download=True,
                               transform=transform)
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)

test_dataset = datasets.MNIST(root='../dataset/mnist',
                              train=False,
                              download=True,
                              transform=transform)
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)


# 此处定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        # 定义输入的通道数
        self.channels = channels
        # 这里为保证最后x能够与F（x）相加，需保证通道数不变，故设置卷积层的输入输出不变。
        # 同时设置padding=1保证W和H不变
        self.conv1 = nn.Conv2d(channels, channels,
                               kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(channels, channels,
                               kernel_size=(3, 3), padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        # 注意此处顺序，先相加，后激活
        return F.relu(x + y)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(5, 5))
        self.mp = nn.MaxPool2d(2)

        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)

        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.rblock1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.rblock2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)

        return x


model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 2000))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy on test set: %d %%' % (100 * correct / total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
```

# 11.循环神经网络

RNN思想：针对带有时序特征的数据，使用权重共享（例如CNN中，对一张图像使用相同的卷积核进行滑动，因此权重数量少）的概念，减少权重运算的计算量。考虑数据序列前后具有依赖关系。

## RNNCell

在RNN Cell中进行循环计算，每次接收上次计算的输出和本次输入进行运算。

<img src="pic\e4957b6e-00ce-4d77-8a61-4a935bf18760.png" alt="e4957b6e-00ce-4d77-8a61-4a935bf18760" style="zoom:33%;" />

1.输入数据xt维度是i，输出隐藏层ht维度是h，则wih表示线性变换矩阵是i*h

2.输入上个隐层ht-1的维度是h，输出隐藏层ht维度是h，则whh表示线性变换矩阵是h*h

3.先将1和2中进行线性变换的结果相加，再通过tanh进行非线性激活（取值-1至+1），输出隐层ht，并作为下次循环运算的输入

4.此处实际整体上只进行了一次线性层运算：

<img src="pic\5152085a-dfbc-412e-a30a-15dafc2481b1.png" alt="5152085a-dfbc-412e-a30a-15dafc2481b1" style="zoom: 50%;" />

<img src="pic\59d88b0c-3b9a-462a-ac53-6a09d03dab4a.png" alt="59d88b0c-3b9a-462a-ac53-6a09d03dab4a" style="zoom:33%;" />

在Pytorch中，实现RNN有2种方式（按照输入分别设置为数据列表中单个向量/全部向量）：

1.定义RNN Cell，并自己实现循环

```python
import torch

batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2

# 定义RNNCell，需要的定义参数有输入维度和隐藏层维度
cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

# 设置数据集：序列长度，批量大小，输入维度
dataset = torch.randn(seq_len, batch_size, input_size)
# 设置隐藏层（初始设置为0）：批量大小，隐藏层维度
hidden = torch.zeros(batch_size, hidden_size)
print(dataset)

# 遍历数据集序列，将每次的序列数据和上次输出作为当前输入
for idx, input in enumerate(dataset):
    print('=' * 20, idx, '=' * 20)
    # 将当前序列数据、隐藏层输入RNNCell中计算，获取hidden作为输出（也是下次的输入）
    hidden = cell(input, hidden)
    print('outputs size: ', hidden.shape)
    print(hidden)

```

由观察可知，RNNCell只要设置输入输出的维度即可，**无需设置Batch**。而输入数据和输出数据则均需要设置维度和批量大小（输入还需设置序列长度）。总结：

**RNNCell：Inputsize、Hiddensize**

**Input：Batchsize、Seqlen、Inputsize**

**Out：Batchsize、Hiddensize**

## RNN

2.直接使用RNN，不用自己实现循环。

其中，输入设置为整个序列集合，输出包括隐藏层序列集合(out)+最终隐藏层输出结果(hidden)，且可设置RNN层数。

<img src="pic\cf875432-c9bb-4dd9-98e6-859b7572babc.png" alt="cf875432-c9bb-4dd9-98e6-859b7572babc" style="zoom:33%;" />

本质上，多层RNN在单个RNN的基础上，将当前时刻输入修改为上层输出。这样RNN便完成了空间上的数据变换。

<img src="pic\1d47b07a-8478-4035-b374-5b264264d637.png" alt="1d47b07a-8478-4035-b374-5b264264d637" style="zoom:33%;" />

```python
import torch

batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2
num_layers = 1

# 直接定义RNN，参数：输入维度、隐藏层维度、RNN层数
cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size,
                    num_layers=num_layers)

# 定义输入，参数：序列长度、批量大小、输入维度
inputs = torch.randn(seq_len, batch_size, input_size)
# 定义初始隐藏层，参数：RNN层数、批量大小、隐藏层维度
hidden = torch.zeros(num_layers, batch_size, hidden_size)

# 获取结果：最后一层Cell的输出结果集合（上），最终时序于不同层下的输出结果（右）
# 这里，out的最后一个输出应该等于hidden的最后一个输出
out, hidden = cell(inputs, hidden)
print('Output size: ', out.shape)
print('Output: ', out)
print('Hidden size: ', hidden.shape)
print('Hidden: ', hidden)
```

RNN可以设置参数batch_first使得BatchSize和序列长度进行位置交换：

<img src="pic\bd304dfb-a619-4d44-9a7b-3f8d9898ffda.png" alt="bd304dfb-a619-4d44-9a7b-3f8d9898ffda" style="zoom: 33%;" />

```python
import torch

batch_size = 1
seq_len = 3
input_size = 4
hidden_size = 2
num_layers = 1

# 设置交换顺序
cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size,
                    num_layers=num_layers, batch_first=True)

# 输入数据：交换顺序
inputs = torch.randn(batch_size, seq_len, input_size)
# 注意这里不要交换顺序
hidden = torch.zeros(num_layers, batch_size, hidden_size)

out, hidden = cell(inputs, hidden)
print('Output size: ', out.shape)
print('Output: ', out)
print('Hidden size: ', hidden.shape)
print('Hidden: ', hidden)
```

总结：

**RNN：Inputsize、Hiddensize、Numlayers**

**Input：Batchsize、Seqlen、Inputsize**

**Out：Numlayers、Batchsize、Hiddensize**

## RNNCell-Seq2Seq

以针对“hello”单词的处理为例，进行序列到序列（Sequence-to-Sequence，常简写为Seq2Seq）的转换任务：

RNN只能处理向量形式，如何将“hello”单词转换为向量序列？

<img src="pic\84f724e2-7e56-4362-b404-c4c20768e220.png" alt="84f724e2-7e56-4362-b404-c4c20768e220" style="zoom:33%;" />

1.使用RNNCell进行实现

```python
import torch

batch_size = 1
input_size = 4
hidden_size = 4

# 定义字典
idx2char = ['e', 'h', 'l', 'o']
# 对应输入：hello
x_data = [1, 0, 2, 2, 3]
# 对应输出：0hlol
y_data = [3, 1, 2, 3, 2]

# 构造one_hot索引矩阵
one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
# 定义输入的向量序列，维度：Seqlen, inputsize
x_one_hot = [one_hot_lookup[x] for x in x_data]

inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
labels = torch.LongTensor(y_data).view(-1, 1)


class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(Model, self).__init__()
        # self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 注意这里是RNNCell而不是RNN
        self.rnncell = torch.nn.RNNCell(input_size=self.input_size,
                                        hidden_size=self.hidden_size)

    # 1次前向传播，就是进行1次RNN的input输入和hidden输出
    def forward(self, input, hidden):
        hidden = self.rnncell(input, hidden)
        return hidden

    # 在init中设置batch_size的目的，就是在此处初始化hidden，实际上这里的初始化操作也可放在函数外部
    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size)


# 分别定义模型、损失函数、优化器
net = Model(input_size, hidden_size, batch_size)
# 将其看作多分类问题
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

for epoch in range(15):
    loss = 0
    optimizer.zero_grad() 
    # 初始化输入的hidden
    hidden = net.init_hidden()
    print('Predicted string: ', end='')
    # 这里实际上就是按seq_len在遍历向量列表
    for input, label in zip(inputs, labels):
        hidden = net(input, hidden)
        # 损失函数？计算hidden和label的对比
        # 在序列建模任务中（如文本生成、时间序列预测），一个样本通常指一个完整的序列。
        #（此处示例中，相当于1次完整循环中只有1个样本“hello”）
        #（可发现，RNN中，单个样本序列元素中每个元素都有label标签，但是整体也算作一个样本）
        #（序列中的每个元素（如每个字符）对应一个时间步。每个时间步的输入和输出可能都有标签，但这些时间步的标签共同构成一个样本的标签序列。）
        # 注意这里，loss是在单个样本（"hello"）中进行累加计算，且没有使用.item()，因为需要构造计算图。
        # 通过下图说明了RNN和CNN在loss值更新中的差异。
        loss += criterion(hidden, label)
        _, idx = hidden.max(dim=1)
        print(idx2char[idx.item()], end='')


    loss.backward()
    optimizer.step()
    print(', Epoch [%d/15] loss = %.4f' % (epoch + 1, loss.item()))

```

<img src="pic\0c83098f-6e5a-4d7a-a05d-608ee3a4df6e.png" alt="0c83098f-6e5a-4d7a-a05d-608ee3a4df6e" style="zoom: 50%;" />

## RNN-Seq2Seq

2.使用RNN进行实现

```python
import torch

batch_size = 1
# 添加定义了序列长度
seq_len = 5
input_size = 4
hidden_size = 4
# 添加定义了RNN层次
num_layers = 1

idx2char = ['e', 'h', 'l', 'o']
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2]

one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]
inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)
labels = torch.LongTensor(y_data)


class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
        super(Model, self).__init__()
        # 此处就是多定义了一个RNN层次
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                num_layers=num_layers)

    def forward(self, input):
        # 初始化hidden输入
        hidden = torch.zeros(self.num_layers,
                             self.batch_size,
                             self.hidden_size)
        # RNN针对整个数据序列（就算是整个序列，也属于单个样本）进行处理，可忽视中间对序列的遍历过程
        # 最终输出：最后1层每个Cell的输出集合，最后1个时间步上的各层输出
        # 这里的输出结果：out(seqlen, batchsize, hiddensize)
        out, _ = self.rnn(input, hidden)
        # 返回结果：(seqlen × batchsize, hiddensize)，这里将前2个维度拼接在一起，方便后续计算
        return out.view(-1, self.hidden_size)


net = Model(input_size, hidden_size, batch_size, num_layers)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

for epoch in range(15):
    # 这里因为使用的是RNN，发现没有对序列的循环过程
    optimizer.zero_grad()
    # 返回结果：(seqlen × batchsize, hiddensize)
    outputs = net(inputs)
    # 这里labels的维度：(seqlen × batchsize, 1)，直接与(seqlen × batchsize, hiddensize)进行交叉熵计算
    # 注意：CrossEntropyLoss此处处理的是两个序列向量集合，而不是先前RNNCell中成双的向量，
    # 因此默认对所有时间步的损失求平均。若需与RNNCell的累加结果一致，需设置reduction='sum'。
    # 一般采用求均值的方式更优。
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()
    print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
    print(', Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))

```

## Embedding和线性变换

在进行自然语言处理时，独热向量存在哪些缺点？

1.维度过高（例如，词级？每个向量都是万级维度，维度诅咒）

2.分布稀疏（万级节点映射至坐标轴上一个点）

3.硬编码（一 一对应）

能否变为：1.低维度 2.稠密 3.可学习？

为此引入**EMBEDDING**（嵌入层），进行数据降维。

<img src="pic\f5e5109b-256d-4307-a9f0-0c6c58ebf6ff.png" alt="f5e5109b-256d-4307-a9f0-0c6c58ebf6ff" style="zoom:33%;" />

关键参数：词汇表大小（inputSize）和嵌入维度（embeddingSize）。

根据对应元素在词汇表中位置，寻找对应向量即可。

<img src="pic\f997ae82-0557-4f35-bd1b-bd24f407a38b.png" alt="f997ae82-0557-4f35-bd1b-bd24f407a38b" style="zoom:33%;" />

<img src="pic\039d3583-c19c-422d-a3f7-5555c62fed7a.png" alt="039d3583-c19c-422d-a3f7-5555c62fed7a" style="zoom:33%;" />

使用时，在序列输入RNN之前，对序列元素分别进行embedding映射（输入需要是长整形）即可。

最后可以接一个线性层，将hiddensize映射至需要分类的维度即可。

（这里，隐藏层维度hiddensize未必等于需分类维度。有时隐藏层需足够大，以编码中间特征）

这里的线性层，可对长序列中每个元素进行处理（参考先前corssenpty）

<img src="pic\f68aa152-7027-457a-9bcc-9fd9a83d6df1.png" alt="f68aa152-7027-457a-9bcc-9fd9a83d6df1" style="zoom:33%;" />

案例实现

```python
import torch

# 批量大小设置1
batch_size = 1
# 序列长度设置5（例：样本hello）
seq_len = 5
# 词汇表维度设置4
input_size = 4
# 隐藏层设置8（这里不是分类类别数）
hidden_size = 8
# 2层RNN
num_layers = 2
# 嵌入维度设置10
embedding_size = 10
# 定义分类类别数
num_class = 4

idx2char = ['e', 'h', 'l', 'o']
# 因为之后设置batch_first，输入数据需将batchsize设置为第1个维度
x_data = [[1, 0, 2, 2, 3]]
y_data = [3, 1, 2, 3, 2]
# embedding的输入需要长整形
inputs = torch.LongTensor(x_data)
labels = torch.LongTensor(y_data)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 定义embedding层
        self.emb = torch.nn.Embedding(input_size, embedding_size)
        # 设置RNN，batch_first=True
        self.rnn = torch.nn.RNN(input_size=embedding_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                batch_first=True)
        # 设置线性层
        self.fc = torch.nn.Linear(hidden_size, num_class)

    def forward(self, x):
        # 设置初始隐层
        hidden = torch.zeros(num_layers, batch_size, hidden_size)
        # 将初始数据x(batch_size,seqlen)输入Embedding(input_size,embedding_size)，
        # 由Embedding转换为x(batch_size,seqlen,embedding_size)
        x = self.emb(x)
        # 将x(batch_size,seqlen,embedding_size)输入RNN，
        # 获取各个时间步上最后层输出(batch_size,seqlen,hidden_size)
        x, _ = self.rnn(x, hidden)
        # 将最后层输出(batch_size,seqlen,hidden_size)基于线性层变换为(batch_size,seqlen,num_class)
        x = self.fc(x)
        # 返回(batch_size×seqlen,num_class)
        return x.view(-1, num_class)


net = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

for epoch in range(15):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()

    print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
    print(', Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))

```

