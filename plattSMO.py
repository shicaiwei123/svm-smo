import sys
from numpy import *
from svm import *
from os import listdir
import datetime


class PlattSMO:
    def __init__(self, dataMat, classlabels, C, toler, maxIter, **kernelargs):
        """
        构造函数
        :param dataMat: 特征向量，二维数组
        :param classlabels: 数据标签，一维数组
        :param C: 对松弛变量的容忍程度，越大越不容忍
        :param toler: 完成一次迭代维度误差要求
        :param maxIter: 迭代次数
        :param kernelargs: 核参数，特别注意高斯核会有两个参数，但是都是通过这一个变量传进来的
        """
        self.x = array(dataMat)
        self.label = array(classlabels).transpose()
        self.C = C
        self.toler = toler
        self.maxIter = maxIter
        self.m = shape(dataMat)[0]  # 输入的行数，也即是表示了有多少个输入
        self.n = shape(dataMat)[1]  # 列数，表示每一个输入有多少个特征向量
        self.alpha = array(zeros(self.m), dtype='float64')  # 初始化alpha
        self.b = 0.0
        self.eCache = array(zeros((self.m, 2)))  # 错误缓存
        self.K = zeros((self.m, self.m), dtype='float64')  # 先求内积，
        self.kwargs = kernelargs
        self.SV = ()  # 最后保留的支持向量
        self.SVIndex = None  # 支持向量的索引
        # wx本就是内积本就是输入一个特征向量，和已有的所有特征向量内积（或者说支持向量），然后再加权求和求出一个数
        # 训练时候，x即是输入的特征向量，也是所有特征向量的一部分，所以可以这样
        # 一个确定xi就是输入x，和所有的特征向量，也就是xj求内积，放在一行当中，那应该是按行取为什么按列
        # 为了用矩阵求和，而且其实这个是一个对称矩阵，所以可以直接这样，至于为什么对称，emmm
        for i in range(self.m):
            for j in range(self.m):
                self.K[i, j] = self.kernelTrans(self.x[i, :], self.x[j, :])

    def calcEK(self, k):
        """
        计算第k个数据的误差
        :param k:
        :return:
        """
        # 因为这是训练阶段，用数据集的数据，所以可以直接这样做，这里先把内积全部求了，
        fxk = dot(self.alpha * self.label, self.K[:, k]) + self.b
        Ek = fxk - float(self.label[k])
        return Ek

    def updateEK(self, k):
        Ek = self.calcEK(k)

        self.eCache[k] = [1, Ek]

    def selectJ(self, i, Ei):
        """
        在确定了一个参数的前提下，按照最大步长取另一个参数
        :param i:
        :param Ei:
        :return:
        """
        maxE = 0.0
        selectJ = 0
        Ej = 0.0
        validECacheList = nonzero(self.eCache[:, 0])[0]
        if len(validECacheList) > 1:
            for k in validECacheList:
                if k == i: continue
                Ek = self.calcEK(k)
                deltaE = abs(Ei - Ek)
                if deltaE > maxE:
                    selectJ = k
                    maxE = deltaE
                    Ej = Ek
            return selectJ, Ej
        else:
            selectJ = selectJrand(i, self.m)
            Ej = self.calcEK(selectJ)
            return selectJ, Ej

    def innerL(self, i):
        """
        选择参数之后，更新参数
        """
        Ei = self.calcEK(i)

        if (self.label[i] * Ei < -self.toler and self.alpha[i] < self.C) or \
                (self.label[i] * Ei > self.toler and self.alpha[i] > 0):

            self.updateEK(i)

            j, Ej = self.selectJ(i, Ei)

            alphaIOld = self.alpha[i].copy()
            alphaJOld = self.alpha[j].copy()

            if self.label[i] != self.label[j]:
                L = max(0, self.alpha[j] - self.alpha[i])
                H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
            else:
                L = max(0, self.alpha[j] + self.alpha[i] - self.C)
                H = min(self.C, self.alpha[i] + self.alpha[j])
            if L == H:
                return 0

            eta = 2 * self.K[i, j] - self.K[i, i] - self.K[j, j]
            if eta >= 0:
                return 0

            self.alpha[j] -= self.label[j] * (Ei - Ej) / eta
            self.alpha[j] = clipAlpha(self.alpha[j], H, L)
            self.updateEK(j)

            if abs(alphaJOld - self.alpha[j]) < 0.00001:  # 目标迭代完成，不需要再迭代，而且所有参数已经保留，理论上是不应该保留当前的参数的，但是也正因为反正相差不多，所以可以
                return 0

            self.alpha[i] += self.label[i] * self.label[j] * (alphaJOld - self.alpha[j])
            self.updateEK(i)

            b1 = self.b - Ei - self.label[i] * self.K[i, i] * (self.alpha[i] - alphaIOld) - \
                 self.label[j] * self.K[i, j] * (self.alpha[j] - alphaJOld)
            b2 = self.b - Ej - self.label[i] * self.K[i, j] * (self.alpha[i] - alphaIOld) - \
                 self.label[j] * self.K[j, j] * (self.alpha[j] - alphaJOld)
            if 0 < self.alpha[i] and self.alpha[i] < self.C:
                self.b = b1
            elif 0 < self.alpha[j] and self.alpha[j] < self.C:
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.0
            return 1
        else:
            return 0

    def smoP(self):
        """
        外层大循环，
        :return:
        """
        iter = 0
        entrySet = True
        alphaPairChanged = 0
        while iter < self.maxIter and ((alphaPairChanged > 0) or (entrySet)):
            alphaPairChanged = 0
            if entrySet:
                for i in range(self.m):
                    alphaPairChanged += self.innerL(i)
                iter += 1
            else:
                nonBounds = nonzero((self.alpha > 0) * (self.alpha < self.C))[0]
                for i in nonBounds:
                    alphaPairChanged += self.innerL(i)
                iter += 1
            if entrySet:
                entrySet = False
            elif alphaPairChanged == 0:
                entrySet = True
        # 保存模型参数
        self.SVIndex = nonzero(self.alpha)[0]  # 取非0部分把应该
        self.SV = self.x[self.SVIndex]
        self.SVAlpha = self.alpha[self.SVIndex]
        self.SVLabel = self.label[self.SVIndex]

        # 清空中间变量
        self.x = None
        self.K = None
        self.label = None
        self.alpha = None
        self.eCache = None

    #   def K(self,i,j):
    #       return self.x[i,:]*self.x[j,:].T
    def kernelTrans(self, x, z):
        """
        核函数说到底就是求内积，求输入x和已有数据标签的内积，最后的结果是一个常数
        内积有两个，一个是向量形式的对应相乘相加一个是矩阵形式的矩阵乘积
        输入是两个要求内积的一维数组，也就是说要拆成一个个来做
        :param x:
        :param z:
        :return:
        """
        if array(x).ndim != 1 or array(x).ndim != 1:
            raise Exception("input vector is not 1 dim")
        if self.kwargs['name'] == 'linear':
            return sum(x * z)
        elif self.kwargs['name'] == 'rbf':
            theta = self.kwargs['theta']
            return exp(sum((x - z) * (x - z)) / (-1 * theta ** 2))

    def calcw(self):
        """
        计算w，结果是一个数组，长度是特征向量的长度
        :return:
        """
        for i in range(self.m):
            self.w += dot(self.alpha[i] * self.label[i], self.x[i, :])

    def predict(self, testData):
        """
        输入待预测的数据，输出结果
        :param testData: 待预测数据，要是数组形式，二维数组
        :return: 一个列表，包含结果
        """
        test = array(testData)
        # return (test * self.w + self.b).getA()
        result = []
        m = shape(test)[0]
        for i in range(m):
            tmp = self.b
            for j in range(len(self.SVIndex)):
                # wx+b,w和SVAlpha，Label以及核函数有关
                # 求和可以是直接求和，也可以转成矩阵，这里就是直接求和
                # 计算支持向量的数目，用他们来进行估计
                tmp += self.SVAlpha[j] * self.SVLabel[j] * self.kernelTrans(self.SV[j], test[i, :])

            # 不可分的情况下，其实就可以，直接指定想要的情况，如果需要的话，工程中会有这个要求
            while tmp == 0:
                tmp = random.uniform(-1, 1)
            if tmp > 0:
                tmp = 1
            else:
                tmp = -1
            result.append(tmp)
        return result


def plotBestfit(data, label, w, b):
    import matplotlib.pyplot as plt
    n = shape(data)[0]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x1 = []
    x2 = []
    y1 = []
    y2 = []
    for i in range(n):
        if int(label[i]) == 1:
            x1.append(data[i][0])
            y1.append(data[i][1])
        else:
            x2.append(data[i][0])
            y2.append(data[i][1])
    ax.scatter(x1, y1, s=10, c='red', marker='s')
    ax.scatter(x2, y2, s=10, c='green', marker='s')
    x = arange(-2, 10, 0.1)
    y = ((-b - w[0] * x) / w[1])
    plt.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()


def loadImage(dir, maps=None):
    dirList = listdir(dir)
    data = []
    label = []
    for file in dirList:
        label.append(file.split('_')[0])
        lines = open(dir + '/' + file).readlines()
        row = len(lines)
        col = len(lines[0].strip())
        line = []
        for i in range(row):
            for j in range(col):
                line.append(float(lines[i][j]))
        data.append(line)
        if maps != None:
            label[-1] = float(maps[label[-1]])
        else:
            label[-1] = float(label[-1])
    return array(data), array(label)


def main():
    '''
    data,label = loadDataSet('testSetRBF.txt')
    smo = PlattSMO(data,label,200,0.0001,10000,name = 'rbf',theta = 1.3)
    smo.smoP()
    smo.calcw()
    print smo.predict(data)
    '''
    # 数据读取
    # 特征是二维数组，标签是一维数组，记住这个就可以，而且标签不是实际标签，而都要转变成-1,1，正类和负类。，谁正谁负没关系。
    maps = {'1': 1.0, '9': -1.0}
    data, label = loadImage("digits/trainingDigits", maps)
    test, testLabel = loadImage("digits/testDigits", maps)

    # 训练
    smo = PlattSMO(data, label, 1, 0.0001, 10000, name='rbf', theta=20)
    begin = datetime.datetime.now()
    smo.smoP()
    end = datetime.datetime.now()
    time_sub = end - begin
    print("keneral_svm time", time_sub.total_seconds())
    print(len(smo.SVIndex))

    # 预测
    testResult = smo.predict(test)
    m = shape(test)[0]
    count = 0.0
    for i in range(m):
        if testLabel[i] != testResult[i]:
            count += 1
    print("classfied error rate is:", count / m)
    # smo.kernelTrans(data,smo.SV[0])


if __name__ == "__main__":
    sys.exit(main())
