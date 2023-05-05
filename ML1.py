import numpy as np
np.random.seed(114514)


def sigmoid(x):
    """激活函数"""
    return 1 / (1 + np.exp(-x))

def MSE_loss(y_hat,y):
    """L2损失函数 """
    return np.mean((y_hat-y.reshape(y_hat.shape))**2/2)

def auto_grad(y_hat,y):
    """反向传播"""
    y = y.reshape(y_hat.shape)
    return y_hat*(1-y_hat)*(y-y_hat)

class Perceptor():
    """指定a、b、c、d下的异或感知机"""
    def __init__(self,a,b,c,d,lr=5,epochs=50):
        # 训练数据集
        self.X = np.array([[0, 0, a], [0, 1, b], [1, 0, c], [1, 1, d]])
        self.Y = np.array([0, 1, 1, 0])
        #模型可学习参数
        self.weight = np.random.normal(0,0.1,size=(self.X.shape[1],1))
        self.bias = np.random.normal(0,0.1,size=(1,1))

        self.epochs = epochs#迭代次数
        self.lr = lr#学习率

    def forward(self, X):
        """前向传播"""
        Y_hat = X@self.weight + self.bias
        Y_hat = sigmoid(Y_hat)
        return Y_hat

    def train(self):
        """训练"""
        for epoch in range(self.epochs):
            #前向传播
            Y_hat = self.forward(self.X)
            grad = auto_grad(Y_hat,self.Y)
            # print(f"epoch {epoch+1}: {MSE_loss(Y_hat,self.Y)}")
            #梯度下降
            self.weight += self.lr*(self.X.T@grad)
            self.bias += self.lr*np.sum(grad,axis=0,keepdims=True)

    def predict(self):
        """预测"""
        Y_hat = self.forward(self.X)
        return Y_hat.round()

    def accuracy(self):
        """计算预测准确率"""
        cmp = self.predict().reshape(self.Y.shape)==self.Y
        return float(cmp.sum())/self.Y.shape[0]

if __name__ == "__main__":
    true_arg_lst = []#存储可能的abcd组合
    for flag in range(16):
        arg = [(flag//(2**i))%2 for i in range(4)]#将i转为二进制作为abcd的组合
        model = Perceptor(*arg)
        model.train()
        # print(arg," accuracy: ",model.accuracy())
        if model.accuracy() == 1.0:
            true_arg_lst.append(arg)

    print(f"误差为0的组合总数: {len(true_arg_lst)}")
    print("组合 [a,b,c,d] 可为: ")
    for item in true_arg_lst: print(item)