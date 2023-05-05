import numpy as np
import random
random.seed(114514)
np.random.seed(114514)


def sigmoid(x):
    return 1/(1+np.exp(-x))


class DataSet:
    def __init__(self,m=1000,batch_size=40,shuffle=True):
        self.A = np.array([[0.8,0.6],[0.47,0.88]])
        self.len = m
        self.batch_size = batch_size
        self.shuffle = shuffle

        laplace_sample = np.random.laplace(0,1,(m,1))
        sigmoid_sample = self.get_sigmoid_sample((m,1))
        self.features = np.concatenate((laplace_sample,sigmoid_sample),axis=1)
        self.features = self.features@self.A.T

    def get_gt(self):
        return self.A

    def get_sigmoid_sample(self,size):
        sample = np.random.uniform(0,1,size)
        return np.log(sample) - np.log(1-sample)

    def loader(self, *args, **kwds):
        cnt = (self.len//self.batch_size)*self.batch_size
        idx_lst = list(range(cnt))
        if self.shuffle: random.shuffle(idx_lst)

        for i in range(0,cnt,self.batch_size):
            op = self.features[idx_lst[i:i+self.batch_size]]
            yield op

class ICA:
    def __init__(self,num_sample=2000,epochs=100,batch_size=10,lr=0.1,err = 0.015):
        self.dataset = DataSet(
            m=num_sample,batch_size=batch_size,shuffle=True
        )
        self.dataloader = self.dataset.loader()
        self.A = self.dataset.A
        print("laplace and sigmoid signal")
        print("signal num: ",num_sample)
        print("SGD batch size: ",batch_size)
        print("epochs: ",epochs)
        print("error tolerence: ",err)
        print("learning rate",lr,"\n")
        print("ground truth A: \n",self.A,"\n")
        self.W = np.eye(2)
        self.batch_size = batch_size
        
        self.epochs = epochs
        self.lr = lr
        self.err = err

    def train(self):
        print("estimate A: ",)
        for i in range(self.epochs):
            for X in self.dataloader:
                dw = self.lr*( ((1-2*sigmoid(self.W@X.T))@X) / self.batch_size+np.linalg.inv(self.W.T))
                self.W += dw
                if np.linalg.norm(dw)<self.err:

                    Anew = np.linalg.inv(self.W).T
                    print(Anew/np.linalg.norm(Anew,axis=1,keepdims=True),"\n")

if __name__=="__main__":
    ica = ICA()
    ica.train()