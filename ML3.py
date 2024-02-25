import numpy as np

class DP:
    def __init__(self,err=1e-6,gamma=0.9):
        self.gamma = gamma
        self.err = err

        self.Vs = np.zeros((8,7))# V(s)
        self.Pi_A = np.ones((8,7,4))*0.25# Pi(a|s)
        self.P = np.zeros((8,7,4,8,7))# P(s'|s,a)
        self.R = np.zeros((8,7,8,7))# R(s'|s)

        self.init_state()

    def init_state(self):
        """initial the episode"""
        node_lst = [[2,3],[5,2],[3,5],[3,1],[5,4]]; l = len(node_lst)
        r_lst = [1,9,3,5,41]
        dx=[1,0,-1,0]; dy=[0,1,0,-1]

        for i in range(1,7):
            for j in range(1,6):
                for k in range(4):
                    self.P[i][j][k][i+dy[k]][j+dx[k]]=1

        for idx in range(l):
            y,x = node_lst[idx]; nx,ny = node_lst[(idx+1)%l]
            self.P[y][x]*=0
            self.P[y,x,:,ny,nx]+=1

            self.R[y][x][ny][nx] = r_lst[idx]

        self.R[1,:,0,:]-=1; self.R[6,:,7,:]-=1; self.R[:,1,:,0]-=1; self.R[:,5,:,6]-=1

    def adj_opt(self):
        """a trick to handle the boundary problem"""
        self.Vs[0,:] = self.Vs[1,:]; self.Vs[7,:] = self.Vs[6,:] 
        self.Vs[:,0] = self.Vs[:,1]; self.Vs[:,6] = self.Vs[:,5]

    def esti_V(self):
        """estimate the V(s)"""
        tmp_err=114514
        iter_cnt=0
        while(tmp_err>self.err):
            iter_cnt+=1
            tmp_err=0
            self.adj_opt()
            for i in range(1,7):
                for j in range(1,6):
                    new_v = np.sum(self.P[i][j] * (self.R[i][j] + self.gamma*self.Vs), axis=(1,2))
                    new_v = np.sum(new_v * self.Pi_A[i][j])
                    tmp_err = max(tmp_err,abs(self.Vs[i][j]-new_v))
                    # if(t%2500==0): print(abs(self.Vs[i][j]-new_v),":",self.Vs[i][j]," ",end="")
                    self.Vs[i][j] = new_v
                    
            #     if(t%2500==0): print("\n")
            # if(t%2500==0): print("-----------------------------------");
        return iter_cnt
        
    def optim_pi(self):
        """fix the V(s),find a better Pi(a|s)"""
        self.adj_opt()
        for i in range(1,7):
            for j in range(1,6):
                self.Pi_A[i][j]*=0
                vij = np.sum(self.P[i][j] * (self.R[i][j] + self.gamma*self.Vs), axis=(1,2))
                self.Pi_A[i][j][np.argmax(vij)]+=1

    def train(self):
        """optimize Pi(a|s) until model is convergent"""
        iter_cnt=114514
        while(iter_cnt>1):
            iter_cnt = self.esti_V()
            self.optim_pi()

    def show_result(self):
        """show V(s) and Pi(a|s)"""
        print("V(s): \n")
        for i in range(1,7):
            for j in range(1,6):
                print(f"{self.Vs[i][j]} ",end="")
            print("\n")
        print("Pi(a|s): \n")
        for i in range(1,7):
            for j in range(1,6):
                print(np.argmax(self.Pi_A[i][j]),end=" ")
            print("\n")

        
if __name__=="__main__":
    dp = DP()
    dp.train()
    dp.show_result()
        

