import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

class VonNeumannNetwork:
    def __init__(self, N):
        self.N = N
        self.A = np.zeros((N,N), dtype=int)
        self.t = 0
    
    def ones(self):
        self.A = np.ones((self.N,self.N), dtype=int)
        
    def alternate(self):
        self.A = np.zeros((self.N,self.N), dtype=int)
        
        for i in range(1,self.N):
            self.A[i] += i
            self.A[:,i] += i
            
        self.A = self.A % 2
        
    def left(self, i, j):
        return self.A[(i - 1) % self.N, j]
    
    def right(self, i, j):
        return self.A[(i + 1) % self.N, j]
    
    def up(self, i, j):
        return self.A[i, (j - 1) % self.N]
    
    def down(self, i, j):
        return self.A[i, (j + 1) % self.N]
        
    def node_step(self, i, j):
        neighbors = [
            self.left(i,j), 
            self.right(i,j),
            self.up(i,j),
            self.down(i,j)]
        
        return sum(neighbors) >= 4
    
    def run_step(self):
        tmp = np.empty((self.N, self.N), dtype=int)
        
        for i in range(self.N):
            for j in range(self.N):
                tmp[i,j] = self.node_step(i, j)
                
        self.A = tmp
        self.t += 1

    def plot1d(self, cols, figsize, fn):
        _, axs = plt.subplots(1,cols,figsize=figsize)

        for i in range(cols):
            sns.heatmap(
                g.A, 
                vmin=0, 
                vmax=1,
                cbar=False,
                xticklabels=False,
                yticklabels=False,
                ax=axs[i])

            axs[i].set_xlabel(f't={g.t}')
            g.run_step()

        plt.savefig(fn)

    
    def plot2d(self, rows, cols, figsize, fn):
        _, axs = plt.subplots(rows,cols,figsize=figsize)

        for i in range(rows):
            for j in range(cols):
                sns.heatmap(
                    g.A, 
                    vmin=0, 
                    vmax=1,
                    cbar=False,
                    xticklabels=False,
                    yticklabels=False,
                    ax=axs[i,j])

                axs[i,j].set_xlabel(f't={g.t}')
                g.run_step()

        plt.savefig(fn)

        

if __name__ == "__main__":

    # plot telephone tag with missing "on"
    g = VonNeumannNetwork(10)
    g.alternate()
    g.A[4,5] = 0
    g.plot2d(3,3, (7,7), "images/alt_minus.png")

    # plot telephone tag with extra "on"
    g = VonNeumannNetwork(10)
    g.alternate()
    g.A[5,5] = 1
    g.plot1d(3, (7,2), "images/alt_plus.png")

    # plot nobody going with one "on"
    g = VonNeumannNetwork(10)
    g.A[5,5] = 1
    g.plot1d(3, (7,2), "images/zero_plus.png")

    # plot everybody going with one missing
    g = VonNeumannNetwork(10)
    g.ones()
    g.A[4,4] = 0
    g.plot2d(3,3, (7,7), "images/one_minus.png")



            