import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

def plot_trajectories():
    slsqp = [[6.06, 9.8, 14.3145, 19.4221],
               [11.6, 18.4, 27, 36.6],
               [22, 35.4, 52, 70.7],]
    SNR = [5,10,15,20]
    for j, (N,K) in enumerate([(4,4),(8,8),(16,16)]):
        path = './resultsN'+str(N)+'K'+str(K)+'.npy'
        res = np.load(path)[1:]
        m = np.mean(res, axis=1)
        
        colors = ['r','g','b','k','orange']
        res_sorted = np.sort(res,axis=1)
        tmax = res.shape[-1]
        fig=plt.figure()
        legend = []
        legend += [Line2D([0], [0], color='k', linestyle='--', label='SLSQP')]  
        legend += [Line2D([0], [0], color='k', linestyle='-', label='LSTM')]  
        for i in range(4):
            plt.plot(m[i], color=colors[i])
            plt.plot(range(tmax),[slsqp[j][i]]*tmax, color=colors[i], linestyle='--')
            plt.fill_between(range(tmax), res_sorted[i,15], res_sorted[i,85], alpha=0.2, color=colors[i], label='_nolegend_')
            legend+=[Line2D([0], [0], color=colors[i], label='SNR = '+str(SNR[i])+' dB', linestyle='none', marker='o')]

        plt.grid(linestyle='--')
        plt.title('N = '+str(N)+', K = '+str(K))
        plt.legend(handles=legend)
        plt.xlabel('Step')
        plt.ylabel('Sum Rate')
        fig.set_size_inches(8,7)
        
        plt.savefig('./trajectoriesN'+str(N)+'K'+str(K)+'.pdf', format='pdf')

    plt.show()

if __name__ == '__main__':
    plot_trajectories()