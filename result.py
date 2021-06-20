import os, sys
sys.path.append(os.curdir)
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

'''
from torch.utils.tensorboard import SummaryWriter
import datetime

class commonLogger:

   def __init__(self,prefix):
       self.writer = SummaryWriter('runs/{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

   def log_avg_reward(self,avg_reward,i_episode):
       self.writer.add_scalar('avg_reward/test', avg_reward, i_episode)
'''

if __name__ == "__main__":
    result = np.load(f"results/TD3_Walker2DBulletEnv-v0_lr_45e-5.npy")
    setp_size = 5e3
    steps = np.arange(0, len(result),1) * setp_size

    if not os.path.exists("./figures"):
        os.makedirs("./figures")    
    plt.plot(steps,result)
    plt.ylabel('Average Score', fontsize=12)
    plt.xlabel('Time steps', fontsize=12)
    plt.title('TD3(lr=0.00045)', fontsize=12)
    plt.axis([0, 1e7, 0, 3000])
    plt.savefig('figures/TD3_lr_45e-5.png')
    plt.close()