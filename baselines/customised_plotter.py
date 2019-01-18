import matplotlib
matplotlib.use('Agg') # Can change to 'Agg' for non-interactive mode
import matplotlib.pyplot as plt
import argparse
import os
from baselines.bench.monitor import load_results
import numpy as np

def ts2xy(ts):
    x = np.cumsum(ts.l.values)
    y = ts.r.values
    return x, y

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def window_func(x, y, window, func):
    yw_func = []
    for i in range(len(y)):
        yw = rolling_window(y[i], window)
        yw_func.append(func(yw, axis=-1))
    return x[window-1:], yw_func

def plot_single_directory (env_id, directory, method_name, num_folds, EPISODES_WINDOW=100):
   directory_name = directory + '/' + 'logs_' + env_id + '_'
   if not os.path.isdir(directory_name+'0'):
       print ('Warning: directory ' + directory_name + '0' + 'does not exist, skipping...')
       return
   results_x = []
   results_y = []
   results_x_all = []
   for i in range (0, num_folds):
       directory_name_i = directory_name + str(i)
       current_results = load_results(directory_name_i)
       current_results_x, current_results_y = ts2xy (current_results)
       results_x.append (current_results_x)
       results_y.append (current_results_y)
       results_x_all.extend (current_results_x)
       #plt.plot (current_results_x, current_results_y)
   results_x_all = np.sort (results_x_all)
   results_y_all = []
   for i in range (num_folds):
       np.append(results_x[i], results_x_all[-1])
       np.append(results_y[i], results_y[i][-1])
       results_y_all.append (np.interp(results_x_all, results_x[i], results_y[i]))
   results_x_all, results_y_all = window_func(results_x_all, results_y_all, EPISODES_WINDOW, np.mean) 
   plt.plot (results_x_all, np.mean (results_y_all, 0), label=method_name)
   plt.fill_between (results_x_all, np.mean(results_y_all, 0) - np.std (results_y_all, 0), np.mean(results_y_all, 0) + np.std (results_y_all, 0), alpha = 0.3)

def plot_results (env_id, directories, method_names, num_folds, postfix=''):
   plt.clf()
   for i in range(len(directories)):
       directory = directories[i]
       method_name = method_names[i]
       plot_single_directory (env_id, directory, method_name, num_folds)
   plt.gca().set_xlabel('Steps')
   plt.gca().set_ylabel('Rewards')
   plt.legend()
   #plt.show ()
   plt.savefig (env_id + postfix + '.png', bbox_inches='tight', pad_inches=0)

def main():
   #directories = ['experimental_gradient/log_bak_01_05', 'acktr']
   #directories = ['experimental_gradient', 'ddpg/log_10_05', 'acktr/log_results', 'experimental_gradient_09_08_buffer_3', 'trpo_mpi']
   #method_names = ['Proposed method', 'DDPG', 'ACKTR', 'Proposed_3', 'TRPO_MPI']
   #directories = ['experimental_gradient_09_08_buffer_3', 'experimental_gradient_12_08_buffer_5', 'trpo_mpi/log_bak', 'ppo1', 'acktr/log_results']
   #method_names = ['Proposed, Buffer Size=3', 'Proposed, Buffer Size=5', 'TRPO_MPI', 'PPO', 'ACKTR']
   #env_ids = {'HumanoidStandup-v2', 'Striker-v2', 'Thrower-v2', 'Pusher-v2', 'Reacher-v2', 'HalfCheetah-v2', 'Swimmer-v2', 'Ant-v2', 'Humanoid-v2', 'Hopper-v2', 'Walker2d-v2', 'InvertedPendulum-v2', 'InvertedDoublePendulum-v2'}
   directories = ['trpo_replay']
   method_names = ['Proposed']
   env_ids = {'Ant-v2'}
   #env_ids = {'Reacher-v2', 'HalfCheetah-v2', 'Swimmer-v2', 'Ant-v2', 'Humanoid-v2', 'Hopper-v2', 'Walker2d-v2', 'InvertedPendulum-v2', 'InvertedDoublePendulum-v2', 'swimmer_swimmer6', 'fish_swim', 'walker_stand', 'ball_in_cup_catch', 'humanoid_stand', 'fish_upright', 'finger_spin', 'cheetah_run', 'walker_walk', 'walker_run'}
   num_folds = 1
   for env_id in env_ids:
       plot_results (env_id, directories, method_names, num_folds, postfix='_3_minus_cov_acktr')
   
   #directories = ['experimental_gradient'] 
   #method_names = ['Proposed method']
   #env_ids = {'Reacher-v2', 'HalfCheetah-v2', 'Swimmer-v2', 'Ant-v2', 'Humanoid-v2', 'Hopper-v2', 'Walker2d-v2', 'InvertedPendulum-v2', 'InvertedDoublePendulum-v2'} 
   #num_folds = 1
   #postfix = '_tr_perf'
   #for env_id in env_ids:
   #    plot_results (env_id, directories, method_names, num_folds, postfix)
if __name__ == '__main__':
      main()

