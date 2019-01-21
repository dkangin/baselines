# TRPO-REPLAY
The implementation is based on the OpenAI baselines library and ACKTR implementation in particular. 
Original paper: [https://arxiv.org/abs/1901.06212]

Dmitry Kangin, Nicolas Pugeault (2018) On-Policy Trust Region Policy Optimisation with Replay Buffers, arXiv:1901.06212
 
## Requirements
 OpenAI baselines library (see README.md in the root directory of this repository for details)
## Usage
To launch the experiments, use the following command: ```LD_LIBRARY_PATH=$HOME/.mujoco/mjpro150/bin DISABLE_MUJOCO_RENDERING=True python3 launch_test.py```

The test environment ids must be listed in launch_test.py as follows: ```env_ids = {'Ant-v2'}```
```NUM_FOLDS``` specifies the number of repetitive experiments for the same environment
The experiment results are logged in the subdirectory named according to the following pattern:  ```logs_{ENVIRONMENT_ID}_{FOLD_INDEX}```, so that ```logs_Ant-v2_0``` means that this is the folder with logs for the environment Ant-v2, fold index is 0. 

## Results visualisation
The original visualiser, used for the paper, is located in ../customised_plotter.py . It outputs the file for every task in the listed directory.  
Parameters of visualiser, including the listed directories and the names of the algorithms, are set in the main function of the file. The file is used as : python3 customised_plotter.py

