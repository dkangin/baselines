#!/usr/bin/env python3

import tensorflow as tf
from baselines import logger
from baselines.common.cmd_util import mujoco_arg_parser
from baselines.experimental_gradient.acktr_cont import train
from algorithm_parameters import algorithm_parameters

def main():
    args = mujoco_arg_parser().parse_args()
    logger.configure()
    parameters = algorithm_parameters()
    train(args.env, parameters=parameters, seed=args.seed)

if __name__ == "__main__":
    main()
