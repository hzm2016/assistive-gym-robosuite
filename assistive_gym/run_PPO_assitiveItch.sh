#!/bin/zsh
export PYTHONPATH=$PYTHONPATH:
python run_robosuite_main.py --env_name ScratchItchJaco-v1 --alg PPO --seed 1
python run_robosuite_main.py --env_name ScratchItchJaco-v1 --alg PPO --seed 2
python run_robosuite_main.py --env_name ScratchItchJaco-v1 --alg PPO --seed 3
python run_robosuite_main.py --env_name ScratchItchJaco-v1 --alg PPO --seed 4
python run_robosuite_main.py --env_name ScratchItchJaco-v1 --alg PPO --seed 5
