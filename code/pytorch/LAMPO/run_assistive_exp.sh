export PYTHONPATH=$PYTHONPATH:
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/zhimin/.mujoco/mujoco200/bin
python code/pytorch/LAMPO/assitive_experiment.py --param_file IMOGICAssitiveOne.json --alg REPS --not_dr True

