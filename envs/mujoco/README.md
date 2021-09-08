## mujoco platfroms
The assets model are in envs/assets generated from the urdf file. 

1. generate .urdf file from solidworks or by ros package, [instruction](https://www.cnblogs.com/21207-ihome/p/7821269.html) 
2. transfer .urdf file into .xml file 
```
./compile .urdf .xml
```
3. check new model by .xml file
```
./simulate .xml
```

Basic controller of robots are at '.controllers' 
```
cd .controller/test
run vis_impedance_controller.py
```

Path planner based on DMP can be utilzied to generate desired trajectories
