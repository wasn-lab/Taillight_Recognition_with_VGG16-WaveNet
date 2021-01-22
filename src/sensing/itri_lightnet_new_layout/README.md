### itri_lightnet class_type

0: Unknown, 
1: Red, 
2: Yellow, 
3: Green, 
4: Straight Green,
5: Straight and Right Green
6: SLR Green,
7: Red and Right,
8: Red and Left,
9: Flashing Red,
10: Flashing Yellow

###=============boyu add color_light, direction, distance=============

color_light
0=unknown, 
1=Red, 
2=Yellow, 
3=Green

direction = (MSB)000(LSB)

MSB bit = Turn Right
Middle bit = Go Ahead
LSB bit = Turn Left

## 啟動方式
``` sh
./devel/lib/itri_lightnet_new_layout/itri_lightnet_new_layout_node
```

## dependencies
OpenCV V4.2
TensorRT 5.0.2.6/4.0.1.6