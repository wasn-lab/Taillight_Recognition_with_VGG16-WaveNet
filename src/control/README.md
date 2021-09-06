### control checker

```使用車輛：B and C```

檢查dspace與xavier溝通之topic是否斷訊,若此斷訊在fail safe中會顯示can通訊異常

註：此主要是檢查B車,C車理論上此checker會常駐true

### dspace_code

```使用車輛：C```

此部份主要為從B車dspace的simulink code轉譯成C++,故此模組僅在C車上執行

1. dspace_tx : 將dspace_code的topic合成B車dspace傳到xavier上之code,可使control checker順利執行
2. flag_management : 產生各類型flag,如light_flag,bus_stop_flag,idle_flag等
3. lateral_control : lateral controller,主要調整側向控制器的參數,所使用的方法為target and control
4. long_control : longitude controller,主要調整縱向控制器參數,包含throttle和brake,其中switching_module為計算切換throttle/brake之時機點
5. speed_profile : 計算最終車速cmd,其中包含acc和acc_pp,並且從flag_management接收flag狀態,如light_flag,bus_stop_flag,idle_flag,static_flag等,去調整車速cmd

### from_dspace

```使用車輛：B```

此為接收dspace can訊號並且轉發成topic供ROS module使用

### geofence

```使用車輛：B and C```

產生自駕車前方路徑與最近障礙物之交點位置柵欄,更詳細內容可看geofence裡面的README.md

### geofence_map_pp

```使用車輛：B and C```

產生自駕車前方路徑與障礙物map_pp之最近交點位置柵欄

### geofence_map_pp_filter

```使用車輛：B and C```

濾除機率低的map pp,交由豪里補充

### geofence_pp

```已無使用```

### gnss_utility

```使用車輛：B and C```

各種常用的gnss座標系,包含TWD97,UTM等
目前使用UTM,原因為autoware lanelet2使用此格式

### lidar_location_send

```使用車輛：B```

將localization_to_veh,imu_data,front_vehicle_target_point,rear_vehicle_target_point的topic轉成can id傳給dspace

註：ukf_mm_topic已無使用

### lidarxyz2lla

```使用車輛：B and C```

將current_pose的座標轉成wgs84(即為經緯度座標)

### mm_tp_code

```已無使用```

### occ_grid_lane

```使用車輛：B and C```

根據自車車道繪製出drivalbe area,並且將左邊變寬,此grid map使用在obstacle avoidance需要車道變寬時

* 重要參數
 1. right_waylength:從車道中心線向右延伸距離
 2. left_waylength:從車道中心線向左延伸距離

### planning_initial

```使用車輛：B and C```

幾乎所有需要接進planning module的topic都由此node轉換
詳細到planning_initial的README.md


### plc_fatek

```使用車輛：B and C```

與PLC溝通之程式

### rad_grab

```使用車輛：B and C```

rad grabber,將radar 資料拋出,須注意是否有PathPredictionOutput/radar這個topic

### smash

```已無使用```

### to_dspace

```使用車輛：B```

1. to_dspace:將topic轉成can傳給dspace

```使用車輛：B and C```
2. bus_stop_info:判斷預約站號及傳CAN到dspace

備註：詳細內容可至to_dspace裡的README.md

### trimble_grabber

```使用車輛：B and C```

gnss grabber for trimble bx992,若換gnss則須重寫grabber,其中此grabber包含座標轉換,目前主要以UTM格式為主

### vehinfo_pub

```使用車輛：B```

從dspace讀取can訊號並且轉發成veh_info,目前此topic僅有ego_speed有在使用
