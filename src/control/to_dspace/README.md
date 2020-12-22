to_dspace package
(1) to_dspace
(2) bus_stop_info





to_dspace
### Input
n.subscribe("/ADV_op/req_run_stop", 1, chatterCallback_02);
n.subscribe("/ADV_op/sys_ready", 1, chatterCallback_03);
n.subscribe("LightResultOutput", 1, chatterCallback_04);
n.subscribe("/traffic", 1, chatterCallback_05);

### Output
CAN

### Description
接收rostopic並轉成CAN丟到dspace




bus_stop_info
### Input
n.subscribe("/reserve/request", 1, chatterCallback_01);
n.subscribe("/NextStop/Info", 1, chatterCallback_02);
n.subscribe("/reserve/route", 1, chatterCallback_03);

### Output
n.advertise<msgs::Flag_Info>("/BusStop/Info", 1);
n.advertise<std_msgs::Int32>("/BusStop/Round", 1);
CAN

### Description
(1) 將後台預約資訊透過CAN傳送到dsapce
(2) 計算趟次並傳給後台
(3) 紀錄每個站點是否停靠並傳給dspace和後台


