### decision_maker

```使用車輛：B and C```

主要決定要將繞越輸出給哪個模組,以及決定繞越結束點,詳細看decision_maker/README.md

### path_transfer

```使用車輛：B and C```

轉發planning module最終出來的path topic給其他模組,並且計算path的各種屬性,如轉彎或是爬坡等,詳細看path_transfer/README.md

### scene_register_checker

```使用車輛：B and C```

根據behavior planning出來的register轉發給flag_managememt, 包含traffic及busstop

### target_planner

```使用車輛：B and C```

計算車輛側向控制的target point

### veh_predictway_point

```使用車輛：B and C```

根據車速及yawrate計算出車輛可能行走軌跡
