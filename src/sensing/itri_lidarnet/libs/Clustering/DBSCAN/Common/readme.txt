how to use:

DBSCAN dbscan;
dbscan.setEpsilon (0.65 + GlobalVariable::UI_DBSCAN_EPS);
dbscan.setMinpts (3 + GlobalVariable::UI_DBSCAN_MINPT);
dbscan.setInputCloud (input);
dbscan.segment (raw_cluster);