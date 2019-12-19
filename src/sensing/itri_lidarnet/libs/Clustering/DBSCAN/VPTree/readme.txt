how to use:

DBSCAN_VPTree dbscan;
dbscan.setEpsilon (0.70);
dbscan.setMinpts (1);
dbscan.setInputCloud (input);
dbscan.segment (raw_cluster);