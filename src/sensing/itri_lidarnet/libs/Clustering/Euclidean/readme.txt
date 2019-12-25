how to use:

pcl::EuclideanClusterExtraction<PointXYZ> ec;
ec.setInputCloud (input);
ec.setClusterTolerance (1);  // the distance to scan for cluster candidates unit:m      traffic flow : 1
ec.setMinClusterSize (1);    // unit:points     traffic flow : 15
ec.setMaxClusterSize (10000);  // unit:points
ec.extract (raw_cluster);


