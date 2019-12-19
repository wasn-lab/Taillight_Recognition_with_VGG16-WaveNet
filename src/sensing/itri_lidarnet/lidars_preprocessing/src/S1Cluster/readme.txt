how to use:

pcl::ModelCoefficients MC;
MC = get_plane_coefficients_sac (cloud_ground, 3.7);  //baby car 0.65 luxgen MVP7 0.723, ITRI bus A 2.68
float dis = pointToPlaneDistance (PointXYZ (0, 0, 0), MC.values[0], MC.values[1], MC.values[2], MC.values[3]);
cout << "[Sensor to ground  ]:" << timer_algorithm_running.getTimeSeconds () << "s " << dis << "m" << endl;

int cur_cluster_num = 0;

S1Cluster S1cluster;
S1cluster = S1Cluster (input_viewer, input_viewID);

CLUSTER_INFO* cur_cluster;
S1cluster.setPlaneParameter (MC);
cur_cluster = S1cluster.getClusters (GlobalVariable::ENABLE_DEBUG_MODE, ptr_cur_cloud, &cur_cluster_num);
cout << "[S1cluster         ]:" << timer_algorithm_running.getTimeSeconds () << "s" << endl;
delete[] cur_cluster;
