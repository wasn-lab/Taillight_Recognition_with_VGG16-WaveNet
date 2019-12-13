how to use:

vector<PointCloud<PointXYZ>> roi_cloud;
roi_cloud = separate_cloud (ptr_cur_cloud, -20, 70, -10, 10, -5, -2);
PointCloud<PointXYZ>::Ptr cloud_roi_inside (new PointCloud<PointXYZ>);
PointCloud<PointXYZ>::Ptr cloud_roi_outside (new PointCloud<PointXYZ>);
*cloud_roi_inside = roi_cloud.at (0);
*cloud_roi_outside = roi_cloud.at (1);
cout << "[get ground ROI    ]:" << timer_algorithm_running.getTimeSeconds () << "s" << endl;


