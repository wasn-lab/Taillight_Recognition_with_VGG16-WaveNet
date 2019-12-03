how to use:

*ptr_cur_cloud = VoxelFilter_CUDA().compute(cloud_non_ground,1,1);
*ptr_cur_cloud = radius_outlier_removal (cloud_non_ground, 1+ GlobalVariable::UI_PARA[0], 3+ GlobalVariable::UI_PARA[1]);

cout << "[remove noise]:" << ptr_cur_cloud->size () << "," << timer_algorithm_running.getTimeSeconds () << "s" << endl;
