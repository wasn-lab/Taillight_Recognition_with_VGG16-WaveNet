how to use:

*ptr_cur_cloud = remove_ground (ptr_cur_cloud, -0.000490628, -0.0249136, 0.999689, 0.723542);

pcl::PointIndicesPtr indices_ground (new pcl::PointIndices);
*indices_ground = remove_ground_sac_segmentation (ptr_cur_cloud);
*indices_ground = remove_ground_sample_consensus_model (ptr_cur_cloud);

cout << "[remove_plane]:" << ptr_cur_cloud->size () << "," << timer_algorithm_running.getTimeSeconds () << "s" << endl;
      
      