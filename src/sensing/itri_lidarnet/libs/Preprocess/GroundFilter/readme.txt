how to use:

*ptr_cur_cloud = remove_ground (ptr_cur_cloud, -0.000490628, -0.0249136, 0.999689, 0.723542);

pcl::PointIndicesPtr indices_ground (new pcl::PointIndices);
*indices_ground = remove_ground_sac_segmentation (ptr_cur_cloud);
*indices_ground = remove_ground_sample_consensus_model (ptr_cur_cloud);

cout << "[remove_plane]:" << ptr_cur_cloud->size () << "," << timer_algorithm_running.getTimeSeconds () << "s" << endl;
      
// filter
if (use_filter)
{
    *input_cloud_tmp = CuboidFilter().hollow_removal<PointXYZI>(input_cloud_tmp, -6.0, 0.5, -1.2, 1.2, -3.0, 0.1);
    *input_cloud_tmp = NoiseFilter().runRadiusOutlierRemoval<PointXYZI>(input_cloud_tmp, 0.22, 1);
    cout << "Top Filter-- "
    << "Points: " << input_cloud_tmp->size() << "; Time Took: " << stopWatch_T.getTimeSeconds() << 's' << endl;
}