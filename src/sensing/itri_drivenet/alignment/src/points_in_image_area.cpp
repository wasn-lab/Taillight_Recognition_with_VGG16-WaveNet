#include "points_in_image_area.h"

using namespace DriveNet;

void getPointCloudInImageFOV(pcl::PointCloud<pcl::PointXYZI>::Ptr lidarall_ptr,
                             pcl::PointCloud<pcl::PointXYZI>::Ptr cams_points_ptr,
                             std::vector<PixelPosition>& cam_pixels, int image_w, int image_h, Alignment alignment)
{
  // std::cout << "===== getPointCloudInImageFOV... =====" << std::endl;
  /// create variable
  pcl::PointCloud<pcl::PointXYZI> cam_points;
  int cloud_sizes = 0;
  std::vector<std::vector<pcl::PointXYZI>> point_cloud(
      std::vector<std::vector<pcl::PointXYZI>>(image_w, std::vector<pcl::PointXYZI>(image_h)));

  /// copy from source
  pcl::copyPointCloud(*lidarall_ptr, *cams_points_ptr);
  cam_points = *cams_points_ptr;

  /// find 3d points in image coverage
  for (size_t i = 0; i < lidarall_ptr->size(); i++)
  {
    if (lidarall_ptr->points[i].x > 0)
    {
      PixelPosition pixel_position{ -1, -1 };
      pixel_position = alignment.projectPointToPixel(lidarall_ptr->points[i]);
      if (pixel_position.u >= 0 && pixel_position.v >= 0)
      {
        if (point_cloud[pixel_position.u][pixel_position.v].x > lidarall_ptr->points[i].x ||
            point_cloud[pixel_position.u][pixel_position.v].x == 0)
        {
          point_cloud[pixel_position.u][pixel_position.v] = lidarall_ptr->points[i];
        }
      }
    }
  }
  /// record the 2d points and 3d points.
  for (int u = 0; u < image_w; u++)
  {
    for (int v = 0; v < image_h; v++)
    {
      PixelPosition pixel_position{ -1, -1 };
      pixel_position.u = u;
      pixel_position.v = v;
      if (point_cloud[u][v].x != 0 && point_cloud[u][v].y != 0 && point_cloud[u][v].z != 0)
      {
        // cam_pixels.push_back(pixel_position);
        cam_points.points[cloud_sizes] = point_cloud[u][v];
        cloud_sizes++;
      }
    }
  }
  /// copy to destination
  cam_points.resize(cloud_sizes);
  *cams_points_ptr = cam_points;
}

void getPointCloudInBoxFOV(msgs::DetectedObjectArray& objects, pcl::PointCloud<pcl::PointXYZI>::Ptr cams_points_ptr,
                           pcl::PointCloud<pcl::PointXYZI>::Ptr cams_bbox_points_ptr,
                           std::vector<PixelPosition>& cam_pixels, std::vector<pcl_cube>& cams_bboxs_cube_min_max,
                           Alignment alignment, bool is_enable_default_3d_bbox)
{
  // std::cout << "===== getPointCloudInBoxFOV... =====" << std::endl;
  /// create variable
  pcl::PointCloud<pcl::PointXYZI> cam_points;
  int cloud_sizes = 0;
  pcl::PointCloud<pcl::PointXYZI> point_cloud_src;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  std::vector<pcl::PointXYZI> point_vector_object;
  std::vector<pcl::PointXYZI> point_vector_objects;
  std::vector<pcl::PointCloud<pcl::PointXYZI>> cams_bboxs_points;

  /// copy from source
  pcl::copyPointCloud(*cams_points_ptr, *cams_bbox_points_ptr);
  cam_points = *cams_bbox_points_ptr;

  /// main
  std::vector<pcl_cube> bboxs_cube_min_max;
  for (const auto& obj : objects.objects)
  {
    pcl::PointCloud<pcl::PointXYZI> bbox_points;
    pcl_cube cube_min_max;  // object min and max point
    for (size_t i = 0; i < cam_points.points.size(); i++)
    {
      // get the 2d box
      std::vector<PixelPosition> bbox_positions(2);
      bbox_positions[0].u = obj.camInfo.u;
      bbox_positions[0].v = obj.camInfo.v;
      bbox_positions[1].u = obj.camInfo.u + obj.camInfo.width;
      bbox_positions[1].v = obj.camInfo.v + obj.camInfo.height;
      transferPixelScaling(bbox_positions);

      // get points in the 2d box
      PixelPosition pixel_position{ -1, -1 };
      pixel_position = alignment.projectPointToPixel(cam_points.points[i]);
      if (pixel_position.u >= bbox_positions[0].u && pixel_position.v >= bbox_positions[0].v &&
          pixel_position.u <= bbox_positions[1].u && pixel_position.v <= bbox_positions[1].v)
      {
        cam_pixels.push_back(pixel_position);
        point_vector_object.push_back(cam_points.points[i]);
        bbox_points.push_back(cam_points.points[i]);
      }
    }
    // vector to point cloud
    pcl::PointCloud<pcl::PointXYZI> point_cloud_object;
    point_cloud_object.points.resize(point_vector_object.size());
    for (size_t i = 0; i < point_vector_object.size(); i++)
    {
      point_cloud_object.points[i] = point_vector_object[i];
    }
    point_vector_object.clear();

    // get points in the 3d box
    if (is_enable_default_3d_bbox)
    {
      getPointCloudIn3DBox(point_cloud_object, obj.classId, cloud_filtered_ptr);
    }
    else
    {
      cloud_filtered_ptr = point_cloud_object.makeShared();
    }

    // point cloud to vector
    for (const auto& point : cloud_filtered_ptr->points)
    {
      point_vector_object.push_back(point);
    }

    // concatenate the points of objects
    point_vector_objects.insert(point_vector_objects.begin(), point_vector_object.begin(), point_vector_object.end());
    point_vector_object.clear();
    pcl::getMinMax3D(bbox_points, cube_min_max.p_min, cube_min_max.p_max);
    object_box bbox;
    bbox = getDefaultObjectBox(obj.classId);
    cube_min_max.p_max.x = cube_min_max.p_min.x + bbox.length;
    cube_min_max.p_max.y = cube_min_max.p_min.y + bbox.width;
    cube_min_max.p_max.z = cube_min_max.p_min.z + bbox.height;
    bboxs_cube_min_max.push_back(cube_min_max);
  }
  removeDuplePoints(point_vector_objects);
  for (size_t i = 0; i < point_vector_objects.size(); i++)
  {
    cam_points.points[i] = point_vector_objects[i];
    cloud_sizes++;
  }
  point_vector_objects.clear();
  cams_bboxs_cube_min_max = bboxs_cube_min_max;

  /// copy to destination
  cam_points.resize(cloud_sizes);
  *cams_bbox_points_ptr = cam_points;
}

void getPointCloudIn3DBox(const pcl::PointCloud<pcl::PointXYZI> cloud_src, int object_class_id,
                          pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered_ptr)
{
  // std::cout << "===== getPointCloudIn3DBox... =====" << std::endl;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointXYZI minPt, maxPt;

  /// get the box length of object
  pcl::getMinMax3D(cloud_src, minPt, maxPt);
  object_box bbox;
  bbox = getDefaultObjectBox(object_class_id);

  /// build the condition
  pcl::ConditionAnd<pcl::PointXYZI>::Ptr range_cond(new pcl::ConditionAnd<pcl::PointXYZI>());
  range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZI>::ConstPtr(
      new pcl::FieldComparison<pcl::PointXYZI>("x", pcl::ComparisonOps::GT, minPt.x)));
  range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZI>::ConstPtr(
      new pcl::FieldComparison<pcl::PointXYZI>("x", pcl::ComparisonOps::LT, minPt.x + bbox.length)));

  /// build the filter
  pcl::ConditionalRemoval<pcl::PointXYZI> condrem;
  condrem.setCondition(range_cond);
  cloud_ptr = cloud_src.makeShared();
  condrem.setInputCloud(cloud_ptr);
  condrem.setKeepOrganized(false);

  /// apply filter
  condrem.filter(*cloud_filtered_ptr);
}