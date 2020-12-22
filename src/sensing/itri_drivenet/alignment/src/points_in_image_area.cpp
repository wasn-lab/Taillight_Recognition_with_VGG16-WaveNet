#include "points_in_image_area.h"

using namespace DriveNet;

void getPointCloudInAllImageRectCoverage(const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidarall_ptr,
                                         std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& cams_points_ptr,
                                         std::vector<Alignment>& alignment)
{
  // std::cout << "===== getPointCloudInAllImageRectCoverage... =====" << std::endl;
  /// create variable
  std::vector<pcl::PointCloud<pcl::PointXYZI>> cam_points(cams_points_ptr.size());

  /// find 3d points in image coverage
  for (size_t i = 0; i < lidarall_ptr->size(); i++)
  {
    for (size_t cam_order = 0; cam_order < cams_points_ptr.size(); cam_order++)
    {
      if (alignment[cam_order].checkPointInCoverage(lidarall_ptr->points[i]))
      {
        cam_points[cam_order].push_back(lidarall_ptr->points[i]);
      }
    }
  }
  /// copy to destination
  for (size_t cam_order = 0; cam_order < cams_points_ptr.size(); cam_order++)
  {
    *cams_points_ptr[cam_order] = cam_points[cam_order];
  }
}

void getPointCloudInImageRectCoverage(const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidarall_ptr,
                                      pcl::PointCloud<pcl::PointXYZI>::Ptr& cams_points_ptr, Alignment& alignment)
{
  // std::cout << "===== getPointCloudInImageRectCoverage... =====" << std::endl;
  /// create variable
  pcl::PointCloud<pcl::PointXYZI> cam_points;

  /// find 3d points in image coverage
  for (size_t i = 0; i < lidarall_ptr->size(); i++)
  {
    if (alignment.checkPointInCoverage(lidarall_ptr->points[i]))
    {
      cam_points.push_back(lidarall_ptr->points[i]);
    }
  }

  /// copy to destination
  *cams_points_ptr = cam_points;
}
void getPointCloudInImageFOV(const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidarall_ptr,
                             pcl::PointCloud<pcl::PointXYZI>::Ptr& cams_points_ptr,
                             std::vector<PixelPosition>& cam_pixels, int image_w, int image_h, Alignment& alignment)
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
#pragma omp parallel for
  for (size_t i = 0; i < lidarall_ptr->size(); i++)
  {
    if (alignment.checkPointInCoverage(lidarall_ptr->points[i]))
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

  /// record the 2d points(cam_pixels) and 3d points(cam_points)
  for (int u = 0; u < image_w; u++)
  {
    for (int v = 0; v < image_h; v++)
    {
      PixelPosition pixel_position{ -1, -1 };
      pixel_position.u = u;
      pixel_position.v = v;
      if (point_cloud[u][v].x != 0 && point_cloud[u][v].y != 0 && point_cloud[u][v].z != 0)
      {
        cam_points.points[cloud_sizes] = point_cloud[u][v];
        cam_pixels.push_back(pixel_position);
        cloud_sizes++;
      }
    }
  }
  /// copy to destination
  cam_points.resize(cloud_sizes);
  *cams_points_ptr = cam_points;
}

void getPointCloudInAllImageFOV(const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidarall_ptr,
                                std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& cams_points_ptr,
                                std::vector<std::vector<PixelPosition>>& cam_pixels, int image_w, int image_h,
                                std::vector<Alignment>& alignment)
{
  // std::cout << "===== getPointCloudInImageFOV... =====" << std::endl;
  /// create variable
  std::vector<pcl::PointCloud<pcl::PointXYZI>> cam_points(cams_points_ptr.size());
  std::vector<std::vector<std::vector<pcl::PointXYZI>>> point_cloud(
      cams_points_ptr.size(), std::vector<std::vector<pcl::PointXYZI>>(image_w, std::vector<pcl::PointXYZI>(image_h)));

/// find 3d points in image coverage
#pragma omp parallel for collapse(2)
  for (size_t i = 0; i < lidarall_ptr->size(); i++)
  {
    for (size_t cam_order = 0; cam_order < cams_points_ptr.size(); cam_order++)
    {
      if (alignment[cam_order].checkPointInCoverage(lidarall_ptr->points[i]))
      {
        PixelPosition pixel_position{ -1, -1 };
        pixel_position = alignment[cam_order].projectPointToPixel(lidarall_ptr->points[i]);
        if (pixel_position.u >= 0 && pixel_position.v >= 0)
        {
          if (point_cloud[cam_order][pixel_position.u][pixel_position.v].x < lidarall_ptr->points[i].x ||
              point_cloud[cam_order][pixel_position.u][pixel_position.v].x == 0)
          {
            point_cloud[cam_order][pixel_position.u][pixel_position.v] = lidarall_ptr->points[i];
          }
        }
      }
    }
  }
  PixelPosition pixel_position_tmp{ -1, -1 };

  /// record the 3d points)
  for (int u = 0; u < image_w; u++)
  {
    for (int v = 0; v < image_h; v++)
    {
      for (size_t cam_order = 0; cam_order < cams_points_ptr.size(); cam_order++)
      {
        if (point_cloud[cam_order][u][v].x != 0 && point_cloud[cam_order][u][v].y != 0 &&
            point_cloud[cam_order][u][v].z != 0)
        {
          cam_points[cam_order].push_back(point_cloud[cam_order][u][v]);
          pixel_position_tmp.u = u;
          pixel_position_tmp.v = v;
          cam_pixels[cam_order].push_back(pixel_position_tmp);
        }
      }
    }
  }
  /// copy to destination
  for (size_t cam_order = 0; cam_order < cams_points_ptr.size(); cam_order++)
  {
    *cams_points_ptr[cam_order] = cam_points[cam_order];
  }
}
void getPointCloudInImageFOV(const pcl::PointCloud<pcl::PointXYZI>::Ptr& lidarall_ptr,
                             pcl::PointCloud<pcl::PointXYZI>::Ptr& cams_points_ptr,
                             /*std::vector<PixelPosition>& cam_pixels,*/ int image_w, int image_h, Alignment& alignment)
{
  // std::cout << "===== getPointCloudInImageFOV... =====" << std::endl;
  /// create variable
  pcl::PointCloud<pcl::PointXYZI> cam_points;
  std::vector<std::vector<pcl::PointXYZI>> point_cloud(
      std::vector<std::vector<pcl::PointXYZI>>(image_w, std::vector<pcl::PointXYZI>(image_h)));

/// find 3d points in image coverage
#pragma omp parallel for
  for (size_t i = 0; i < lidarall_ptr->size(); i++)
  {
    PixelPosition pixel_position{ -1, -1 };
    if (alignment.checkPointInCoverage(lidarall_ptr->points[i]))
    {
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
  /// record the 3d points)
  for (int u = 0; u < image_w; u++)
  {
    for (int v = 0; v < image_h; v++)
    {
      if (point_cloud[u][v].x != 0 && point_cloud[u][v].y != 0 && point_cloud[u][v].z != 0)
      {
        cam_points.push_back(point_cloud[u][v]);
      }
    }
  }
  /// copy to destination
  *cams_points_ptr = cam_points;
}

void getPointCloudInBoxFOV(
    const msgs::DetectedObjectArray& objects, const pcl::PointCloud<pcl::PointXYZI>::Ptr& cams_points_ptr,
    pcl::PointCloud<pcl::PointXYZI>::Ptr& cams_bbox_points_ptr, const std::vector<PixelPosition>& cam_pixels_cam,
    std::vector<std::vector<PixelPosition>>& cam_pixels_obj, msgs::DetectedObjectArray& objects_2d_bbox,
    std::vector<pcl::PointCloud<pcl::PointXYZI>>& cam_bboxs_points, Alignment& alignment, CloudCluster& cloud_cluster,
    bool is_enable_default_3d_bbox, bool do_clustering, bool do_display)
{
  // std::cout << "===== getPointCloudInBoxFOV... =====" << std::endl;
  /// create variable
  pcl::PointCloud<pcl::PointXYZI> cam_points;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI> point_cloud_object;
  std::vector<pcl::PointXYZI> point_vector_object;
  std::vector<pcl::PointXYZI> point_vector_objects;
  std::vector<PixelPosition> cam_pixels_object;

  // std::cout << "objects.objects size: " << objects.objects.size() << std::endl;
  /// main
  for (auto& obj : objects.objects)
  {
    msgs::DetectedObject obj_tmp = obj;
    obj_tmp.header = objects.header;
    int pixel_index = 0;

    for(size_t i = 0; i < obj_tmp.camInfo.size(); i++)
    {
      // get the 2d box
      std::vector<PixelPosition> bbox_positions(2);
      bbox_positions[0].u = obj_tmp.camInfo[i].u;
      bbox_positions[0].v = obj_tmp.camInfo[i].v;
      bbox_positions[1].u = obj_tmp.camInfo[i].u + obj_tmp.camInfo[i].width;
      bbox_positions[1].v = obj_tmp.camInfo[i].v + obj_tmp.camInfo[i].height;
      transferPixelScaling(bbox_positions);

      for (const auto& pixel_position : cam_pixels_cam)
      {
        // get points in the 2d box
        if (pixel_position.u >= bbox_positions[0].u && pixel_position.v >= bbox_positions[0].v &&
            pixel_position.u <= bbox_positions[1].u && pixel_position.v <= bbox_positions[1].v)
        {
          if (do_display)
          {
            cam_pixels_object.push_back(pixel_position);
          }
          point_cloud_object.push_back(cams_points_ptr->points[pixel_index]);
        }
        pixel_index++;
      }
    }
    // std::cout << "point_vector_object size: " << point_vector_object.size() << std::endl;

    if (!point_cloud_object.empty())
    {
      // get points in the 3d box
      if (is_enable_default_3d_bbox)
      {
        std::vector<int> inliers_remove;
        getPointCloudIn3DBox(point_cloud_object, obj.classId, cloud_filtered_ptr, inliers_remove);

        if (do_display)
        {
          if (!cloud_filtered_ptr->points.empty())
          {
            for (size_t index = 0; index < inliers_remove.size(); index++)
            {
              cam_pixels_object.erase(cam_pixels_object.begin() + inliers_remove[index] - static_cast<int>(index));
            }
          }
        }
      }
      else
      {
        cloud_filtered_ptr = point_cloud_object.makeShared();
      }
      point_cloud_object.clear();

      // std::cout << "cloud_filtered_ptr->points size: " << cloud_filtered_ptr->points.size() << std::endl;

      if (!cloud_filtered_ptr->points.empty())
      {
        if (do_clustering)
        {
          bool do_downsampling = true;  // default is ture
          std::vector<pcl::PointCloud<pcl::PointXYZI>> cluster_points;
          cluster_points = cloud_cluster.getClusters(cloud_filtered_ptr, do_downsampling);
          for (const auto& points : cluster_points)
          {
            cam_bboxs_points.push_back(points);
          }
        }
        else
        {
          cam_bboxs_points.push_back(*cloud_filtered_ptr);
        }
        objects_2d_bbox.objects.push_back(obj_tmp);

        if (do_display)
        {
          cam_pixels_obj.push_back(cam_pixels_object);
          cam_pixels_object.clear();
          // point cloud to vector
          for (const auto& point : cloud_filtered_ptr->points)
          {
            point_vector_object.push_back(point);
          }
          // concatenate the points of objects
          point_vector_objects.insert(point_vector_objects.begin(), point_vector_object.begin(),
                                      point_vector_object.end());
          point_vector_object.clear();
        }
      }
    }
  }

  if (do_display)
  {
    removeDuplePoints(point_vector_objects);
    for (size_t i = 0; i < point_vector_objects.size(); i++)
    {
      cam_points.push_back(point_vector_objects[i]);
    }
    /// copy to destination
    *cams_bbox_points_ptr = cam_points;
  }
}

void getPointCloudInBoxFOV(const msgs::DetectedObjectArray& objects, msgs::DetectedObjectArray& remaining_objects,
                           const pcl::PointCloud<pcl::PointXYZI>::Ptr& cams_points_ptr,
                           pcl::PointCloud<pcl::PointXYZI>::Ptr& cams_bbox_points_ptr,
                           const std::vector<PixelPosition>& cam_pixels_cam,
                           std::vector<std::vector<PixelPosition>>& cam_pixels_obj,
                           msgs::DetectedObjectArray& objects_2d_bbox,
                           std::vector<pcl::PointCloud<pcl::PointXYZI>>& cam_bboxs_points, Alignment& alignment,
                           CloudCluster& cloud_cluster, bool is_enable_default_3d_bbox, bool do_clustering,
                           bool do_display)
{
  // std::cout << "===== getPointCloudInBoxFOV... =====" << std::endl;
  /// create variable
  pcl::PointCloud<pcl::PointXYZI> cam_points;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI> point_cloud_object;
  std::vector<pcl::PointXYZI> point_vector_object;
  std::vector<pcl::PointXYZI> point_vector_objects;
  std::vector<PixelPosition> cam_pixels_object;

  /// main
  if (!remaining_objects.objects.empty())
  {
    remaining_objects.objects.clear();
  }
  if (!objects_2d_bbox.objects.empty())
  {
    objects_2d_bbox.objects.clear();
  }

  for (const auto& obj : objects.objects)
  {
    msgs::DetectedObject obj_tmp = obj;
    obj_tmp.header = objects.header;
    int pixel_index = 0;

    for(size_t i = 0; i < obj_tmp.camInfo.size(); i++)
    {
      // get the 2d box
      std::vector<PixelPosition> bbox_positions(2);
      bbox_positions[0].u = obj_tmp.camInfo[i].u;
      bbox_positions[0].v = obj_tmp.camInfo[i].v;
      bbox_positions[1].u = obj_tmp.camInfo[i].u + obj_tmp.camInfo[i].width;
      bbox_positions[1].v = obj_tmp.camInfo[i].v + obj_tmp.camInfo[i].height;
      transferPixelScaling(bbox_positions);

      for (const auto& pixel_position : cam_pixels_cam)
      {
        // get points in the 2d box
        if (pixel_position.u >= bbox_positions[0].u && pixel_position.v >= bbox_positions[0].v &&
            pixel_position.u <= bbox_positions[1].u && pixel_position.v <= bbox_positions[1].v)
        {
          if (do_display)
          {
            cam_pixels_object.push_back(pixel_position);
          }
          point_cloud_object.push_back(cams_points_ptr->points[pixel_index]);
        }
        pixel_index++;
      }
    }
    // std::cout << "point_cloud_object size: " << point_cloud_object.size() << std::endl;

    if (!point_cloud_object.empty())
    {
      std::vector<int> inliers_remove;

      // get points in the 3d box
      if (is_enable_default_3d_bbox)
      {
        getPointCloudIn3DBox(point_cloud_object, obj.classId, cloud_filtered_ptr, inliers_remove);
        if (do_display)
        {
          if (!cloud_filtered_ptr->points.empty())
          {
            for (size_t index = 0; index < inliers_remove.size(); index++)
            {
              cam_pixels_object.erase(cam_pixels_object.begin() + inliers_remove[index] - static_cast<int>(index));
            }
          }
        }
      }
      else
      {
        cloud_filtered_ptr = point_cloud_object.makeShared();
      }
      point_cloud_object.clear();
      // std::cout << "cloud_filtered_ptr->points size: " << cloud_filtered_ptr->points.size() << std::endl;

      if (!cloud_filtered_ptr->points.empty())
      {
        if (do_clustering)
        {
          bool do_downsampling = true;  // default is ture
          std::vector<pcl::PointCloud<pcl::PointXYZI>> cluster_points;
          cluster_points = cloud_cluster.getClusters(cloud_filtered_ptr, do_downsampling);
          for (const auto& points : cluster_points)
          {
            cam_bboxs_points.push_back(points);
          }
        }
        else
        {
          cam_bboxs_points.push_back(*cloud_filtered_ptr);
        }
        objects_2d_bbox.objects.push_back(obj_tmp);

        if (do_display)
        {
          cam_pixels_obj.push_back(cam_pixels_object);
          cam_pixels_object.clear();
          // point cloud to vector
          for (const auto& point : cloud_filtered_ptr->points)
          {
            point_vector_object.push_back(point);
          }
          // concatenate the points of objects
          point_vector_objects.insert(point_vector_objects.begin(), point_vector_object.begin(),
                                      point_vector_object.end());
          point_vector_object.clear();
        }
      }
      else
      {
        remaining_objects.objects.push_back(obj);
      }
    }
    else
    {
      remaining_objects.objects.push_back(obj);
    }
  }
  // std::cout << "remaining_objects.objects.size(): " << remaining_objects.objects.size() << std::endl;
  if (do_display)
  {
    removeDuplePoints(point_vector_objects);
    for (const auto& point : point_vector_objects)
    {
      cam_points.push_back(point);
    }
    /// copy to destination
    *cams_bbox_points_ptr = cam_points;
  }
}

void getPointCloudIn3DBox(const pcl::PointCloud<pcl::PointXYZI>& cloud_src, int object_class_id,
                          pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_filtered_ptr, std::vector<int>& inliers_remove)
{
  // std::cout << "===== getPointCloudIn3DBox... =====" << std::endl;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointXYZI min_pt, max_pt;

  /// get the box length of object
  pcl::getMinMax3D(cloud_src, min_pt, max_pt);
  object_box bbox{};
  bbox = getDefaultObjectBox(object_class_id);

  /// build the condition
  pcl::ConditionAnd<pcl::PointXYZI>::Ptr range_cond(new pcl::ConditionAnd<pcl::PointXYZI>());
  range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZI>::ConstPtr(
      new pcl::FieldComparison<pcl::PointXYZI>("x", pcl::ComparisonOps::GT, min_pt.x - 0.0001)));
  range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZI>::ConstPtr(
      new pcl::FieldComparison<pcl::PointXYZI>("x", pcl::ComparisonOps::LT, min_pt.x + bbox.length)));

  /// build the filter
  pcl::ConditionalRemoval<pcl::PointXYZI> condrem(true);
  condrem.setCondition(range_cond);
  cloud_ptr = cloud_src.makeShared();
  condrem.setInputCloud(cloud_ptr);
  condrem.setKeepOrganized(false);

  /// set input index
  pcl::PointIndicesPtr inliers_input(new pcl::PointIndices);
  if (!cloud_src.empty())
  {
    for (size_t i = 0; i < cloud_src.size(); i++)
    {
      inliers_input->indices.push_back(static_cast<int>(i));
    }
    condrem.setIndices(inliers_input);
  }
  /// apply filter
  condrem.filter(*cloud_filtered_ptr);

  /// get remove index
  if (!inliers_input->indices.empty())
  {
    pcl::IndicesConstPtr indices_remove;
    indices_remove = condrem.getRemovedIndices();
    for (size_t i = 0; i < indices_remove->size(); i++)
    {
      // std::cout<<(*indices_remove)[i]<<": ";
      inliers_remove.push_back((*indices_remove)[i]);
    }
  }
}