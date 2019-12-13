
#ifndef S4FLOWDETECT_H_
#define S4FLOWDETECT_H_

#include <string>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/geometry.h>

#include "../UserDefine.h"

class S4FlowDetect
{
  public:
    S4FlowDetect ();
    S4FlowDetect (boost::shared_ptr<pcl::visualization::PCLVisualizer> input_viewer,
                  int *input_viewID);
    virtual
    ~S4FlowDetect ();

    void
    add_gateway_rectangle (pcl::PointXYZ rectangle_min,
                           pcl::PointXYZ rectangle_max);

    void
    update (bool is_debug,
            CLUSTER_INFO* cluster_info,
            int cluster_number,
            int *counter);

    void
    show_info (int counter[],
               double system_running_time_second,
               double frames_running_time_second);

  private:
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
    int *viewID;

    pcl::PointXYZ pt_min;
    pcl::PointXYZ pt_max;
};

#endif /* S4FLOWDETECT_H_ */
