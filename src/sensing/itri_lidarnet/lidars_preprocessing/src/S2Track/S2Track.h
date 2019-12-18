
#ifndef S2TRACK_H_
#define S2TRACK_H_

#include "../all_header.h"

class S2Track
{
  public:
    S2Track ();
    S2Track (boost::shared_ptr<pcl::visualization::PCLVisualizer> input_viewer,
             int *input_viewID);
    virtual
    ~S2Track ();

    void
    update (bool is_debug,
            CLUSTER_INFO* cluster_info,
            int cluster_size);

  private:
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
    int *viewID;

    pcl::StopWatch frame_time;
    vector<CLUSTER_INFO> pre_vehicle_table;
    int counter_direction;
    int tracking_id_count = 0;
};

#endif /* S2TRACK_H_ */
