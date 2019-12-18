#ifndef SVMWRAPPER_H_
#define SVMWRAPPER_H_

#include "../all_header.h"
#include "SvmWrapperModify.h"
#include "../debug_tool.h"

class SvmWrapper
{
  public:
    SvmWrapper ();
    virtual
    ~SvmWrapper ();

    void
    initialize (string input_name);
    bool
    calculate (CLUSTER_INFO *single_cluster_info);

    static void
    labelingTool (CLUSTER_INFO *single_cluster_info,
                  boost::shared_ptr<pcl::visualization::PCLVisualizer> input_viewer,
                  int *input_viewID);

  private:
    pcl::SVMClassify SVMclassify;
    bool is_loaded = false;
    string type_name;

    static void
    keyboard_event_occurred (const pcl::visualization::KeyboardEvent &event,
                             void* viewer_void);
    static PointCloud<PointXYZ>::Ptr cloud_training;
    static string single_label;
};

#endif /* SVMWRAPPER_H_ */
