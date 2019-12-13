
#ifndef S3CLASSIFY_H_
#define S3CLASSIFY_H_

#include "../all_header.h"
#include "../GlobalVariable.h"

#include "SvmWrapper.h"

class S3Classify
{
  public:
    S3Classify ();
    S3Classify (boost::shared_ptr<pcl::visualization::PCLVisualizer> input_viewer,
                int *input_viewID);
    virtual
    ~S3Classify ();

    void
    update (bool is_debug,
            CLUSTER_INFO* cluster_info,
            int cluster_size);

  private:
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
    int *viewID;

    SvmWrapper svm_pedestrian;
    SvmWrapper svm_motorcycle;
    SvmWrapper svm_car;
    SvmWrapper svm_bus;
};

#endif /* S3CLASSIFY_H_ */
