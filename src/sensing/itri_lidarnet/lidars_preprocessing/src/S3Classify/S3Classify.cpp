#include "S3Classify.h"

S3Classify::S3Classify () :
    viewer (NULL),
    viewID (NULL)
{
}

S3Classify::S3Classify (boost::shared_ptr<pcl::visualization::PCLVisualizer> input_viewer,
                        int *input_viewID) :
    viewer (input_viewer),
    viewID (input_viewID)
{
  svm_pedestrian.initialize ("pedestrian");
  svm_motorcycle.initialize ("motorcycle");
  svm_car.initialize ("car");
  svm_bus.initialize ("bus");
}

S3Classify::~S3Classify ()
{
}

void
S3Classify::update (bool is_debug,
                    CLUSTER_INFO* cluster_info,
                    int cluster_size)
{
//#pragma omp parallel for
  for (int i = 0; i < cluster_size; ++i)
  {

/*    if (svm_pedestrian.calculate (&cluster_info[i]))
    {
      cluster_info[i].cluster_tag = 2;
    }
    else if (svm_motorcycle.calculate (&cluster_info[i]))
    {
      cluster_info[i].cluster_tag = 3;
    }
    else if (svm_car.calculate (&cluster_info[i]))
    {
      cluster_info[i].cluster_tag = 4;
    }
    else if (svm_bus.calculate (&cluster_info[i]))
    {
      cluster_info[i].cluster_tag = 5;
    }
    else
    {
      cluster_info[i].cluster_tag = 1;
    }*/

    if (GlobalVariable::ENABLE_LABEL_TOOL)
    {
      SvmWrapper::labelingTool (&cluster_info[i], viewer, viewID);
    }

  }

  if (is_debug && !GlobalVariable::ENABLE_LABEL_TOOL)
  {
    for (int i = 0; i < cluster_size; i++)
    {
      if (cluster_info[i].cluster_tag > 0)
      {
        /*
        int rgb[3];
        switch (cluster_info[i].cluster_tag)
        {
          case 1:  // white for unknown object
            rgb[0] = 255;
            rgb[1] = 255;
            rgb[2] = 255;
            break;
          case 2:  // aqua for pedestrian
            rgb[0] = 0;
            rgb[1] = 255;
            rgb[2] = 234;
            break;
          case 3:  // yellow for motorcycle
            rgb[0] = 255;
            rgb[1] = 255;
            rgb[2] = 0;
            break;
          case 4:  // green for car
            rgb[0] = 0;
            rgb[1] = 255;
            rgb[2] = 0;
            break;
          case 5:  // blue for bus
            rgb[0] = 0;
            rgb[1] = 0;
            rgb[2] = 255;
            break;
        }

                viewer->addLine (PointXYZ(0.6,-27,0), PointXYZ(0.6,27,0), 0, 255, 0, to_string (*viewID));
         ++*viewID;
         viewer->addLine (PointXYZ(6,-27,0), PointXYZ(6,27,0), 0, 255, 0, to_string (*viewID));
         ++*viewID;
         viewer->addLine (PointXYZ(6,-27,0), PointXYZ(0.6,-27,0), 0, 255, 0, to_string (*viewID));
         ++*viewID;
         viewer->addLine (PointXYZ(6, 27,0), PointXYZ(0.6,27,0), 0, 255, 0, to_string (*viewID));
         ++*viewID;*/

      }
    }
  }
}

