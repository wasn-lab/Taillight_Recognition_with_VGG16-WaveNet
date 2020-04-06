#include "eval_utility.h"

VPointCloud pointcloudIL2I(VPointCloudXYZIL::Ptr cloud_il)
{

	VPointCloud pcdExtract;
	pcdExtract.points.reserve(cloud_il->points.size());

	for (size_t i = 0; i < cloud_il->points.size(); ++i)
	{
		VPoint pointinfo;
		pointinfo.x = cloud_il->points.at(i).x;
		pointinfo.y = cloud_il->points.at(i).y;
		pointinfo.z = cloud_il->points.at(i).z;
		pointinfo.intensity = cloud_il->points.at(i).intensity;

		pcdExtract.points.push_back(pointinfo);
	}

	return pcdExtract;
}

VPointCloudXYZIL pcdExtract_byClass(VPointCloudXYZIL::Ptr cloud_il, int class_index)
{

	VPointCloudXYZIL pcdExtract;
	pcdExtract.points.reserve(cloud_il->points.size());

	for (size_t i = 0; i < cloud_il->points.size(); ++i)
	{
    if (cloud_il->points.at(i).label == class_index)
    {
      pcdExtract.points.push_back(cloud_il->points.at(i));
    }
  }

	return pcdExtract;
}

VPointCloudXYZIL pcdExtract_allLabelObj(VPointCloudXYZIL::ConstPtr cloud_il)
{

	VPointCloudXYZIL pcdExtract;
	pcdExtract.points.reserve(cloud_il->points.size());

	for (size_t i = 0; i < cloud_il->points.size(); ++i)
	{
    if (cloud_il->points.at(i).label > 0)
    {
      pcdExtract.points.push_back(cloud_il->points.at(i));
    }
  }

	return pcdExtract;
}

pointcloudEval::pointcloudEval() {}
pointcloudEval::pointcloudEval (VPointCloudXYZIL::Ptr input_cloud_GT, VPointCloudXYZIL::Ptr input_cloud_PD)
{
	cloud_GT = input_cloud_GT;
	cloud_PD = input_cloud_PD;
}

pointcloudEval::~pointcloudEval() {}

void pointcloudEval::IOUcal_pointwise(int class_index)
{

	VPointCloudXYZIL GT_extract = pcdExtract_byClass(cloud_GT,class_index);
	VPointCloudXYZIL PD_extract = pcdExtract_byClass(cloud_PD,class_index);

	int intersect_cnt = 0;
	int GTnonInter_cnt = 0;
	int PDnonInter_cnt = 0;

	for(size_t i=0; i<GT_extract.points.size(); i++)
	{
		for(size_t j=0; i<PD_extract.points.size(); j++)
		{
			if( (GT_extract.points.at(i).x == PD_extract.points.at(j).x) && (GT_extract.points.at(i).y == PD_extract.points.at(j).y) && (GT_extract.points.at(i).z == PD_extract.points.at(j).z) )
			{
				intersect_cnt++;
				break;
			}
		}
	}

	GTnonInter_cnt = GT_extract.points.size() - intersect_cnt;
	PDnonInter_cnt = PD_extract.points.size() - intersect_cnt;

  if (GT_extract.points.size() == 0)
  {
    GT_cloudExist = false;
  }

  if (PD_extract.points.size() == 0)
  {
    PD_cloudExist = false;
  }

  if (GT_extract.points.size() == 0 && PD_extract.points.size() == 0)
  {
    iou = 0;
  }
  else
  {
    iou = intersect_cnt/(intersect_cnt+GTnonInter_cnt+PDnonInter_cnt);
  }
}


