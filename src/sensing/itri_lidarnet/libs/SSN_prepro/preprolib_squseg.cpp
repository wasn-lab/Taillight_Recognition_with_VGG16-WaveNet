#include "preprolib_squseg.h"

int getdir(string dir, vector<string> &filenames)
{
	DIR *dp;
	struct dirent *dirp;
	if ((dp = opendir(dir.c_str())) == NULL)
	{
		cout << "Error(" << errno << ") opening " << dir << endl;
		return errno;
	}

	while ((dirp = readdir(dp)) != NULL)
	{
		if (string(dirp->d_name).compare(".") != 0 && string(dirp->d_name).compare("..") != 0)
			filenames.push_back(string(dirp->d_name));
	}
	closedir(dp);
	return 0;
}

int ILcomb(string inputname_i, string inputname_l, VPointCloudXYZIL::Ptr cloud_il)
{

	// ======== read original pointcloud with XYZ and I ========
	VPointCloud::Ptr cloud_i(new VPointCloud);
	if (pcl::io::loadPCDFile<VPoint>(inputname_i, *cloud_i) == -1) //* load the file
	{
		std::cout << "Couldn't read file" << inputname_i << std::endl;
		return (-1);
	}

	// ======== read labelled pointcloud with XYZ and LO ========
	int header_offset = 10;

	ifstream fin(inputname_l.c_str());
	vector<vector<float>> cloud_l; //2 維陣列

	float tmp;			 // temp
	istringstream iline; // input stream class to operate on strings
	string line;		 
	while (getline(fin, line))
	{

		cloud_l.push_back(vector<float>());
		iline.str(line);

		while (iline >> tmp)
		{
			cloud_l.rbegin()->push_back(tmp);
		}

		iline.clear();
	}
	fin.close();

	if (cloud_i->points.size() == cloud_l.size() - header_offset)
	{
		// Fill in the cloud data
		cloud_il->width = cloud_i->width;
		cloud_il->height = cloud_i->height;
		cloud_il->points.reserve(cloud_il->width * cloud_il->height);

		for (size_t i = 0; i < cloud_i->points.size(); ++i)
		{
			VPointXYZIL pointinfo;
			pointinfo.x = cloud_i->points.at(i).x;
			pointinfo.y = cloud_i->points.at(i).y;
			pointinfo.z = cloud_i->points.at(i).z;
			pointinfo.intensity = cloud_i->points.at(i).intensity;

			switch (int(cloud_l[header_offset + i][3]))
			{
			case 5: // class of Person in Hitachi sse
				pointinfo.label = 2;
				break;
			case 6: // class of Rider in Hitachi sse
				pointinfo.label = 3;
				break;
			case 7: // class of Car in Hitachi sse
				pointinfo.label = 1;
				break;
			default:
				pointinfo.label = 0; // for unknown
			}
			cloud_il->points.push_back(pointinfo);
		}

		return (0);
	}
	else
	{
		return (-2);
	}
	
}

VPointCloudXYZIDL sph_proj(VPointCloudXYZIL::Ptr cloud_il, const float phi_center, const float phi_range, const float imageWidth)
{
	// ======== select 90 deg. of front view ========
	// float phi_range = 90.0;
	float phi_Nbound = phi_center - phi_range/2;
	float dphi = phi_range / (imageWidth - 1);
	float phi_Pbound = phi_Nbound + dphi * (imageWidth - 1);

	// ======== selection of VOV ========
	// float theta_range = 2.4-(-24.8);      // for KITTI dataset
	float theta_range = 41.0;         // self-defined
	float theta_UPbound = 18;
	float dtheta = theta_range / (imageHeight - 1);
	float theta_LOWbound = theta_UPbound - dtheta * (imageHeight - 1);

	// ======== Remapping ========
	VPointCloudXYZIDL filtered_cloud;

	filtered_cloud.width = imageWidth;
	filtered_cloud.height = imageHeight;
	filtered_cloud.points.resize(imageHeight * imageWidth); // 64 * 512

	for (size_t i = 0; i < cloud_il->points.size(); i++)
	{
		float x = cloud_il->points[i].x;
		float y = cloud_il->points[i].y;
		float z = cloud_il->points[i].z;
		float d = sqrt(x * x + y * y + z * z);
		//float r = sqrt(x * x + y * y);
		float phi = atan2(y, x) * R2D; // compute arc tangent for 4 quadrant

		if (phi_Nbound <= -180)
		{
			phi = atan2(-y, -x) * R2D - 180; // compute arc tangent for 4 quadrant
		}
		if (phi_Pbound >= 180)
		{
			phi = atan2(-y, -x) * R2D + 180; // compute arc tangent for 4 quadrant
		}
		
		float theta = asin(z / d) * R2D;

		if (phi >= phi_Nbound && phi <= phi_Pbound && theta <= theta_UPbound && theta >= theta_LOWbound)
		{

			int phi_d = (int)((phi - phi_Nbound) / dphi);
			int theta_d = (int)((theta_UPbound - theta) / dtheta);

			//	cout<< theta_UPstart <<endl;
			//	cout<< theta_d <<endl;

			if (phi_d >= 0 && phi_d < imageWidth && theta_d >= 0 && theta_d < imageHeight)
			{
				if ((filtered_cloud.points.at(theta_d * imageWidth + phi_d).d) == 0 || d < (filtered_cloud.points.at(theta_d * imageWidth + phi_d).d))
				{
					filtered_cloud.points.at(theta_d * imageWidth + phi_d).x = cloud_il->points.at(i).x;
					filtered_cloud.points.at(theta_d * imageWidth + phi_d).y = cloud_il->points.at(i).y;
					filtered_cloud.points.at(theta_d * imageWidth + phi_d).z = cloud_il->points.at(i).z;
					filtered_cloud.points.at(theta_d * imageWidth + phi_d).intensity = cloud_il->points.at(i).intensity;
					filtered_cloud.points.at(theta_d * imageWidth + phi_d).d = d;
					filtered_cloud.points.at(theta_d * imageWidth + phi_d).label = cloud_il->points.at(i).label;
				}
			}
		}

	}

	return filtered_cloud;
}

VPointCloudXYZID sph_proj(VPointCloud::Ptr cloud_i, const float phi_center, const float phi_range, const float imageWidth)
{

	// ======== select 90 deg. of front view ========
	// float phi_range = 90.0;
	float phi_Nbound = phi_center - phi_range/2;
	float dphi = phi_range / (imageWidth - 1);
	float phi_Pbound = phi_Nbound + dphi * (imageWidth - 1);

	// ======== selection of VOV ========
	// float theta_range = 2.4-(-24.8);      // for KITTI dataset
	float theta_range = 41.0;         // self-defined
	float theta_UPbound = 18;
	float dtheta = theta_range / (imageHeight - 1);
	float theta_LOWbound = theta_UPbound - dtheta * (imageHeight - 1);

	// ======== Remapping ========
	VPointCloudXYZID filtered_cloud;

	filtered_cloud.width = imageWidth;
	filtered_cloud.height = imageHeight;
	filtered_cloud.points.resize(imageHeight * imageWidth); // 64 * 512

	int fill_count = 0;
	for (size_t i = 0; i < cloud_i->points.size(); i++)
	{
		float x = cloud_i->points[i].x;
		float y = cloud_i->points[i].y;
		float z = cloud_i->points[i].z;
		float d = sqrt(x * x + y * y + z * z);
		//float r = sqrt(x * x + y * y);
		float phi = atan2(y, x) * R2D; // compute arc tangent for 4 quadrant

		if (phi_Nbound <= -180)
		{
			phi = atan2(-y, -x) * R2D - 180; // compute arc tangent for 4 quadrant
		}
		if (phi_Pbound >= 180)
		{
			phi = atan2(-y, -x) * R2D + 180; // compute arc tangent for 4 quadrant
		}

		float theta = asin(z / d) * R2D;

		if (phi >= phi_Nbound && phi <= phi_Pbound && theta <= theta_UPbound && theta >= theta_LOWbound)
		{

			int phi_d = (int)((phi - phi_Nbound) / dphi);
			int theta_d = (int)((theta_UPbound - theta) / dtheta);

			//	cout<< theta_UPstart <<endl;
			//	cout<< theta_d <<endl;

			if (phi_d >= 0 && phi_d < imageWidth && theta_d >= 0 && theta_d < imageHeight)
			{
				if ((filtered_cloud.points.at(theta_d * imageWidth + phi_d).d) == 0 || d < (filtered_cloud.points.at(theta_d * imageWidth + phi_d).d))
				{
					filtered_cloud.points.at(theta_d * imageWidth + phi_d).x = cloud_i->points.at(i).x;
					filtered_cloud.points.at(theta_d * imageWidth + phi_d).y = cloud_i->points.at(i).y;
					filtered_cloud.points.at(theta_d * imageWidth + phi_d).z = cloud_i->points.at(i).z;
					filtered_cloud.points.at(theta_d * imageWidth + phi_d).intensity = cloud_i->points.at(i).intensity;
					filtered_cloud.points.at(theta_d * imageWidth + phi_d).d = d;

					fill_count++;
				}
			}
		}

	}

	// cout << "filled_percentage (val over all) = " <<  (float) fill_count/ (float) cloud_i->points.size() * 100.0 << "%" << endl;
	// cout << "filled_percentage (val over filtered_cloud) = " <<  (float) fill_count/ (float) (imageHeight * imageWidth) * 100.0 << "%" << endl;

	return filtered_cloud;
}

void SSNspan_config(float *OUT_ptr, const char ViewType, const float phi_center)
{
	
	switch (ViewType)
	{
	case 'X':
	{
		const float SPAN_PARA[2] = {90, 512};            // {span, imagewidth}
		for(size_t i = 0; i < int(sizeof(SPAN_PARA)/sizeof(float)); i++)
			OUT_ptr[i] = SPAN_PARA[i];

		break;
	}
	case 'T':
	{
		const float SPAN_PARA[3][2] = {{90, 512},
		                         {180, 1024},
								 {90, 512}};      // {span, imagewidth}
		int phi_center_ind;

		switch (int(phi_center))
		{
		case -135:
		{
			phi_center_ind = 0;
			break;
		}
		case 0:
		{
			phi_center_ind = 1;
			break;
		}
		case 135:
		{
			phi_center_ind = 2;
			break;
		}
		default:
			cout << "No matched phi_center found !!!!!!!!!!" << endl;
		}

		for (size_t i = 0; i < int(sizeof(SPAN_PARA[0])/sizeof(float)); i++)
			OUT_ptr[i] = SPAN_PARA[phi_center_ind][i];

		break;
	}	
	default:
		cout << "No matched ViewType found !!!!!!!!!!" << endl;
	}
}