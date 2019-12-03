#include "preprolib_squseg.h"
#include <pcl/console/parse.h>
#include "PlaneGroundFilter.h"
#include "extract_Indices.h"

int main(int argc, char **argv)
{
	std::string dataset;
	char ViewType = 'X';
	
	dataset = argv[1];
	if (argc >= 3)
	{
	ViewType = *argv[2];
	}

	// ======== PROJECTION PARAMETERS (should be changed accordding to car-type) =========
	float x_projCenter = -2.0;
	float z_projCenter = -1.4;
	float theta_UPbound = 18.0;
	float theta_range = 41.0;


	// ======== list filenames in directory ========
	string dir_i = "../../../src/squeezesegnet/lidar_squseg_detect/data/" + dataset + "/oriFile/";
	string dir_l = "../../../src/squeezesegnet/lidar_squseg_detect/data/" + dataset + "/labelFile/";
	string dir_il = "../../../src/squeezesegnet/lidar_squseg_detect/data/" + dataset + "/txtFile/";

	vector<string> filenames = vector<string>();

	getdir(dir_i, filenames);

	for (unsigned int i_file = 0; i_file < filenames.size(); i_file++)
	{
		int filename_length = filenames[i_file].length();
		string file_name;
		file_name = file_name.assign(filenames[i_file], 0, filename_length - 4);

		// cout << filenames[i_file] << endl;
		// cout << file_name << endl;

		// ======== read original pointcloud of XYZI and XYZLO and combine to XYZIL ========
		std::string inputname_i;
		inputname_i = dir_i + file_name + ".pcd";

		std::string inputname_l;
		inputname_l = dir_l + file_name + ".txt";

		cout << inputname_i << endl;

		VPointCloudXYZIL::Ptr cloud_il(new VPointCloudXYZIL);

		int ferror = ILcomb(inputname_i, inputname_l, cloud_il);

		if (ferror != 0)
		{
			cout << "Error code:" << ferror << " (-1: Couldn't read pcd file, -2: point number mismatch)" << endl;
			continue;
		}
		else
		{
			cout << "OK!!!" << endl;
		}

		// ======== coordinate transform to projection center for spherical projection ======
		for (size_t i = 0; i < cloud_il->points.size(); i++)
		{
			cloud_il->points[i].x = cloud_il->points[i].x - x_projCenter;
			// float release_Cloud->points[i].y = release_Cloud->points[i].y;
			cloud_il->points[i].z = cloud_il->points[i].z - z_projCenter;
		}

		// ======== remove ground ========
		VPointCloud::Ptr cloud_i(new VPointCloud);
		if (pcl::io::loadPCDFile<VPoint>(inputname_i, *cloud_i) == -1) //* load the file
		{
			std::cout << "Couldn't read file" << inputname_i << std::endl;
			return (-1);
		}

		pcl::PointIndicesPtr indices_ground(new pcl::PointIndices);
		*indices_ground = PlaneGroundFilter().runMorphological<PointXYZI>(cloud_i, 0.3, 2, 1, 0.9, 0.4, 0.5);
		//*indices_ground = RayGroundFilter (3.6,5.8,9.0,0.01,0.01,0.15,0.3,0.8,0.175).compute (ptr_cur_cloud);
		//*indices_ground = remove_ground_sac_segmentation (ptr_cur_cloud);
		//*indices_ground = remove_ground_sample_consensus_model (ptr_cur_cloud);
		PointCloud<PointXYZIL>::Ptr cloud_non_ground(new PointCloud<PointXYZIL>);
		PointCloud<PointXYZIL>::Ptr cloud_ground(new PointCloud<PointXYZIL>);

		// cloud_non_ground->width = indices_ground->indices.size();
		// cloud_non_ground->height = cloud_il->height;
		// cout<< cloud_il->width << endl;
		// cout << indices_ground->indices.size() << endl;
		// cloud_non_ground->points.resize(indices_ground->indices.size() * cloud_il->height);

		extract_Indices<PointXYZIL>(cloud_il, indices_ground, *cloud_ground, *cloud_non_ground);
		// for (size_t i = 0; i < indices_ground->indices.size(); ++i)
		// {
		// 	cout << indices_ground->indices[i] << endl;
		// 	cloud_non_ground->points.at(i) = cloud_il->points.at(indices_ground->indices[i]);
		// }

		// ======== projection to filtered cloud =========
		vector<float> phi_center_all;

		switch (ViewType)
		{
		case 'X':
		{
			phi_center_all = {-90, 0, 90, 180};
			break;
		}
		case 'T':
		{
			phi_center_all = {-135, 0, 135};
			break;
		}
		default:
			cout << "No matched ViewType found !!!!!!!!!!" << endl;
		}

		for (size_t i_HFOVnumber = 0; i_HFOVnumber < phi_center_all.size(); i_HFOVnumber++)
		{
			float phi_center = phi_center_all.at(i_HFOVnumber);
			float SPAN_PARA[2];         // {span, imagewidth}
			SSNspan_config(SPAN_PARA, ViewType, phi_center);

			VPointCloudXYZIDL filtered_cloud = sph_proj(cloud_non_ground, phi_center, SPAN_PARA[0], SPAN_PARA[1], theta_UPbound, theta_range); 

			std::string output_pcdname;
			std::string output_txtname;

			if (phi_center >= 0)
			{
				stringstream ss;
				ss << phi_center;
				output_txtname = dir_il + file_name + "_P" + ss.str() + ".txt";
				output_pcdname = dir_il + file_name + "_P" + ss.str() + ".pcd";
			}
			else
			{
				stringstream ss;
				ss << phi_center * -1;
				output_txtname = dir_il + file_name + "_N" + ss.str() + ".txt";
				output_pcdname = dir_il + file_name + "_N" + ss.str() + ".pcd";
			}

			// pcl::io::savePCDFileASCII(output_pcdname, filtered_cloud);

			std::fstream fs;
			fs.open(output_txtname.c_str(), std::fstream::out);
			for (size_t i = 0; i < filtered_cloud.points.size(); i++)
			{
				fs << setprecision(8) << filtered_cloud.points[i].x << "\t"
				   << setprecision(8) << filtered_cloud.points[i].y << "\t"
				   << setprecision(8) << filtered_cloud.points[i].z << "\t"
				   << setprecision(8) << filtered_cloud.points[i].intensity << "\t"
				   << setprecision(8) << filtered_cloud.points[i].d << "\t"
				   << setprecision(8) << filtered_cloud.points[i].label << "\n";
			}
			fs.close();
			filtered_cloud.points.clear();
		}
	}

	return (0);
}
