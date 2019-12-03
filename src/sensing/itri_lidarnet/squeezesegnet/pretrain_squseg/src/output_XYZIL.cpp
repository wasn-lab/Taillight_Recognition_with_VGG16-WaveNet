#include "preprolib_squseg.h"
#include <pcl/console/parse.h>
#include "PlaneGroundFilter.h"
#include "extract_Indices.h"
#include<boost/filesystem.hpp> 

 namespace BFS = boost::filesystem;


int main(int argc, char **argv)
{
	std::string source_dir;
	
	source_dir = argv[1];

	// ======== list filenames in directory ========
	// string dir_i = source_dir + "/ori/";
	// string dir_l = source_dir + "/OK/";
	// string dir_il = source_dir + "/txtFile/";

	BFS::path dir_i(source_dir + "/ori");
	BFS::path dir_l(source_dir + "/OK");
	BFS::path dir_il(source_dir + "/XYZIL");

	std::cout << "Files in " << dir_i << std::endl;

	int pcd_count = 0;
	int miss_label_count = 0;
	int error_count = 0;
	int OK_count = 0;
	int exist_count = 0;
	int rename_count = 0;

	vector<int> existFile_cnt;
	vector<string> existFile_name;


	for (BFS::directory_iterator it = BFS::directory_iterator(dir_i); it != BFS::directory_iterator(); ++it)
	{
		BFS::path dir_i_sub = it->path();

		if( BFS::is_directory(dir_i_sub) && BFS::exists(dir_l/dir_i_sub.filename()) )
		{

			for (BFS::directory_iterator it2 = BFS::directory_iterator(dir_i_sub); it2 != BFS::directory_iterator(); ++it2)
			{
				BFS::path dir_i_sub_file = it2->path();

				if (dir_i_sub_file.extension().string().compare(".pcd") == 0)
				{
					pcd_count++;
					string file_name = dir_i_sub_file.stem().string();

					// ======== read original pointcloud of XYZI and XYZLO and combine to XYZIL ========
					std::string inputname_i;
					inputname_i = (dir_i / dir_i_sub.filename()).string() + "/" + file_name + ".pcd";
					cout << inputname_i << endl;

					std::string inputname_l;
					inputname_l = (dir_l / dir_i_sub.filename()).string() + "/" + file_name + ".txt";

					if (!(BFS::exists(inputname_l)))
					{
						miss_label_count++;
						cout << "pcd_all: " << pcd_count << " miss_label: " << miss_label_count << " error: " << error_count << " OK: " << OK_count << " exist:" << exist_count << " rename: " << rename_count << endl;	
						continue;
					}

					VPointCloudXYZIL::Ptr cloud_il(new VPointCloudXYZIL);

					int ferror = ILcomb(inputname_i, inputname_l, cloud_il);

					if (ferror != 0)
					{
						error_count++;
						cout << "Error code:" << ferror << " (-1: Couldn't read pcd file, -2: point number mismatch)" << endl;
						cout << "pcd_all: " << pcd_count << " miss_label: " << miss_label_count << " error: " << error_count << " OK: " << OK_count << " exist:" << exist_count << " rename: " << rename_count << endl;	
						continue;
					}
					else
					{
						OK_count++;
						cout << "OK!!!" << endl;
					}				

					// // ======== position of projection center ======
					// float x_projCenter = -2;
					// float z_projCenter = -1.4;

					// for (size_t i = 0; i < cloud_il->points.size(); i++)
					// {
					// 	cloud_il->points[i].x = cloud_il->points[i].x - x_projCenter;
					// 	// float release_Cloud->points[i].y = release_Cloud->points[i].y;
					// 	cloud_il->points[i].z = cloud_il->points[i].z - z_projCenter;
					// }

					// // ======== remove ground ========
					// VPointCloud::Ptr cloud_i(new VPointCloud);
					// if (pcl::io::loadPCDFile<VPoint>(inputname_i, *cloud_i) == -1) //* load the file
					// {
					// 	std::cout << "Couldn't read file" << inputname_i << std::endl;
					// 	return (-1);
					// }

					// pcl::PointIndicesPtr indices_ground(new pcl::PointIndices);
					// *indices_ground = PlaneGroundFilter().runMorphological<PointXYZI>(cloud_i, 0.3, 2, 1, 0.9, 0.4, 0.5);
					// //*indices_ground = RayGroundFilter (3.6,5.8,9.0,0.01,0.01,0.15,0.3,0.8,0.175).compute (ptr_cur_cloud);
					// //*indices_ground = remove_ground_sac_segmentation (ptr_cur_cloud);
					// //*indices_ground = remove_ground_sample_consensus_model (ptr_cur_cloud);
					// PointCloud<PointXYZIL>::Ptr cloud_non_ground(new PointCloud<PointXYZIL>);
					// PointCloud<PointXYZIL>::Ptr cloud_ground(new PointCloud<PointXYZIL>);

					// // cloud_non_ground->width = indices_ground->indices.size();
					// // cloud_non_ground->height = cloud_il->height;
					// // cout<< cloud_il->width << endl;
					// // cout << indices_ground->indices.size() << endl;
					// // cloud_non_ground->points.resize(indices_ground->indices.size() * cloud_il->height);

					// extract_Indices<PointXYZIL>(cloud_il, indices_ground, *cloud_ground, *cloud_non_ground);
					// // for (size_t i = 0; i < indices_ground->indices.size(); ++i)
					// // {
					// // 	cout << indices_ground->indices[i] << endl;
					// // 	cloud_non_ground->points.at(i) = cloud_il->points.at(indices_ground->indices[i]);
					// // }

					string output_pcdname_test = dir_il.string() + "/" + file_name + ".pcd";

					if (BFS::exists(output_pcdname_test))
					{
						exist_count++;
						if (existFile_name.empty())
						{
							existFile_name.push_back(file_name);
							existFile_cnt.push_back(1);
							stringstream ss_cnt;
							ss_cnt << existFile_cnt.back();
							file_name = file_name + ss_cnt.str();
						}
						else
						{
							int search_flag;

							for (size_t i = 0; i < existFile_name.size(); i++)
							{
								search_flag = i;
								// cout << i << "/" << existFile_name.size() << endl;

								if (file_name.compare(existFile_name.at(i))==0)
								{
									existFile_cnt.at(i)++;
									stringstream ss_cnt;
									ss_cnt << existFile_cnt.at(i);
									file_name = file_name + ss_cnt.str();
									break;
								}
							}

							if ((search_flag == (existFile_name.size() - 1)) && (file_name.compare(existFile_name.at(search_flag)) != 0))
							{
								existFile_name.push_back(file_name);
								existFile_cnt.push_back(1);
								stringstream ss_cnt;
								ss_cnt << existFile_cnt.back();
								file_name = file_name + ss_cnt.str();
							}
						}

					}

					// ======== summation of rename count ============
					int rename_count = 0;

					for (vector<int>::iterator it_existFile_cnt = existFile_cnt.begin(); it_existFile_cnt != existFile_cnt.end(); ++it_existFile_cnt)
						rename_count = rename_count + *it_existFile_cnt;

					cout << "pcd_all: " << pcd_count << " miss_label: " << miss_label_count << " error: " << error_count << " OK: " << OK_count << " exist:" << exist_count << " rename: " << rename_count << endl;

					std::string output_pcdname;
					output_pcdname = dir_il.string() + "/" + file_name + ".pcd";

					pcl::io::savePCDFileASCII(output_pcdname, *cloud_il);
				}

			}

		}

	}

	std::fstream fs_exist;
	fs_exist.open( (source_dir + "/ExistFiles_xyzil.txt").c_str(), std::fstream::out);

	for (size_t i = 0; i < existFile_name.size(); i++)
		fs_exist << existFile_name.at(i) << ": " << existFile_cnt.at(i) << "\n";

	fs_exist.close();

	return (0);
}
