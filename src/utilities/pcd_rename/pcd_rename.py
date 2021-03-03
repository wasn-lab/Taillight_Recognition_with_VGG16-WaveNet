import sys
import glob
import time
import string
import rospy
from pathlib import Path
import shutil
import fire
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import PointCloud2
from os.path import getsize

def rename(imageset_file_path, pcd_dir, start_idx, test_num):
    path = Path(imageset_file_path)
    assert path.exists() == True, "Imageset file path not correct"
    output_dir = Path ("output").resolve()
    if not output_dir.is_dir():
        output_dir.mkdir()
    with open(path) as f:
        lines = f.readlines()
        assert len(lines) == start_idx + test_num,  \
            " Number of lines does not meet the input parameter, \
                the former is %d while the latter is %d" % (len(lines), start_idx + test_num)
        for i in range(start_idx, start_idx + test_num):
            line_lst = lines[i].split(' ')
            bin_path = line_lst[1]
            filename = bin_path.split('/')[-1]
            file_str = filename.split('.')[0]
            pcd_path = Path(pcd_dir) / (file_str + ".pcd")
            pcd_path = pcd_path.resolve()
            output_path = output_dir / (file_str + ".pcd")

            shutil.copy(pcd_path, output_dir)
            assert output_path.is_file(), \
                "Such file %s does not exist." % (output_path)
            output_path = output_path.replace(output_dir / ('%06d.pcd' % (i)))
        
        return ("Rename work done. Please check if the file name is correct.")

if __name__ == '__main__':
    fire.Fire()
            
            
            
