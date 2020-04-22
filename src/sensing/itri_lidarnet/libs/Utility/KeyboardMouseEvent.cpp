#include "KeyboardMouseEvent.h"

KeyboardMouseEvent::KeyboardMouseEvent()
{
}

KeyboardMouseEvent::~KeyboardMouseEvent()
{
}

void KeyboardMouseEvent::setCloudToPCD(PointCloud<PointXYZ> input)
{
  cloud_save_to_pcd = input;
}

bool KeyboardMouseEvent::getPauseState()
{
  return pause_state;
}

bool KeyboardMouseEvent::getResultState()
{
  return result_mode_state;
}

bool KeyboardMouseEvent::getCHState()
{
  return ch_mode_state;
}

bool KeyboardMouseEvent::getBBoxState()
{
  return bbox_mode_state;
}

bool KeyboardMouseEvent::getLidarAllState()
{
  return lidar_all_state;
}

void KeyboardMouseEvent::mouseCallback(const pcl::visualization::MouseEvent& event, void* viewer_void)
{
  // pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *> (viewer_void);
  if (event.getButton() == pcl::visualization::MouseEvent::LeftButton &&
      event.getType() == pcl::visualization::MouseEvent::MouseButtonRelease)
  {
    // char str[512];
    // viewer->addText ("clicked here", event.getX (), event.getY (), str);
  }
}

void KeyboardMouseEvent::keyboardCallback(const pcl::visualization::KeyboardEvent& event, void* viewer_void)
{
  pcl::visualization::PCLVisualizer* viewer = static_cast<pcl::visualization::PCLVisualizer*>(viewer_void);

  if (event.getKeySym() == "a" && event.keyDown())  // switch mode to showing result point cloud only
  {
    if (result_mode_state)
    {
      result_mode_state = false;
      cout << "[INFO]: change display mode to general point clouds" << endl;
    }
    else
    {
      result_mode_state = true;
      cout << "[INFO]: change display mode to output results only" << endl;
    }
  }
  if (event.getKeySym() == "t" && event.keyDown())  // switch mode to showing bounding box or convex hull
  {
    if (ch_mode_state)
    {
      ch_mode_state = false;
      cout << "[INFO]: shut down the convexhull showing" << endl;
    }
    else
    {
      ch_mode_state = true;
      cout << "[INFO]: showing convex hull" << endl;
    }
  }
  if (event.getKeySym() == "y" && event.keyDown())
  {
    if (bbox_mode_state)
    {
      bbox_mode_state = false;
      cout << "[INFO]: shut down the bounding box showing" << endl;
    }
    else
    {
      bbox_mode_state = true;
      cout << "[INFO]: showing bounding boxes" << endl;
    }
  }
  if (event.getKeySym() == "k" && event.keyDown())
  {
    if (lidar_all_state)
    {
      lidar_all_state = false;
      cout << "[INFO]: showing 4 Lidars respectively" << endl;
    }
    else
    {
      lidar_all_state = true;
      cout << "[INFO]: showing /LidarAll" << endl;
    }
  }
  if (event.getKeySym() == "s" && event.keyDown())  // save normal PCD
  {
    static int num2 = 0;
    struct stat buf;
    while (stat((to_string(num2) + ".pcd").c_str(), &buf) == 0)
    {
      num2++;
    }

    pcl::io::savePCDFileASCII(to_string(num2) + ".pcd", cloud_save_to_pcd);

    cout << "[INFO]: output PCD file" << endl;
    num2++;
  }

  if (event.getKeySym() == "b" && event.keyDown())  // save Stitching PCD
  {
    struct stat buf;
    if (stat("StitchingBackground.pcd", &buf) == 0)
    {
      PointCloud<PointXYZ> background_cloud;
      pcl::io::loadPCDFile<PointXYZ>("StitchingBackground.pcd", background_cloud);
      background_cloud = background_cloud + cloud_save_to_pcd;
      pcl::io::savePCDFileASCII("StitchingBackground.pcd", background_cloud);
    }
    else
    {
      pcl::io::savePCDFileASCII("StitchingBackground.pcd", cloud_save_to_pcd);
    }
    cout << "[INFO]: output Stitching Background file" << endl;
  }

  if (event.getKeySym() == "d" && event.keyDown())  // User: stop stream
  {
    if (pause_state)
    {
      pause_state = false;
    }
    else
    {
      pause_state = true;
    }
    cout << "[INFO]: stop stream" << pause_state << endl;
  }

  if (event.getKeySym() == "x" && event.keyDown())  // print cam para
  {
    vector<pcl::visualization::Camera> cam;
    viewer->getCameras(cam);

    cout << "Camera Auto Code: " << endl
         << "cam.pos[0]=" << cam[0].pos[0] << ";cam.pos[1]=" << cam[0].pos[1] << ";cam.pos[2]=" << cam[0].pos[2] << ";"
         << endl
         << "cam.view[0]=" << cam[0].view[0] << ";cam.view[1]=" << cam[0].view[1] << ";cam.view[2]=" << cam[0].view[2]
         << ";" << endl
         << "cam.focal[0]=" << cam[0].focal[0] << ";cam.focal[1]=" << cam[0].focal[1]
         << ";cam.focal[2]=" << cam[0].focal[2] << ";" << endl
         << "cam.clip[0]=" << cam[0].clip[0] << ";cam.clip[1]=" << cam[0].clip[1]
         << ";" << endl
         << "cam.fovy=" << cam[0].fovy << ";" << endl
         << "cam.window_pos[0]=" << cam[0].window_pos[0] << ";cam.window_pos[1]=" << cam[0].window_pos[1] << ";" << endl
         << "cam.window_size[0]=" << cam[0].window_size[0] << ";cam.window_size[1]=" << cam[0].window_size[1] << ";"
         << endl
         << endl;

    cout << "Camera Auto Code: " << endl
         << "cam = CamPara(" << cam[0].pos[0] << "," << cam[0].pos[1] << "," << cam[0].pos[2] << "," << cam[0].view[0]
         << "," << cam[0].view[1] << "," << cam[0].view[2] << "," << cam[0].focal[0] << "," << cam[0].focal[1] << ","
         << cam[0].focal[2] << "," << cam[0].clip[0] << "," << cam[0].clip[1] << ","
         << cam[0].fovy << "," << cam[0].window_pos[0] << "," << cam[0].window_pos[1] << "," << cam[0].window_size[0]
         << "," << cam[0].window_size[1] << ");" << endl;
  }
}

PointCloud<PointXYZ> KeyboardMouseEvent::cloud_save_to_pcd;
bool KeyboardMouseEvent::pause_state;
bool KeyboardMouseEvent::result_mode_state;
bool KeyboardMouseEvent::ch_mode_state;
bool KeyboardMouseEvent::bbox_mode_state;
bool KeyboardMouseEvent::lidar_all_state;

/*
 * PCL default key
 *

 p, P   : switch to a point-based representation, the coordinates of the arrow will disappear
 w, W   : switch to a wireframe-based representation (where available), the coordinates of the arrow will be thick
 s, S   : switch to a surface-based representation (where available)

 j, J   : take a .PNG snapshot of the current window view
 c, C   : display current camera/window parameters
 f, F   : fly to point mode

 e, E   : exit the interactor
 q, Q   : stop and call VTK's TerminateApp, quit application

 +/-   : increment/decrement overall point size
 +/- [+ ALT] : zoom in/out

 g, G   : display scale grid (on/off)
 u, U   : display lookup table (on/off)

 o, O         : switch between perspective/parallel projection (default = perspective)
 r, R [+ ALT] : reset camera [to viewpoint = {0, 0, 0} -> center_{x, y, z}]
 CTRL + s, S  : save camera parameters
 CTRL + r, R  : restore camera parameters

 ALT + s, S   : turn stereo mode on/off
 ALT + f, F   : switch between maximized window mode and original size

 l, L           : list all available geometric and color handlers for the current actor map
 ALT + 0..9 [+ CTRL]  : switch between different geometric handlers (where available)
 0..9 [+ CTRL]  : switch between different color handlers (where available)

 SHIFT + left click   : select a point (start with -use_point_picking)

 x, X   : toggle rubber band selection mode for left mouse button

 */
