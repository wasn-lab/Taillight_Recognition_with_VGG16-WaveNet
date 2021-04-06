#include "Main.hpp"

extern std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> inputCloud;
extern std::vector<BoxInfo> inputBox;

void loadPCDFile(std::string filename) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr inputPcd(
      new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(filename, *inputPcd) == -1) {
    PCL_ERROR("Couldn't read file \n");
  } else {
    std::cout << "Load PCD File : " << filename << std::endl;
    if (inputCloud.empty() == false) inputCloud.erase(inputCloud.begin());
    inputCloud.push_back(inputPcd);
  }
}

void loadXMLFile(std::string filename) {
  if (inputBox.empty() == false)
  {
    inputBox.clear();
  }
  // Initialize xerces
  try { XMLPlatformUtils::Initialize();}
  catch (const XMLException& toCatch) 
  {
    char* message = XMLString::transcode(toCatch.getMessage());
    std::cout << "Error During XML parser initialization! :\n"
                       << message << "\n";
    XMLString::release(&message);
  }

  // Create parser, set parser values for validation
  XercesDOMParser* parser = new XercesDOMParser();
  parser->setValidationScheme(XercesDOMParser::Val_Never);
  parser->setDoNamespaces(true);
  parser->setDoSchema(true);
  parser->setValidationConstraintFatal(true);

  parser->parse(XMLString::transcode(filename.c_str()));

  // DOMElement* docRootNode;
  DOMNodeList* docCurrentNode;
  DOMDocument* doc;
  // DOMNodeIterator* walker;
  doc = parser->getDocument();
  // docRootNode = doc->getDocumentElement();
  docCurrentNode = doc->getDocumentElement()->getElementsByTagName(XMLString::transcode("object"));
  // std::cout << doc->getNodeType() << std::endl;
  std::cout << docCurrentNode->getLength() << std::endl;
  // try 
  // {
  //   walker = doc->createNodeIterator(docRootNode, DOMNodeFilter::SHOW_ELEMENT, NULL, true);
  // }
  // catch (const XMLException& toCatch) 
  // {
  //   char* message = XMLString::transcode(toCatch.getMessage());
  //   std::cout << "Error During creating node iterator :\n"
  //                      << message << "\n";
  //   XMLString::release(&message);
  // }

  DOMNode* current_node = NULL;
  std::string thisNodeName, nextNodeName;

  for (XMLSize_t i = 0; i < docCurrentNode->getLength(); i++)
  {
    current_node = docCurrentNode->item(i);
    thisNodeName = XMLString::transcode(current_node->getNodeName());
    if (thisNodeName == "object")
    {
      std::cout << thisNodeName << std::endl;
      BoxInfo bbox;
      pcl::PointCloud<pcl::PointXYZ>::Ptr bbox_corner(new pcl::PointCloud<pcl::PointXYZ>);
      bbox.corner = bbox_corner;
      // class id
      DOMNodeList* obj_nodelist = current_node->getChildNodes();
      DOMNode* class_node = obj_nodelist->item(3);
      std::cout << XMLString::transcode(class_node->getTextContent()) << std::endl;
      std::string obj_class = XMLString::transcode(class_node->getTextContent());
      bbox.classId = obj_class;
      // corners
      // <pcd_bndbox></pcd_bndbox>->items under<pcd_bndbox></pcd_bndbox>
      thisNodeName = XMLString::transcode(obj_nodelist->item(5)->getNodeName());
      std::cout << thisNodeName << std::endl;
      std::cout << nextNodeName << std::endl;
      if (thisNodeName == "track_id")
      {
        nextNodeName = XMLString::transcode(obj_nodelist->item(7)->getNodeName());
        if (nextNodeName == "vehicle_info")
        {
          obj_nodelist = obj_nodelist->item(9)->getChildNodes();
        }
        else
        {
          obj_nodelist = obj_nodelist->item(7)->getChildNodes();
        }
      }
      else
      {
        obj_nodelist = obj_nodelist->item(5)->getChildNodes();
      }

      DOMNode* rotation_node = obj_nodelist->item(17);
      float rotation_y = atof(XMLString::transcode(rotation_node->getTextContent()));
      float center_x = atof(XMLString::transcode(obj_nodelist->item(9)->getTextContent()));
      float center_y = atof(XMLString::transcode(obj_nodelist->item(11)->getTextContent()));
      float center_z = atof(XMLString::transcode(obj_nodelist->item(13)->getTextContent()));

      for (XMLSize_t j = 21; j < 37; j=j+2)
      {
        // <pN></pN>
        DOMNode* corner_node = obj_nodelist->item(j);
        std::cout << XMLString::transcode(corner_node->getNodeName()) << std::endl;
        // <x></x><y></y><z></z>
        DOMNodeList* corner_nodelist = corner_node->getChildNodes();
        float x = atof(XMLString::transcode(corner_nodelist->item(1)->getTextContent()));
        float y = atof(XMLString::transcode(corner_nodelist->item(3)->getTextContent()));
        float z = atof(XMLString::transcode(corner_nodelist->item(5)->getTextContent()));
        std::cout << x << " " << y << " " << z << std::endl;

        pcl::PointXYZ p;
        p.x = x;
        p.y = y;
        p.z = z;

        bbox.corner->push_back(p);
      }

      Eigen::Matrix4f transform_1 = Eigen::Matrix4f::Identity();
      Eigen::Matrix4f transform_2 = Eigen::Matrix4f::Identity();
      Eigen::Matrix4f transform_3 = Eigen::Matrix4f::Identity();
      transform_1(0, 3) = -center_x;
      transform_1(1, 3) = -center_y;
      transform_1(2, 3) = -center_z;
      transform_2(0, 0) = std::cos(rotation_y);
      transform_2(0, 1) = -std::sin(rotation_y);
      transform_2(1, 0) = std::sin(rotation_y);
      transform_2(1, 1) = std::cos(rotation_y);
      transform_3(0, 3) = center_x;
      transform_3(1, 3) = center_y;
      transform_3(2, 3) = center_z;
      pcl::transformPointCloud(*bbox.corner, *bbox.corner, transform_1);
      pcl::transformPointCloud(*bbox.corner, *bbox.corner, transform_2);
      pcl::transformPointCloud(*bbox.corner, *bbox.corner, transform_3);

      inputBox.push_back(bbox);
    }
    else
    {
      std::cout << thisNodeName << " is not object" << std::endl;
    }
  }
  XMLPlatformUtils::Terminate();
}

void loadTXTFile(std::string filename) {
  if (inputBox.empty() == false)
  {
    inputBox.clear();
  }

  // declare line variables
  std::string classID;
  float dumb[7];
  float length, width, height;
  float x_center, y_center, z_center; 
  float y_rotation;
  float score;
  // flag for judging groundtruth 
  bool gt = false;
  // open txt file
  std::fstream txt_file;
  txt_file.open(filename);
  // declare line stream and read lines
  std::string line;
  while (std::getline(txt_file, line))
  {
    // declare temporary line stream
    std::string line_tmp = line;
    std::istringstream iss_tmp(line_tmp);
    std::vector<std::string> tokens;
    std::string token;
    // declare bbox and assign classID
    // first count the number of elements
    // if there are 16 elements,  that means the column of txt file includes score, which means the file is produced by pointpillars
    // if there are 15,  the txt file is groundtruth file
    while (std::getline(iss_tmp, token, ' '))
    {
      tokens.push_back(token);
    }
    if (tokens.size() == 16)
    {
      std::istringstream iss(line);
      if (!(iss >> classID >> dumb[0] >> dumb[1] >> dumb[2] >> dumb[3] >> dumb[4] >> dumb[5] >> dumb[6]
       >> length >> width >> height >> x_center >> y_center >> z_center >> y_rotation >> score))
       {
         break;
       }
       // the inference code of motorbike model has been modified so the dimension may be in different order
       // so we need to fix it
       if (classID == "Motorbike")
       {
         float tmp;
         tmp = length;
         length = width;
         width = height;
         height = tmp;
       }
    }
    else
    {
      std::istringstream iss(line);
      if (!(iss >> classID >> dumb[0] >> dumb[1] >> dumb[2] >> dumb[3] >> dumb[4] >> dumb[5] >> dumb[6]
       >> height >> length >> width >> x_center >> y_center >> z_center >> y_rotation))
       {
         break;
       }
       gt = true;
       score = 1.0f;
    }
    
    std::cout << classID << " " << dumb[0] << " " << dumb[1] << " " << dumb[2] << " " << dumb[3] << " " << dumb[4] << " " << dumb[5] << " " << dumb[6] << " " 
      << length << " " << width << " " << height << " " << x_center << " " << y_center << " " << z_center << " " << y_rotation << " " << score << " " << std::endl;
    BoxInfo bbox;
    pcl::PointCloud<pcl::PointXYZ>::Ptr bbox_corner(new pcl::PointCloud<pcl::PointXYZ>);
    bbox.corner = bbox_corner;
    bbox.classId = classID;

    // assign corner
    int x_rel_pos[8] = {-1, 1, 1, -1, -1, 1, 1, -1};
    int y_rel_pos[8] = {-1, -1, 1, 1, -1, -1, 1, 1};
    int z_rel_pos[8] = {1, 1, 1, 1, -1, -1, -1, -1};
    for (size_t i = 0; i < 8; i++)
    {
      pcl::PointXYZ p;
      p.x = x_center + length * 0.5 * x_rel_pos[i];
      p.y = y_center + width * 0.5 * y_rel_pos[i];
      p.z = z_center + height * 0.5 * z_rel_pos[i];
      bbox.corner->push_back(p);
    }
    // transform corners according to rotation
    Eigen::Matrix4f transform_1 = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f transform_2 = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f transform_3 = Eigen::Matrix4f::Identity();
    transform_1(0, 3) = -x_center;
    transform_1(1, 3) = -y_center;
    transform_1(2, 3) = -z_center;
    transform_2(0, 0) = std::cos(y_rotation);
    transform_2(0, 1) = -std::sin(y_rotation);
    transform_2(1, 0) = std::sin(y_rotation);
    transform_2(1, 1) = std::cos(y_rotation);
    // the inference code has added PI/2 to y rotation of motorbike detection
    if (bbox.classId == "Motorbike" && gt == false)
    {
      transform_2(0, 0) = std::cos(y_rotation+M_PI_2);
      transform_2(0, 1) = -std::sin(y_rotation+M_PI_2);
      transform_2(1, 0) = std::sin(y_rotation+M_PI_2);
      transform_2(1, 1) = std::cos(y_rotation+M_PI_2);
    }
    transform_3(0, 3) = x_center;
    transform_3(1, 3) = y_center;
    transform_3(2, 3) = z_center;
    pcl::transformPointCloud(*bbox.corner, *bbox.corner, transform_1);
    pcl::transformPointCloud(*bbox.corner, *bbox.corner, transform_2);
    pcl::transformPointCloud(*bbox.corner, *bbox.corner, transform_3);

    inputBox.push_back(bbox);
  }
}

void displayPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  for (auto point : cloud->points) {
    glColor3f(1.0, 1.0, 1.0);
    glVertex3f(point.x, point.y, point.z);
  }
}

void displayBoundingBox(BoxInfo &bbox)
{
  if (bbox.classId == "Car")
  {
    glColor3f(0.0, 1.0, 0.0);
  }
  else if (bbox.classId == "Pedestrian")
  {
    glColor3f(0.0, 1.0, 1.0);
  }
  else if (bbox.classId == "Motorbike")
  {
    glColor3f(1.0, 0.0, 1.0);
  }
  else if (bbox.classId == "Bus")
  {
    glColor3f(0.0, 0.0, 1.0);
  }
  else
  {
    glColor3f(1.0, 0.0, 0.0);
  }

  for (int i = 0; i < 4; i++)
  {
    pcl::PointXYZ cur_corner, next_corner;
    if (i == 3)
    {
      cur_corner = bbox.corner->points[i];
      next_corner = bbox.corner->points[0];
    }
    else
    {
      cur_corner = bbox.corner->points[i];
      next_corner = bbox.corner->points[i+1];
    }
    
    glVertex3f(cur_corner.x, cur_corner.y, cur_corner.z);
    glVertex3f(next_corner.x, next_corner.y, next_corner.z);
  }
  for (int i = 4; i < 8; i++)
  {
    pcl::PointXYZ cur_corner, next_corner;
    if (i == 7)
    {
      cur_corner = bbox.corner->points[i];
      next_corner = bbox.corner->points[4];
    }
    else
    {
      cur_corner = bbox.corner->points[i];
      next_corner = bbox.corner->points[i+1];
    }
    
    glVertex3f(cur_corner.x, cur_corner.y, cur_corner.z);
    glVertex3f(next_corner.x, next_corner.y, next_corner.z);
  }
  for (int i = 0; i < 4; i++)
  {
    pcl::PointXYZ cur_corner, next_corner;
    cur_corner = bbox.corner->points[i];
    next_corner = bbox.corner->points[i+4];
    
    glVertex3f(cur_corner.x, cur_corner.y, cur_corner.z);
    glVertex3f(next_corner.x, next_corner.y, next_corner.z);
  }
}
