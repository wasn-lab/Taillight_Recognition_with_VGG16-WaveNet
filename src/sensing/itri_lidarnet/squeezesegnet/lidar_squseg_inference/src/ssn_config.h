#ifndef SSN_CONFIG_H_
#define SSN_CONFIG_H_


#include <string>

void norm_mean(float* mean_ptr,string data_set,char ViewType,float phi_center)
{

  if (data_set.compare("hino"))
  {
    switch (ViewType)
    {
    case 'X':
    {
      float INPUT_MEAN[4][5] = {{0.10, -2.95, 0.22, 3.15, 3.32},
                                {3.58, -0.07, 0.23, 1.51, 4.04},
                                {0.14, 3.29, 0.22, 2.22, 3.69},
                                {-3.51, 0.07, 0.26, 1.49, 3.92}};

      float phi_range = 90.0;
      int phi_center_ind = int(phi_center / phi_range) + 1;

      for (size_t i = 0; i < 5; i++)
        mean_ptr[i] = INPUT_MEAN[phi_center_ind][i];
        
      break;
    }
    case 'T':
    {
      float INPUT_MEAN[3][5] = {{-1.63, -1.57, 0.16, 1.54, 2.52},
                                {1.91, 0.00, 0.18, 2.04, 2.97},
                                {-1.67, 1.57, 0.17, 1.54, 2.55}};

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

      for (size_t i = 0; i < 5; i++)
        mean_ptr[i] = INPUT_MEAN[phi_center_ind][i];

      break;
    }
    default:
      cout << "No matched ViewType found !!!!!!!!!!" << endl;
    }
  }
  else if(data_set.compare("kitti"))
  {
    float INPUT_MEAN[5] = {10.88, 0.23, -1.04, 0.21, 12.12};
    
    for(size_t i=0; i<5; i++)
      mean_ptr[i] = INPUT_MEAN[i];
  }
  else
  {
    float INPUT_MEAN[4][5] = {{0.10, -2.95, 0.22, 3.15, 3.32},
                              {3.58, -0.07, 0.23, 1.51, 4.04},
                              {0.14, 3.29, 0.22, 2.22, 3.69},
                              {-3.51, 0.07, 0.26, 1.49, 3.92}};

    float phi_range = 90.0;
    int phi_center_ind = int(phi_center / phi_range) + 1;

    for (size_t i = 0; i < 5; i++)
      mean_ptr[i] = INPUT_MEAN[phi_center_ind][i];
  }
  
}

void norm_std(float* std_ptr, string data_set,char ViewType,float phi_center)
{

  if (data_set.compare("hino"))
  {
    switch (ViewType)
    {
    case 'X':
    {
      float INPUT_STD[4][5] = {{3.35, 6.26, 0.86, 8.39, 7.01},
                               {8.04, 4.00, 1.02, 5.82, 8.86},
                               {3.60, 6.75, 0.89, 7.09, 7.53},
                               {8.78, 3.93, 0.95, 6.15, 9.52}};

      float phi_range = 90.0;
      int phi_center_ind = int(phi_center / phi_range) + 1;

      for (size_t i = 0; i < 5; i++)
        std_ptr[i] = INPUT_STD[phi_center_ind][i];

      break;
    }
    case 'T':
    {
      float INPUT_STD[3][5] = {{5.46, 4.49, 0.69, 5.99, 7.12},
                               {5.74, 5.50, 0.87, 7.77, 7.71},
                               {5.43, 4.24, 0.70, 5.95, 6.90}};

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

      for (size_t i = 0; i < 5; i++)
        std_ptr[i] = INPUT_STD[phi_center_ind][i];

      break;
    }
    default:
      cout << "No matched ViewType found !!!!!!!!!!" << endl;
    }
  }
  else if (data_set.compare("kitti"))
  {
    float INPUT_STD[5] = {11.47, 6.91, 0.86, 0.16, 12.32};

    for (size_t i = 0; i < 5; i++)
      std_ptr[i] = INPUT_STD[i];
    }
    else
    {
      float INPUT_STD[4][5] = {{3.35, 6.26, 0.86, 8.39, 7.01},
                               {8.04, 4.00, 1.02, 5.82, 8.86},
                               {3.60, 6.75, 0.89, 7.09, 7.53},
                               {8.78, 3.93, 0.95, 6.15, 9.52}};

      float phi_range = 90.0;
      int phi_center_ind = int(phi_center / phi_range) + 1;

      for (size_t i = 0; i < 5; i++)
        std_ptr[i] = INPUT_STD[phi_center_ind][i];
    }
  
}

float proj_center(string data_set, int index)
{

  if (data_set.compare("hino"))
  {
    float CENTER[2] = {-2, -1.4};   // {x,z}
    return CENTER[index];
  }
  else if(data_set.compare("kitti"))
  {
    float CENTER[2] = {0, 0};       // {x,z}
    return CENTER[index];
  }
  else
  {
    float CENTER[2] = {-2.5, -0.6};   // {x,z}, temporaly used for b1 test, wil be reset to {-2.5,-1.4} for letting hino as default
    return CENTER[index];
  }
  
}


#endif /* SSN_CONFIG_H_ */
