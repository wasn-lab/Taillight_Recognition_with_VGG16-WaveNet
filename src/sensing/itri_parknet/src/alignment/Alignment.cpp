#include "Alignment.h"

// cv::Mat alig_CameraExtrinsicMat(3,3,CV_64F);
// cv::Mat alig_DistCoeff_(1,5,CV_64F);
// cv::Mat alig_CameraExtrinsicMat_tr(1,3,CV_64F);
// cv::Mat alig_CameraMat_(3,3,CV_64F);

Alignment::Alignment()
{
}

void Alignment::a(int a)
{
  cout << "123" << endl;
  cout << a << endl;
}
void Alignment::camera_alignment_module(int num, double& cosfi_min, double& cosfi_max, double& costhta_min,
                                        double& costhta_max)
{
  int cmd_id = num;

  cv::Mat in_CameraExtrinsicMat = cv::Mat::zeros(3, 3, CV_64F);
  cv::Mat in_DistCoeff_ = cv::Mat::zeros(1, 5, CV_64F);
  cv::Mat in_CameraExtrinsicMat_tr = cv::Mat::zeros(1, 3, CV_64F);
  cv::Mat in_CameraMat_ = cv::Mat::zeros(3, 3, CV_64F);

  switch (cmd_id)
  {
    case 1:  // 60 left LidarFront .

      cosfi_min = 30;
      cosfi_max = 90;

      costhta_min = 60;
      costhta_max = 120;

      in_CameraExtrinsicMat.at<double>(0, 0) = 0.872960;
      in_CameraExtrinsicMat.at<double>(0, 1) = -0.026354;
      in_CameraExtrinsicMat.at<double>(0, 2) = 0.487080;
      in_CameraExtrinsicMat.at<double>(1, 0) = -0.486966;
      in_CameraExtrinsicMat.at<double>(1, 1) = 0.011060;
      in_CameraExtrinsicMat.at<double>(1, 2) = 0.873351;
      in_CameraExtrinsicMat.at<double>(2, 0) = -0.028404;
      in_CameraExtrinsicMat.at<double>(2, 1) = -0.999591;
      in_CameraExtrinsicMat.at<double>(2, 2) = -0.003178;

      in_CameraExtrinsicMat_tr.at<double>(0) = 1.06461;
      in_CameraExtrinsicMat_tr.at<double>(1) = 0.245148;
      in_CameraExtrinsicMat_tr.at<double>(2) = -1.01763;

      in_DistCoeff_.at<double>(0) = -1.3234694514851916e-01;
      in_DistCoeff_.at<double>(1) = -2.2877508585825934e+00;
      in_DistCoeff_.at<double>(2) = -2.6973306893876632e-03;
      in_DistCoeff_.at<double>(3) = 9.8102015044200194e-04;
      in_DistCoeff_.at<double>(4) = 8.4113059659251626e+00;

      in_CameraMat_.at<double>(0, 0) = 1.8828750696345512e+03;
      in_CameraMat_.at<double>(0, 1) = 0;
      in_CameraMat_.at<double>(0, 2) = 9.7726620903028095e+02;
      in_CameraMat_.at<double>(1, 0) = 0;
      in_CameraMat_.at<double>(1, 1) = 1.8708459663727708e+03;
      in_CameraMat_.at<double>(1, 2) = 5.4527594054228973e+02;
      in_CameraMat_.at<double>(2, 0) = 0;
      in_CameraMat_.at<double>(2, 1) = 0;
      in_CameraMat_.at<double>(2, 2) = 1;

      alig_CameraExtrinsicMat = in_CameraExtrinsicMat;
      alig_CameraExtrinsicMat_tr = in_CameraExtrinsicMat_tr;
      alig_DistCoeff_ = in_DistCoeff_;
      alig_CameraMat_ = in_CameraMat_;

      break;

    case 2:  // 60 LidarFront .
      cosfi_min = -30;
      cosfi_max = 30;

      costhta_min = 60;
      costhta_max = 120;

      in_CameraExtrinsicMat.at<double>(0, 0) = -0.00243068;
      in_CameraExtrinsicMat.at<double>(0, 1) = 0.0012012;
      in_CameraExtrinsicMat.at<double>(0, 2) = 0.999996;
      in_CameraExtrinsicMat.at<double>(1, 0) = -0.999964;
      in_CameraExtrinsicMat.at<double>(1, 1) = -0.00812897;
      in_CameraExtrinsicMat.at<double>(1, 2) = -0.00242083;
      in_CameraExtrinsicMat.at<double>(2, 0) = 0.00812603;
      in_CameraExtrinsicMat.at<double>(2, 1) = -0.999966;
      in_CameraExtrinsicMat.at<double>(2, 2) = 0.00122091;

      in_CameraExtrinsicMat_tr.at<double>(0) = 0.594071;  // 0.0838637;
      in_CameraExtrinsicMat_tr.at<double>(1) = 0.16804;
      in_CameraExtrinsicMat_tr.at<double>(2) = -1.06013;

      in_DistCoeff_.at<double>(0) = -0.2801951;
      in_DistCoeff_.at<double>(1) = 0.0400105;
      in_DistCoeff_.at<double>(2) = -0.00253047;
      in_DistCoeff_.at<double>(3) = 0.00201554;
      in_DistCoeff_.at<double>(4) = 0.0851216;

      in_CameraMat_.at<double>(0, 0) = 1864.08;
      in_CameraMat_.at<double>(0, 1) = 0;
      in_CameraMat_.at<double>(0, 2) = 958.784;
      in_CameraMat_.at<double>(1, 0) = 0;
      in_CameraMat_.at<double>(1, 1) = 1873.96;
      in_CameraMat_.at<double>(1, 2) = 604.981;
      in_CameraMat_.at<double>(2, 0) = 0;
      in_CameraMat_.at<double>(2, 1) = 0;
      in_CameraMat_.at<double>(2, 2) = 1;

      alig_CameraExtrinsicMat = in_CameraExtrinsicMat;
      alig_CameraExtrinsicMat_tr = in_CameraExtrinsicMat_tr;
      alig_DistCoeff_ = in_DistCoeff_;
      alig_CameraMat_ = in_CameraMat_;

      break;

    case 3:  // 60 right LidarFront .
      cosfi_min = -90;
      cosfi_max = -30;

      costhta_min = 60;
      costhta_max = 120;

      in_CameraExtrinsicMat.at<double>(0, 0) = -0.87665;
      in_CameraExtrinsicMat.at<double>(0, 1) = 0.0372601;
      in_CameraExtrinsicMat.at<double>(0, 2) = 0.479683;
      in_CameraExtrinsicMat.at<double>(1, 0) = -0.480141;
      in_CameraExtrinsicMat.at<double>(1, 1) = -0.0039296;
      in_CameraExtrinsicMat.at<double>(1, 2) = -0.877182;
      in_CameraExtrinsicMat.at<double>(2, 0) = -0.030799;
      in_CameraExtrinsicMat.at<double>(2, 1) = -0.999298;
      in_CameraExtrinsicMat.at<double>(2, 2) = 0.021335;

      in_CameraExtrinsicMat_tr.at<double>(0) = 1.31721 - 0.5;
      in_CameraExtrinsicMat_tr.at<double>(1) = -0.447222 + 0.7;
      in_CameraExtrinsicMat_tr.at<double>(2) = -0.781829 - 0.3;

      in_DistCoeff_.at<double>(0) = -1.3234694514851916e-01;
      in_DistCoeff_.at<double>(1) = -2.2877508585825934e+00;
      in_DistCoeff_.at<double>(2) = -2.6973306893876632e-03;
      in_DistCoeff_.at<double>(3) = 9.8102015044200194e-04;
      in_DistCoeff_.at<double>(4) = 8.4113059659251626e+00;

      in_CameraMat_.at<double>(0, 0) = 1.8828750696345512e+03;
      in_CameraMat_.at<double>(0, 1) = 0;
      in_CameraMat_.at<double>(0, 2) = 9.7726620903028095e+02;
      in_CameraMat_.at<double>(1, 0) = 0;
      in_CameraMat_.at<double>(1, 1) = 1.8708459663727708e+03;
      in_CameraMat_.at<double>(1, 2) = 5.4527594054228973e+02;
      in_CameraMat_.at<double>(2, 0) = 0;
      in_CameraMat_.at<double>(2, 1) = 0;
      in_CameraMat_.at<double>(2, 2) = 1;

      alig_CameraExtrinsicMat = in_CameraExtrinsicMat;
      alig_CameraExtrinsicMat_tr = in_CameraExtrinsicMat_tr;
      alig_DistCoeff_ = in_DistCoeff_;
      alig_CameraMat_ = in_CameraMat_;

      break;

    case 4:
      // 120 side left
      // x-aixs:95~1735 is correct
      cosfi_min = 65;
      cosfi_max = 160;

      costhta_min = 98;
      costhta_max = 148;

      in_CameraExtrinsicMat.at<double>(0, 0) = 0.925797;
      in_CameraExtrinsicMat.at<double>(0, 1) = -0.175469;
      in_CameraExtrinsicMat.at<double>(0, 2) = -0.334829;
      in_CameraExtrinsicMat.at<double>(1, 0) = 0.0419238;
      in_CameraExtrinsicMat.at<double>(1, 1) = -0.83262;
      in_CameraExtrinsicMat.at<double>(1, 2) = 0.552256;
      in_CameraExtrinsicMat.at<double>(2, 0) = -0.375689;
      in_CameraExtrinsicMat.at<double>(2, 1) = -0.525314;
      in_CameraExtrinsicMat.at<double>(2, 2) = -0.763481;

      in_CameraExtrinsicMat_tr.at<double>(0) = 0.0837403;
      in_CameraExtrinsicMat_tr.at<double>(1) = 1.15435;
      in_CameraExtrinsicMat_tr.at<double>(2) = -0.253806;

      in_DistCoeff_.at<double>(0) = -0.384279;
      in_DistCoeff_.at<double>(1) = 0.188464;
      in_DistCoeff_.at<double>(2) = -0.00160702;
      in_DistCoeff_.at<double>(3) = 0.000320484;
      in_DistCoeff_.at<double>(4) = -0.048941;

      in_CameraMat_.at<double>(0, 0) = 1006.37;
      in_CameraMat_.at<double>(0, 1) = 0;
      in_CameraMat_.at<double>(0, 2) = 947.756;
      in_CameraMat_.at<double>(1, 0) = 0;
      in_CameraMat_.at<double>(1, 1) = 1011.26;
      in_CameraMat_.at<double>(1, 2) = 615.522;
      in_CameraMat_.at<double>(2, 0) = 0;
      in_CameraMat_.at<double>(2, 1) = 0;
      in_CameraMat_.at<double>(2, 2) = 1;

      alig_CameraExtrinsicMat = in_CameraExtrinsicMat;
      alig_CameraExtrinsicMat_tr = in_CameraExtrinsicMat_tr;
      alig_DistCoeff_ = in_DistCoeff_;
      alig_CameraMat_ = in_CameraMat_;

      break;

    case 5:  // 120 LidarFront .port_b/cam_1 ....

      cosfi_min = -80;
      cosfi_max = 80;

      costhta_min = 90;
      costhta_max = 150;

      in_CameraExtrinsicMat.at<double>(0, 0) = -0.0772879;
      in_CameraExtrinsicMat.at<double>(0, 1) = -0.804028;
      in_CameraExtrinsicMat.at<double>(0, 2) = 0.589547;
      in_CameraExtrinsicMat.at<double>(1, 0) = -0.996919;
      in_CameraExtrinsicMat.at<double>(1, 1) = 0.0543901;
      in_CameraExtrinsicMat.at<double>(1, 2) = -0.0565157;
      in_CameraExtrinsicMat.at<double>(2, 0) = 0.0133747;
      in_CameraExtrinsicMat.at<double>(2, 1) = -0.592098;
      in_CameraExtrinsicMat.at<double>(2, 2) = -0.805755;

      in_CameraExtrinsicMat_tr.at<double>(0) = -0.799889;  // 0.0838637;
      in_CameraExtrinsicMat_tr.at<double>(1) = 0.492527;
      in_CameraExtrinsicMat_tr.at<double>(2) = 4.2695;

      in_DistCoeff_.at<double>(0) = -0.117567;
      in_DistCoeff_.at<double>(1) = -0.134832;
      in_DistCoeff_.at<double>(2) = 0.0100694;
      in_DistCoeff_.at<double>(3) = 0.00339625;
      in_DistCoeff_.at<double>(4) = 0.0829011;

      in_CameraMat_.at<double>(0, 0) = 1382.82;
      in_CameraMat_.at<double>(0, 1) = 0;
      in_CameraMat_.at<double>(0, 2) = 962.425;
      in_CameraMat_.at<double>(1, 0) = 0;
      in_CameraMat_.at<double>(1, 1) = 1327.62;
      in_CameraMat_.at<double>(1, 2) = 607.539;
      in_CameraMat_.at<double>(2, 0) = 0;
      in_CameraMat_.at<double>(2, 1) = 0;
      in_CameraMat_.at<double>(2, 2) = 1;

      alig_CameraExtrinsicMat = in_CameraExtrinsicMat;
      alig_CameraExtrinsicMat_tr = in_CameraExtrinsicMat_tr;
      alig_DistCoeff_ = in_DistCoeff_;
      alig_CameraMat_ = in_CameraMat_;
      break;

    case 6:
      // 120 side right
      cosfi_min = -170;
      cosfi_max = -30;

      costhta_min = 90;
      costhta_max = 150;

      in_CameraExtrinsicMat.at<double>(0, 0) = -0.853910;
      in_CameraExtrinsicMat.at<double>(0, 1) = -0.220861;
      in_CameraExtrinsicMat.at<double>(0, 2) = -0.471230;
      in_CameraExtrinsicMat.at<double>(1, 0) = -0.023933;
      in_CameraExtrinsicMat.at<double>(1, 1) = 0.921187;
      in_CameraExtrinsicMat.at<double>(1, 2) = -0.388383;
      in_CameraExtrinsicMat.at<double>(2, 0) = 0.519870;
      in_CameraExtrinsicMat.at<double>(2, 1) = -0.320366;
      in_CameraExtrinsicMat.at<double>(2, 2) = -0.791898;

      in_CameraExtrinsicMat_tr.at<double>(0) = 1.3171;
      in_CameraExtrinsicMat_tr.at<double>(1) = -1.26605;
      in_CameraExtrinsicMat_tr.at<double>(2) = 0.565924;

      in_DistCoeff_.at<double>(0) = -0.269525;
      in_DistCoeff_.at<double>(1) = -0.0169836;
      in_DistCoeff_.at<double>(2) = -0.00484475;
      in_DistCoeff_.at<double>(3) = 0.00881316;
      in_DistCoeff_.at<double>(4) = 0.0491113;

      in_CameraMat_.at<double>(0, 0) = 1270.75;
      in_CameraMat_.at<double>(0, 1) = 0;
      in_CameraMat_.at<double>(0, 2) = 957.399;
      in_CameraMat_.at<double>(1, 0) = 0;
      in_CameraMat_.at<double>(1, 1) = 1256.34;
      in_CameraMat_.at<double>(1, 2) = 614.28;
      in_CameraMat_.at<double>(2, 0) = 0;
      in_CameraMat_.at<double>(2, 1) = 0;
      in_CameraMat_.at<double>(2, 2) = 1;

      alig_CameraExtrinsicMat = in_CameraExtrinsicMat;
      alig_CameraExtrinsicMat_tr = in_CameraExtrinsicMat_tr;
      alig_DistCoeff_ = in_DistCoeff_;
      alig_CameraMat_ = in_CameraMat_;
      break;

    case 7:  // 30 left_back LidarLeft ...

      cosfi_min = 150;
      cosfi_max = 180;

      costhta_min = 90;
      costhta_max = 120;

      in_CameraExtrinsicMat.at<double>(0, 0) = 0.143614;
      in_CameraExtrinsicMat.at<double>(0, 1) = -0.131303;
      in_CameraExtrinsicMat.at<double>(0, 2) = -0.980885;
      in_CameraExtrinsicMat.at<double>(1, 0) = 0.0257818;
      in_CameraExtrinsicMat.at<double>(1, 1) = -0.990326;
      in_CameraExtrinsicMat.at<double>(1, 2) = 0.136342;
      in_CameraExtrinsicMat.at<double>(2, 0) = -0.989298;
      in_CameraExtrinsicMat.at<double>(2, 1) = -0.0448696;
      in_CameraExtrinsicMat.at<double>(2, 2) = -0.13884;

      in_CameraExtrinsicMat_tr.at<double>(0) = -0.68018;
      in_CameraExtrinsicMat_tr.at<double>(1) = 1.19149;
      in_CameraExtrinsicMat_tr.at<double>(2) = -0.407725;

      in_DistCoeff_.at<double>(0) = 0.265471;
      in_DistCoeff_.at<double>(1) = -13.0192;
      in_DistCoeff_.at<double>(2) = -0.00588032;
      in_DistCoeff_.at<double>(3) = 0.00378908;
      in_DistCoeff_.at<double>(4) = 114.774;

      in_CameraMat_.at<double>(0, 0) = 3682.51;
      in_CameraMat_.at<double>(0, 1) = 0;
      in_CameraMat_.at<double>(0, 2) = 992.373;
      in_CameraMat_.at<double>(1, 0) = 0;
      in_CameraMat_.at<double>(1, 1) = 3707.93;
      in_CameraMat_.at<double>(1, 2) = 448.949;
      in_CameraMat_.at<double>(2, 0) = 0;
      in_CameraMat_.at<double>(2, 1) = 0;
      in_CameraMat_.at<double>(2, 2) = 1;

      alig_CameraExtrinsicMat = in_CameraExtrinsicMat;
      alig_CameraExtrinsicMat_tr = in_CameraExtrinsicMat_tr;
      alig_DistCoeff_ = in_DistCoeff_;
      alig_CameraMat_ = in_CameraMat_;

      break;

    case 8:  // 30 LidarFrontTop .

      cosfi_min = -105;
      cosfi_max = 50;

      costhta_min = 90;
      costhta_max = 150;

      in_CameraExtrinsicMat.at<double>(0, 0) = 8.0878741856220682e-03;
      in_CameraExtrinsicMat.at<double>(0, 1) = -3.5082994716428617e-02;
      in_CameraExtrinsicMat.at<double>(0, 2) = 9.9935167472361019e-01;
      in_CameraExtrinsicMat.at<double>(1, 0) = -9.9973091670208647e-01;
      in_CameraExtrinsicMat.at<double>(1, 1) = -2.2012231201564036e-02;
      in_CameraExtrinsicMat.at<double>(1, 2) = 7.3181874487222021e-03;
      in_CameraExtrinsicMat.at<double>(2, 0) = 2.1741216184089085e-02;
      in_CameraExtrinsicMat.at<double>(2, 1) = -9.9914195445855225e-01;
      in_CameraExtrinsicMat.at<double>(2, 2) = -3.5251586625012710e-02;

      in_CameraExtrinsicMat_tr.at<double>(0) = 4.3653121001898632e-01;  // 0.0838637;
      in_CameraExtrinsicMat_tr.at<double>(1) = -1.2899014412823920e-01;
      in_CameraExtrinsicMat_tr.at<double>(2) = -3.4014022740197092e-01;

      in_DistCoeff_.at<double>(0) = -1.3234694514851916e-01;
      in_DistCoeff_.at<double>(1) = -2.2877508585825934e+00;
      in_DistCoeff_.at<double>(2) = -2.6973306893876632e-03;
      in_DistCoeff_.at<double>(3) = 9.8102015044200194e-04;
      in_DistCoeff_.at<double>(4) = 8.4113059659251626e+00;

      in_CameraMat_.at<double>(0, 0) = 1.8828750696345512e+03;
      in_CameraMat_.at<double>(0, 1) = 0;
      in_CameraMat_.at<double>(0, 2) = 9.7726620903028095e+02;
      in_CameraMat_.at<double>(1, 0) = 0;
      in_CameraMat_.at<double>(1, 1) = 1.8708459663727708e+03;
      in_CameraMat_.at<double>(1, 2) = 5.4527594054228973e+02;
      in_CameraMat_.at<double>(2, 0) = 0;
      in_CameraMat_.at<double>(2, 1) = 0;
      in_CameraMat_.at<double>(2, 2) = 1;

      alig_CameraExtrinsicMat = in_CameraExtrinsicMat;
      alig_CameraExtrinsicMat_tr = in_CameraExtrinsicMat_tr;
      alig_DistCoeff_ = in_DistCoeff_;
      alig_CameraMat_ = in_CameraMat_;

      break;

    case 9:  // 30 righ_backt LidarRight ...

      cosfi_min = -180;
      cosfi_max = -150;

      costhta_min = 90;
      costhta_max = 120;

      // fixed
      in_CameraExtrinsicMat.at<double>(0, 0) = -0.276814;
      in_CameraExtrinsicMat.at<double>(0, 1) = -0.256694;
      in_CameraExtrinsicMat.at<double>(0, 2) = -0.926003;
      in_CameraExtrinsicMat.at<double>(1, 0) = -0.028933;
      in_CameraExtrinsicMat.at<double>(1, 1) = 0.965449;
      in_CameraExtrinsicMat.at<double>(1, 2) = -0.258980;
      in_CameraExtrinsicMat.at<double>(2, 0) = 0.960488;
      in_CameraExtrinsicMat.at<double>(2, 1) = -0.044897;
      in_CameraExtrinsicMat.at<double>(2, 2) = -0.274677;
      // original
      // CameraExtrinsicMat.at<double>(0,0)=-0.110303;    CameraExtrinsicMat.at<double>(0,1)=-0.217173;
      // CameraExtrinsicMat.at<double>(0,2)=-0.969881;
      // CameraExtrinsicMat.at<double>(1,0)=-0.0163458;    CameraExtrinsicMat.at<double>(1,1)=0.9761;
      // CameraExtrinsicMat.at<double>(1,2)=-0.216707;
      // CameraExtrinsicMat.at<double>(2,0)=0.993764;    CameraExtrinsicMat.at<double>(2,1)=-0.00805004;
      // CameraExtrinsicMat.at<double>(2,2)=-0.111217;

      in_CameraExtrinsicMat_tr.at<double>(0) = -1.003;
      in_CameraExtrinsicMat_tr.at<double>(1) = -1.21877;
      in_CameraExtrinsicMat_tr.at<double>(2) = -0.491312;

      in_DistCoeff_.at<double>(0) = -0.206623;
      in_DistCoeff_.at<double>(1) = -0.511431;
      in_DistCoeff_.at<double>(2) = -0.00909592;
      in_DistCoeff_.at<double>(3) = 5.11685e-05;
      in_DistCoeff_.at<double>(4) = 6.7519;

      in_CameraMat_.at<double>(0, 0) = 3537.66;
      in_CameraMat_.at<double>(0, 1) = 0;
      in_CameraMat_.at<double>(0, 2) = 1043.23;
      in_CameraMat_.at<double>(1, 0) = 0;
      in_CameraMat_.at<double>(1, 1) = 3576.59;
      in_CameraMat_.at<double>(1, 2) = 419.304;
      in_CameraMat_.at<double>(2, 0) = 0;
      in_CameraMat_.at<double>(2, 1) = 0;
      in_CameraMat_.at<double>(2, 2) = 1;

      alig_CameraExtrinsicMat = in_CameraExtrinsicMat;
      alig_CameraExtrinsicMat_tr = in_CameraExtrinsicMat_tr;
      alig_DistCoeff_ = in_DistCoeff_;
      alig_CameraMat_ = in_CameraMat_;

      break;
    default:
      cosfi_min = -99999;
      cosfi_max = 99999;

      costhta_min = -99999;
      costhta_max = 99999;
  }
}

double* Alignment::value_distance_array(pcl::PointCloud<pcl::PointXYZ> release_cloud, double image_x, double image_y,
                                        int num)
{
  static double r[3];

  // cv::Mat alig_CameraExtrinsicMat= cv::Mat::zeros(3,3,CV_64F);
  // cv::Mat alig_DistCoeff_= cv::Mat::zeros(1,5,CV_64F);
  // cv::Mat alig_CameraExtrinsicMat_tr= cv::Mat::zeros(1,3,CV_64F);
  // cv::Mat alig_CameraMat_= cv::Mat::zeros(3,3,CV_64F);

  bool search_value = false;
  bool range_search_value = false;

  double costhta_max, costhta_min, cosfi_max, cosfi_min;

  camera_alignment_module(num, cosfi_min, cosfi_max, costhta_min, costhta_max);

  cv::Mat invR = alig_CameraExtrinsicMat(cv::Rect(0, 0, 3, 3)).t();
  // cv::Mat invR = alig_CameraExtrinsicMat(cv::Rect(0,0,3,3)).inv();//same as
  // alig_CameraExtrinsicMat(cv::Rect(0,0,3,3)).t()

  cv::Mat invT = -invR * alig_CameraExtrinsicMat_tr.t();

  // invT = -invR*(invT).t();
  cv::Mat invR_T = invR.t();
  cv::Mat invT_T = invT.t();

  int tmp_min_px = 1920;
  int tmp_min_py = 1208;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud = release_cloud.makeShared();

  // cv::Mat M_LIDAR_2_MID = cv::Mat::zeros(1208, 1920, CV_32FC3); //for spatial alignment
  double size_cloud = cloud->size();
  for (int i = 0; i < cloud->size(); i++)
  {
    double dist_raw = sqrt(cloud->points[i].x * cloud->points[i].x + cloud->points[i].y * cloud->points[i].y +
                           cloud->points[i].z * cloud->points[i].z);

    double costhta = acos(cloud->points[i].z / dist_raw) / 2 / PI * 360;
    double cosfi = atan2(cloud->points[i].y, cloud->points[i].x) / 2 / PI * 360;

    if (cosfi < cosfi_max && cosfi > cosfi_min && costhta < costhta_max && costhta > costhta_min)
    {
      // This calibration tool assumes that the Velodyne is installed with the default order of axes for the Velodyne
      // sensor.
      // X axis points to the front
      // Y axis points to the left
      // Z axis points upwards

      cv::Mat point(1, 3, CV_64F);
      point.at<double>(0) = (double)cloud->points[i].x;
      point.at<double>(1) = (double)cloud->points[i].y;
      point.at<double>(2) = (double)cloud->points[i].z;

      point = point * invR_T + invT_T;

      double dist = (double)(point.at<double>(0) * point.at<double>(0)) +
                    (double)(point.at<double>(1) * point.at<double>(1)) +
                    (double)(point.at<double>(2) * point.at<double>(2));
      dist = sqrt(dist);

      double tmpx = point.at<double>(0) / point.at<double>(2);
      double tmpy = point.at<double>(1) / point.at<double>(2);
      double r2 = tmpx * tmpx + tmpy * tmpy;
      double tmpdist = 1 + alig_DistCoeff_.at<double>(0) * r2 + alig_DistCoeff_.at<double>(1) * r2 * r2 +
                       alig_DistCoeff_.at<double>(4) * r2 * r2 * r2;
      cv::Point2d imagepoint;

      imagepoint.x = tmpx * tmpdist + 2 * alig_DistCoeff_.at<double>(2) * tmpx * tmpy +
                     alig_DistCoeff_.at<double>(3) * (r2 + 2 * tmpx * tmpx);
      imagepoint.y = tmpy * tmpdist + alig_DistCoeff_.at<double>(2) * (r2 + 2 * tmpy * tmpy) +
                     2 * alig_DistCoeff_.at<double>(3) * tmpx * tmpy;

      imagepoint.x = alig_CameraMat_.at<double>(0, 0) * imagepoint.x + alig_CameraMat_.at<double>(0, 2);
      imagepoint.y = alig_CameraMat_.at<double>(1, 1) * imagepoint.y + alig_CameraMat_.at<double>(1, 2);

      if (imagepoint.x > 0 && imagepoint.y > 0)
      {
        int px = int(imagepoint.x + 0.5);
        int py = int(imagepoint.y + 0.5);

        double range_x_min, range_x_max, range_y_min, range_y_max;
        if (num == 7 || num == 9)
        {
          range_x_min = (image_x > 80) ? (image_x - 80) : 0;
          range_x_max = ((image_x + 80) < 1920) ? (image_x + 80) : 1920;

          range_y_min = (image_y > 10) ? (image_y - 10) : 0;
          range_y_max = ((image_y + 10) < 1208) ? (image_y + 10) : 1208;
        }
        else
        {
          range_x_min = (image_x > 10) ? (image_x - 10) : 0;
          range_x_max = ((image_x + 10) < 1920) ? (image_x + 10) : 1920;

          range_y_min = (image_y > 65) ? (image_y - 65) : 0;
          range_y_max = ((image_y + 65) < 1208) ? (image_y + 65) : 1208;
        }

        if (px == image_x && py == image_y)
        {
          r[0] = (double)cloud->points[i].x;
          r[1] = (double)cloud->points[i].y;
          r[2] = (double)cloud->points[i].z;

          search_value = true;
          return r;
        }
        else
        {
          if (px > range_x_min && px < range_x_max && search_value == false)
          {
            if (py > range_y_min && py < range_y_max)
            {
              double now = sqrt((px - image_x) * (px - image_x) + (py - image_y) * (py - image_y));
              double pre = sqrt((tmp_min_px - image_x) * (tmp_min_px - image_x) +
                                (tmp_min_py - image_y) * (tmp_min_py - image_y));
              if (now < pre)
              {
                r[0] = (double)cloud->points[i].x;
                r[1] = (double)cloud->points[i].y;
                r[2] = (double)cloud->points[i].z;
                range_search_value = true;
              }
            }
          }
        }
      }
    }
  }

  if (search_value == false)
  {
    r[0] = 0;
    r[1] = 0;
    r[2] = 0;
  }
  return r;
}
double* Alignment::value_distance_array_2pixel(pcl::PointCloud<pcl::PointXYZ> release_cloud, double image_x,
                                               double image_y, double image_x2, double image_y2, int num)
{
  static double r[6];

  // cv::Mat alig_CameraExtrinsicMat= cv::Mat::zeros(3,3,CV_64F);
  // cv::Mat alig_DistCoeff_= cv::Mat::zeros(1,5,CV_64F);
  // cv::Mat alig_CameraExtrinsicMat_tr= cv::Mat::zeros(1,3,CV_64F);
  // cv::Mat alig_CameraMat_= cv::Mat::zeros(3,3,CV_64F);

  bool search_value_pixel_1 = false;
  bool search_value_pixel_1_range = false;

  bool search_value_pixel_2 = false;
  bool search_value_pixel_2_range = false;

  double costhta_max, costhta_min, cosfi_max, cosfi_min;

  camera_alignment_module(num, cosfi_min, cosfi_max, costhta_min, costhta_max);

  cv::Mat invR = alig_CameraExtrinsicMat(cv::Rect(0, 0, 3, 3)).t();
  // cv::Mat invR = alig_CameraExtrinsicMat(cv::Rect(0,0,3,3)).inv();//same as
  // alig_CameraExtrinsicMat(cv::Rect(0,0,3,3)).t()

  cv::Mat invT = -invR * alig_CameraExtrinsicMat_tr.t();

  // invT = -invR*(invT).t();
  cv::Mat invR_T = invR.t();
  cv::Mat invT_T = invT.t();

  double count = 0;

  int tmp_min_px = 1920;
  int tmp_min_py = 1208;

  int tmp_min_px2 = 1920;
  int tmp_min_py2 = 1208;

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  cloud = release_cloud.makeShared();

  // cv::Mat M_LIDAR_2_MID = cv::Mat::zeros(1208, 1920, CV_32FC3); //for spatial alignment
  double size_cloud = cloud->size();
  for (int i = 0; i < cloud->size(); i++)
  {
    double dist_raw = sqrt(cloud->points[i].x * cloud->points[i].x + cloud->points[i].y * cloud->points[i].y +
                           cloud->points[i].z * cloud->points[i].z);

    double costhta = acos(cloud->points[i].z / dist_raw) / 2 / PI * 360;
    double cosfi = atan2(cloud->points[i].y, cloud->points[i].x) / 2 / PI * 360;

    if (cloud->points[i].x == 0)
    {
      if (cloud->points[i].y > 0)
      {
        cosfi = 90;
      }
      else if (cloud->points[i].y < 0)
      {
        cosfi = -90;
      }
      else
      {
        cosfi = -10000;
      }
    }
    if (cosfi < cosfi_max && cosfi > cosfi_min && costhta < costhta_max && costhta > costhta_min)
    {
      count++;

      // This calibration tool assumes that the Velodyne is installed with the default order of axes for the Velodyne
      // sensor.
      // X axis points to the front
      // Y axis points to the left
      // Z axis points upwards

      cv::Mat point(1, 3, CV_64F);
      point.at<double>(0) = (double)cloud->points[i].x;
      point.at<double>(1) = (double)cloud->points[i].y;
      point.at<double>(2) = (double)cloud->points[i].z;

      point = point * invR_T + invT_T;

      double dist = (double)(point.at<double>(0) * point.at<double>(0)) +
                    (double)(point.at<double>(1) * point.at<double>(1)) +
                    (double)(point.at<double>(2) * point.at<double>(2));
      dist = sqrt(dist);

      double tmpx = point.at<double>(0) / point.at<double>(2);
      double tmpy = point.at<double>(1) / point.at<double>(2);
      double r2 = tmpx * tmpx + tmpy * tmpy;
      double tmpdist = 1 + alig_DistCoeff_.at<double>(0) * r2 + alig_DistCoeff_.at<double>(1) * r2 * r2 +
                       alig_DistCoeff_.at<double>(4) * r2 * r2 * r2;
      cv::Point2d imagepoint;

      imagepoint.x = tmpx * tmpdist + 2 * alig_DistCoeff_.at<double>(2) * tmpx * tmpy +
                     alig_DistCoeff_.at<double>(3) * (r2 + 2 * tmpx * tmpx);
      imagepoint.y = tmpy * tmpdist + alig_DistCoeff_.at<double>(2) * (r2 + 2 * tmpy * tmpy) +
                     2 * alig_DistCoeff_.at<double>(3) * tmpx * tmpy;

      imagepoint.x = alig_CameraMat_.at<double>(0, 0) * imagepoint.x + alig_CameraMat_.at<double>(0, 2);
      imagepoint.y = alig_CameraMat_.at<double>(1, 1) * imagepoint.y + alig_CameraMat_.at<double>(1, 2);

      if (imagepoint.x > 0 && imagepoint.y > 0)
      {
        int px = int(imagepoint.x + 0.5);
        int py = int(imagepoint.y + 0.5);

        double range_x_min, range_x_max, range_y_min, range_y_max;

        double range_x2_min, range_x2_max, range_y2_min, range_y2_max;

        if (num == 7 || num == 9)
        {
          // point1
          range_x_min = (image_x > 80) ? (image_x - 80) : 0;
          range_x_max = ((image_x + 80) < 1920) ? (image_x + 80) : 1920;

          range_y_min = (image_y > 10) ? (image_y - 10) : 0;
          range_y_max = ((image_y + 10) < 1208) ? (image_y + 10) : 1208;

          // point2
          range_x2_min = (image_x2 > 80) ? (image_x2 - 80) : 0;
          range_x2_max = ((image_x2 + 80) < 1920) ? (image_x2 + 80) : 1920;

          range_y2_min = (image_y2 > 10) ? (image_y2 - 10) : 0;
          range_y2_max = ((image_y2 + 10) < 1208) ? (image_y2 + 10) : 1208;
        }
        else
        {
          // point1
          range_x_min = (image_x > 10) ? (image_x - 10) : 0;
          range_x_max = ((image_x + 10) < 1920) ? (image_x + 10) : 1920;

          range_y_min = (image_y > 65) ? (image_y - 65) : 0;
          range_y_max = ((image_y + 65) < 1208) ? (image_y + 65) : 1208;

          // point2
          range_x2_min = (image_x2 > 10) ? (image_x2 - 10) : 0;
          range_x2_max = ((image_x2 + 10) < 1920) ? (image_x2 + 10) : 1920;

          range_y2_min = (image_y2 > 65) ? (image_y2 - 65) : 0;
          range_y2_max = ((image_y2 + 65) < 1208) ? (image_y2 + 65) : 1208;
        }

        if (px == image_x && py == image_y)
        {
          r[0] = (double)cloud->points[i].x;
          r[1] = (double)cloud->points[i].y;
          r[2] = (double)cloud->points[i].z;

          search_value_pixel_1 = true;
        }
        else
        {
          if (px > range_x_min && px < range_x_max && search_value_pixel_1 == false)
          {
            if (py > range_y_min && py < range_y_max)
            {
              double dis_pixel = sqrt((px - image_x) * (px - image_x) + (py - image_y) * (py - image_y));
              double temp_dis_pixel = sqrt((tmp_min_px - image_x) * (tmp_min_px - image_x) +
                                           (tmp_min_py - image_y) * (tmp_min_py - image_y));

              if (dis_pixel < temp_dis_pixel)
              {
                r[0] = (double)cloud->points[i].x;
                r[1] = (double)cloud->points[i].y;
                r[2] = (double)cloud->points[i].z;

                search_value_pixel_1_range = true;
                tmp_min_px = px;
                tmp_min_py = py;
              }
            }
          }
        }

        if (px == image_x2 && py == image_y2)
        {
          r[3] = (double)cloud->points[i].x;
          r[4] = (double)cloud->points[i].y;
          r[5] = (double)cloud->points[i].z;

          search_value_pixel_2 = true;
        }
        else
        {
          if (px > range_x2_min && px < range_x2_max && search_value_pixel_2 == false)
          {
            if (py > range_y2_min && py < range_y2_max)
            {
              double dis_pixel_2 = sqrt((px - image_x2) * (px - image_x2) + (py - image_y2) * (py - image_y2));
              double temp_dis_pixel_2 = sqrt((tmp_min_px2 - image_x2) * (tmp_min_px2 - image_x2) +
                                             (tmp_min_py2 - image_y2) * (tmp_min_py2 - image_y2));

              if (dis_pixel_2 < temp_dis_pixel_2)
              {
                r[3] = (double)cloud->points[i].x;
                r[4] = (double)cloud->points[i].y;
                r[5] = (double)cloud->points[i].z;

                search_value_pixel_2_range = true;
                tmp_min_px2 = px;
                tmp_min_py2 = py;
              }
            }
          }
        }
      }
    }
  }

  cout << "count:" << count << endl;

  if (search_value_pixel_1 == false && search_value_pixel_1_range == false)
  {
    r[0] = 0;
    r[1] = 0;
    r[2] = 0;
  }
  if (search_value_pixel_2 == false && search_value_pixel_2_range == false)
  {
    r[3] = 0;
    r[4] = 0;
    r[5] = 0;
  }
  return r;
}
// double* Alignment::3d_to_get_2d(double x,double y,double z,int num)
// {
//   static double r[2];

//   // cv::Mat alig_CameraExtrinsicMat= cv::Mat::zeros(3,3,CV_64F);
//   // cv::Mat alig_DistCoeff_= cv::Mat::zeros(1,5,CV_64F);
//   // cv::Mat alig_CameraExtrinsicMat_tr= cv::Mat::zeros(1,3,CV_64F);
//   // cv::Mat alig_CameraMat_= cv::Mat::zeros(3,3,CV_64F);

//   bool search_value=false;

//   double costhta_max,costhta_min,cosfi_max,cosfi_min;

//   camera_alignment_module(num,cosfi_min,cosfi_max,costhta_min,costhta_max);

//   cv::Mat invR = alig_CameraExtrinsicMat(cv::Rect(0,0,3,3)).t();
//   //cv::Mat invR = alig_CameraExtrinsicMat(cv::Rect(0,0,3,3)).inv();//same as
//   alig_CameraExtrinsicMat(cv::Rect(0,0,3,3)).t()

// 	cv::Mat invT =-invR*alig_CameraExtrinsicMat_tr.t();

//   //invT = -invR*(invT).t();
//    cv::Mat invR_T = invR.t();
//    cv::Mat invT_T = invT.t();

//     cv::Mat point(1, 3, CV_64F);
// 		point.at<double>(0) = x;
// 		point.at<double>(1) = y;
// 		point.at<double>(2) = z;

//     point = point * invR_T + invT_T;

//     double tmpx = point.at<double>(0) / point.at<double>(2);
// 		double tmpy = point.at<double>(1)/point.at<double>(2);
// 		double r2 = tmpx * tmpx + tmpy * tmpy;
// 		double tmpdist = 1 + alig_DistCoeff_.at<double>(0) * r2
// 			                 + alig_DistCoeff_.at<double>(1) * r2 * r2
//                        + alig_DistCoeff_.at<double>(4) * r2 * r2 * r2;
//     cv::Point2d imagepoint;

//         imagepoint.x = tmpx * tmpdist
//           + 2 * alig_DistCoeff_.at<double>(2) * tmpx * tmpy
//           + alig_DistCoeff_.at<double>(3) * (r2 + 2 * tmpx * tmpx);
//         imagepoint.y = tmpy * tmpdist
//           + alig_DistCoeff_.at<double>(2) * (r2 + 2 * tmpy * tmpy)
//           + 2 * alig_DistCoeff_.at<double>(3) * tmpx * tmpy;

//     imagepoint.x = alig_CameraMat_.at<double>(0,0) * imagepoint.x + alig_CameraMat_.at<double>(0,2);
//     imagepoint.y = alig_CameraMat_.at<double>(1,1) * imagepoint.y + alig_CameraMat_.at<double>(1,2);

//     if(imagepoint.x>0&&imagepoint.y>0){
//       int px = int(imagepoint.x + 0.5);
//       int py = int(imagepoint.y + 0.5);
//       r[0]=px;
//       r[1]=py;
//     }else{
//       r[0]=-9999;
//       r[1]=-9999;
//     }

//   return r;
// }
