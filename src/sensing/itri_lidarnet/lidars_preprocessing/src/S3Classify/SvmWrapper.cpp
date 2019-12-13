#include "SvmWrapper.h"

SvmWrapper::SvmWrapper ()
{
}

SvmWrapper::~SvmWrapper ()
{
}

void
SvmWrapper::initialize (string input_name)
{
  type_name = input_name;
  pcl::SVMTrain SVMtrain;               //--> our trainer, to be used for store training data or for a new training procedure
  pcl::SVMModel SVMmodel;               //--> classifier model, this is automatically generated after the training or loaded for the classification
  vector<pcl::SVMData> training_set;    //--> the training set is a vector of data

  // load the train data file
  if (SVMtrain.loadProblem ( ("train_" + type_name + ".dat").c_str ()))
  {
    SVMtrain.setDebugMode (false);

    // check the training set loaded
    SVMtrain.adaptProbToInput ();
    training_set = SVMtrain.getInputTrainingSet ();
    cout << "[SVM] " + type_name + " Training set size = " << training_set.size () << endl;

    // configure parameters
    pcl::SVMParam SVMparam;     //--> our own configuration parameters
    SVMparam.kernel_type = RBF;
    SVMparam.shrinking = 1;
    SVMparam.gamma = 5;
    SVMparam.C = 10;
    SVMparam.probability = 1;  //0 off, 1 on. Using probability parameter will always results in high number of unclassified sample

    SVMtrain.setParameters (SVMparam);  // set the parameters for the trainer

    // train the classifier
    if (SVMtrain.trainClassifier ())
    {
      // check the model for the classifier
      SVMmodel = SVMtrain.getClassifierModel ();

      cout << "[SVM] Model parameters summary :" << endl;
      if ( (SVMparam.probability ? true : false))
      {
        cout << "[SVM]   Probability support and active --> " << "ProbA = " << *SVMmodel.probA << "  " << "ProbB = " << *SVMmodel.probB << endl;
      }
      else
      {
        std::cout << "[SVM]   Probability support!" << endl;
      }

      cout << "[SVM]   the number of support vectors " << SVMmodel.l << endl;
      cout << "[SVM]   number of classes " << SVMmodel.nr_class << endl;
      cout << "[SVM]   sv_coef " << * (*SVMmodel.sv_coef) << endl;
      cout << "[SVM]   Rho " << *SVMmodel.rho << endl;
      cout << "[SVM]   label " << *SVMmodel.label << endl;
      cout << "[SVM]   nSV " << *SVMmodel.nSV << endl;

      // save the generated model
      if (SVMtrain.saveClassifierModel ( ("model_" + type_name + ".dat").c_str ()))
      {
        cout << "[SVM] output model file OK" << endl;
      }
      else
      {
        cout << "[SVM] error! output Model file failed" << endl;
      }

      // save the training set
      /*      if (SVMtrain.saveTrainingSet ("train_out.dat"))
       {
       cout << "[SVM] output training file OK" << endl;
       }
       else
       {
       cout << "[SVM] error! output training file failed" << endl;
       }*/

      // try to export some of the training data from the std::vector<dataset>
      /*      pcl::SVMData my_svm_dataA;  //--> dataset A
       if (training_set.size () > 1)
       {
       my_svm_dataA = training_set.at (0);
       for (size_t i = 0; i < my_svm_dataA.SV.size (); i++)
       {
       cout << my_svm_dataA.label << my_svm_dataA.SV.at (i).idx << my_svm_dataA.SV.at (i).value << endl;
       }
       }*/

    }
    else
    {
      cout << "[SVM] error! train data failed" << endl;
    }
  }
  else
  {
    cout << "[SVM] error! can't load training file" << endl;
  }

}

bool
SvmWrapper::calculate (CLUSTER_INFO *single_cluster_info)
{
  if (!is_loaded)
  {
    if (SVMclassify.loadClassifierModel ( ("model_" + type_name + ".dat").c_str ()))
    {
      std::cout << "[SVM] model has been successfully loaded \n";
    }
    else
    {
      std::cout << "[SVM] model can not be loaded \n";
    }
    is_loaded = true;
  }

  string parameter;
  parameter.append ("1 ");
  parameter.append ("1:" + to_string (single_cluster_info->dx) + " ");
  parameter.append ("2:" + to_string (single_cluster_info->dy) + " ");
  parameter.append ("3:" + to_string (single_cluster_info->dz) + " ");
  parameter.append ("4:" + to_string ((float) single_cluster_info->covariance (0, 0)) + " ");
  parameter.append ("5:" + to_string ((float) single_cluster_info->covariance (0, 1)) + " ");
  parameter.append ("6:" + to_string ((float) single_cluster_info->covariance (0, 2)) + " ");
  parameter.append ("7:" + to_string ((float) single_cluster_info->covariance (1, 0)) + " ");
  parameter.append ("8:" + to_string ((float) single_cluster_info->covariance (1, 1)) + " ");
  parameter.append ("9:" + to_string ((float) single_cluster_info->covariance (1, 2)) + " ");
  parameter.append ("10:" + to_string ((float) single_cluster_info->covariance (2, 0)) + " ");
  parameter.append ("11:" + to_string ((float) single_cluster_info->covariance (2, 1)) + " ");
  parameter.append ("12:" + to_string ((float) single_cluster_info->covariance (2, 2)) + " ");
  //------------------------------------update------------------------------------------------
  parameter.append ("13:" + to_string (single_cluster_info->hull_vol) + " ");
  parameter.append ("14:" + to_string (single_cluster_info->GRSD21[0]) + " ");
  parameter.append ("15:" + to_string (single_cluster_info->GRSD21[1]) + " ");
  parameter.append ("16:" + to_string (single_cluster_info->GRSD21[2]) + " ");
  parameter.append ("17:" + to_string (single_cluster_info->GRSD21[3]) + " ");
  parameter.append ("18:" + to_string (single_cluster_info->GRSD21[4]) + " ");
  parameter.append ("19:" + to_string (single_cluster_info->GRSD21[5]) + " ");
  parameter.append ("20:" + to_string (single_cluster_info->GRSD21[6]) + " ");
  parameter.append ("21:" + to_string (single_cluster_info->GRSD21[7]) + " ");
  parameter.append ("22:" + to_string (single_cluster_info->GRSD21[8]) + " ");
  parameter.append ("23:" + to_string (single_cluster_info->GRSD21[9]) + " ");
  parameter.append ("24:" + to_string (single_cluster_info->GRSD21[10]) + " ");
  parameter.append ("25:" + to_string (single_cluster_info->GRSD21[11]) + " ");
  parameter.append ("26:" + to_string (single_cluster_info->GRSD21[12]) + " ");
  parameter.append ("27:" + to_string (single_cluster_info->GRSD21[13]) + " ");
  parameter.append ("28:" + to_string (single_cluster_info->GRSD21[14]) + " ");
  parameter.append ("29:" + to_string (single_cluster_info->GRSD21[15]) + " ");
  parameter.append ("30:" + to_string (single_cluster_info->GRSD21[16]) + " ");
  parameter.append ("31:" + to_string (single_cluster_info->GRSD21[17]) + " ");
  parameter.append ("32:" + to_string (single_cluster_info->GRSD21[18]) + " ");
  parameter.append ("33:" + to_string (single_cluster_info->GRSD21[19]) + " ");
  parameter.append ("34:" + to_string (single_cluster_info->GRSD21[20]) + " ");
  parameter.append ("\n");

  if (SVMclassify.loadClassProblemITRI (parameter))
  {
    //test data have been successfully loaded;
    pcl::SVMParam SVMparam = SVMclassify.getParameters ();
    SVMclassify.setProbabilityEstimates ( (SVMparam.probability ? false : true));

    if (SVMclassify.classification ())
    {
      // classification done
      vector<vector<double> > classification_result;

      if (SVMclassify.hasLabelledTrainingSet ())
      {
        // Loaded dataset has labels, the classification test will run

        if (SVMclassify.classificationTest ())
        {
          SVMclassify.getClassificationResult (classification_result);

          // run the classification and return the number of positive/negative samples

          for (size_t i = 0; i < classification_result.size (); i++)
          {
            if (classification_result.at (i).size () == 3)
            {
              single_cluster_info->confidence = classification_result.at (i).at (1) * 100;
              if (classification_result.at (i).at (0) == 1)
                return true;
              else
                return false;
            }
            else
            {
              if (classification_result.at (i).at (0) == 1)
              {
                return true;  //positive
              }
              else if (classification_result.at (i).at (0) == -1)
              {
                return false;  //negative
              }
              else
              {
                return false;  //unclassified
              }
            }
          }

          //pcl::SVMtestReport svm_test_report = SVMclassify.getClassificationTestReport ();
          //pcl::console::print_value (" - Accuracy (classification) = %g%% (%d/%d)\n", svm_test_report.accuracy, svm_test_report.correctPredictionsIdx, svm_test_report.totalSamples);
        }
        else
        {
          cout << "[SVM] error! Classification test NOT SUCCESS " << endl;
        }
      }
      else
      {
        cout << "[SVM] error! Loaded dataset has NO labels, the classification test cannot be executed" << endl;
      }

    }
    else
    {
      cout << "[SVM] error! classification ERROR" << endl;
    }
  }
  else
  {
    cout << "[SVM] test data can not be loaded \n";
  }
  return false;
}

PointCloud<PointXYZ>::Ptr SvmWrapper::cloud_training (new PointCloud<PointXYZ>);
string SvmWrapper::single_label;

void
SvmWrapper::labelingTool (CLUSTER_INFO *single_cluster_info,
                          boost::shared_ptr<pcl::visualization::PCLVisualizer> input_viewer,
                          int *input_viewID)
{

  // training format = <target> <feature>:<value> <feature>:<value> ... <feature>:<value> # <info>
  // format example  = -1 1:0.43 3:0.12 9284:0.2 # abcdef

  single_label.append ("1:" + to_string (single_cluster_info->dx) + " ");
  single_label.append ("2:" + to_string (single_cluster_info->dy) + " ");
  single_label.append ("3:" + to_string (single_cluster_info->dz) + " ");
  single_label.append ("4:" + to_string ((float) single_cluster_info->covariance (0, 0)) + " ");
  single_label.append ("5:" + to_string ((float) single_cluster_info->covariance (0, 1)) + " ");
  single_label.append ("6:" + to_string ((float) single_cluster_info->covariance (0, 2)) + " ");
  single_label.append ("7:" + to_string ((float) single_cluster_info->covariance (1, 0)) + " ");
  single_label.append ("8:" + to_string ((float) single_cluster_info->covariance (1, 1)) + " ");
  single_label.append ("9:" + to_string ((float) single_cluster_info->covariance (1, 2)) + " ");
  single_label.append ("10:" + to_string ((float) single_cluster_info->covariance (2, 0)) + " ");
  single_label.append ("11:" + to_string ((float) single_cluster_info->covariance (2, 1)) + " ");
  single_label.append ("12:" + to_string ((float) single_cluster_info->covariance (2, 2)) + " ");
  //------------------------------------update------------------------------------------------
  single_label.append ("13:" + to_string (single_cluster_info->hull_vol) + " ");
  single_label.append ("14:" + to_string (single_cluster_info->GRSD21[0]) + " ");
  single_label.append ("15:" + to_string (single_cluster_info->GRSD21[1]) + " ");
  single_label.append ("16:" + to_string (single_cluster_info->GRSD21[2]) + " ");
  single_label.append ("17:" + to_string (single_cluster_info->GRSD21[3]) + " ");
  single_label.append ("18:" + to_string (single_cluster_info->GRSD21[4]) + " ");
  single_label.append ("19:" + to_string (single_cluster_info->GRSD21[5]) + " ");
  single_label.append ("20:" + to_string (single_cluster_info->GRSD21[6]) + " ");
  single_label.append ("21:" + to_string (single_cluster_info->GRSD21[7]) + " ");
  single_label.append ("22:" + to_string (single_cluster_info->GRSD21[8]) + " ");
  single_label.append ("23:" + to_string (single_cluster_info->GRSD21[9]) + " ");
  single_label.append ("24:" + to_string (single_cluster_info->GRSD21[10]) + " ");
  single_label.append ("25:" + to_string (single_cluster_info->GRSD21[11]) + " ");
  single_label.append ("26:" + to_string (single_cluster_info->GRSD21[12]) + " ");
  single_label.append ("27:" + to_string (single_cluster_info->GRSD21[13]) + " ");
  single_label.append ("28:" + to_string (single_cluster_info->GRSD21[14]) + " ");
  single_label.append ("29:" + to_string (single_cluster_info->GRSD21[15]) + " ");
  single_label.append ("30:" + to_string (single_cluster_info->GRSD21[16]) + " ");
  single_label.append ("31:" + to_string (single_cluster_info->GRSD21[17]) + " ");
  single_label.append ("32:" + to_string (single_cluster_info->GRSD21[18]) + " ");
  single_label.append ("33:" + to_string (single_cluster_info->GRSD21[19]) + " ");
  single_label.append ("34:" + to_string (single_cluster_info->GRSD21[20]) + " ");

  *cloud_training = single_cluster_info->cloud;

  static bool flag = false;
  if (!flag)
  {
    input_viewer->registerKeyboardCallback (SvmWrapper::keyboard_event_occurred, (void*) input_viewer.get ());
    flag = true;
  }

  input_viewer->addPointCloud<PointXYZ> (cloud_training, to_string (*input_viewID), 0);
  input_viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, to_string (*input_viewID), 0);

  while (cloud_training->size () > 0)
  {
    boost::this_thread::sleep (boost::posix_time::microseconds (10000));
    input_viewer->spinOnce ();
  }
  input_viewer->removePointCloud (to_string (*input_viewID));
}

void
SvmWrapper::keyboard_event_occurred (const pcl::visualization::KeyboardEvent &event,
                                     void* viewer_void)
{
  if ( (event.getKeySym () == "y" || event.getKeySym () == "n" || event.getKeySym () == "m") && event.keyDown () && cloud_training->size () > 0)
  {
    if (event.getKeySym () == "m")
    {
      cloud_training->clear ();
      single_label.clear ();
    }
    else
    {
      ofstream file_training ("train.dat", std::ios_base::app | std::ios_base::out);

      if (file_training.is_open ())
      {

        if (event.getKeySym () == "y")    // Positive
        {
          single_label = "1 " + single_label;
        }
        if (event.getKeySym () == "n")    // Negative
        {
          single_label = "-1 " + single_label;
        }

        cout << single_label << endl;
        file_training << single_label << endl;
        file_training.close ();
        cloud_training->clear ();
        single_label.clear ();
      }
    }
  }
}

