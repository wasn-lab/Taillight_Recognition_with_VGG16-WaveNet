#ifndef PCL_SVM_WRAPPER_H_
#define PCL_SVM_WRAPPER_H_

#include "../all_header.h"

#define Malloc(type, n) static_cast<type*>(malloc((n) * sizeof(type)))

namespace pcl
{
/*
 * \brief The structure stores the parameters for the classificationa nd
 * must be initialized and passed to the training method pcl::SVMTrain.
 *  \param svm_type {C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR}
 *  \param kernel_type {LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED}
 *  \param probability sets the probability estimates
 */

struct SVMParam : svm_parameter
{
  SVMParam()
  {
    svm_type = C_SVC;   // C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR
    kernel_type = RBF;  // LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED
    degree = 3;         // for poly
    gamma = 0;          // 1/num_features {for poly/rbf/sigmoid}
    coef0 = 0;          //  for poly/sigmoid

    nu = 0.5;          // for NU_SVC, ONE_CLASS, and NU_SVR
    cache_size = 100;  // in MB
    C = 1;             // for C_SVC, EPSILON_SVR and NU_SVR
    eps = 1e-3;        // stopping criteria
    p = 0.1;           // for EPSILON_SVR
    shrinking = 0;     // use the shrinking heuristics
    probability = 0;   // do probability estimates

    nr_weight = 0;        // for C_SVC
    weight_label = NULL;  // for C_SVC
    weight = NULL;        // for C_SVC
  }
};

/*  * \brief The structure initialize a model crated by the SVM (Support Vector Machines) classifier (pcl::SVMTrain)*/

struct SVMModel : svm_model
{
  SVMModel()
  {
    l = 0;
    probA = NULL;
    probB = NULL;
  }

  /*
   * \brief check for the validity of the model,
   *         l represents the number of support vectors,
   *         without SVs the model is not condiered as not valid
   *  \return true if SMV has a valid model
   */

  bool isValid()
  {
    if (this->l > 0)
    {
      return true;
    }
    else
    {
      return false;
    }
  }
};

/*  * \brief The structure initialize a single feature value for the classification using SVM (Support Vector
 * Machines).*/

struct SVMDataPoint
{
  int idx;      // It's the feature index. It has to be an integer number greater or equal to zero.
  float value;  // The value assigned to the correspondent feature.

  SVMDataPoint() : idx(-1), value(0)
  {
  }
};

/*
 * \brief The structure stores the features and the label of a single sample which has to be used
 * for the training or the classification of the SVM (Support Vector Machines).
 */

struct SVMData
{
  double label;                       // Pointer to the label value. It is a mandatory to train the classifier.
  std::vector<pcl::SVMDataPoint> SV;  // Vector of features for the specific sample.

  SVMData() : label(std::numeric_limits<double>::signaling_NaN())
  {
  }
};

/*
 * \brief The structure stores the test report of the SVM classifier.
 *  easy to use during the training/classification procedures
 */

struct SVMtestReport
{
  int correctPredictionsIdx;  // store the number of correct predictions after a classification test
  double MSE;                 // Mean squared error (regression) - only for NU_SVR and EPSILON_SVR
  double SCC;                 // Squared correlation coefficient (regression) - only for NU_SVR and EPSILON_SVR
  double accuracy;            // accuracy in percentage
  int totalSamples;           // number of tested samples

  SVMtestReport() : correctPredictionsIdx(0), MSE(0.0), SCC(0.0), accuracy(0.0), totalSamples(0)
  {
  }
};

/*  * \brief Base class for SVM SVM (Support Vector Machines).*/

class PCL_EXPORTS SVM
{
protected:
  std::vector<SVMData> training_set_;  // Basic training set
  svm_problem prob_;                   // contains the problem (vector of samples with their features)
  SVMModel model_;                     // model of the classifier
  svm_scaling scaling_;  // for the best model training, the input dataset is scaled and the scaling factors are stored
                         // here
  SVMParam param_;       // it stores the training parameters
  std::string class_name_;  // The SVM class name.

  svm_node* x_space_;  // ITRI

  char* line_;                  // buffer for line reading
  int max_line_len_;            // max line length in the input file
  bool labelled_training_set_;  // it stores whether the input set of samples is labelled
  /*      * \brief Set for output printings during classification.*/
  static void printNull(const char*){};

  /*      * \brief To read a line from the input file. Stored in "line_".*/
  char* readline(FILE* input);

  /*      * \brief Outputs an error in file reading.*/
  void exitInputError(int line_num)
  {
    fprintf(stderr, "Wrong input format at line %d\n", line_num);
    exit(1);
  }

  /*      * \brief Get a string representation of the name of this class.*/
  inline const std::string& getClassName() const
  {
    return (class_name_);
  }

  /*      * \brief Convert the input format (vector of SVMData) into a readable format for libSVM.*/
  void adaptInputToLibSVM(std::vector<SVMData> training_set, svm_problem& prob);

  /*      * \brief Convert the libSVM format (svm_problem) into a easier output format.*/
  void adaptLibSVMToInput(std::vector<SVMData>& training_set, svm_problem prob);

  /*      * \brief Load a problem from an extern file.*/
  bool loadProblem(const char* filename, svm_problem& prob);

  bool loadProblemITRI(string input, svm_problem& prob);

  /*      * \brief Save the raw problem in an extern file.*/
  bool saveProblem(const char* filename, bool labelled);

  /*      * \brief Save the problem (with normalized values) in an extern file.*/
  bool saveProblemNorm(const char* filename, svm_problem prob_, bool labelled);

public:
  /*      * \brief  Constructor.*/
  SVM()
    : training_set_()
    , prob_()
    , model_()
    , scaling_()
    , param_()
    , class_name_()
    , x_space_(0)
    , line_(NULL)
    , max_line_len_(10000)
    , labelled_training_set_(1)
  {
  }

  /*      * \brief Destructor.*/
  ~SVM()
  {
    svm_destroy_param(&param_);  // delete parameters

    if (scaling_.max > 0)
    {
      free(scaling_.obj);  // delete scaling factors
    }

    // delete the problem
    if (prob_.l > 0)
    {
      free(prob_.x);
      free(prob_.y);
    }
  }

  /*      * \brief Return the labels order from the classifier model.*/
  void getLabel(std::vector<int>& labels)
  {
    int nr_class = svm_get_nr_class(&model_);
    int* labels_ = static_cast<int*>(malloc(nr_class * sizeof(int)));
    svm_get_labels(&model_, labels_);

    for (int j = 0; j < nr_class; j++)
    {
      labels.push_back(labels_[j]);
    }

    free(labels_);
  };

  /*      * \brief Return the current training parameters.*/
  SVMParam getParameters()
  {
    return param_;
  }

  /*      * \brief Return the result of the training.*/
  SVMModel getClassifierModel()
  {
    return model_;
  }

  /*      * \brief Save the classifier model in an extern file (in svmlight format).
   \return false if fails.*/
  bool saveClassifierModel(const char* filename)
  {
    // exit if model has no data
    if (model_.l == 0)
    {
      fprintf(stderr, "model has no data to save in %s\n", filename);
      return 0;
    }

    if (svm_save_model(filename, &model_))
    {
      fprintf(stderr, "can't save model to file %s\n", filename);
      return 0;  // exit (1);
    }
    else
    {
      return 1;
    }
  };
};

/*
 * \brief SVM (Support Vector Machines) training class for the SVM machine learning.
 * It creates a model for the classifier from a labelled input dataset.
 * OPTIONAL: pcl::SVMParam has to be given as input to vary the default training method and parameters.
 */

class PCL_EXPORTS SVMTrain : public SVM
{
protected:
  using SVM::class_name_;
  using SVM::labelled_training_set_;
  using SVM::line_;
  using SVM::max_line_len_;
  using SVM::model_;
  using SVM::param_;
  using SVM::prob_;
  using SVM::scaling_;
  using SVM::training_set_;

  bool debug_;            // Set to 1 to see the training output
  int cross_validation_;  // Set too 1 for cross validating the classifier
  int nr_fold_;  // Number of folds to be used during cross validation. It indicates in how many parts is split the
                 // input training set.

  /*      * \brief To cross validate the classifier. It is automatic for probability estimate.*/
  void doCrossValidation();

  /*
   * \brief It extracts scaling factors from the input training_set.
   *  The scaling of the training_set is a mandatory for a good training of the classifier.
   */
  void scaleFactors(std::vector<SVMData> training_set, svm_scaling& scaling);

public:
  /*      * \brief Constructor.*/
  SVMTrain() : debug_(0), cross_validation_(0), nr_fold_(0)
  {
    class_name_ = "SVMTrain";
    svm_set_print_string_function(&printNull);  // Default to NULL to not print debugging info
    labelled_training_set_ = false;
    line_ = NULL;
    max_line_len_ = 10000;
  }

  /*      * \brief Destructor.*/
  ~SVMTrain()
  {
    if (model_.l > 0)
    {
      svm_free_model_content(&model_);
    }
  }

  /*      * \brief Change default training parameters (pcl::SVMParam).*/
  void setParameters(SVMParam param)
  {
    param_ = param;
  }

  /*      * \brief It adds/store the training set with labelled data.*/
  void setInputTrainingSet(std::vector<SVMData> training_set)
  {
    training_set_.insert(training_set_.end(), training_set.begin(), training_set.end());
  }

  /*      * \brief Return the current training set.*/
  std::vector<SVMData> getInputTrainingSet()
  {
    return training_set_;
  }

  /*      * \brief Reset the training set.*/
  void resetTrainingSet()
  {
    training_set_.clear();
  }

  /*      * \brief Start the training of the SVM classifier.
   \return false if fails.*/
  bool trainClassifier();

  /*
   * \brief Read in a problem (in svmlight format).
   * \return false if fails.
   */
  bool loadProblem(const char* filename)
  {
    return SVM::loadProblem(filename, prob_);
  };

  /*      * \brief Adapt the input prob_ (in svmlight format) to training_set (as vector of SVMData) .*/
  void adaptProbToInput()
  {
    adaptLibSVMToInput(training_set_, prob_);
  };

  /*      * \brief Set to 1 for debugging info.*/
  void setDebugMode(bool in)
  {
    debug_ = in;

    if (in)
    {
      svm_set_print_string_function(NULL);
    }
    else
    {
      svm_set_print_string_function(&printNull);
    }
  };

  /*
   * \brief Save the raw training set in a file (in svmlight format).
   * \return false if fails.
   */
  bool saveTrainingSet(const char* filename)
  {
    return SVM::saveProblem(filename, 1);
  };

  /*
   * \brief Save the normalized training set in a file (in svmlight format).
   * \return false if fails.
   */
  bool saveNormTrainingSet(const char* filename)
  {
    return SVM::saveProblemNorm(filename, prob_, 1);
  };
};

/*
 * \brief SVM (Support Vector Machines) classification of a dataset.
 * It can be used both for testing a classifier model and for classify of new data.
 */

class PCL_EXPORTS SVMClassify : public SVM
{
protected:
  using SVM::labelled_training_set_;
  using SVM::line_;
  using SVM::model_;

  using SVM::x_space_;  // ITRI

  using SVM::class_name_;
  using SVM::max_line_len_;
  using SVM::param_;
  using SVM::prob_;
  using SVM::scaling_;
  using SVM::training_set_;

  bool model_extern_copied_;                      // Set to 0 if the model is loaded from an extern file.
  bool predict_probability_;                      // Set to 1 to predict probabilities.
  std::vector<std::vector<double> > prediction_;  // It stores the resulting prediction.

  SVMtestReport testClassReport_;  // test reports

  /*      * \brief It scales the input dataset using the model information.*/
  void scaleProblem(svm_problem& input, svm_scaling scaling);

public:
  /*      * \brief Constructor.*/
  SVMClassify() : model_extern_copied_(0), predict_probability_(0)
  {
    class_name_ = "SvmClassify";
    labelled_training_set_ = false;
    line_ = NULL;
    max_line_len_ = 10000;
    x_space_ = NULL;
  }

  /*      * \brief Destructor.*/
  ~SVMClassify()
  {
    if (!model_extern_copied_ && model_.l > 0)
    {
      svm_free_model_content(&model_);
    }
  }

  /*
   * \brief It adds/store the training set with labelled data.
   * \brief It adds/store the training set with/without labelled data.
   *
   * \note this will actually work also with not labelled data
   *       so the function should be setInputSet as it can be used
   *       for training of for classifying
   */

  void setInputTrainingSet(std::vector<SVMData> training_set)
  {
    assert(training_set.size() > 0);

    if (scaling_.max == 0)
    {
      // to be sure to have loaded the scaling
      PCL_ERROR("[pcl::%s::setInputTrainingSet] Classifier model not loaded!\n", getClassName().c_str());
      return;
    }

    // just a check if we have labels
    if (training_set.at(0).label <= 1 && training_set.at(0).label >= -1)
    {
      labelled_training_set_ = true;
      PCL_WARN("[pcl::%s::setInputTrainingSet] Training data has labels!\n", getClassName().c_str());
    }
    else
    {
      PCL_ERROR("[pcl::%s::setInputTrainingSet] Training data has NO labels!\n", getClassName().c_str());
    }

    training_set_.insert(training_set_.end(), training_set.begin(), training_set.end());
    SVM::adaptInputToLibSVM(training_set_, prob_);
  }

  /*      * \brief Return the current training set.*/
  std::vector<SVMData> getInputTrainingSet()
  {
    return training_set_;
  }

  /*      * \brief Reset the training set.*/
  void resetTrainingSet()
  {
    training_set_.clear();
  }

  /*
   * \brief Read in a classifier model (in svmlight format).
   * \return false if fails.
   */
  bool loadClassifierModel(const char* filename);

  /*      * \brief Get the result of the classification.*/
  void getClassificationResult(std::vector<std::vector<double> >& out)
  {
    out.clear();
    out.insert(out.begin(), prediction_.begin(), prediction_.end());

    free(line_);  // ITRI
    free(prob_.x);
    free(prob_.y);
    free(x_space_);
  }

  /*      * \brief Get the result of the classification test .*/
  SVMtestReport getClassificationTestReport()
  {
    return testClassReport_;
  }

  /*      * \brief Save the classification result in an extern file.*/
  void saveClassificationResult(const char* filename);

  /*      * \brief Set the classifier model.*/
  void setClassifierModel(SVMModel model)
  {
    // model (inner pointers are references)
    model_ = model;
    int i = 0;

    while (model_.scaling[i].index != -1)
    {
      i++;
    }

    scaling_.max = i;
    scaling_.obj = Malloc(struct svm_node, i + 1);
    scaling_.obj[i].index = -1;

    // Performing full scaling copy
    for (int j = 0; j < i; j++)
    {
      scaling_.obj[j] = model_.scaling[j];
    }

    model_extern_copied_ = 1;
  };

  /*
   * \brief Read in a raw classification problem (in svmlight format).
   *  The values are normalized using the classifier model information.
   * \return false if fails.
   */
  bool loadClassProblem(const char* filename)
  {
    assert(model_.l != 0);

    bool out = SVM::loadProblem(filename, prob_);
    SVM::adaptLibSVMToInput(training_set_, prob_);
    scaleProblem(prob_, scaling_);
    return out;
  };

  /*
   * \brief Read in a raw classification problem (in svmlight format).
   *  The values are normalized using the classifier model information.
   * \return false if fails.
   */
  bool loadClassProblemITRI(string input)
  {
    assert(model_.l != 0);

    bool out = SVM::loadProblemITRI(input, prob_);
    SVM::adaptLibSVMToInput(training_set_, prob_);
    scaleProblem(prob_, scaling_);
    return out;
  };

  /*
   * \brief Read in a normalized classification problem (in svmlight format).
   * The data are kept whitout normalizing.
   * \return false if fails.
   */
  bool loadNormClassProblem(const char* filename)
  {
    bool out = SVM::loadProblem(filename, prob_);
    SVM::adaptLibSVMToInput(training_set_, prob_);
    return out;
  };

  /*
   * \brief Check if the training set is labelled.
   *  Usefull to perform the classificationTest
   * \return true if the dataset is labelled
   */
  bool hasLabelledTrainingSet()
  {
    return labelled_training_set_;
  }

  /*
   * \brief Set whether the classification has to be done with the probability estimate.
   * (the classifier model has to support it).
   */
  void setProbabilityEstimates(bool set)
  {
    predict_probability_ = set;
  };

  /*
   * \brief Start the classification on labelled input dataset. It returns the accuracy percentage.
   * To get the classification result, use getClassificationResult.
   * \return false if fails.
   */
  bool classificationTest();

  /*
   * \brief Start the classification on un-labelled input dataset.
   * To get the classification result, use getClassificationResult.
   * \return false if fails.
   */
  bool classification();

  /*      * \brief Start the classification on a single set.*/
  std::vector<double> classification(SVMData in);

  /*
   * \brief Save the raw classification problem in a file (in svmlight format).
   * \return false if fails.
   */
  bool saveClassProblem(const char* filename)
  {
    return SVM::saveProblem(filename, 0);
  };

  /*
   * \brief Save the normalized classification problem in a file (in svmlight format).
   * \return false if fails.
   */
  bool saveNormClassProblem(const char* filename)
  {
    return SVM::saveProblemNorm(filename, prob_, 0);
  };
};
}  // namespace pcl

#endif  // PCL_SVM_WRAPPER_H_
