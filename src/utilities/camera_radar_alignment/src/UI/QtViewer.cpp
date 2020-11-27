#include "QtViewer.h"

QtViewer::QtViewer(QWidget* parent) : QMainWindow(parent), ui(new Ui::QtViewer)
{
  ui->setupUi(this);
  this->setWindowTitle("QtViewer");

  // Set up the QVTK window
  // viewer.reset (new pcl::visualization::PCLVisualizer ("viewer", false));
  // ui->qvtkWidget->SetRenderWindow (viewer->getRenderWindow ());
  // viewer->setupInteractor (ui->qvtkWidget->GetInteractor (),
  // ui->qvtkWidget->GetRenderWindow ());
  ui->qvtkWidget->update();

  // UI function initialization
  // viewer->resetCamera ();
  ui->qvtkWidget->update();

  // Connect functions

  // ------TAB 1
  connect(ui->horizontalScrollBar0, SIGNAL(valueChanged(int)), this, SLOT(HorizontalScrollBar0(int)));
  connect(ui->horizontalScrollBar1, SIGNAL(valueChanged(int)), this, SLOT(HorizontalScrollBar1(int)));
  connect(ui->horizontalScrollBar2, SIGNAL(valueChanged(int)), this, SLOT(HorizontalScrollBar2(int)));
  connect(ui->horizontalScrollBar3, SIGNAL(valueChanged(int)), this, SLOT(HorizontalScrollBar3(int)));
  connect(ui->horizontalScrollBar4, SIGNAL(valueChanged(int)), this, SLOT(HorizontalScrollBar4(int)));
  connect(ui->horizontalScrollBar5, SIGNAL(valueChanged(int)), this, SLOT(HorizontalScrollBar5(int)));

  connect(ui->radioButton0, SIGNAL(clicked()), this, SLOT(RadioButton0()));
  connect(ui->radioButton1, SIGNAL(clicked()), this, SLOT(RadioButton1()));
  connect(ui->radioButton2, SIGNAL(clicked()), this, SLOT(RadioButton2()));

  connect(ui->checkBox1, SIGNAL(stateChanged(int)), this, SLOT(CheckBox1(int)));
  connect(ui->checkBox3, SIGNAL(stateChanged(int)), this, SLOT(CheckBox3(int)));

  connect(ui->lineEdit12, SIGNAL(textChanged(const QString&)), this, SLOT(LineEditor12(const QString&)));

  // ------TAB 2
  connect(ui->checkBox2, SIGNAL(stateChanged(int)), this, SLOT(CheckBox2(int)));
  connect(ui->lineEdit7, SIGNAL(textChanged(const QString&)), this, SLOT(LineEditor7(const QString&)));
  connect(ui->lineEdit13, SIGNAL(textChanged(const QString&)), this, SLOT(LineEditor13(const QString&)));
  connect(ui->lineEdit14, SIGNAL(textChanged(const QString&)), this, SLOT(LineEditor14(const QString&)));
  connect(ui->lineEdit15, SIGNAL(textChanged(const QString&)), this, SLOT(LineEditor15(const QString&)));
  connect(ui->lineEdit16, SIGNAL(textChanged(const QString&)), this, SLOT(LineEditor16(const QString&)));

  // ------TAB 3
  connect(ui->checkBox0, SIGNAL(stateChanged(int)), this, SLOT(CheckBox0(int)));
  connect(ui->lineEdit2, SIGNAL(textChanged(const QString&)), this, SLOT(LineEditor2(const QString&)));
  connect(ui->lineEdit3, SIGNAL(textChanged(const QString&)), this, SLOT(LineEditor3(const QString&)));
  connect(ui->lineEdit4, SIGNAL(textChanged(const QString&)), this, SLOT(LineEditor4(const QString&)));
  connect(ui->lineEdit5, SIGNAL(textChanged(const QString&)), this, SLOT(LineEditor5(const QString&)));
  connect(ui->lineEdit6, SIGNAL(textChanged(const QString&)), this, SLOT(LineEditor6(const QString&)));

  // ------TAB 4
  connect(ui->lineEdit0, SIGNAL(textChanged(const QString&)), this, SLOT(LineEditor0(const QString&)));
  connect(ui->lineEdit1, SIGNAL(textChanged(const QString&)), this, SLOT(LineEditor1(const QString&)));
  connect(ui->pushButton_random, SIGNAL(clicked()), this, SLOT(ButtonPressed0()));

  // ------TAB 5
  connect(ui->horizontalScrollBar6, SIGNAL(valueChanged(int)), this, SLOT(HorizontalScrollBar6(int)));
  connect(ui->horizontalScrollBar7, SIGNAL(valueChanged(int)), this, SLOT(HorizontalScrollBar7(int)));
  connect(ui->horizontalScrollBar8, SIGNAL(valueChanged(int)), this, SLOT(HorizontalScrollBar8(int)));
  connect(ui->horizontalScrollBar9, SIGNAL(valueChanged(int)), this, SLOT(HorizontalScrollBar9(int)));
  connect(ui->horizontalScrollBar10, SIGNAL(valueChanged(int)), this, SLOT(HorizontalScrollBar10(int)));
  connect(ui->horizontalScrollBar11, SIGNAL(valueChanged(int)), this, SLOT(HorizontalScrollBar11(int)));

  // Config file
  boost::property_tree::ini_parser::read_ini("config.ini", boost_ptree);

  // ------TAB 1
  ui->horizontalScrollBar0->setValue(boost_ptree.get<int>("UI_LATERAL_RANGE", 10));
  ui->horizontalScrollBar1->setValue(boost_ptree.get<double>("UI_UNIFORM_SAMPLING", 0.1));
  ui->horizontalScrollBar2->setValue(boost_ptree.get<double>("UI_DBSCAN_EPS", 1));
  ui->horizontalScrollBar3->setValue(boost_ptree.get<int>("UI_DBSCAN_MINPT", 1));
  ui->lineEdit12->setText(QString::fromStdString(boost_ptree.get<string>("SENSOR_TO_GROUND", "")));

  // ------TAB 2
  ui->lineEdit7->setText(QString::fromStdString(boost_ptree.get<string>("UI_LANE_INTENSITY_I", "")));
  ui->lineEdit13->setText(QString::fromStdString(boost_ptree.get<string>("UI_LANE_DETECTION_WITH_OUTER", "")));
  ui->lineEdit14->setText(QString::fromStdString(boost_ptree.get<string>("UI_LANE_DETECTION_WITH_INNER", "")));
  ui->lineEdit15->setText(QString::fromStdString(boost_ptree.get<string>("UI_LANE_DETECTION_LENGTH", "")));
  ui->lineEdit16->setText(QString::fromStdString(boost_ptree.get<string>("UI_LANE_IDEA_WIDTH", "")));

  // ------TAB 3
  ui->lineEdit2->setText(QString::fromStdString(boost_ptree.get<string>("UI_PARKING_INTENSITY_I", "")));
  ui->lineEdit4->setText(QString::fromStdString(boost_ptree.get<string>("UI_PARKING_SPACE_DEGREE", "")));
  ui->lineEdit5->setText(QString::fromStdString(boost_ptree.get<string>("UI_PARKING_SLOT_WIDTH", "")));
  ui->lineEdit6->setText(QString::fromStdString(boost_ptree.get<string>("UI_PARKING_SLOT_LENGTH", "")));

  // ------TAB 4
  ui->lineEdit0->setText(QString::fromStdString(boost_ptree.get<string>("UDP_IP", "192.168.0.1")));
  ui->lineEdit1->setText(QString::fromStdString(boost_ptree.get<string>("UDP_Port", "8888")));

  // Initial

  // ------TAB 1
  ui->radioButton0->setChecked(false);
  ui->radioButton1->setChecked(false);
  ui->radioButton2->setChecked(true);

  QDoubleValidator* validator = new QDoubleValidator(this);
  validator->setBottom(-9999.99);
  validator->setTop(9999.99);
  validator->setDecimals(2);
  validator->setNotation(QDoubleValidator::StandardNotation);

  ui->lineEdit2->setValidator(validator);
  ui->lineEdit3->setValidator(validator);
  ui->lineEdit4->setValidator(validator);
  ui->lineEdit5->setValidator(validator);
  ui->lineEdit6->setValidator(validator);
  ui->lineEdit7->setValidator(validator);
  ui->lineEdit12->setValidator(validator);
  ui->lineEdit13->setValidator(validator);
  ui->lineEdit14->setValidator(validator);
  ui->lineEdit15->setValidator(validator);
  ui->lineEdit16->setValidator(validator);
}

QtViewer::~QtViewer()
{
  delete ui;
}

//------------------------------------------------------------------ TAB 1
void QtViewer::HorizontalScrollBar0(int value)
{
  GlobalVariable::UI_PARA[6] = double(value) / 10;
}

void QtViewer::HorizontalScrollBar1(int value)
{
  if (value != 0)
  {
    GlobalVariable::UI_UNIFORM_SAMPLING = double(value) / 1000;
    boost_ptree.put("UI_UNIFORM_SAMPLING", value);
    boost::property_tree::write_ini("config.ini", boost_ptree);
  }
}

void QtViewer::HorizontalScrollBar2(int value)
{
  if (value != 0)
  {
    GlobalVariable::UI_DBSCAN_EPS = double(value) / 200;
    boost_ptree.put("UI_DBSCAN_EPS", value);
    boost::property_tree::write_ini("config.ini", boost_ptree);
  }
}

void QtViewer::HorizontalScrollBar3(int value)
{
  if (value != 0)
  {
    GlobalVariable::UI_DBSCAN_MINPT = (unsigned int)value;
    boost_ptree.put("UI_DBSCAN_MINPT", value);
    boost::property_tree::write_ini("config.ini", boost_ptree);
  }
}

void QtViewer::HorizontalScrollBar4(int value)
{
}

void QtViewer::HorizontalScrollBar5(int value)
{
}

void QtViewer::RadioButton0()
{
  GlobalVariable::UI_DBSCAN_SELECT = 0;
}

void QtViewer::RadioButton1()
{
  GlobalVariable::UI_DBSCAN_SELECT = 1;
}

void QtViewer::RadioButton2()
{
  GlobalVariable::UI_DBSCAN_SELECT = 2;
}

void QtViewer::CheckBox1(int value)
{
  if (value == 0)
    GlobalVariable::UI_ENABLE_TRAINING_TOOL = false;
  else if (value == 2)
    GlobalVariable::UI_ENABLE_TRAINING_TOOL = true;
}

void QtViewer::CheckBox3(int value)
{
  if (value == 0)
    GlobalVariable::UI_ENABLE_OBJECTS = false;
  else if (value == 2)
    GlobalVariable::UI_ENABLE_OBJECTS = true;
}

void QtViewer::LineEditor12(const QString& str)
{
  std::istringstream iss(str.toStdString());
  float f;
  iss >> noskipws >> f;

  if (iss.eof() && !iss.fail())
  {
    GlobalVariable::SENSOR_TO_GROUND = str.toFloat();
    boost_ptree.put("SENSOR_TO_GROUND", GlobalVariable::SENSOR_TO_GROUND);
    boost::property_tree::write_ini("config.ini", boost_ptree);
  }
}
//------------------------------------------------------------------ TAB 2
void QtViewer::CheckBox2(int value)
{
  if (value == 0)
    GlobalVariable::UI_ENABLE_LANE = false;
  else if (value == 2)
    GlobalVariable::UI_ENABLE_LANE = true;
}
void QtViewer::LineEditor7(const QString& str)
{
  std::istringstream iss(str.toStdString());
  float f;
  iss >> noskipws >> f;

  if (iss.eof() && !iss.fail())
  {
    GlobalVariable::UI_LANE_INTENSITY_I = str.toFloat();
    boost_ptree.put("UI_LANE_INTENSITY_I", GlobalVariable::UI_LANE_INTENSITY_I);
    boost::property_tree::write_ini("config.ini", boost_ptree);
  }
}

void QtViewer::LineEditor13(const QString& str)
{
  std::istringstream iss(str.toStdString());
  float f;
  iss >> noskipws >> f;

  if (iss.eof() && !iss.fail())
  {
    GlobalVariable::UI_LANE_DETECTION_WITH_OUTER = str.toFloat();
    boost_ptree.put("UI_LANE_DETECTION_WITH_OUTER", GlobalVariable::UI_LANE_DETECTION_WITH_OUTER);
    boost::property_tree::write_ini("config.ini", boost_ptree);
  }
}

void QtViewer::LineEditor14(const QString& str)
{
  std::istringstream iss(str.toStdString());
  float f;
  iss >> noskipws >> f;

  if (iss.eof() && !iss.fail())
  {
    GlobalVariable::UI_LANE_DETECTION_WITH_INNER = str.toFloat();
    boost_ptree.put("UI_LANE_DETECTION_WITH_INNER", GlobalVariable::UI_LANE_DETECTION_WITH_INNER);
    boost::property_tree::write_ini("config.ini", boost_ptree);
  }
}

void QtViewer::LineEditor15(const QString& str)
{
  std::istringstream iss(str.toStdString());
  float f;
  iss >> noskipws >> f;

  if (iss.eof() && !iss.fail())
  {
    GlobalVariable::UI_LANE_DETECTION_LENGTH = str.toFloat();
    boost_ptree.put("UI_LANE_DETECTION_LENGTH", GlobalVariable::UI_LANE_DETECTION_LENGTH);
    boost::property_tree::write_ini("config.ini", boost_ptree);
  }
}

void QtViewer::LineEditor16(const QString& str)
{
  std::istringstream iss(str.toStdString());
  float f;
  iss >> noskipws >> f;

  if (iss.eof() && !iss.fail())
  {
    GlobalVariable::UI_LANE_IDEA_WIDTH = str.toFloat();
    boost_ptree.put("UI_LANE_IDEA_WIDTH", GlobalVariable::UI_LANE_IDEA_WIDTH);
    boost::property_tree::write_ini("config.ini", boost_ptree);
  }
}
//------------------------------------------------------------------ TAB 3
void QtViewer::CheckBox0(int value)
{
  if (value == 0)
    GlobalVariable::UI_ENABLE_PARKING = false;
  else if (value == 2)
    GlobalVariable::UI_ENABLE_PARKING = true;
}
void QtViewer::LineEditor2(const QString& str)
{
  std::istringstream iss(str.toStdString());
  float f;
  iss >> noskipws >> f;

  if (iss.eof() && !iss.fail())
  {
    GlobalVariable::UI_PARKING_INTENSITY_I = str.toFloat();
    boost_ptree.put("UI_PARKING_INTENSITY_I", GlobalVariable::UI_PARKING_INTENSITY_I);
    boost::property_tree::write_ini("config.ini", boost_ptree);
  }
}
void QtViewer::LineEditor3(const QString& str)
{
}
void QtViewer::LineEditor4(const QString& str)
{
  std::istringstream iss(str.toStdString());
  float f;
  iss >> noskipws >> f;

  if (iss.eof() && !iss.fail())
  {
    GlobalVariable::UI_PARKING_SPACE_DEGREE = str.toFloat();
    boost_ptree.put("UI_PARKING_SPACE_DEGREE", GlobalVariable::UI_PARKING_SPACE_DEGREE);
    boost::property_tree::write_ini("config.ini", boost_ptree);
  }
}
void QtViewer::LineEditor5(const QString& str)
{
  std::istringstream iss(str.toStdString());
  float f;
  iss >> noskipws >> f;

  if (iss.eof() && !iss.fail())
  {
    GlobalVariable::UI_PARKING_SLOT_WIDTH = str.toFloat();
    boost_ptree.put("UI_PARKING_SLOT_WIDTH", GlobalVariable::UI_PARKING_SLOT_WIDTH);
    boost::property_tree::write_ini("config.ini", boost_ptree);
  }
}
void QtViewer::LineEditor6(const QString& str)
{
  std::istringstream iss(str.toStdString());
  float f;
  iss >> noskipws >> f;

  if (iss.eof() && !iss.fail())
  {
    GlobalVariable::UI_PARKING_SLOT_LENGTH = str.toFloat();
    boost_ptree.put("UI_PARKING_SLOT_LENGTH", GlobalVariable::UI_PARKING_SLOT_LENGTH);
    boost::property_tree::write_ini("config.ini", boost_ptree);
  }
}
//------------------------------------------------------------------ TAB 4
void QtViewer::LineEditor0(const QString& str)
{
  boost::system::error_code boost_error_code;
  boost::asio::ip::address::from_string(str.toStdString(), boost_error_code);
  if (boost_error_code.value() == 0)
  {
    GlobalVariable::UI_UDP_IP = str.toStdString();
    boost_ptree.put("UDP_IP", GlobalVariable::UI_UDP_IP);
    boost::property_tree::write_ini("config.ini", boost_ptree);
  }
}

void QtViewer::LineEditor1(const QString& str)
{
  string s = str.toStdString();
  std::string::const_iterator it = s.begin();
  while (it != s.end() && std::isdigit(*it))
    ++it;
  if (!s.empty() && it == s.end())
  {
    GlobalVariable::UI_UDP_Port = stoi(s);
    boost_ptree.put("UDP_Port", GlobalVariable::UI_UDP_Port);
    boost::property_tree::write_ini("config.ini", boost_ptree);
  }
}

void QtViewer::ButtonPressed0()
{
  std::cout << GlobalVariable::UI_TESTING_BUTTOM << std::endl;
  GlobalVariable::UI_TESTING_BUTTOM = true;
  std::cout << GlobalVariable::UI_TESTING_BUTTOM << std::endl;
}
//------------------------------------------------------------------ TAB 5

void QtViewer::HorizontalScrollBar6(int value)
{
  GlobalVariable::UI_PARA[0] = double(value);
}

void QtViewer::HorizontalScrollBar7(int value)
{
  GlobalVariable::UI_PARA[1] = double(value) / 10;
}

void QtViewer::HorizontalScrollBar8(int value)
{
  GlobalVariable::UI_PARA[2] = double(value) / 10;
}

void QtViewer::HorizontalScrollBar9(int value)
{
  GlobalVariable::UI_PARA[3] = double(value) / 10;
}

void QtViewer::HorizontalScrollBar10(int value)
{
  GlobalVariable::UI_PARA[4] = double(value);
}

void QtViewer::HorizontalScrollBar11(int value)
{
  GlobalVariable::UI_PARA[5] = double(value);
}

/*void
 QtViewer::SliderReleased ()
 {
 ui->qvtkWidget->update ();
 }*/
