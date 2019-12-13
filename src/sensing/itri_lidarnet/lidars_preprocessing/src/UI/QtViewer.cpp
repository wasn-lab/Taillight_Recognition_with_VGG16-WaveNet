#include "QtViewer.h"

QtViewer::QtViewer (QWidget *parent) :
    QMainWindow (parent),
    ui (new Ui::QtViewer)
{
  ui->setupUi (this);
  this->setWindowTitle ("QtViewer");

  // Set up the QVTK window
  viewer.reset (new pcl::visualization::PCLVisualizer ("viewer", false));
  ui->qvtkWidget->SetRenderWindow (viewer->getRenderWindow ());
  viewer->setupInteractor (ui->qvtkWidget->GetInteractor (), ui->qvtkWidget->GetRenderWindow ());
  ui->qvtkWidget->update ();

  // UI function initialization
  viewer->resetCamera ();
  ui->qvtkWidget->update ();

  // Connect functions

  // ------TAB 1
  connect (ui->horizontalScrollBar0, SIGNAL(valueChanged (int)), this, SLOT(HorizontalScrollBar0 (int)));
  connect (ui->horizontalScrollBar1, SIGNAL(valueChanged (int)), this, SLOT(HorizontalScrollBar1 (int)));
  connect (ui->horizontalScrollBar2, SIGNAL(valueChanged (int)), this, SLOT(HorizontalScrollBar2 (int)));
  connect (ui->horizontalScrollBar3, SIGNAL(valueChanged (int)), this, SLOT(HorizontalScrollBar3 (int)));
  connect (ui->horizontalScrollBar4, SIGNAL(valueChanged (int)), this, SLOT(HorizontalScrollBar4 (int)));
  connect (ui->horizontalScrollBar5, SIGNAL(valueChanged (int)), this, SLOT(HorizontalScrollBar5 (int)));

  connect (ui->radioButton0, SIGNAL(clicked ()), this, SLOT(RadioButton0 ()));
  connect (ui->radioButton1, SIGNAL(clicked ()), this, SLOT(RadioButton1 ()));
  connect (ui->radioButton2, SIGNAL(clicked ()), this, SLOT(RadioButton2 ()));

  connect (ui->checkBox1, SIGNAL(stateChanged (int)), this, SLOT(CheckBox1(int)));
  connect (ui->checkBox3, SIGNAL(stateChanged (int)), this, SLOT(CheckBox3(int)));

  connect (ui->lineEdit12, SIGNAL(textChanged(const QString &)), this, SLOT(LineEditor12 (const QString &)));

  // ------TAB 2
  connect (ui->horizontalScrollBar6, SIGNAL(valueChanged (int)), this, SLOT(HorizontalScrollBar6 (int)));
  connect (ui->horizontalScrollBar7, SIGNAL(valueChanged (int)), this, SLOT(HorizontalScrollBar7 (int)));
  connect (ui->horizontalScrollBar8, SIGNAL(valueChanged (int)), this, SLOT(HorizontalScrollBar8 (int)));
  connect (ui->horizontalScrollBar9, SIGNAL(valueChanged (int)), this, SLOT(HorizontalScrollBar9 (int)));
  connect (ui->horizontalScrollBar10, SIGNAL(valueChanged (int)), this, SLOT(HorizontalScrollBar10 (int)));
  connect (ui->horizontalScrollBar11, SIGNAL(valueChanged (int)), this, SLOT(HorizontalScrollBar11 (int)));

  // Config file
  try
  {
    boost::property_tree::ini_parser::read_ini (GlobalVariable::CONFIG_FILE_NAME, boost_ptree);
  }
  catch (exception &e)
  {
    cout << e.what ();
    boost::property_tree::ini_parser::write_ini (GlobalVariable::CONFIG_FILE_NAME, boost_ptree);
  }

  // ------TAB 1
  ui->horizontalScrollBar0->setValue (boost_ptree.get<int> ("UI_LATERAL_RANGE", 10));
  ui->horizontalScrollBar1->setValue (boost_ptree.get<double> ("UI_UNIFORM_SAMPLING", 0.1));
  ui->horizontalScrollBar2->setValue (boost_ptree.get<double> ("UI_DBSCAN_EPS", 1));
  ui->horizontalScrollBar3->setValue (boost_ptree.get<int> ("UI_DBSCAN_MINPT", 1));
  ui->lineEdit12->setText (QString::fromStdString (boost_ptree.get<string> ("SENSOR_TO_GROUND", "")));

  // Initial

  ui->radioButton0->setChecked (false);
  ui->radioButton1->setChecked (false);
  ui->radioButton2->setChecked (true);

  QDoubleValidator *validator = new QDoubleValidator (this);
  validator->setBottom (-9999.99);
  validator->setTop (9999.99);
  validator->setDecimals (2);
  validator->setNotation (QDoubleValidator::StandardNotation);

  ui->lineEdit12->setValidator (validator);

}

QtViewer::~QtViewer ()
{
  delete ui;
}

//------------------------------------------------------------------ TAB 1
void
QtViewer::HorizontalScrollBar0 (int value)
{

}

void
QtViewer::HorizontalScrollBar1 (int value)
{

}

void
QtViewer::HorizontalScrollBar2 (int value)
{

}

void
QtViewer::HorizontalScrollBar3 (int value)
{

}

void
QtViewer::HorizontalScrollBar4 (int value)
{
}

void
QtViewer::HorizontalScrollBar5 (int value)
{
}

void
QtViewer::RadioButton0 ()
{
}

void
QtViewer::RadioButton1 ()
{
}

void
QtViewer::RadioButton2 ()
{
}

void
QtViewer::CheckBox1 (int value)
{
}

void
QtViewer::CheckBox3 (int value)
{
}

void
QtViewer::LineEditor12 (const QString & str)
{

}
//------------------------------------------------------------------ TAB 2
void
QtViewer::CheckBox2 (int value)
{

}
void
QtViewer::LineEditor7 (const QString & str)
{

}

void
QtViewer::LineEditor13 (const QString & str)
{

}

void
QtViewer::LineEditor14 (const QString & str)
{

}

void
QtViewer::LineEditor15 (const QString & str)
{

}

void
QtViewer::LineEditor16 (const QString & str)
{

}
//------------------------------------------------------------------ TAB 3
void
QtViewer::CheckBox0 (int value)
{
}
void
QtViewer::LineEditor2 (const QString & str)
{
}
void
QtViewer::LineEditor3 (const QString & str)
{

}
void
QtViewer::LineEditor4 (const QString & str)
{
}
void
QtViewer::LineEditor5 (const QString & str)
{
}
void
QtViewer::LineEditor6 (const QString & str)
{
}
//------------------------------------------------------------------ TAB 4
void
QtViewer::LineEditor0 (const QString & str)
{

}

void
QtViewer::LineEditor1 (const QString & str)
{

}

void
QtViewer::ButtonPressed0 ()
{
}
//------------------------------------------------------------------ TAB 2

void
QtViewer::HorizontalScrollBar6 (int value)
{
  GlobalVariable::UI_PARA[0] = double (value) / 100;
}

void
QtViewer::HorizontalScrollBar7 (int value)
{
  GlobalVariable::UI_PARA[1] = double (value) / 100;
}

void
QtViewer::HorizontalScrollBar8 (int value)
{
  GlobalVariable::UI_PARA[2] = double (value) / 100;
}

void
QtViewer::HorizontalScrollBar9 (int value)
{
  GlobalVariable::UI_PARA[3] = double (value) / 100;
}

void
QtViewer::HorizontalScrollBar10 (int value)
{
  GlobalVariable::UI_PARA[4] = double (value) / 100;
}

void
QtViewer::HorizontalScrollBar11 (int value)
{
  GlobalVariable::UI_PARA[5] = double (value) / 100;
}

