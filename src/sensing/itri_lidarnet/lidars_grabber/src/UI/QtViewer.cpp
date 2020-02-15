#include "QtViewer.h"
#include <QMessageBox>
#include <QString>
#include <QStringList>
#include <QFile>
#include <QTextStream>
#include <ros/ros.h>
#include <QFileDialog>
#include <string>
#include <vector>

// global varible
int carType = -1;  // {Truck: 0, Bus: 1}
QString srcfile;
string LaunchFileName;

QtViewer::QtViewer(QWidget* parent) : QMainWindow(parent), ui(new Ui::QtViewer)
{
  ui->setupUi(this);
  this->setWindowTitle("QtViewer");

  ui->tab_Truck->setVisible(false);
  ui->tab_Bus->setVisible(false);

  QMessageBox msgBox;
  // check launch file name
  if (ros::param::has("LaunchFileName"))
  {
    ros::param::get("LaunchFileName", LaunchFileName);
    srcfile = "../../../src/itriadv/src/sensing/itri_lidarnet/launch/launch/" + QString::fromStdString(LaunchFileName);

    if (LaunchFileName == "hino1.launch" or LaunchFileName == "hino2.launch")
    {
      ui->mode_Bus->setEnabled(false);
      ui->tab_Truck->setVisible(true);
      ui->tab_Bus->setVisible(false);
    }
    else if (LaunchFileName == "b1.launch" or LaunchFileName == "b1_new.launch" or
             LaunchFileName == "b1_new_write.launch")
    {
      ui->mode_Truck->setEnabled(false);
      ui->tab_Bus->setVisible(true);
      ui->tab_Truck->setVisible(false);
    }
    msgBox.setText("Src path:" + srcfile);
    msgBox.exec();
  }
  else
  {
    msgBox.setText("Can't get Launch File Name!");
    msgBox.exec();
  }

  //============================================================== Hino initial
  //-------------------------------------------------Left
  ui->Btn_Left_Reset->setEnabled(true);
  ui->Btn_Left_Accept->setEnabled(true);

  ui->Btn_Left_Redo->setEnabled(false);
  ui->Btn_Left_Save->setEnabled(false);

  ui->Label_Left_Fine_TX->setText("000.0000");
  ui->Label_Left_Fine_TY->setText("000.0000");
  ui->Label_Left_Fine_TZ->setText("000.0000");
  ui->Label_Left_Fine_RX->setText("000.0000");
  ui->Label_Left_Fine_RY->setText("000.0000");
  ui->Label_Left_Fine_RZ->setText("000.0000");
  //-------------------------------------------------Right
  ui->Btn_Right_Reset->setEnabled(true);
  ui->Btn_Right_Accept->setEnabled(true);

  ui->Btn_Right_Redo->setEnabled(false);
  ui->Btn_Right_Save->setEnabled(false);

  ui->Label_Right_Fine_TX->setText("000.0000");
  ui->Label_Right_Fine_TY->setText("000.0000");
  ui->Label_Right_Fine_TZ->setText("000.0000");
  ui->Label_Right_Fine_RX->setText("000.0000");
  ui->Label_Right_Fine_RY->setText("000.0000");
  ui->Label_Right_Fine_RZ->setText("000.0000");
  //-------------------------------------------------Front
  ui->Btn_Front_Reset->setEnabled(true);
  ui->Btn_Front_Accept->setEnabled(true);

  ui->Btn_Front_Redo->setEnabled(false);
  ui->Btn_Front_Save->setEnabled(false);

  ui->Label_Front_Fine_TX->setText("000.0000");
  ui->Label_Front_Fine_TY->setText("000.0000");
  ui->Label_Front_Fine_TZ->setText("000.0000");
  ui->Label_Front_Fine_RX->setText("000.0000");
  ui->Label_Front_Fine_RY->setText("000.0000");
  ui->Label_Front_Fine_RZ->setText("000.0000");

  //============================================================== B1 initial
  //-------------------------------------------------FrontLeft
  ui->Btn_FrontLeft_Reset->setEnabled(true);
  ui->Btn_FrontLeft_Accept->setEnabled(true);

  ui->Btn_FrontLeft_Redo->setEnabled(false);
  ui->Btn_FrontLeft_Save->setEnabled(false);

  ui->Label_FrontLeft_Fine_TX->setText("000.0000");
  ui->Label_FrontLeft_Fine_TY->setText("000.0000");
  ui->Label_FrontLeft_Fine_TZ->setText("000.0000");
  ui->Label_FrontLeft_Fine_RX->setText("000.0000");
  ui->Label_FrontLeft_Fine_RY->setText("000.0000");
  ui->Label_FrontLeft_Fine_RZ->setText("000.0000");
  //-------------------------------------------------FrontRight
  ui->Btn_FrontRight_Reset->setEnabled(true);
  ui->Btn_FrontRight_Accept->setEnabled(true);

  ui->Btn_FrontRight_Redo->setEnabled(false);
  ui->Btn_FrontRight_Save->setEnabled(false);

  ui->Label_FrontRight_Fine_TX->setText("000.0000");
  ui->Label_FrontRight_Fine_TY->setText("000.0000");
  ui->Label_FrontRight_Fine_TZ->setText("000.0000");
  ui->Label_FrontRight_Fine_RX->setText("000.0000");
  ui->Label_FrontRight_Fine_RY->setText("000.0000");
  ui->Label_FrontRight_Fine_RZ->setText("000.0000");
  //-------------------------------------------------RearLeft
  ui->Btn_RearLeft_Reset->setEnabled(true);
  ui->Btn_RearLeft_Accept->setEnabled(true);

  ui->Btn_RearLeft_Redo->setEnabled(false);
  ui->Btn_RearLeft_Save->setEnabled(false);

  ui->Label_RearLeft_Fine_TX->setText("000.0000");
  ui->Label_RearLeft_Fine_TY->setText("000.0000");
  ui->Label_RearLeft_Fine_TZ->setText("000.0000");
  ui->Label_RearLeft_Fine_RX->setText("000.0000");
  ui->Label_RearLeft_Fine_RY->setText("000.0000");
  ui->Label_RearLeft_Fine_RZ->setText("000.0000");

  //-------------------------------------------------RearRight
  ui->Btn_RearRight_Reset->setEnabled(true);
  ui->Btn_RearRight_Accept->setEnabled(true);

  ui->Btn_RearRight_Redo->setEnabled(false);
  ui->Btn_RearRight_Save->setEnabled(false);

  ui->Label_RearRight_Fine_TX->setText("000.0000");
  ui->Label_RearRight_Fine_TY->setText("000.0000");
  ui->Label_RearRight_Fine_TZ->setText("000.0000");
  ui->Label_RearRight_Fine_RX->setText("000.0000");
  ui->Label_RearRight_Fine_RY->setText("000.0000");
  ui->Label_RearRight_Fine_RZ->setText("000.0000");
}
//============================================================== End of initial
QtViewer::~QtViewer()
{
  delete ui;
}

//============================================================= Mode Switch
void QtViewer::on_mode_Truck_clicked()
{
}

void QtViewer::on_mode_Bus_clicked()
{
}

//============================================================== Hino Slot
//--------------------------------------------------- Lidar_Left
void QtViewer::on_Slider_Left_TX_valueChanged(int value)
{
  ui->LCD_Left_TX->display(value);
  GlobalVariable::UI_PARA[0] = double(value) / 100;
  ui->Btn_Left_Redo->setEnabled(false);
}

void QtViewer::on_Slider_Left_TY_valueChanged(int value)
{
  ui->LCD_Left_TY->display(value);
  GlobalVariable::UI_PARA[1] = double(value) / 100;
  ui->Btn_Left_Redo->setEnabled(false);
}

void QtViewer::on_Slider_Left_TZ_valueChanged(int value)
{
  ui->LCD_Left_TZ->display(value);
  GlobalVariable::UI_PARA[2] = double(value) / 100;
  ui->Btn_Left_Redo->setEnabled(false);
}

void QtViewer::on_Slider_Left_RX_valueChanged(int value)
{
  ui->LCD_Left_RX->display(value);
  GlobalVariable::UI_PARA[3] = double(value) / 100;
  ui->Btn_Left_Redo->setEnabled(false);
}

void QtViewer::on_Slider_Left_RY_valueChanged(int value)
{
  ui->LCD_Left_RY->display(value);
  GlobalVariable::UI_PARA[4] = double(value) / 100;
  ui->Btn_Left_Redo->setEnabled(false);
}

void QtViewer::on_Slider_Left_RZ_valueChanged(int value)
{
  ui->LCD_Left_RZ->display(value);
  GlobalVariable::UI_PARA[5] = double(value) / 100;
  ui->Btn_Left_Redo->setEnabled(false);
}

void QtViewer::on_Btn_Left_Load_clicked()
{
  ui->Btn_Left_Reset->click();
  if (ros::param::has("LidarLeft_Fine_Param"))
  {
    vector<double> LidarLeft_Fine_Param;
    ros::param::get("LidarLeft_Fine_Param", LidarLeft_Fine_Param);

    ui->LCD_Left_TX->display(LidarLeft_Fine_Param[0]);
    ui->LCD_Left_TY->display(LidarLeft_Fine_Param[1]);
    ui->LCD_Left_TZ->display(LidarLeft_Fine_Param[2]);
    ui->LCD_Left_RX->display(LidarLeft_Fine_Param[3]);
    ui->LCD_Left_RY->display(LidarLeft_Fine_Param[4]);
    ui->LCD_Left_RZ->display(LidarLeft_Fine_Param[5]);

    ui->Slider_Left_TX->setValue(LidarLeft_Fine_Param[0] * 100);
    ui->Slider_Left_TY->setValue(LidarLeft_Fine_Param[1] * 100);
    ui->Slider_Left_TZ->setValue(LidarLeft_Fine_Param[2] * 100);
    ui->Slider_Left_RX->setValue(LidarLeft_Fine_Param[3] * 100);
    ui->Slider_Left_RY->setValue(LidarLeft_Fine_Param[4] * 100);
    ui->Slider_Left_RZ->setValue(LidarLeft_Fine_Param[5] * 100);

    GlobalVariable::UI_PARA[0] = LidarLeft_Fine_Param[0];
    GlobalVariable::UI_PARA[1] = LidarLeft_Fine_Param[1];
    GlobalVariable::UI_PARA[2] = LidarLeft_Fine_Param[2];
    GlobalVariable::UI_PARA[3] = LidarLeft_Fine_Param[3];
    GlobalVariable::UI_PARA[4] = LidarLeft_Fine_Param[4];
    GlobalVariable::UI_PARA[5] = LidarLeft_Fine_Param[5];

    ui->Btn_Left_Refresh->click();
  }
  else
  {
    QMessageBox msgBox;
    msgBox.setText("Left Parameter Not Found!");
    msgBox.exec();
  }
}

void QtViewer::on_Btn_Left_Reset_clicked()
{
  ui->LCD_Left_TX->display(0);
  ui->LCD_Left_TY->display(0);
  ui->LCD_Left_TZ->display(0);
  ui->LCD_Left_RX->display(0);
  ui->LCD_Left_RY->display(0);
  ui->LCD_Left_RZ->display(0);

  ui->Slider_Left_TX->setValue(0);
  ui->Slider_Left_TY->setValue(0);
  ui->Slider_Left_TZ->setValue(0);
  ui->Slider_Left_RX->setValue(0);
  ui->Slider_Left_RY->setValue(0);
  ui->Slider_Left_RZ->setValue(0);

  GlobalVariable::UI_PARA[0] = 0.0;
  GlobalVariable::UI_PARA[1] = 0.0;
  GlobalVariable::UI_PARA[2] = 0.0;
  GlobalVariable::UI_PARA[3] = 0.0;
  GlobalVariable::UI_PARA[4] = 0.0;
  GlobalVariable::UI_PARA[5] = 0.0;

  ui->Btn_Left_Refresh->click();
}

void QtViewer::on_Btn_Left_Accept_clicked()
{
  for (int i = 0; i < 6; i++)
  {
    GlobalVariable::UI_PARA_BK[i] = GlobalVariable::UI_PARA[i];
  }

  ui->Btn_Left_Redo->setEnabled(true);
  ui->Btn_Left_Save->setEnabled(true);

  ui->Btn_Left_Refresh->click();
}

void QtViewer::on_Btn_Left_Refresh_clicked()
{
  ui->Label_Left_Fine_TX->setText(QString("%1").arg(GlobalVariable::UI_PARA[0]));
  ui->Label_Left_Fine_TY->setText(QString("%1").arg(GlobalVariable::UI_PARA[1]));
  ui->Label_Left_Fine_TZ->setText(QString("%1").arg(GlobalVariable::UI_PARA[2]));
  ui->Label_Left_Fine_RX->setText(QString("%1").arg(GlobalVariable::UI_PARA[3]));
  ui->Label_Left_Fine_RY->setText(QString("%1").arg(GlobalVariable::UI_PARA[4]));
  ui->Label_Left_Fine_RZ->setText(QString("%1").arg(GlobalVariable::UI_PARA[5]));
}

void QtViewer::on_Btn_Left_Redo_clicked()
{
  GlobalVariable::Left_FineTune_Trigger = true;
}

void QtViewer::on_Btn_Left_Save_clicked()
{
  int lineNum = 4;

  QString new_content = QString("<rosparam param=\"LidarLeft_Fine_Param\">[%1, %2, %3, %4, %5, %6]</rosparam>")
                            .arg(GlobalVariable::UI_PARA[0])
                            .arg(GlobalVariable::UI_PARA[1])
                            .arg(GlobalVariable::UI_PARA[2])
                            .arg(GlobalVariable::UI_PARA[3])
                            .arg(GlobalVariable::UI_PARA[4])
                            .arg(GlobalVariable::UI_PARA[5]);

  QString filePath = srcfile;
  QString curPath = QDir::currentPath();

  QMessageBox msgBox;
  msgBox.setWindowTitle("CHECK PARAMETER!");
  msgBox.setText(new_content);
  msgBox.setStandardButtons(QMessageBox::Yes);
  msgBox.addButton(QMessageBox::No);
  msgBox.setDefaultButton(QMessageBox::No);
  if (msgBox.exec() == QMessageBox::Yes)
  {
    QMessageBox infoBox;
    infoBox.setText("Save to " + filePath);
    infoBox.exec();

    QString strAll;
    QStringList strList;
    QFile readFile(filePath);

    if (!readFile.open(QIODevice::ReadOnly | QIODevice::Text))
    {
      infoBox.setText("Src can't be found.");
      infoBox.exec();
    }
    else
    {
      QTextStream stream(&readFile);
      strAll = stream.readAll();
      infoBox.setText("Src read.");
      infoBox.exec();
    }
    readFile.close();

    QString bkStr;
    int bkNum = 8;
    QFile writeFile(filePath);
    if (writeFile.open(QIODevice::WriteOnly | QIODevice::Text))
    {
      QTextStream stream(&writeFile);
      strList = strAll.split("\n");
      for (int i = 0; i < strList.count(); i++)
      {
        if (i == lineNum - 1)
        {
          QString tempStr = strList.at(i);
          bkStr = tempStr;
          tempStr.replace(0, tempStr.length(), new_content);
          stream << tempStr << '\n';
        }
        else if (i == bkNum - 1)
        {
          QString tempStr = strList.at(i);
          tempStr.replace(0, tempStr.length(), bkStr);
          stream << tempStr << '\n';
        }
        else
        {
          stream << strList.at(i) << '\n';
        }
      }

      infoBox.setText(filePath + " has been modified.");
      infoBox.exec();
    }
    else
    {
      infoBox.setText("Saved Failed!! Cancel!!");
      infoBox.exec();
    }
    writeFile.close();
  }
  else
  {
    QMessageBox infoBox;
    infoBox.setText("Cancel");
    infoBox.exec();
  }
}

//--------------------------------------------------- Lidar_Right
void QtViewer::on_Slider_Right_TX_valueChanged(int value)
{
  ui->LCD_Right_TX->display(value);
  GlobalVariable::UI_PARA[6] = double(value) / 100;
  ui->Btn_Right_Redo->setEnabled(false);
}

void QtViewer::on_Slider_Right_TY_valueChanged(int value)
{
  ui->LCD_Right_TY->display(value);
  GlobalVariable::UI_PARA[7] = double(value) / 100;
  ui->Btn_Right_Redo->setEnabled(false);
}

void QtViewer::on_Slider_Right_TZ_valueChanged(int value)
{
  ui->LCD_Right_TZ->display(value);
  GlobalVariable::UI_PARA[8] = double(value) / 100;
  ui->Btn_Right_Redo->setEnabled(false);
}

void QtViewer::on_Slider_Right_RX_valueChanged(int value)
{
  ui->LCD_Right_RX->display(value);
  GlobalVariable::UI_PARA[9] = double(value) / 100;
  ui->Btn_Right_Redo->setEnabled(false);
}

void QtViewer::on_Slider_Right_RY_valueChanged(int value)
{
  ui->LCD_Right_RY->display(value);
  GlobalVariable::UI_PARA[10] = double(value) / 100;
  ui->Btn_Right_Redo->setEnabled(false);
}

void QtViewer::on_Slider_Right_RZ_valueChanged(int value)
{
  ui->LCD_Right_RZ->display(value);
  GlobalVariable::UI_PARA[11] = double(value) / 100;
  ui->Btn_Right_Redo->setEnabled(false);
}

void QtViewer::on_Btn_Right_Load_clicked()
{
  ui->Btn_Right_Reset->click();
  if (ros::param::has("LidarRight_Fine_Param"))
  {
    vector<double> LidarRight_Fine_Param;
    ros::param::get("LidarRight_Fine_Param", LidarRight_Fine_Param);

    ui->LCD_Right_TX->display(LidarRight_Fine_Param[0]);
    ui->LCD_Right_TY->display(LidarRight_Fine_Param[1]);
    ui->LCD_Right_TZ->display(LidarRight_Fine_Param[2]);
    ui->LCD_Right_RX->display(LidarRight_Fine_Param[3]);
    ui->LCD_Right_RY->display(LidarRight_Fine_Param[4]);
    ui->LCD_Right_RZ->display(LidarRight_Fine_Param[5]);

    ui->Slider_Right_TX->setValue(LidarRight_Fine_Param[0] * 100);
    ui->Slider_Right_TY->setValue(LidarRight_Fine_Param[1] * 100);
    ui->Slider_Right_TZ->setValue(LidarRight_Fine_Param[2] * 100);
    ui->Slider_Right_RX->setValue(LidarRight_Fine_Param[3] * 100);
    ui->Slider_Right_RY->setValue(LidarRight_Fine_Param[4] * 100);
    ui->Slider_Right_RZ->setValue(LidarRight_Fine_Param[5] * 100);

    GlobalVariable::UI_PARA[6] = LidarRight_Fine_Param[0];
    GlobalVariable::UI_PARA[7] = LidarRight_Fine_Param[1];
    GlobalVariable::UI_PARA[8] = LidarRight_Fine_Param[2];
    GlobalVariable::UI_PARA[9] = LidarRight_Fine_Param[3];
    GlobalVariable::UI_PARA[10] = LidarRight_Fine_Param[4];
    GlobalVariable::UI_PARA[11] = LidarRight_Fine_Param[5];

    ui->Btn_Right_Refresh->click();
  }
  else
  {
    QMessageBox msgBox;
    msgBox.setText("Right Parameter Not Found!");
    msgBox.exec();
  }
}

void QtViewer::on_Btn_Right_Reset_clicked()
{
  ui->LCD_Right_TX->display(0);
  ui->LCD_Right_TY->display(0);
  ui->LCD_Right_TZ->display(0);
  ui->LCD_Right_RX->display(0);
  ui->LCD_Right_RY->display(0);
  ui->LCD_Right_RZ->display(0);

  ui->Slider_Right_TX->setValue(0);
  ui->Slider_Right_TY->setValue(0);
  ui->Slider_Right_TZ->setValue(0);
  ui->Slider_Right_RX->setValue(0);
  ui->Slider_Right_RY->setValue(0);
  ui->Slider_Right_RZ->setValue(0);

  GlobalVariable::UI_PARA[6] = 0.0;
  GlobalVariable::UI_PARA[7] = 0.0;
  GlobalVariable::UI_PARA[8] = 0.0;
  GlobalVariable::UI_PARA[9] = 0.0;
  GlobalVariable::UI_PARA[10] = 0.0;
  GlobalVariable::UI_PARA[11] = 0.0;

  ui->Btn_Right_Refresh->click();
}

void QtViewer::on_Btn_Right_Accept_clicked()
{
  for (int i = 6; i < 12; i++)
  {
    GlobalVariable::UI_PARA_BK[i] = GlobalVariable::UI_PARA[i];
  }

  ui->Btn_Right_Redo->setEnabled(true);
  ui->Btn_Right_Save->setEnabled(true);

  ui->Btn_Right_Refresh->click();
}

void QtViewer::on_Btn_Right_Refresh_clicked()
{
  ui->Label_Right_Fine_TX->setText(QString("%1").arg(GlobalVariable::UI_PARA[6]));
  ui->Label_Right_Fine_TY->setText(QString("%1").arg(GlobalVariable::UI_PARA[7]));
  ui->Label_Right_Fine_TZ->setText(QString("%1").arg(GlobalVariable::UI_PARA[8]));
  ui->Label_Right_Fine_RX->setText(QString("%1").arg(GlobalVariable::UI_PARA[9]));
  ui->Label_Right_Fine_RY->setText(QString("%1").arg(GlobalVariable::UI_PARA[10]));
  ui->Label_Right_Fine_RZ->setText(QString("%1").arg(GlobalVariable::UI_PARA[11]));
}

void QtViewer::on_Btn_Right_Save_clicked()
{
  int lineNum = 5;
  QString new_content = QString("<rosparam param=\"LidarRight_Fine_Param\">[%1, %2, %3, %4, %5, %6]</rosparam>")
                            .arg(GlobalVariable::UI_PARA[6])
                            .arg(GlobalVariable::UI_PARA[7])
                            .arg(GlobalVariable::UI_PARA[8])
                            .arg(GlobalVariable::UI_PARA[9])
                            .arg(GlobalVariable::UI_PARA[10])
                            .arg(GlobalVariable::UI_PARA[11]);

  QString filePath = srcfile;
  QString curPath = QDir::currentPath();

  QMessageBox msgBox;
  msgBox.setWindowTitle("CHECK PARAMETER!");
  msgBox.setText(new_content);
  msgBox.setStandardButtons(QMessageBox::Yes);
  msgBox.addButton(QMessageBox::No);
  msgBox.setDefaultButton(QMessageBox::No);
  if (msgBox.exec() == QMessageBox::Yes)
  {
    QMessageBox infoBox;
    infoBox.setText("Save to " + filePath);
    infoBox.exec();

    QString strAll;
    QStringList strList;
    QFile readFile(filePath);

    if (!readFile.open(QIODevice::ReadOnly | QIODevice::Text))
    {
      infoBox.setText("Src can't be found.");
      infoBox.exec();
    }

    else
    {
      QTextStream stream(&readFile);
      strAll = stream.readAll();
      infoBox.setText("Src read.");
      infoBox.exec();
    }
    readFile.close();

    QString bkStr;
    int bkNum = 9;
    QFile writeFile(filePath);
    if (writeFile.open(QIODevice::WriteOnly | QIODevice::Text))
    {
      QTextStream stream(&writeFile);
      strList = strAll.split("\n");
      for (int i = 0; i < strList.count(); i++)
      {
        if (i == lineNum - 1)
        {
          QString tempStr = strList.at(i);
          bkStr = tempStr;
          tempStr.replace(0, tempStr.length(), new_content);
          stream << tempStr << '\n';
        }
        else if (i == bkNum - 1)
        {
          QString tempStr = strList.at(i);
          tempStr.replace(0, tempStr.length(), bkStr);
          stream << tempStr << '\n';
        }
        else
        {
          stream << strList.at(i) << '\n';
        }
      }

      infoBox.setText(filePath + " has been modified.");
      infoBox.exec();
    }
    else
    {
      infoBox.setText("Saved Failed!! Cancel!!");
      infoBox.exec();
    }
    writeFile.close();
  }
  else
  {
    QMessageBox infoBox;
    infoBox.setText("Cancel");
    infoBox.exec();
  }
}
void QtViewer::on_Btn_Right_Redo_clicked()
{
  GlobalVariable::Right_FineTune_Trigger = true;
}

//--------------------------------------------------- Lidar_Front
void QtViewer::on_Slider_Front_TX_valueChanged(int value)
{
  ui->LCD_Front_TX->display(value);
  GlobalVariable::UI_PARA[12] = double(value) / 100;
  ui->Btn_Front_Redo->setEnabled(false);
}

void QtViewer::on_Slider_Front_TY_valueChanged(int value)
{
  ui->LCD_Front_TY->display(value);
  GlobalVariable::UI_PARA[13] = double(value) / 100;
  ui->Btn_Front_Redo->setEnabled(false);
}

void QtViewer::on_Slider_Front_TZ_valueChanged(int value)
{
  ui->LCD_Front_TZ->display(value);
  GlobalVariable::UI_PARA[14] = double(value) / 100;
  ui->Btn_Front_Redo->setEnabled(false);
}

void QtViewer::on_Slider_Front_RX_valueChanged(int value)
{
  ui->LCD_Front_RX->display(value);
  GlobalVariable::UI_PARA[15] = double(value) / 100;
  ui->Btn_Front_Redo->setEnabled(false);
}

void QtViewer::on_Slider_Front_RY_valueChanged(int value)
{
  ui->LCD_Front_RY->display(value);
  GlobalVariable::UI_PARA[16] = double(value) / 100;
  ui->Btn_Front_Redo->setEnabled(false);
}

void QtViewer::on_Slider_Front_RZ_valueChanged(int value)
{
  ui->LCD_Front_RZ->display(value);
  GlobalVariable::UI_PARA[17] = double(value) / 100;
  ui->Btn_Front_Redo->setEnabled(false);
}

void QtViewer::on_Btn_Front_Load_clicked()
{
  ui->Btn_Front_Reset->click();
  if (ros::param::has("LidarFront_Fine_Param"))
  {
    vector<double> LidarFront_Fine_Param;
    ros::param::get("LidarFront_Fine_Param", LidarFront_Fine_Param);

    ui->LCD_Front_TX->display(LidarFront_Fine_Param[0]);
    ui->LCD_Front_TY->display(LidarFront_Fine_Param[1]);
    ui->LCD_Front_TZ->display(LidarFront_Fine_Param[2]);
    ui->LCD_Front_RX->display(LidarFront_Fine_Param[3]);
    ui->LCD_Front_RY->display(LidarFront_Fine_Param[4]);
    ui->LCD_Front_RZ->display(LidarFront_Fine_Param[5]);

    ui->Slider_Front_TX->setValue(LidarFront_Fine_Param[0] * 100);
    ui->Slider_Front_TY->setValue(LidarFront_Fine_Param[1] * 100);
    ui->Slider_Front_TZ->setValue(LidarFront_Fine_Param[2] * 100);
    ui->Slider_Front_RX->setValue(LidarFront_Fine_Param[3] * 100);
    ui->Slider_Front_RY->setValue(LidarFront_Fine_Param[4] * 100);
    ui->Slider_Front_RZ->setValue(LidarFront_Fine_Param[5] * 100);

    GlobalVariable::UI_PARA[12] = LidarFront_Fine_Param[0];
    GlobalVariable::UI_PARA[13] = LidarFront_Fine_Param[1];
    GlobalVariable::UI_PARA[14] = LidarFront_Fine_Param[2];
    GlobalVariable::UI_PARA[15] = LidarFront_Fine_Param[3];
    GlobalVariable::UI_PARA[16] = LidarFront_Fine_Param[4];
    GlobalVariable::UI_PARA[17] = LidarFront_Fine_Param[5];

    ui->Btn_Front_Refresh->click();
  }
  else
  {
    QMessageBox msgBox;
    msgBox.setText("Front Parameter Not Found!");
    msgBox.exec();
  }
}

void QtViewer::on_Btn_Front_Reset_clicked()
{
  ui->LCD_Front_TX->display(0);
  ui->LCD_Front_TY->display(0);
  ui->LCD_Front_TZ->display(0);
  ui->LCD_Front_RX->display(0);
  ui->LCD_Front_RY->display(0);
  ui->LCD_Front_RZ->display(0);

  ui->Slider_Front_TX->setValue(0);
  ui->Slider_Front_TY->setValue(0);
  ui->Slider_Front_TZ->setValue(0);
  ui->Slider_Front_RX->setValue(0);
  ui->Slider_Front_RY->setValue(0);
  ui->Slider_Front_RZ->setValue(0);

  GlobalVariable::UI_PARA[12] = 0.0;
  GlobalVariable::UI_PARA[13] = 0.0;
  GlobalVariable::UI_PARA[14] = 0.0;
  GlobalVariable::UI_PARA[15] = 0.0;
  GlobalVariable::UI_PARA[16] = 0.0;
  GlobalVariable::UI_PARA[17] = 0.0;

  ui->Btn_Front_Refresh->click();
}

void QtViewer::on_Btn_Front_Accept_clicked()
{
  ui->Btn_Front_Redo->setEnabled(true);
  ui->Btn_Front_Save->setEnabled(true);

  for (int i = 12; i < 18; i++)
  {
    GlobalVariable::UI_PARA_BK[i] = GlobalVariable::UI_PARA[i];
  }

  ui->Btn_Front_Refresh->click();
}

void QtViewer::on_Btn_Front_Refresh_clicked()
{
  ui->Label_Front_Fine_TX->setText(QString("%1").arg(GlobalVariable::UI_PARA[12]));
  ui->Label_Front_Fine_TY->setText(QString("%1").arg(GlobalVariable::UI_PARA[13]));
  ui->Label_Front_Fine_TZ->setText(QString("%1").arg(GlobalVariable::UI_PARA[14]));
  ui->Label_Front_Fine_RX->setText(QString("%1").arg(GlobalVariable::UI_PARA[15]));
  ui->Label_Front_Fine_RY->setText(QString("%1").arg(GlobalVariable::UI_PARA[16]));
  ui->Label_Front_Fine_RZ->setText(QString("%1").arg(GlobalVariable::UI_PARA[17]));
}

void QtViewer::on_Btn_Front_Save_clicked()
{
  int lineNum = 6;
  QString new_content = QString("<rosparam param=\"LidarFront_Fine_Param\">[%1, %2, %3, %4, %5, %6]</rosparam>")
                            .arg(GlobalVariable::UI_PARA[12])
                            .arg(GlobalVariable::UI_PARA[13])
                            .arg(GlobalVariable::UI_PARA[14])
                            .arg(GlobalVariable::UI_PARA[15])
                            .arg(GlobalVariable::UI_PARA[16])
                            .arg(GlobalVariable::UI_PARA[17]);

  QString filePath = srcfile;
  QString curPath = QDir::currentPath();

  QMessageBox msgBox;
  msgBox.setWindowTitle("CHECK PARAMETER!");
  msgBox.setText(new_content);
  msgBox.setStandardButtons(QMessageBox::Yes);
  msgBox.addButton(QMessageBox::No);
  msgBox.setDefaultButton(QMessageBox::No);
  if (msgBox.exec() == QMessageBox::Yes)
  {
    QMessageBox infoBox;
    infoBox.setText("Save to " + filePath);
    infoBox.exec();

    QString strAll;
    QStringList strList;
    QFile readFile(filePath);

    if (!readFile.open(QIODevice::ReadOnly | QIODevice::Text))
    {
      infoBox.setText("Src can't be found.");
      infoBox.exec();
    }
    else
    {
      QTextStream stream(&readFile);
      strAll = stream.readAll();
      infoBox.setText("Src read.");
      infoBox.exec();
    }
    readFile.close();

    QString bkStr;
    int bkNum = 10;
    QFile writeFile(filePath);
    if (writeFile.open(QIODevice::WriteOnly | QIODevice::Text))
    {
      QTextStream stream(&writeFile);
      strList = strAll.split("\n");
      for (int i = 0; i < strList.count(); i++)
      {
        if (i == lineNum - 1)
        {
          QString tempStr = strList.at(i);
          bkStr = tempStr;
          tempStr.replace(0, tempStr.length(), new_content);
          stream << tempStr << '\n';
        }
        else if (i == bkNum - 1)
        {
          QString tempStr = strList.at(i);
          tempStr.replace(0, tempStr.length(), bkStr);
          stream << tempStr << '\n';
        }
        else
        {
          stream << strList.at(i) << '\n';
        }
      }

      infoBox.setText(filePath + " has been modified.");
      infoBox.exec();
    }
    else
    {
      infoBox.setText("Saved Failed!! Cancel!!");
      infoBox.exec();
    }
    writeFile.close();
  }
  else
  {
    QMessageBox infoBox;
    infoBox.setText("Cancel");
    infoBox.exec();
  }
}
void QtViewer::on_Btn_Front_Redo_clicked()
{
  GlobalVariable::Front_FineTune_Trigger = true;
}

//============================================================== B1 Slot
//--------------------------------------------------- Front-Left
void QtViewer::on_Slider_FrontLeft_TX_valueChanged(int value)
{
  ui->LCD_FrontLeft_TX->display(value);
  GlobalVariable::UI_PARA[0] = double(value) / 100;
  ui->Btn_FrontLeft_Redo->setEnabled(false);
}

void QtViewer::on_Slider_FrontLeft_TY_valueChanged(int value)
{
  ui->LCD_FrontLeft_TY->display(value);
  GlobalVariable::UI_PARA[1] = double(value) / 100;
  ui->Btn_FrontLeft_Redo->setEnabled(false);
}

void QtViewer::on_Slider_FrontLeft_TZ_valueChanged(int value)
{
  ui->LCD_FrontLeft_TZ->display(value);
  GlobalVariable::UI_PARA[2] = double(value) / 100;
  ui->Btn_FrontLeft_Redo->setEnabled(false);
}

void QtViewer::on_Slider_FrontLeft_RX_valueChanged(int value)
{
  ui->LCD_FrontLeft_RX->display(value);
  GlobalVariable::UI_PARA[3] = double(value) / 100;
  ui->Btn_FrontLeft_Redo->setEnabled(false);
}

void QtViewer::on_Slider_FrontLeft_RY_valueChanged(int value)
{
  ui->LCD_FrontLeft_RY->display(value);
  GlobalVariable::UI_PARA[4] = double(value) / 100;
  ui->Btn_FrontLeft_Redo->setEnabled(false);
}

void QtViewer::on_Slider_FrontLeft_RZ_valueChanged(int value)
{
  ui->LCD_FrontLeft_RZ->display(value);
  GlobalVariable::UI_PARA[5] = double(value) / 100;
  ui->Btn_FrontLeft_Redo->setEnabled(false);
}
void QtViewer::on_Btn_FrontLeft_Load_clicked()
{
  ui->Btn_FrontLeft_Reset->click();
  if (ros::param::has("LidarFrontLeft_Fine_Param"))
  {
    vector<double> LidarFrontLeft_Fine_Param;
    ros::param::get("LidarFrontLeft_Fine_Param", LidarFrontLeft_Fine_Param);

    ui->LCD_FrontLeft_TX->display(LidarFrontLeft_Fine_Param[0]);
    ui->LCD_FrontLeft_TY->display(LidarFrontLeft_Fine_Param[1]);
    ui->LCD_FrontLeft_TZ->display(LidarFrontLeft_Fine_Param[2]);
    ui->LCD_FrontLeft_RX->display(LidarFrontLeft_Fine_Param[3]);
    ui->LCD_FrontLeft_RY->display(LidarFrontLeft_Fine_Param[4]);
    ui->LCD_FrontLeft_RZ->display(LidarFrontLeft_Fine_Param[5]);

    ui->Slider_FrontLeft_TX->setValue(LidarFrontLeft_Fine_Param[0] * 100);
    ui->Slider_FrontLeft_TY->setValue(LidarFrontLeft_Fine_Param[1] * 100);
    ui->Slider_FrontLeft_TZ->setValue(LidarFrontLeft_Fine_Param[2] * 100);
    ui->Slider_FrontLeft_RX->setValue(LidarFrontLeft_Fine_Param[3] * 100);
    ui->Slider_FrontLeft_RY->setValue(LidarFrontLeft_Fine_Param[4] * 100);
    ui->Slider_FrontLeft_RZ->setValue(LidarFrontLeft_Fine_Param[5] * 100);

    GlobalVariable::UI_PARA[0] = LidarFrontLeft_Fine_Param[0];
    GlobalVariable::UI_PARA[1] = LidarFrontLeft_Fine_Param[1];
    GlobalVariable::UI_PARA[2] = LidarFrontLeft_Fine_Param[2];
    GlobalVariable::UI_PARA[3] = LidarFrontLeft_Fine_Param[3];
    GlobalVariable::UI_PARA[4] = LidarFrontLeft_Fine_Param[4];
    GlobalVariable::UI_PARA[5] = LidarFrontLeft_Fine_Param[5];

    ui->Btn_FrontLeft_Refresh->click();
  }
  else
  {
    QMessageBox msgBox;
    msgBox.setText("FrontLeft Parameter Not Found!");
    msgBox.exec();
  }
}

void QtViewer::on_Btn_FrontLeft_Reset_clicked()
{
  ui->LCD_FrontLeft_TX->display(0);
  ui->LCD_FrontLeft_TY->display(0);
  ui->LCD_FrontLeft_TZ->display(0);
  ui->LCD_FrontLeft_RX->display(0);
  ui->LCD_FrontLeft_RY->display(0);
  ui->LCD_FrontLeft_RZ->display(0);

  ui->Slider_FrontLeft_TX->setValue(0);
  ui->Slider_FrontLeft_TY->setValue(0);
  ui->Slider_FrontLeft_TZ->setValue(0);
  ui->Slider_FrontLeft_RX->setValue(0);
  ui->Slider_FrontLeft_RY->setValue(0);
  ui->Slider_FrontLeft_RZ->setValue(0);

  GlobalVariable::UI_PARA[0] = 0.0;
  GlobalVariable::UI_PARA[1] = 0.0;
  GlobalVariable::UI_PARA[2] = 0.0;
  GlobalVariable::UI_PARA[3] = 0.0;
  GlobalVariable::UI_PARA[4] = 0.0;
  GlobalVariable::UI_PARA[5] = 0.0;

  ui->Btn_FrontLeft_Refresh->click();
}

void QtViewer::on_Btn_FrontLeft_Accept_clicked()
{
  for (int i = 0; i < 6; i++)
  {
    GlobalVariable::UI_PARA_BK[i] = GlobalVariable::UI_PARA[i];
  }

  ui->Btn_FrontLeft_Redo->setEnabled(true);
  ui->Btn_FrontLeft_Save->setEnabled(true);

  ui->Btn_FrontLeft_Refresh->click();
}

void QtViewer::on_Btn_FrontLeft_Refresh_clicked()
{
  ui->Label_FrontLeft_Fine_TX->setText(QString("%1").arg(GlobalVariable::UI_PARA[0]));
  ui->Label_FrontLeft_Fine_TY->setText(QString("%1").arg(GlobalVariable::UI_PARA[1]));
  ui->Label_FrontLeft_Fine_TZ->setText(QString("%1").arg(GlobalVariable::UI_PARA[2]));
  ui->Label_FrontLeft_Fine_RX->setText(QString("%1").arg(GlobalVariable::UI_PARA[3]));
  ui->Label_FrontLeft_Fine_RY->setText(QString("%1").arg(GlobalVariable::UI_PARA[4]));
  ui->Label_FrontLeft_Fine_RZ->setText(QString("%1").arg(GlobalVariable::UI_PARA[5]));
}

void QtViewer::on_Btn_FrontLeft_Save_clicked()
{
  QString filePath = srcfile;
  QString curPath = QDir::currentPath();

  QString new_content = QString("<rosparam param=\"LidarFrontLeft_Fine_Param\">[%1, %2, %3, %4, %5, %6]</rosparam>")
                            .arg(GlobalVariable::UI_PARA[0])
                            .arg(GlobalVariable::UI_PARA[1])
                            .arg(GlobalVariable::UI_PARA[2])
                            .arg(GlobalVariable::UI_PARA[3])
                            .arg(GlobalVariable::UI_PARA[4])
                            .arg(GlobalVariable::UI_PARA[5]);

  QMessageBox msgBox;
  msgBox.setWindowTitle("CHECK PARAMETER!");
  msgBox.setText(new_content);
  msgBox.setStandardButtons(QMessageBox::Yes);
  msgBox.addButton(QMessageBox::No);
  msgBox.setDefaultButton(QMessageBox::No);

  if (msgBox.exec() == QMessageBox::Yes)
  {
    QMessageBox infoBox;
    infoBox.setText("Save to " + filePath);
    infoBox.exec();

    QString strAll;
    QStringList strList;
    QFile readFile(filePath);

    if (!readFile.open(QIODevice::ReadOnly | QIODevice::Text))
    {
      infoBox.setText("Src can't be read.");
      infoBox.exec();
    }
    else
    {
      QTextStream stream(&readFile);
      strAll = stream.readAll();
    }
    readFile.close();

    QFile writeFile(filePath);
    if (writeFile.open(QIODevice::WriteOnly | QIODevice::Text))
    {
      QTextStream stream(&writeFile);
      strList = strAll.split("\n");
      QString bkStr;

      int param_tick = 0;
      for (int i = 0; i < strList.length(); i++)
      {
        if (strList.at(i).startsWith("<rosparam param=\"LidarFrontLeft_Fine_Param\">"))
        {
          if (param_tick == 0)
          {
            QString tempStr = strList.at(i);
            bkStr = tempStr;
            tempStr.replace(0, tempStr.length(), new_content);
            stream << tempStr << '\n';
            param_tick += 1;
          }
          else
          {
            stream << bkStr << '\n';
            param_tick = 0;
          }
        }
        else
        {
          {
            stream << strList.at(i) << '\n';
          }
        }
      }
      infoBox.setText(filePath + " has been modified.");
      infoBox.exec();
    }
    writeFile.close();
  }
  else
  {
    QMessageBox infoBox;
    infoBox.setText("Cancel");
    infoBox.exec();
  }
}

void QtViewer::on_Btn_FrontLeft_Redo_clicked()
{
  GlobalVariable::FrontLeft_FineTune_Trigger = true;
}

//--------------------------------------------------- Front-Right
void QtViewer::on_Slider_FrontRight_TX_valueChanged(int value)
{
  ui->LCD_FrontRight_TX->display(value);
  GlobalVariable::UI_PARA[6] = double(value) / 100;
  ui->Btn_FrontRight_Redo->setEnabled(false);
}

void QtViewer::on_Slider_FrontRight_TY_valueChanged(int value)
{
  ui->LCD_FrontRight_TY->display(value);
  GlobalVariable::UI_PARA[7] = double(value) / 100;
  ui->Btn_FrontRight_Redo->setEnabled(false);
}

void QtViewer::on_Slider_FrontRight_TZ_valueChanged(int value)
{
  ui->LCD_FrontRight_TZ->display(value);
  GlobalVariable::UI_PARA[8] = double(value) / 100;
  ui->Btn_FrontRight_Redo->setEnabled(false);
}

void QtViewer::on_Slider_FrontRight_RX_valueChanged(int value)
{
  ui->LCD_FrontRight_RX->display(value);
  GlobalVariable::UI_PARA[9] = double(value) / 100;
  ui->Btn_FrontRight_Redo->setEnabled(false);
}

void QtViewer::on_Slider_FrontRight_RY_valueChanged(int value)
{
  ui->LCD_FrontRight_RY->display(value);
  GlobalVariable::UI_PARA[10] = double(value) / 100;
  ui->Btn_FrontRight_Redo->setEnabled(false);
}

void QtViewer::on_Slider_FrontRight_RZ_valueChanged(int value)
{
  ui->LCD_FrontRight_RZ->display(value);
  GlobalVariable::UI_PARA[11] = double(value) / 100;
  ui->Btn_FrontRight_Redo->setEnabled(false);
}

void QtViewer::on_Btn_FrontRight_Load_clicked()
{
  ui->Btn_FrontRight_Reset->click();
  if (ros::param::has("LidarFrontRight_Fine_Param"))
  {
    vector<double> LidarFrontRight_Fine_Param;
    ros::param::get("LidarFrontRight_Fine_Param", LidarFrontRight_Fine_Param);

    ui->LCD_FrontRight_TX->display(LidarFrontRight_Fine_Param[0]);
    ui->LCD_FrontRight_TY->display(LidarFrontRight_Fine_Param[1]);
    ui->LCD_FrontRight_TZ->display(LidarFrontRight_Fine_Param[2]);
    ui->LCD_FrontRight_RX->display(LidarFrontRight_Fine_Param[3]);
    ui->LCD_FrontRight_RY->display(LidarFrontRight_Fine_Param[4]);
    ui->LCD_FrontRight_RZ->display(LidarFrontRight_Fine_Param[5]);

    ui->Slider_FrontRight_TX->setValue(LidarFrontRight_Fine_Param[0] * 100);
    ui->Slider_FrontRight_TY->setValue(LidarFrontRight_Fine_Param[1] * 100);
    ui->Slider_FrontRight_TZ->setValue(LidarFrontRight_Fine_Param[2] * 100);
    ui->Slider_FrontRight_RX->setValue(LidarFrontRight_Fine_Param[3] * 100);
    ui->Slider_FrontRight_RY->setValue(LidarFrontRight_Fine_Param[4] * 100);
    ui->Slider_FrontRight_RZ->setValue(LidarFrontRight_Fine_Param[5] * 100);

    GlobalVariable::UI_PARA[6] = LidarFrontRight_Fine_Param[0];
    GlobalVariable::UI_PARA[7] = LidarFrontRight_Fine_Param[1];
    GlobalVariable::UI_PARA[8] = LidarFrontRight_Fine_Param[2];
    GlobalVariable::UI_PARA[9] = LidarFrontRight_Fine_Param[3];
    GlobalVariable::UI_PARA[10] = LidarFrontRight_Fine_Param[4];
    GlobalVariable::UI_PARA[11] = LidarFrontRight_Fine_Param[5];

    ui->Btn_FrontRight_Refresh->click();
  }
  else
  {
    QMessageBox msgBox;
    msgBox.setText("FrontRight Parameter Not Found!");
    msgBox.exec();
  }
}

void QtViewer::on_Btn_FrontRight_Reset_clicked()
{
  ui->LCD_FrontRight_TX->display(0);
  ui->LCD_FrontRight_TY->display(0);
  ui->LCD_FrontRight_TZ->display(0);
  ui->LCD_FrontRight_RX->display(0);
  ui->LCD_FrontRight_RY->display(0);
  ui->LCD_FrontRight_RZ->display(0);

  ui->Slider_FrontRight_TX->setValue(0);
  ui->Slider_FrontRight_TY->setValue(0);
  ui->Slider_FrontRight_TZ->setValue(0);
  ui->Slider_FrontRight_RX->setValue(0);
  ui->Slider_FrontRight_RY->setValue(0);
  ui->Slider_FrontRight_RZ->setValue(0);

  GlobalVariable::UI_PARA[6] = 0.0;
  GlobalVariable::UI_PARA[7] = 0.0;
  GlobalVariable::UI_PARA[8] = 0.0;
  GlobalVariable::UI_PARA[9] = 0.0;
  GlobalVariable::UI_PARA[10] = 0.0;
  GlobalVariable::UI_PARA[11] = 0.0;

  ui->Btn_FrontRight_Refresh->click();
}

void QtViewer::on_Btn_FrontRight_Accept_clicked()
{
  for (int i = 6; i < 12; i++)
  {
    GlobalVariable::UI_PARA_BK[i] = GlobalVariable::UI_PARA[i];
  }

  ui->Btn_FrontRight_Redo->setEnabled(true);
  ui->Btn_FrontRight_Save->setEnabled(true);

  ui->Btn_FrontRight_Refresh->click();
}

void QtViewer::on_Btn_FrontRight_Refresh_clicked()
{
  ui->Label_FrontRight_Fine_TX->setText(QString("%1").arg(GlobalVariable::UI_PARA[6]));
  ui->Label_FrontRight_Fine_TY->setText(QString("%1").arg(GlobalVariable::UI_PARA[7]));
  ui->Label_FrontRight_Fine_TZ->setText(QString("%1").arg(GlobalVariable::UI_PARA[8]));
  ui->Label_FrontRight_Fine_RX->setText(QString("%1").arg(GlobalVariable::UI_PARA[9]));
  ui->Label_FrontRight_Fine_RY->setText(QString("%1").arg(GlobalVariable::UI_PARA[10]));
  ui->Label_FrontRight_Fine_RZ->setText(QString("%1").arg(GlobalVariable::UI_PARA[11]));
}

void QtViewer::on_Btn_FrontRight_Save_clicked()
{
  QString new_content = QString("<rosparam param=\"LidarFrontRight_Fine_Param\">[%1, %2, %3, %4, %5, %6]</rosparam>")
                            .arg(GlobalVariable::UI_PARA[6])
                            .arg(GlobalVariable::UI_PARA[7])
                            .arg(GlobalVariable::UI_PARA[8])
                            .arg(GlobalVariable::UI_PARA[9])
                            .arg(GlobalVariable::UI_PARA[10])
                            .arg(GlobalVariable::UI_PARA[11]);

  QString filePath = srcfile;
  QString curPath = QDir::currentPath();

  QMessageBox msgBox;
  msgBox.setWindowTitle("CHECK PARAMETER!");
  msgBox.setText(new_content);
  msgBox.setStandardButtons(QMessageBox::Yes);
  msgBox.addButton(QMessageBox::No);
  msgBox.setDefaultButton(QMessageBox::No);
  if (msgBox.exec() == QMessageBox::Yes)
  {
    QMessageBox infoBox;
    infoBox.setText("Save to " + filePath);
    infoBox.exec();

    QString strAll;
    QStringList strList;
    QFile readFile(filePath);

    if (!readFile.open(QIODevice::ReadOnly | QIODevice::Text))
    {
      infoBox.setText("Src can't be found.");
      infoBox.exec();
    }
    else
    {
      QTextStream stream(&readFile);
      strAll = stream.readAll();
      infoBox.setText("Src read.");
      infoBox.exec();
    }
    readFile.close();

    QFile writeFile(filePath);
    if (writeFile.open(QIODevice::WriteOnly | QIODevice::Text))
    {
      QTextStream stream(&writeFile);
      strList = strAll.split("\n");
      QString bkStr;

      int param_tick = 0;
      for (int i = 0; i < strList.length(); i++)
      {
        if (strList.at(i).startsWith("<rosparam param=\"LidarFrontRight_Fine_Param\">"))
        {
          if (param_tick == 0)
          {
            QString tempStr = strList.at(i);
            bkStr = tempStr;
            tempStr.replace(0, tempStr.length(), new_content);
            stream << tempStr << '\n';
            param_tick += 1;
          }
          else
          {
            stream << bkStr << '\n';
            param_tick = 0;
          }
        }
        else
        {
          {
            stream << strList.at(i) << '\n';
          }
        }
      }
      infoBox.setText(filePath + " has been modified.");
      infoBox.exec();
    }
    writeFile.close();
  }
  else
  {
    QMessageBox infoBox;
    infoBox.setText("Cancel");
    infoBox.exec();
  }
}

void QtViewer::on_Btn_FrontRight_Redo_clicked()
{
  GlobalVariable::FrontRight_FineTune_Trigger = true;
}

//--------------------------------------------------- Rear-Left
void QtViewer::on_Slider_RearLeft_TX_valueChanged(int value)
{
  ui->LCD_RearLeft_TX->display(value);
  GlobalVariable::UI_PARA[12] = double(value) / 100;
  ui->Btn_RearLeft_Redo->setEnabled(false);
}

void QtViewer::on_Slider_RearLeft_TY_valueChanged(int value)
{
  ui->LCD_RearLeft_TY->display(value);
  GlobalVariable::UI_PARA[13] = double(value) / 100;
  ui->Btn_RearLeft_Redo->setEnabled(false);
}

void QtViewer::on_Slider_RearLeft_TZ_valueChanged(int value)
{
  ui->LCD_RearLeft_TZ->display(value);
  GlobalVariable::UI_PARA[14] = double(value) / 100;
  ui->Btn_RearLeft_Redo->setEnabled(false);
}

void QtViewer::on_Slider_RearLeft_RX_valueChanged(int value)
{
  ui->LCD_RearLeft_RX->display(value);
  GlobalVariable::UI_PARA[15] = double(value) / 100;
  ui->Btn_RearLeft_Redo->setEnabled(false);
}

void QtViewer::on_Slider_RearLeft_RY_valueChanged(int value)
{
  ui->LCD_RearLeft_RY->display(value);
  GlobalVariable::UI_PARA[16] = double(value) / 100;
  ui->Btn_RearLeft_Redo->setEnabled(false);
}

void QtViewer::on_Slider_RearLeft_RZ_valueChanged(int value)
{
  ui->LCD_RearLeft_RZ->display(value);
  GlobalVariable::UI_PARA[17] = double(value) / 100;
  ui->Btn_RearLeft_Redo->setEnabled(false);
}

void QtViewer::on_Btn_RearLeft_Load_clicked()
{
  ui->Btn_RearLeft_Reset->click();
  if (ros::param::has("LidarRearLeft_Fine_Param"))
  {
    vector<double> LidarRearLeft_Fine_Param;
    ros::param::get("LidarRearLeft_Fine_Param", LidarRearLeft_Fine_Param);

    ui->LCD_RearLeft_TX->display(LidarRearLeft_Fine_Param[0]);
    ui->LCD_RearLeft_TY->display(LidarRearLeft_Fine_Param[1]);
    ui->LCD_RearLeft_TZ->display(LidarRearLeft_Fine_Param[2]);
    ui->LCD_RearLeft_RX->display(LidarRearLeft_Fine_Param[3]);
    ui->LCD_RearLeft_RY->display(LidarRearLeft_Fine_Param[4]);
    ui->LCD_RearLeft_RZ->display(LidarRearLeft_Fine_Param[5]);

    ui->Slider_RearLeft_TX->setValue(LidarRearLeft_Fine_Param[0] * 100);
    ui->Slider_RearLeft_TY->setValue(LidarRearLeft_Fine_Param[1] * 100);
    ui->Slider_RearLeft_TZ->setValue(LidarRearLeft_Fine_Param[2] * 100);
    ui->Slider_RearLeft_RX->setValue(LidarRearLeft_Fine_Param[3] * 100);
    ui->Slider_RearLeft_RY->setValue(LidarRearLeft_Fine_Param[4] * 100);
    ui->Slider_RearLeft_RZ->setValue(LidarRearLeft_Fine_Param[5] * 100);

    GlobalVariable::UI_PARA[12] = LidarRearLeft_Fine_Param[0];
    GlobalVariable::UI_PARA[13] = LidarRearLeft_Fine_Param[1];
    GlobalVariable::UI_PARA[14] = LidarRearLeft_Fine_Param[2];
    GlobalVariable::UI_PARA[15] = LidarRearLeft_Fine_Param[3];
    GlobalVariable::UI_PARA[16] = LidarRearLeft_Fine_Param[4];
    GlobalVariable::UI_PARA[17] = LidarRearLeft_Fine_Param[5];

    ui->Btn_RearLeft_Refresh->click();
  }
  else
  {
    QMessageBox msgBox;
    msgBox.setText("RearLeft Parameter Not Found!");
    msgBox.exec();
  }
}

void QtViewer::on_Btn_RearLeft_Reset_clicked()
{
  ui->LCD_RearLeft_TX->display(0);
  ui->LCD_RearLeft_TY->display(0);
  ui->LCD_RearLeft_TZ->display(0);
  ui->LCD_RearLeft_RX->display(0);
  ui->LCD_RearLeft_RY->display(0);
  ui->LCD_RearLeft_RZ->display(0);

  ui->Slider_RearLeft_TX->setValue(0);
  ui->Slider_RearLeft_TY->setValue(0);
  ui->Slider_RearLeft_TZ->setValue(0);
  ui->Slider_RearLeft_RX->setValue(0);
  ui->Slider_RearLeft_RY->setValue(0);
  ui->Slider_RearLeft_RZ->setValue(0);

  GlobalVariable::UI_PARA[12] = 0.0;
  GlobalVariable::UI_PARA[13] = 0.0;
  GlobalVariable::UI_PARA[14] = 0.0;
  GlobalVariable::UI_PARA[15] = 0.0;
  GlobalVariable::UI_PARA[16] = 0.0;
  GlobalVariable::UI_PARA[17] = 0.0;

  ui->Btn_RearLeft_Refresh->click();
}

void QtViewer::on_Btn_RearLeft_Accept_clicked()
{
  for (int i = 12; i < 18; i++)
  {
    GlobalVariable::UI_PARA_BK[i] = GlobalVariable::UI_PARA[i];
  }

  ui->Btn_RearLeft_Redo->setEnabled(true);
  ui->Btn_RearLeft_Save->setEnabled(true);

  ui->Btn_RearLeft_Refresh->click();
}

void QtViewer::on_Btn_RearLeft_Refresh_clicked()
{
  ui->Label_RearLeft_Fine_TX->setText(QString("%1").arg(GlobalVariable::UI_PARA[12]));
  ui->Label_RearLeft_Fine_TY->setText(QString("%1").arg(GlobalVariable::UI_PARA[13]));
  ui->Label_RearLeft_Fine_TZ->setText(QString("%1").arg(GlobalVariable::UI_PARA[14]));
  ui->Label_RearLeft_Fine_RX->setText(QString("%1").arg(GlobalVariable::UI_PARA[15]));
  ui->Label_RearLeft_Fine_RY->setText(QString("%1").arg(GlobalVariable::UI_PARA[16]));
  ui->Label_RearLeft_Fine_RZ->setText(QString("%1").arg(GlobalVariable::UI_PARA[17]));
}

void QtViewer::on_Btn_RearLeft_Save_clicked()
{
  QString new_content = QString("<rosparam param=\"LidarRearLeft_Fine_Param\">[%1, %2, %3, %4, %5, %6]</rosparam>")
                            .arg(GlobalVariable::UI_PARA[12])
                            .arg(GlobalVariable::UI_PARA[13])
                            .arg(GlobalVariable::UI_PARA[14])
                            .arg(GlobalVariable::UI_PARA[15])
                            .arg(GlobalVariable::UI_PARA[16])
                            .arg(GlobalVariable::UI_PARA[17]);
  QString filePath = srcfile;
  QString curPath = QDir::currentPath();

  QMessageBox msgBox;
  msgBox.setWindowTitle("CHECK PARAMETER!");
  msgBox.setText(new_content);
  msgBox.setStandardButtons(QMessageBox::Yes);
  msgBox.addButton(QMessageBox::No);
  msgBox.setDefaultButton(QMessageBox::No);
  if (msgBox.exec() == QMessageBox::Yes)
  {
    QMessageBox infoBox;
    infoBox.setText("Save to " + filePath);
    infoBox.exec();

    QString strAll;
    QStringList strList;
    QFile readFile(filePath);

    if (!readFile.open(QIODevice::ReadOnly | QIODevice::Text))
    {
      infoBox.setText("Src can't be found.");
      infoBox.exec();
    }
    else
    {
      QTextStream stream(&readFile);
      strAll = stream.readAll();
      infoBox.setText("Src read.");
      infoBox.exec();
    }
    readFile.close();

    QFile writeFile(filePath);
    if (writeFile.open(QIODevice::WriteOnly | QIODevice::Text))
    {
      QTextStream stream(&writeFile);
      strList = strAll.split("\n");
      QString bkStr;

      int param_tick = 0;
      for (int i = 0; i < strList.length(); i++)
      {
        if (strList.at(i).startsWith("<rosparam param=\"LidarRearLeft_Fine_Param\">"))
        {
          if (param_tick == 0)
          {
            QString tempStr = strList.at(i);
            bkStr = tempStr;
            tempStr.replace(0, tempStr.length(), new_content);
            stream << tempStr << '\n';
            param_tick += 1;
          }
          else
          {
            stream << bkStr << '\n';
            param_tick = 0;
          }
        }
        else
        {
          {
            stream << strList.at(i) << '\n';
          }
        }
      }
      infoBox.setText(filePath + " has been modified.");
      infoBox.exec();
    }
    writeFile.close();
  }
  else
  {
    QMessageBox infoBox;
    infoBox.setText("Cancel");
    infoBox.exec();
  }
}

void QtViewer::on_Btn_RearLeft_Redo_clicked()
{
  GlobalVariable::RearLeft_FineTune_Trigger = true;
}

//--------------------------------------------------- Rear-Right
void QtViewer::on_Slider_RearRight_TX_valueChanged(int value)
{
  ui->LCD_RearRight_TX->display(value);
  GlobalVariable::UI_PARA[18] = double(value) / 100;
  ui->Btn_RearRight_Redo->setEnabled(false);
}

void QtViewer::on_Slider_RearRight_TY_valueChanged(int value)
{
  ui->LCD_RearRight_TY->display(value);
  GlobalVariable::UI_PARA[19] = double(value) / 100;
  ui->Btn_RearRight_Redo->setEnabled(false);
}

void QtViewer::on_Slider_RearRight_TZ_valueChanged(int value)
{
  ui->LCD_RearRight_TZ->display(value);
  GlobalVariable::UI_PARA[20] = double(value) / 100;
  ui->Btn_RearRight_Redo->setEnabled(false);
}

void QtViewer::on_Slider_RearRight_RX_valueChanged(int value)
{
  ui->LCD_RearRight_RX->display(value);
  GlobalVariable::UI_PARA[21] = double(value) / 100;
  ui->Btn_RearRight_Redo->setEnabled(false);
}

void QtViewer::on_Slider_RearRight_RY_valueChanged(int value)
{
  ui->LCD_RearRight_RY->display(value);
  GlobalVariable::UI_PARA[22] = double(value) / 100;
  ui->Btn_RearRight_Redo->setEnabled(false);
}

void QtViewer::on_Slider_RearRight_RZ_valueChanged(int value)
{
  ui->LCD_RearRight_RZ->display(value);
  GlobalVariable::UI_PARA[23] = double(value) / 100;
  ui->Btn_RearRight_Redo->setEnabled(false);
}

void QtViewer::on_Btn_RearRight_Load_clicked()
{
  ui->Btn_RearRight_Reset->click();
  if (ros::param::has("LidarRearRight_Fine_Param"))
  {
    vector<double> LidarRearRight_Fine_Param;
    ros::param::get("LidarRearRight_Fine_Param", LidarRearRight_Fine_Param);

    ui->LCD_RearRight_TX->display(LidarRearRight_Fine_Param[0]);
    ui->LCD_RearRight_TY->display(LidarRearRight_Fine_Param[1]);
    ui->LCD_RearRight_TZ->display(LidarRearRight_Fine_Param[2]);
    ui->LCD_RearRight_RX->display(LidarRearRight_Fine_Param[3]);
    ui->LCD_RearRight_RY->display(LidarRearRight_Fine_Param[4]);
    ui->LCD_RearRight_RZ->display(LidarRearRight_Fine_Param[5]);

    ui->Slider_RearRight_TX->setValue(LidarRearRight_Fine_Param[0] * 100);
    ui->Slider_RearRight_TY->setValue(LidarRearRight_Fine_Param[1] * 100);
    ui->Slider_RearRight_TZ->setValue(LidarRearRight_Fine_Param[2] * 100);
    ui->Slider_RearRight_RX->setValue(LidarRearRight_Fine_Param[3] * 100);
    ui->Slider_RearRight_RY->setValue(LidarRearRight_Fine_Param[4] * 100);
    ui->Slider_RearRight_RZ->setValue(LidarRearRight_Fine_Param[5] * 100);

    GlobalVariable::UI_PARA[18] = LidarRearRight_Fine_Param[0];
    GlobalVariable::UI_PARA[19] = LidarRearRight_Fine_Param[1];
    GlobalVariable::UI_PARA[20] = LidarRearRight_Fine_Param[2];
    GlobalVariable::UI_PARA[21] = LidarRearRight_Fine_Param[3];
    GlobalVariable::UI_PARA[22] = LidarRearRight_Fine_Param[4];
    GlobalVariable::UI_PARA[23] = LidarRearRight_Fine_Param[5];

    ui->Btn_RearRight_Refresh->click();
  }
  else
  {
    QMessageBox msgBox;
    msgBox.setText("RearRight Parameter Not Found!");
    msgBox.exec();
  }
}

void QtViewer::on_Btn_RearRight_Reset_clicked()
{
  ui->LCD_RearRight_TX->display(0);
  ui->LCD_RearRight_TY->display(0);
  ui->LCD_RearRight_TZ->display(0);
  ui->LCD_RearRight_RX->display(0);
  ui->LCD_RearRight_RY->display(0);
  ui->LCD_RearRight_RZ->display(0);

  ui->Slider_RearRight_TX->setValue(0);
  ui->Slider_RearRight_TY->setValue(0);
  ui->Slider_RearRight_TZ->setValue(0);
  ui->Slider_RearRight_RX->setValue(0);
  ui->Slider_RearRight_RY->setValue(0);
  ui->Slider_RearRight_RZ->setValue(0);

  GlobalVariable::UI_PARA[18] = 0.0;
  GlobalVariable::UI_PARA[19] = 0.0;
  GlobalVariable::UI_PARA[20] = 0.0;
  GlobalVariable::UI_PARA[21] = 0.0;
  GlobalVariable::UI_PARA[22] = 0.0;
  GlobalVariable::UI_PARA[23] = 0.0;

  ui->Btn_RearRight_Refresh->click();
}

void QtViewer::on_Btn_RearRight_Accept_clicked()
{
  for (int i = 18; i < 24; i++)
  {
    GlobalVariable::UI_PARA_BK[i] = GlobalVariable::UI_PARA[i];
  }

  ui->Btn_RearRight_Redo->setEnabled(true);
  ui->Btn_RearRight_Save->setEnabled(true);

  ui->Btn_RearRight_Refresh->click();
}

void QtViewer::on_Btn_RearRight_Refresh_clicked()
{
  ui->Label_RearRight_Fine_TX->setText(QString("%1").arg(GlobalVariable::UI_PARA[18]));
  ui->Label_RearRight_Fine_TY->setText(QString("%1").arg(GlobalVariable::UI_PARA[19]));
  ui->Label_RearRight_Fine_TZ->setText(QString("%1").arg(GlobalVariable::UI_PARA[20]));
  ui->Label_RearRight_Fine_RX->setText(QString("%1").arg(GlobalVariable::UI_PARA[21]));
  ui->Label_RearRight_Fine_RY->setText(QString("%1").arg(GlobalVariable::UI_PARA[22]));
  ui->Label_RearRight_Fine_RZ->setText(QString("%1").arg(GlobalVariable::UI_PARA[23]));
}

void QtViewer::on_Btn_RearRight_Save_clicked()
{
  QString new_content = QString("<rosparam param=\"LidarRearRight_Fine_Param\">[%1, %2, %3, %4, %5, %6]</rosparam>")
                            .arg(GlobalVariable::UI_PARA[18])
                            .arg(GlobalVariable::UI_PARA[19])
                            .arg(GlobalVariable::UI_PARA[20])
                            .arg(GlobalVariable::UI_PARA[21])
                            .arg(GlobalVariable::UI_PARA[22])
                            .arg(GlobalVariable::UI_PARA[23]);

  QString filePath = srcfile;
  QString curPath = QDir::currentPath();

  QMessageBox msgBox;
  msgBox.setWindowTitle("CHECK PARAMETER!");
  msgBox.setText(new_content);
  msgBox.setStandardButtons(QMessageBox::Yes);
  msgBox.addButton(QMessageBox::No);
  msgBox.setDefaultButton(QMessageBox::No);
  if (msgBox.exec() == QMessageBox::Yes)
  {
    QMessageBox infoBox;
    infoBox.setText("Save to " + filePath);
    infoBox.exec();

    QString strAll;
    QStringList strList;
    QFile readFile(filePath);

    if (!readFile.open(QIODevice::ReadOnly | QIODevice::Text))
    {
      infoBox.setText("Src can't be found.");
      infoBox.exec();
    }
    else
    {
      QTextStream stream(&readFile);
      strAll = stream.readAll();
      infoBox.setText("Src read.");
      infoBox.exec();
    }
    readFile.close();
    QFile writeFile(filePath);
    if (writeFile.open(QIODevice::WriteOnly | QIODevice::Text))
    {
      QTextStream stream(&writeFile);
      strList = strAll.split("\n");
      QString bkStr;

      int param_tick = 0;
      for (int i = 0; i < strList.length(); i++)
      {
        if (strList.at(i).startsWith("<rosparam param=\"LidarRearRight_Fine_Param\">"))
        {
          if (param_tick == 0)
          {
            QString tempStr = strList.at(i);
            bkStr = tempStr;
            tempStr.replace(0, tempStr.length(), new_content);
            stream << tempStr << '\n';
            param_tick += 1;
          }
          else
          {
            stream << bkStr << '\n';
            param_tick = 0;
          }
        }
        else
        {
          {
            stream << strList.at(i) << '\n';
          }
        }
      }
      infoBox.setText(filePath + " has been modified.");
      infoBox.exec();
    }
    writeFile.close();
  }
  else
  {
    QMessageBox infoBox;
    infoBox.setText("Cancel");
    infoBox.exec();
  }
}

void QtViewer::on_Btn_RearRight_Redo_clicked()
{
  GlobalVariable::RearRight_FineTune_Trigger = true;
}

//============================================================== End of B1 Slot

//--------------------------------------------------- Write Launch file  Test
void QtViewer::on_Btn_Test_clicked()
{
}
