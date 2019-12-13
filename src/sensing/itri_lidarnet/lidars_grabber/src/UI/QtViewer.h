#ifndef QTVIEWER_H
#define QTVIEWER_H

#include "../all_header.h"
#include "../GlobalVariable.h"


#include "ui_QtViewer.h"


namespace Ui
{
  class QtViewer;
}

class QtViewer : public QMainWindow
{
  Q_OBJECT

  public:
    explicit
    QtViewer (QWidget *parent = 0);
    ~QtViewer ();

  public slots:


  protected:
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

private slots:

    //----------------------------------------------------Menu
    void on_mode_Truck_clicked();
    void on_mode_Bus_clicked();
    void on_Btn_Test_clicked();

    //======================================================== Hino
    //----------------------------------------------------Left
    void on_Btn_Left_Load_clicked();
    void on_Btn_Left_Reset_clicked();
    void on_Btn_Left_Accept_clicked();
    void on_Slider_Left_TX_valueChanged(int value);
    void on_Slider_Left_TY_valueChanged(int value);
    void on_Slider_Left_TZ_valueChanged(int value);
    void on_Slider_Left_RX_valueChanged(int value);
    void on_Slider_Left_RY_valueChanged(int value);
    void on_Slider_Left_RZ_valueChanged(int value);
    void on_Btn_Left_Save_clicked();
    void on_Btn_Left_Redo_clicked();
    void on_Btn_Left_Refresh_clicked();

    //----------------------------------------------------Right
    void on_Btn_Right_Load_clicked();
    void on_Btn_Right_Reset_clicked();
    void on_Slider_Right_TX_valueChanged(int value);
    void on_Slider_Right_TY_valueChanged(int value);
    void on_Slider_Right_TZ_valueChanged(int value);
    void on_Slider_Right_RX_valueChanged(int value);
    void on_Slider_Right_RY_valueChanged(int value);
    void on_Slider_Right_RZ_valueChanged(int value);
    void on_Btn_Right_Accept_clicked();
    void on_Btn_Right_Save_clicked();
    void on_Btn_Right_Redo_clicked();
    void on_Btn_Right_Refresh_clicked();

    //----------------------------------------------------Front
    void on_Btn_Front_Load_clicked();
    void on_Btn_Front_Reset_clicked();
    void on_Slider_Front_TX_valueChanged(int value);
    void on_Slider_Front_TY_valueChanged(int value);
    void on_Slider_Front_TZ_valueChanged(int value);
    void on_Slider_Front_RX_valueChanged(int value);
    void on_Slider_Front_RY_valueChanged(int value);
    void on_Slider_Front_RZ_valueChanged(int value);
    void on_Btn_Front_Accept_clicked();
    void on_Btn_Front_Save_clicked();
    void on_Btn_Front_Redo_clicked();
    void on_Btn_Front_Refresh_clicked();

    //======================================================== B1
    //----------------------------------------------Front-Left
    void on_Btn_FrontLeft_Load_clicked();
    void on_Btn_FrontLeft_Reset_clicked();
    void on_Btn_FrontLeft_Accept_clicked();
    void on_Slider_FrontLeft_TX_valueChanged(int value);
    void on_Slider_FrontLeft_TY_valueChanged(int value);
    void on_Slider_FrontLeft_TZ_valueChanged(int value);
    void on_Slider_FrontLeft_RX_valueChanged(int value);
    void on_Slider_FrontLeft_RY_valueChanged(int value);
    void on_Slider_FrontLeft_RZ_valueChanged(int value);
    void on_Btn_FrontLeft_Save_clicked();
    void on_Btn_FrontLeft_Redo_clicked();
    void on_Btn_FrontLeft_Refresh_clicked();

    //----------------------------------------------Front-Right
    void on_Btn_FrontRight_Load_clicked();
    void on_Btn_FrontRight_Reset_clicked();
    void on_Btn_FrontRight_Accept_clicked();
    void on_Slider_FrontRight_TX_valueChanged(int value);
    void on_Slider_FrontRight_TY_valueChanged(int value);
    void on_Slider_FrontRight_TZ_valueChanged(int value);
    void on_Slider_FrontRight_RX_valueChanged(int value);
    void on_Slider_FrontRight_RY_valueChanged(int value);
    void on_Slider_FrontRight_RZ_valueChanged(int value);
    void on_Btn_FrontRight_Save_clicked();
    void on_Btn_FrontRight_Redo_clicked();
    void on_Btn_FrontRight_Refresh_clicked();

    //----------------------------------------------Rear-Left
    void on_Btn_RearLeft_Load_clicked();
    void on_Btn_RearLeft_Reset_clicked();
    void on_Btn_RearLeft_Accept_clicked();
    void on_Slider_RearLeft_TX_valueChanged(int value);
    void on_Slider_RearLeft_TY_valueChanged(int value);
    void on_Slider_RearLeft_TZ_valueChanged(int value);
    void on_Slider_RearLeft_RX_valueChanged(int value);
    void on_Slider_RearLeft_RY_valueChanged(int value);
    void on_Slider_RearLeft_RZ_valueChanged(int value);
    void on_Btn_RearLeft_Save_clicked();
    void on_Btn_RearLeft_Redo_clicked();
    void on_Btn_RearLeft_Refresh_clicked();

    //----------------------------------------------Rear-Right
    void on_Btn_RearRight_Load_clicked();
    void on_Btn_RearRight_Reset_clicked();
    void on_Btn_RearRight_Accept_clicked();
    void on_Slider_RearRight_TX_valueChanged(int value);
    void on_Slider_RearRight_TY_valueChanged(int value);
    void on_Slider_RearRight_TZ_valueChanged(int value);
    void on_Slider_RearRight_RX_valueChanged(int value);
    void on_Slider_RearRight_RY_valueChanged(int value);
    void on_Slider_RearRight_RZ_valueChanged(int value);
    void on_Btn_RearRight_Save_clicked();
    void on_Btn_RearRight_Redo_clicked();
    void on_Btn_RearRight_Refresh_clicked();

    //======================================================== End of B1

private:
    Ui::QtViewer *ui;

    boost::property_tree::ptree boost_ptree;

};

#endif // QTVIEWER_H

