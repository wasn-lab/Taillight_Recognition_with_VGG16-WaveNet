/********************************************************************************
 ** Form generated from reading UI file 'QtViewer.ui'
 **
 ** Created by: Qt User Interface Compiler version 5.9.1
 **
 ** WARNING! All changes made in this file will be lost when recompiling UI file!
 ********************************************************************************/

#ifndef UI_QTVIEWER_H
#define UI_QTVIEWER_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLCDNumber>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QScrollBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QWidget>
#include "QVTKWidget.h"

QT_BEGIN_NAMESPACE

class Ui_QtViewer
{
  public:
    QWidget *centralwidget;
    QTabWidget *tabWidget;
    QWidget *tab0;
    QScrollBar *horizontalScrollBar0;
    QScrollBar *horizontalScrollBar1;
    QScrollBar *horizontalScrollBar2;
    QScrollBar *horizontalScrollBar3;
    QScrollBar *horizontalScrollBar4;
    QScrollBar *horizontalScrollBar5;
    QLCDNumber *lcdNumber0;
    QLCDNumber *lcdNumber1;
    QLCDNumber *lcdNumber3;
    QLCDNumber *lcdNumber4;
    QLCDNumber *lcdNumber5;
    QLCDNumber *lcdNumber2;
    QLabel *label0;
    QRadioButton *radioButton1;
    QRadioButton *radioButton0;
    QLabel *label1;
    QLabel *label2;
    QLabel *label3;
    QLabel *label4;
    QLabel *label5;
    QCheckBox *checkBox1;
    QRadioButton *radioButton2;
    QCheckBox *checkBox3;
    QLineEdit *lineEdit12;
    QLabel *label6;
    QWidget *tab4;
    QScrollBar *horizontalScrollBar11;
    QLCDNumber *lcdNumber7;
    QLCDNumber *lcdNumber10;
    QScrollBar *horizontalScrollBar7;
    QLCDNumber *lcdNumber8;
    QScrollBar *horizontalScrollBar9;
    QLCDNumber *lcdNumber6;
    QScrollBar *horizontalScrollBar6;
    QScrollBar *horizontalScrollBar8;
    QScrollBar *horizontalScrollBar10;
    QLCDNumber *lcdNumber11;
    QLCDNumber *lcdNumber9;
    QVTKWidget *qvtkWidget;
    QVTKWidget *qvtkWidget_2;

    void
    setupUi (QMainWindow *QtViewer)
    {
      if (QtViewer->objectName ().isEmpty ())
        QtViewer->setObjectName (QStringLiteral("QtViewer"));
      QtViewer->resize (452, 282);
      QtViewer->setMinimumSize (QSize (0, 0));
      QtViewer->setMaximumSize (QSize (5000, 5000));
      centralwidget = new QWidget (QtViewer);
      centralwidget->setObjectName (QStringLiteral("centralwidget"));
      tabWidget = new QTabWidget (centralwidget);
      tabWidget->setObjectName (QStringLiteral("tabWidget"));
      tabWidget->setGeometry (QRect (0, 0, 451, 281));
      tab0 = new QWidget ();
      tab0->setObjectName (QStringLiteral("tab0"));
      horizontalScrollBar0 = new QScrollBar (tab0);
      horizontalScrollBar0->setObjectName (QStringLiteral("horizontalScrollBar0"));
      horizontalScrollBar0->setGeometry (QRect (10, 10, 271, 16));
      horizontalScrollBar0->setMinimum (0);
      horizontalScrollBar0->setMaximum (50);
      horizontalScrollBar0->setPageStep (2);
      horizontalScrollBar0->setSliderPosition (0);
      horizontalScrollBar0->setOrientation (Qt::Horizontal);
      horizontalScrollBar1 = new QScrollBar (tab0);
      horizontalScrollBar1->setObjectName (QStringLiteral("horizontalScrollBar1"));
      horizontalScrollBar1->setGeometry (QRect (10, 40, 271, 16));
      horizontalScrollBar1->setMaximum (999);
      horizontalScrollBar1->setOrientation (Qt::Horizontal);
      horizontalScrollBar2 = new QScrollBar (tab0);
      horizontalScrollBar2->setObjectName (QStringLiteral("horizontalScrollBar2"));
      horizontalScrollBar2->setGeometry (QRect (10, 70, 271, 16));
      horizontalScrollBar2->setMaximum (999);
      horizontalScrollBar2->setOrientation (Qt::Horizontal);
      horizontalScrollBar3 = new QScrollBar (tab0);
      horizontalScrollBar3->setObjectName (QStringLiteral("horizontalScrollBar3"));
      horizontalScrollBar3->setGeometry (QRect (10, 100, 271, 16));
      horizontalScrollBar3->setMaximum (999);
      horizontalScrollBar3->setOrientation (Qt::Horizontal);
      horizontalScrollBar4 = new QScrollBar (tab0);
      horizontalScrollBar4->setObjectName (QStringLiteral("horizontalScrollBar4"));
      horizontalScrollBar4->setGeometry (QRect (10, 130, 271, 16));
      horizontalScrollBar4->setMaximum (999);
      horizontalScrollBar4->setOrientation (Qt::Horizontal);
      horizontalScrollBar5 = new QScrollBar (tab0);
      horizontalScrollBar5->setObjectName (QStringLiteral("horizontalScrollBar5"));
      horizontalScrollBar5->setGeometry (QRect (10, 160, 271, 16));
      horizontalScrollBar5->setMaximum (999);
      horizontalScrollBar5->setOrientation (Qt::Horizontal);
      lcdNumber0 = new QLCDNumber (tab0);
      lcdNumber0->setObjectName (QStringLiteral("lcdNumber0"));
      lcdNumber0->setGeometry (QRect (290, 10, 41, 21));
      lcdNumber0->setDigitCount (3);
      lcdNumber0->setSegmentStyle (QLCDNumber::Flat);
      lcdNumber0->setProperty ("intValue", QVariant (0));
      lcdNumber1 = new QLCDNumber (tab0);
      lcdNumber1->setObjectName (QStringLiteral("lcdNumber1"));
      lcdNumber1->setGeometry (QRect (290, 40, 41, 21));
      lcdNumber1->setDigitCount (3);
      lcdNumber1->setSegmentStyle (QLCDNumber::Flat);
      lcdNumber1->setProperty ("intValue", QVariant (0));
      lcdNumber3 = new QLCDNumber (tab0);
      lcdNumber3->setObjectName (QStringLiteral("lcdNumber3"));
      lcdNumber3->setGeometry (QRect (290, 100, 41, 21));
      lcdNumber3->setDigitCount (3);
      lcdNumber3->setSegmentStyle (QLCDNumber::Flat);
      lcdNumber3->setProperty ("intValue", QVariant (0));
      lcdNumber4 = new QLCDNumber (tab0);
      lcdNumber4->setObjectName (QStringLiteral("lcdNumber4"));
      lcdNumber4->setGeometry (QRect (290, 130, 41, 21));
      lcdNumber4->setDigitCount (3);
      lcdNumber4->setSegmentStyle (QLCDNumber::Flat);
      lcdNumber4->setProperty ("intValue", QVariant (0));
      lcdNumber5 = new QLCDNumber (tab0);
      lcdNumber5->setObjectName (QStringLiteral("lcdNumber5"));
      lcdNumber5->setGeometry (QRect (290, 160, 41, 21));
      lcdNumber5->setDigitCount (3);
      lcdNumber5->setSegmentStyle (QLCDNumber::Flat);
      lcdNumber5->setProperty ("intValue", QVariant (0));
      lcdNumber2 = new QLCDNumber (tab0);
      lcdNumber2->setObjectName (QStringLiteral("lcdNumber2"));
      lcdNumber2->setGeometry (QRect (290, 70, 41, 21));
      lcdNumber2->setDigitCount (3);
      lcdNumber2->setSegmentStyle (QLCDNumber::Flat);
      lcdNumber2->setProperty ("intValue", QVariant (0));
      label0 = new QLabel (tab0);
      label0->setObjectName (QStringLiteral("label0"));
      label0->setGeometry (QRect (340, 10, 91, 17));
      radioButton1 = new QRadioButton (tab0);
      radioButton1->setObjectName (QStringLiteral("radioButton1"));
      radioButton1->setGeometry (QRect (70, 190, 81, 22));
      radioButton0 = new QRadioButton (tab0);
      radioButton0->setObjectName (QStringLiteral("radioButton0"));
      radioButton0->setGeometry (QRect (10, 190, 61, 22));
      label1 = new QLabel (tab0);
      label1->setObjectName (QStringLiteral("label1"));
      label1->setGeometry (QRect (340, 40, 91, 17));
      label2 = new QLabel (tab0);
      label2->setObjectName (QStringLiteral("label2"));
      label2->setGeometry (QRect (340, 70, 91, 17));
      label3 = new QLabel (tab0);
      label3->setObjectName (QStringLiteral("label3"));
      label3->setGeometry (QRect (340, 100, 91, 17));
      label4 = new QLabel (tab0);
      label4->setObjectName (QStringLiteral("label4"));
      label4->setGeometry (QRect (340, 130, 91, 17));
      label5 = new QLabel (tab0);
      label5->setObjectName (QStringLiteral("label5"));
      label5->setGeometry (QRect (340, 160, 91, 17));
      checkBox1 = new QCheckBox (tab0);
      checkBox1->setObjectName (QStringLiteral("checkBox1"));
      checkBox1->setGeometry (QRect (80, 220, 121, 23));
      radioButton2 = new QRadioButton (tab0);
      radioButton2->setObjectName (QStringLiteral("radioButton2"));
      radioButton2->setGeometry (QRect (150, 190, 71, 22));
      checkBox3 = new QCheckBox (tab0);
      checkBox3->setObjectName (QStringLiteral("checkBox3"));
      checkBox3->setGeometry (QRect (10, 220, 71, 23));
      lineEdit12 = new QLineEdit (tab0);
      lineEdit12->setObjectName (QStringLiteral("lineEdit12"));
      lineEdit12->setGeometry (QRect (220, 190, 111, 25));
      lineEdit12->setInputMethodHints (Qt::ImhDigitsOnly | Qt::ImhFormattedNumbersOnly);
      lineEdit12->setMaxLength (9);
      lineEdit12->setFrame (true);
      lineEdit12->setClearButtonEnabled (false);
      label6 = new QLabel (tab0);
      label6->setObjectName (QStringLiteral("label6"));
      label6->setGeometry (QRect (340, 190, 111, 17));
      tabWidget->addTab (tab0, QString ());
      tab4 = new QWidget ();
      tab4->setObjectName (QStringLiteral("tab4"));
      horizontalScrollBar11 = new QScrollBar (tab4);
      horizontalScrollBar11->setObjectName (QStringLiteral("horizontalScrollBar11"));
      horizontalScrollBar11->setGeometry (QRect (10, 160, 271, 16));
      horizontalScrollBar11->setMinimum (-999);
      horizontalScrollBar11->setMaximum (999);
      horizontalScrollBar11->setOrientation (Qt::Horizontal);
      lcdNumber7 = new QLCDNumber (tab4);
      lcdNumber7->setObjectName (QStringLiteral("lcdNumber7"));
      lcdNumber7->setGeometry (QRect (290, 40, 71, 21));
      lcdNumber7->setDigitCount (4);
      lcdNumber7->setSegmentStyle (QLCDNumber::Flat);
      lcdNumber7->setProperty ("intValue", QVariant (0));
      lcdNumber10 = new QLCDNumber (tab4);
      lcdNumber10->setObjectName (QStringLiteral("lcdNumber10"));
      lcdNumber10->setGeometry (QRect (290, 130, 71, 21));
      lcdNumber10->setDigitCount (4);
      lcdNumber10->setSegmentStyle (QLCDNumber::Flat);
      lcdNumber10->setProperty ("intValue", QVariant (0));
      horizontalScrollBar7 = new QScrollBar (tab4);
      horizontalScrollBar7->setObjectName (QStringLiteral("horizontalScrollBar7"));
      horizontalScrollBar7->setGeometry (QRect (10, 40, 271, 16));
      horizontalScrollBar7->setMinimum (-999);
      horizontalScrollBar7->setMaximum (999);
      horizontalScrollBar7->setOrientation (Qt::Horizontal);
      lcdNumber8 = new QLCDNumber (tab4);
      lcdNumber8->setObjectName (QStringLiteral("lcdNumber8"));
      lcdNumber8->setGeometry (QRect (290, 70, 71, 21));
      lcdNumber8->setDigitCount (4);
      lcdNumber8->setSegmentStyle (QLCDNumber::Flat);
      lcdNumber8->setProperty ("intValue", QVariant (0));
      horizontalScrollBar9 = new QScrollBar (tab4);
      horizontalScrollBar9->setObjectName (QStringLiteral("horizontalScrollBar9"));
      horizontalScrollBar9->setGeometry (QRect (10, 100, 271, 16));
      horizontalScrollBar9->setMinimum (-999);
      horizontalScrollBar9->setMaximum (999);
      horizontalScrollBar9->setOrientation (Qt::Horizontal);
      lcdNumber6 = new QLCDNumber (tab4);
      lcdNumber6->setObjectName (QStringLiteral("lcdNumber6"));
      lcdNumber6->setGeometry (QRect (290, 10, 71, 21));
      lcdNumber6->setDigitCount (4);
      lcdNumber6->setSegmentStyle (QLCDNumber::Flat);
      lcdNumber6->setProperty ("intValue", QVariant (0));
      horizontalScrollBar6 = new QScrollBar (tab4);
      horizontalScrollBar6->setObjectName (QStringLiteral("horizontalScrollBar6"));
      horizontalScrollBar6->setGeometry (QRect (10, 10, 271, 16));
      horizontalScrollBar6->setMinimum (-999);
      horizontalScrollBar6->setMaximum (999);
      horizontalScrollBar6->setPageStep (10);
      horizontalScrollBar6->setValue (0);
      horizontalScrollBar6->setSliderPosition (0);
      horizontalScrollBar6->setOrientation (Qt::Horizontal);
      horizontalScrollBar8 = new QScrollBar (tab4);
      horizontalScrollBar8->setObjectName (QStringLiteral("horizontalScrollBar8"));
      horizontalScrollBar8->setGeometry (QRect (10, 70, 271, 16));
      horizontalScrollBar8->setMinimum (-999);
      horizontalScrollBar8->setMaximum (999);
      horizontalScrollBar8->setOrientation (Qt::Horizontal);
      horizontalScrollBar10 = new QScrollBar (tab4);
      horizontalScrollBar10->setObjectName (QStringLiteral("horizontalScrollBar10"));
      horizontalScrollBar10->setGeometry (QRect (10, 130, 271, 16));
      horizontalScrollBar10->setMinimum (-999);
      horizontalScrollBar10->setMaximum (999);
      horizontalScrollBar10->setOrientation (Qt::Horizontal);
      lcdNumber11 = new QLCDNumber (tab4);
      lcdNumber11->setObjectName (QStringLiteral("lcdNumber11"));
      lcdNumber11->setGeometry (QRect (290, 160, 71, 21));
      lcdNumber11->setDigitCount (4);
      lcdNumber11->setSegmentStyle (QLCDNumber::Flat);
      lcdNumber11->setProperty ("intValue", QVariant (0));
      lcdNumber9 = new QLCDNumber (tab4);
      lcdNumber9->setObjectName (QStringLiteral("lcdNumber9"));
      lcdNumber9->setGeometry (QRect (290, 100, 71, 21));
      lcdNumber9->setDigitCount (4);
      lcdNumber9->setSegmentStyle (QLCDNumber::Flat);
      lcdNumber9->setProperty ("intValue", QVariant (0));
      qvtkWidget = new QVTKWidget (tab4);
      qvtkWidget->setObjectName (QStringLiteral("qvtkWidget"));
      qvtkWidget->setGeometry (QRect (380, 10, 41, 171));
      qvtkWidget_2 = new QVTKWidget (tab4);
      qvtkWidget_2->setObjectName (QStringLiteral("qvtkWidget_2"));
      qvtkWidget_2->setGeometry (QRect (380, 210, 41, 21));
      tabWidget->addTab (tab4, QString ());
      QtViewer->setCentralWidget (centralwidget);

      retranslateUi (QtViewer);
      QObject::connect (horizontalScrollBar4, SIGNAL(valueChanged(int)), lcdNumber4, SLOT(display(int)));
      QObject::connect (horizontalScrollBar2, SIGNAL(valueChanged(int)), lcdNumber2, SLOT(display(int)));
      QObject::connect (horizontalScrollBar6, SIGNAL(valueChanged(int)), lcdNumber6, SLOT(display(int)));
      QObject::connect (horizontalScrollBar0, SIGNAL(valueChanged(int)), lcdNumber0, SLOT(display(int)));
      QObject::connect (horizontalScrollBar5, SIGNAL(valueChanged(int)), lcdNumber5, SLOT(display(int)));
      QObject::connect (horizontalScrollBar1, SIGNAL(valueChanged(int)), lcdNumber1, SLOT(display(int)));
      QObject::connect (horizontalScrollBar3, SIGNAL(valueChanged(int)), lcdNumber3, SLOT(display(int)));
      QObject::connect (horizontalScrollBar8, SIGNAL(valueChanged(int)), lcdNumber8, SLOT(display(int)));
      QObject::connect (horizontalScrollBar7, SIGNAL(valueChanged(int)), lcdNumber7, SLOT(display(int)));
      QObject::connect (horizontalScrollBar9, SIGNAL(valueChanged(int)), lcdNumber9, SLOT(display(int)));
      QObject::connect (horizontalScrollBar10, SIGNAL(valueChanged(int)), lcdNumber10, SLOT(display(int)));
      QObject::connect (horizontalScrollBar11, SIGNAL(valueChanged(int)), lcdNumber11, SLOT(display(int)));

      tabWidget->setCurrentIndex (1);

      QMetaObject::connectSlotsByName (QtViewer);
    }  // setupUi

    void
    retranslateUi (QMainWindow *QtViewer)
    {
      QtViewer->setWindowTitle (QApplication::translate ("QtViewer", "PCLViewer", Q_NULLPTR));
      label0->setText (QApplication::translate ("QtViewer", "lateral range", Q_NULLPTR));
      radioButton1->setText (QApplication::translate ("QtViewer", "VP Tree", Q_NULLPTR));
      radioButton0->setText (QApplication::translate ("QtViewer", "CPU", Q_NULLPTR));
      label1->setText (QApplication::translate ("QtViewer", "downsample", Q_NULLPTR));
      label2->setText (QApplication::translate ("QtViewer", "dbscan eps", Q_NULLPTR));
      label3->setText (QApplication::translate ("QtViewer", "dbscan minpt", Q_NULLPTR));
      label4->setText (QApplication::translate ("QtViewer", "reserved", Q_NULLPTR));
      label5->setText (QApplication::translate ("QtViewer", "reserved", Q_NULLPTR));
      checkBox1->setText (QApplication::translate ("QtViewer", "training mode", Q_NULLPTR));
      radioButton2->setText (QApplication::translate ("QtViewer", "CUDA", Q_NULLPTR));
      checkBox3->setText (QApplication::translate ("QtViewer", "Enable", Q_NULLPTR));
      lineEdit12->setText (QString ());
      label6->setText (QApplication::translate ("QtViewer", "Lidar to Ground", Q_NULLPTR));
      tabWidget->setTabText (tabWidget->indexOf (tab0), QApplication::translate ("QtViewer", "Objects", Q_NULLPTR));
      tabWidget->setTabText (tabWidget->indexOf (tab4), QApplication::translate ("QtViewer", "Transform", Q_NULLPTR));
    }  // retranslateUi

};

namespace Ui
{
  class QtViewer : public Ui_QtViewer
  {
  };
}  // namespace Ui

QT_END_NAMESPACE

#endif // UI_QTVIEWER_H
