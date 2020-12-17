/********************************************************************************
 ** Form generated from reading UI file 'QtViewer.ui'
 **
 ** Created by: Qt User Interface Compiler version 5.9.1
 **
 ** WARNING! All changes made in this file will be lost when recompiling UI
 *file!
 ********************************************************************************/

#ifndef UI_QTVIEWER_H
#define UI_QTVIEWER_H

#include "QVTKWidget.h"
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
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QScrollBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_QtViewer
{
public:
  QWidget* centralwidget;
  QTabWidget* tabWidget;
  QWidget* tab0;
  QScrollBar* horizontalScrollBar0;
  QScrollBar* horizontalScrollBar1;
  QScrollBar* horizontalScrollBar2;
  QScrollBar* horizontalScrollBar3;
  QScrollBar* horizontalScrollBar4;
  QScrollBar* horizontalScrollBar5;
  QLCDNumber* lcdNumber0;
  QLCDNumber* lcdNumber1;
  QLCDNumber* lcdNumber3;
  QLCDNumber* lcdNumber4;
  QLCDNumber* lcdNumber5;
  QLCDNumber* lcdNumber2;
  QLabel* label0;
  QRadioButton* radioButton1;
  QRadioButton* radioButton0;
  QLabel* label1;
  QLabel* label2;
  QLabel* label3;
  QLabel* label4;
  QLabel* label5;
  QCheckBox* checkBox1;
  QRadioButton* radioButton2;
  QCheckBox* checkBox3;
  QLineEdit* lineEdit12;
  QLabel* label6;
  QWidget* tab1;
  QLineEdit* lineEdit7;
  QLabel* label15;
  QCheckBox* checkBox2;
  QLineEdit* lineEdit15;
  QLineEdit* lineEdit14;
  QLineEdit* lineEdit13;
  QLabel* label17_3;
  QLabel* label17_4;
  QLabel* label16_3;
  QLineEdit* lineEdit16;
  QLabel* label17_5;
  QWidget* tab2;
  QCheckBox* checkBox0;
  QLineEdit* lineEdit6;
  QLineEdit* lineEdit2;
  QLineEdit* lineEdit5;
  QLineEdit* lineEdit3;
  QLineEdit* lineEdit4;
  QLabel* label10;
  QLabel* label11;
  QLabel* label12;
  QLabel* label13;
  QLabel* label14;
  QWidget* tab3;
  QLabel* label_8;
  QLabel* label_6;
  QLineEdit* lineEdit0;
  QLineEdit* lineEdit1;
  QLabel* label_7;
  QWidget* tab;
  QScrollBar* horizontalScrollBar11;
  QLCDNumber* lcdNumber7;
  QLCDNumber* lcdNumber10;
  QScrollBar* horizontalScrollBar7;
  QLCDNumber* lcdNumber8;
  QScrollBar* horizontalScrollBar9;
  QLCDNumber* lcdNumber6;
  QScrollBar* horizontalScrollBar6;
  QScrollBar* horizontalScrollBar8;
  QScrollBar* horizontalScrollBar10;
  QLCDNumber* lcdNumber11;
  QLCDNumber* lcdNumber9;
  QVTKWidget* qvtkWidget;
  QVTKWidget* qvtkWidget_2;
  QPushButton* pushButton_random;

  void setupUi(QMainWindow* QtViewer)
  {
    if (QtViewer->objectName().isEmpty())
      QtViewer->setObjectName(QStringLiteral("QtViewer"));
    QtViewer->resize(452, 282);
    QtViewer->setMinimumSize(QSize(0, 0));
    QtViewer->setMaximumSize(QSize(5000, 5000));
    centralwidget = new QWidget(QtViewer);
    centralwidget->setObjectName(QStringLiteral("centralwidget"));
    tabWidget = new QTabWidget(centralwidget);
    tabWidget->setObjectName(QStringLiteral("tabWidget"));
    tabWidget->setGeometry(QRect(0, 0, 451, 281));
    tab0 = new QWidget();
    tab0->setObjectName(QStringLiteral("tab0"));
    horizontalScrollBar0 = new QScrollBar(tab0);
    horizontalScrollBar0->setObjectName(QStringLiteral("horizontalScrollBar0"));
    horizontalScrollBar0->setGeometry(QRect(10, 10, 271, 16));
    horizontalScrollBar0->setMinimum(0);
    horizontalScrollBar0->setMaximum(50);
    horizontalScrollBar0->setPageStep(2);
    horizontalScrollBar0->setSliderPosition(0);
    horizontalScrollBar0->setOrientation(Qt::Horizontal);
    horizontalScrollBar1 = new QScrollBar(tab0);
    horizontalScrollBar1->setObjectName(QStringLiteral("horizontalScrollBar1"));
    horizontalScrollBar1->setGeometry(QRect(10, 40, 271, 16));
    horizontalScrollBar1->setMaximum(999);
    horizontalScrollBar1->setOrientation(Qt::Horizontal);
    horizontalScrollBar2 = new QScrollBar(tab0);
    horizontalScrollBar2->setObjectName(QStringLiteral("horizontalScrollBar2"));
    horizontalScrollBar2->setGeometry(QRect(10, 70, 271, 16));
    horizontalScrollBar2->setMaximum(999);
    horizontalScrollBar2->setOrientation(Qt::Horizontal);
    horizontalScrollBar3 = new QScrollBar(tab0);
    horizontalScrollBar3->setObjectName(QStringLiteral("horizontalScrollBar3"));
    horizontalScrollBar3->setGeometry(QRect(10, 100, 271, 16));
    horizontalScrollBar3->setMaximum(999);
    horizontalScrollBar3->setOrientation(Qt::Horizontal);
    horizontalScrollBar4 = new QScrollBar(tab0);
    horizontalScrollBar4->setObjectName(QStringLiteral("horizontalScrollBar4"));
    horizontalScrollBar4->setGeometry(QRect(10, 130, 271, 16));
    horizontalScrollBar4->setMaximum(999);
    horizontalScrollBar4->setOrientation(Qt::Horizontal);
    horizontalScrollBar5 = new QScrollBar(tab0);
    horizontalScrollBar5->setObjectName(QStringLiteral("horizontalScrollBar5"));
    horizontalScrollBar5->setGeometry(QRect(10, 160, 271, 16));
    horizontalScrollBar5->setMaximum(999);
    horizontalScrollBar5->setOrientation(Qt::Horizontal);
    lcdNumber0 = new QLCDNumber(tab0);
    lcdNumber0->setObjectName(QStringLiteral("lcdNumber0"));
    lcdNumber0->setGeometry(QRect(290, 10, 41, 21));
    lcdNumber0->setDigitCount(3);
    lcdNumber0->setSegmentStyle(QLCDNumber::Flat);
    lcdNumber0->setProperty("intValue", QVariant(0));
    lcdNumber1 = new QLCDNumber(tab0);
    lcdNumber1->setObjectName(QStringLiteral("lcdNumber1"));
    lcdNumber1->setGeometry(QRect(290, 40, 41, 21));
    lcdNumber1->setDigitCount(3);
    lcdNumber1->setSegmentStyle(QLCDNumber::Flat);
    lcdNumber1->setProperty("intValue", QVariant(0));
    lcdNumber3 = new QLCDNumber(tab0);
    lcdNumber3->setObjectName(QStringLiteral("lcdNumber3"));
    lcdNumber3->setGeometry(QRect(290, 100, 41, 21));
    lcdNumber3->setDigitCount(3);
    lcdNumber3->setSegmentStyle(QLCDNumber::Flat);
    lcdNumber3->setProperty("intValue", QVariant(0));
    lcdNumber4 = new QLCDNumber(tab0);
    lcdNumber4->setObjectName(QStringLiteral("lcdNumber4"));
    lcdNumber4->setGeometry(QRect(290, 130, 41, 21));
    lcdNumber4->setDigitCount(3);
    lcdNumber4->setSegmentStyle(QLCDNumber::Flat);
    lcdNumber4->setProperty("intValue", QVariant(0));
    lcdNumber5 = new QLCDNumber(tab0);
    lcdNumber5->setObjectName(QStringLiteral("lcdNumber5"));
    lcdNumber5->setGeometry(QRect(290, 160, 41, 21));
    lcdNumber5->setDigitCount(3);
    lcdNumber5->setSegmentStyle(QLCDNumber::Flat);
    lcdNumber5->setProperty("intValue", QVariant(0));
    lcdNumber2 = new QLCDNumber(tab0);
    lcdNumber2->setObjectName(QStringLiteral("lcdNumber2"));
    lcdNumber2->setGeometry(QRect(290, 70, 41, 21));
    lcdNumber2->setDigitCount(3);
    lcdNumber2->setSegmentStyle(QLCDNumber::Flat);
    lcdNumber2->setProperty("intValue", QVariant(0));
    label0 = new QLabel(tab0);
    label0->setObjectName(QStringLiteral("label0"));
    label0->setGeometry(QRect(340, 10, 91, 17));
    radioButton1 = new QRadioButton(tab0);
    radioButton1->setObjectName(QStringLiteral("radioButton1"));
    radioButton1->setGeometry(QRect(70, 190, 81, 22));
    radioButton0 = new QRadioButton(tab0);
    radioButton0->setObjectName(QStringLiteral("radioButton0"));
    radioButton0->setGeometry(QRect(10, 190, 61, 22));
    label1 = new QLabel(tab0);
    label1->setObjectName(QStringLiteral("label1"));
    label1->setGeometry(QRect(340, 40, 91, 17));
    label2 = new QLabel(tab0);
    label2->setObjectName(QStringLiteral("label2"));
    label2->setGeometry(QRect(340, 70, 91, 17));
    label3 = new QLabel(tab0);
    label3->setObjectName(QStringLiteral("label3"));
    label3->setGeometry(QRect(340, 100, 91, 17));
    label4 = new QLabel(tab0);
    label4->setObjectName(QStringLiteral("label4"));
    label4->setGeometry(QRect(340, 130, 91, 17));
    label5 = new QLabel(tab0);
    label5->setObjectName(QStringLiteral("label5"));
    label5->setGeometry(QRect(340, 160, 91, 17));
    checkBox1 = new QCheckBox(tab0);
    checkBox1->setObjectName(QStringLiteral("checkBox1"));
    checkBox1->setGeometry(QRect(80, 220, 121, 23));
    radioButton2 = new QRadioButton(tab0);
    radioButton2->setObjectName(QStringLiteral("radioButton2"));
    radioButton2->setGeometry(QRect(150, 190, 71, 22));
    checkBox3 = new QCheckBox(tab0);
    checkBox3->setObjectName(QStringLiteral("checkBox3"));
    checkBox3->setGeometry(QRect(10, 220, 71, 23));
    lineEdit12 = new QLineEdit(tab0);
    lineEdit12->setObjectName(QStringLiteral("lineEdit12"));
    lineEdit12->setGeometry(QRect(220, 190, 111, 25));
    lineEdit12->setInputMethodHints(Qt::ImhDigitsOnly | Qt::ImhFormattedNumbersOnly);
    lineEdit12->setMaxLength(9);
    lineEdit12->setFrame(true);
    lineEdit12->setClearButtonEnabled(false);
    label6 = new QLabel(tab0);
    label6->setObjectName(QStringLiteral("label6"));
    label6->setGeometry(QRect(340, 190, 111, 17));
    tabWidget->addTab(tab0, QString());
    tab1 = new QWidget();
    tab1->setObjectName(QStringLiteral("tab1"));
    lineEdit7 = new QLineEdit(tab1);
    lineEdit7->setObjectName(QStringLiteral("lineEdit7"));
    lineEdit7->setGeometry(QRect(10, 10, 81, 25));
    lineEdit7->setInputMethodHints(Qt::ImhDigitsOnly | Qt::ImhFormattedNumbersOnly);
    lineEdit7->setMaxLength(9);
    lineEdit7->setFrame(true);
    lineEdit7->setClearButtonEnabled(false);
    label15 = new QLabel(tab1);
    label15->setObjectName(QStringLiteral("label15"));
    label15->setGeometry(QRect(100, 10, 81, 17));
    checkBox2 = new QCheckBox(tab1);
    checkBox2->setObjectName(QStringLiteral("checkBox2"));
    checkBox2->setGeometry(QRect(10, 220, 121, 23));
    lineEdit15 = new QLineEdit(tab1);
    lineEdit15->setObjectName(QStringLiteral("lineEdit15"));
    lineEdit15->setGeometry(QRect(10, 100, 81, 25));
    lineEdit15->setInputMethodHints(Qt::ImhDigitsOnly | Qt::ImhFormattedNumbersOnly);
    lineEdit15->setMaxLength(9);
    lineEdit15->setFrame(true);
    lineEdit15->setClearButtonEnabled(false);
    lineEdit14 = new QLineEdit(tab1);
    lineEdit14->setObjectName(QStringLiteral("lineEdit14"));
    lineEdit14->setGeometry(QRect(10, 70, 81, 25));
    lineEdit14->setInputMethodHints(Qt::ImhDigitsOnly | Qt::ImhFormattedNumbersOnly);
    lineEdit14->setMaxLength(9);
    lineEdit14->setFrame(true);
    lineEdit14->setClearButtonEnabled(false);
    lineEdit13 = new QLineEdit(tab1);
    lineEdit13->setObjectName(QStringLiteral("lineEdit13"));
    lineEdit13->setGeometry(QRect(10, 40, 81, 25));
    lineEdit13->setInputMethodHints(Qt::ImhDigitsOnly | Qt::ImhFormattedNumbersOnly);
    lineEdit13->setMaxLength(9);
    lineEdit13->setFrame(true);
    lineEdit13->setClearButtonEnabled(false);
    label17_3 = new QLabel(tab1);
    label17_3->setObjectName(QStringLiteral("label17_3"));
    label17_3->setGeometry(QRect(100, 40, 151, 17));
    label17_4 = new QLabel(tab1);
    label17_4->setObjectName(QStringLiteral("label17_4"));
    label17_4->setGeometry(QRect(100, 100, 151, 17));
    label16_3 = new QLabel(tab1);
    label16_3->setObjectName(QStringLiteral("label16_3"));
    label16_3->setGeometry(QRect(100, 70, 151, 17));
    lineEdit16 = new QLineEdit(tab1);
    lineEdit16->setObjectName(QStringLiteral("lineEdit16"));
    lineEdit16->setGeometry(QRect(10, 130, 81, 25));
    lineEdit16->setInputMethodHints(Qt::ImhDigitsOnly | Qt::ImhFormattedNumbersOnly);
    lineEdit16->setMaxLength(9);
    lineEdit16->setFrame(true);
    lineEdit16->setClearButtonEnabled(false);
    label17_5 = new QLabel(tab1);
    label17_5->setObjectName(QStringLiteral("label17_5"));
    label17_5->setGeometry(QRect(100, 130, 151, 17));
    tabWidget->addTab(tab1, QString());
    tab2 = new QWidget();
    tab2->setObjectName(QStringLiteral("tab2"));
    checkBox0 = new QCheckBox(tab2);
    checkBox0->setObjectName(QStringLiteral("checkBox0"));
    checkBox0->setGeometry(QRect(10, 220, 92, 23));
    lineEdit6 = new QLineEdit(tab2);
    lineEdit6->setObjectName(QStringLiteral("lineEdit6"));
    lineEdit6->setGeometry(QRect(10, 100, 81, 25));
    lineEdit6->setInputMethodHints(Qt::ImhDigitsOnly | Qt::ImhFormattedNumbersOnly);
    lineEdit6->setMaxLength(9);
    lineEdit6->setFrame(true);
    lineEdit6->setClearButtonEnabled(false);
    lineEdit2 = new QLineEdit(tab2);
    lineEdit2->setObjectName(QStringLiteral("lineEdit2"));
    lineEdit2->setGeometry(QRect(10, 10, 81, 25));
    lineEdit2->setInputMethodHints(Qt::ImhDigitsOnly | Qt::ImhFormattedNumbersOnly);
    lineEdit2->setMaxLength(9);
    lineEdit2->setFrame(true);
    lineEdit2->setClearButtonEnabled(false);
    lineEdit5 = new QLineEdit(tab2);
    lineEdit5->setObjectName(QStringLiteral("lineEdit5"));
    lineEdit5->setGeometry(QRect(10, 70, 81, 25));
    lineEdit5->setInputMethodHints(Qt::ImhDigitsOnly | Qt::ImhFormattedNumbersOnly);
    lineEdit5->setMaxLength(9);
    lineEdit5->setFrame(true);
    lineEdit5->setClearButtonEnabled(false);
    lineEdit3 = new QLineEdit(tab2);
    lineEdit3->setObjectName(QStringLiteral("lineEdit3"));
    lineEdit3->setGeometry(QRect(10, 130, 81, 25));
    lineEdit3->setInputMethodHints(Qt::ImhDigitsOnly | Qt::ImhFormattedNumbersOnly);
    lineEdit3->setMaxLength(9);
    lineEdit3->setFrame(true);
    lineEdit3->setClearButtonEnabled(false);
    lineEdit4 = new QLineEdit(tab2);
    lineEdit4->setObjectName(QStringLiteral("lineEdit4"));
    lineEdit4->setGeometry(QRect(10, 40, 81, 25));
    lineEdit4->setInputMethodHints(Qt::ImhDigitsOnly | Qt::ImhFormattedNumbersOnly);
    lineEdit4->setMaxLength(9);
    lineEdit4->setFrame(true);
    lineEdit4->setClearButtonEnabled(false);
    label10 = new QLabel(tab2);
    label10->setObjectName(QStringLiteral("label10"));
    label10->setGeometry(QRect(100, 10, 151, 17));
    label11 = new QLabel(tab2);
    label11->setObjectName(QStringLiteral("label11"));
    label11->setGeometry(QRect(100, 130, 151, 17));
    label12 = new QLabel(tab2);
    label12->setObjectName(QStringLiteral("label12"));
    label12->setGeometry(QRect(100, 40, 151, 17));
    label13 = new QLabel(tab2);
    label13->setObjectName(QStringLiteral("label13"));
    label13->setGeometry(QRect(100, 70, 151, 17));
    label14 = new QLabel(tab2);
    label14->setObjectName(QStringLiteral("label14"));
    label14->setGeometry(QRect(100, 100, 151, 17));
    tabWidget->addTab(tab2, QString());
    tab3 = new QWidget();
    tab3->setObjectName(QStringLiteral("tab3"));
    label_8 = new QLabel(tab3);
    label_8->setObjectName(QStringLiteral("label_8"));
    label_8->setGeometry(QRect(10, 30, 31, 17));
    label_6 = new QLabel(tab3);
    label_6->setObjectName(QStringLiteral("label_6"));
    label_6->setGeometry(QRect(50, 10, 67, 17));
    lineEdit0 = new QLineEdit(tab3);
    lineEdit0->setObjectName(QStringLiteral("lineEdit0"));
    lineEdit0->setGeometry(QRect(50, 30, 141, 25));
    lineEdit0->setInputMethodHints(Qt::ImhDigitsOnly | Qt::ImhFormattedNumbersOnly);
    lineEdit0->setMaxLength(15);
    lineEdit0->setFrame(true);
    lineEdit0->setClearButtonEnabled(false);
    lineEdit1 = new QLineEdit(tab3);
    lineEdit1->setObjectName(QStringLiteral("lineEdit1"));
    lineEdit1->setGeometry(QRect(200, 30, 61, 25));
    lineEdit1->setInputMethodHints(Qt::ImhFormattedNumbersOnly);
    lineEdit1->setMaxLength(5);
    lineEdit1->setFrame(true);
    lineEdit1->setClearButtonEnabled(false);
    label_7 = new QLabel(tab3);
    label_7->setObjectName(QStringLiteral("label_7"));
    label_7->setGeometry(QRect(200, 10, 67, 17));
    tabWidget->addTab(tab3, QString());
    tab = new QWidget();
    tab->setObjectName(QStringLiteral("tab"));
    horizontalScrollBar11 = new QScrollBar(tab);
    horizontalScrollBar11->setObjectName(QStringLiteral("horizontalScrollBar11"));
    horizontalScrollBar11->setGeometry(QRect(10, 160, 271, 16));
    horizontalScrollBar11->setMinimum(-999);
    horizontalScrollBar11->setMaximum(999);
    horizontalScrollBar11->setOrientation(Qt::Horizontal);
    lcdNumber7 = new QLCDNumber(tab);
    lcdNumber7->setObjectName(QStringLiteral("lcdNumber7"));
    lcdNumber7->setGeometry(QRect(290, 40, 71, 21));
    lcdNumber7->setDigitCount(4);
    lcdNumber7->setSegmentStyle(QLCDNumber::Flat);
    lcdNumber7->setProperty("intValue", QVariant(0));
    lcdNumber10 = new QLCDNumber(tab);
    lcdNumber10->setObjectName(QStringLiteral("lcdNumber10"));
    lcdNumber10->setGeometry(QRect(290, 130, 71, 21));
    lcdNumber10->setDigitCount(4);
    lcdNumber10->setSegmentStyle(QLCDNumber::Flat);
    lcdNumber10->setProperty("intValue", QVariant(0));
    horizontalScrollBar7 = new QScrollBar(tab);
    horizontalScrollBar7->setObjectName(QStringLiteral("horizontalScrollBar7"));
    horizontalScrollBar7->setGeometry(QRect(10, 40, 271, 16));
    horizontalScrollBar7->setMinimum(-999);
    horizontalScrollBar7->setMaximum(999);
    horizontalScrollBar7->setOrientation(Qt::Horizontal);
    lcdNumber8 = new QLCDNumber(tab);
    lcdNumber8->setObjectName(QStringLiteral("lcdNumber8"));
    lcdNumber8->setGeometry(QRect(290, 70, 71, 21));
    lcdNumber8->setDigitCount(4);
    lcdNumber8->setSegmentStyle(QLCDNumber::Flat);
    lcdNumber8->setProperty("intValue", QVariant(0));
    horizontalScrollBar9 = new QScrollBar(tab);
    horizontalScrollBar9->setObjectName(QStringLiteral("horizontalScrollBar9"));
    horizontalScrollBar9->setGeometry(QRect(10, 100, 271, 16));
    horizontalScrollBar9->setMinimum(-999);
    horizontalScrollBar9->setMaximum(999);
    horizontalScrollBar9->setOrientation(Qt::Horizontal);
    lcdNumber6 = new QLCDNumber(tab);
    lcdNumber6->setObjectName(QStringLiteral("lcdNumber6"));
    lcdNumber6->setGeometry(QRect(290, 10, 71, 21));
    lcdNumber6->setDigitCount(4);
    lcdNumber6->setSegmentStyle(QLCDNumber::Flat);
    lcdNumber6->setProperty("intValue", QVariant(0));
    horizontalScrollBar6 = new QScrollBar(tab);
    horizontalScrollBar6->setObjectName(QStringLiteral("horizontalScrollBar6"));
    horizontalScrollBar6->setGeometry(QRect(10, 10, 271, 16));
    horizontalScrollBar6->setMinimum(-999);
    horizontalScrollBar6->setMaximum(999);
    horizontalScrollBar6->setPageStep(10);
    horizontalScrollBar6->setValue(0);
    horizontalScrollBar6->setSliderPosition(0);
    horizontalScrollBar6->setOrientation(Qt::Horizontal);
    horizontalScrollBar8 = new QScrollBar(tab);
    horizontalScrollBar8->setObjectName(QStringLiteral("horizontalScrollBar8"));
    horizontalScrollBar8->setGeometry(QRect(10, 70, 271, 16));
    horizontalScrollBar8->setMinimum(-999);
    horizontalScrollBar8->setMaximum(999);
    horizontalScrollBar8->setOrientation(Qt::Horizontal);
    horizontalScrollBar10 = new QScrollBar(tab);
    horizontalScrollBar10->setObjectName(QStringLiteral("horizontalScrollBar10"));
    horizontalScrollBar10->setGeometry(QRect(10, 130, 271, 16));
    horizontalScrollBar10->setMinimum(-999);
    horizontalScrollBar10->setMaximum(999);
    horizontalScrollBar10->setOrientation(Qt::Horizontal);
    lcdNumber11 = new QLCDNumber(tab);
    lcdNumber11->setObjectName(QStringLiteral("lcdNumber11"));
    lcdNumber11->setGeometry(QRect(290, 160, 71, 21));
    lcdNumber11->setDigitCount(4);
    lcdNumber11->setSegmentStyle(QLCDNumber::Flat);
    lcdNumber11->setProperty("intValue", QVariant(0));
    lcdNumber9 = new QLCDNumber(tab);
    lcdNumber9->setObjectName(QStringLiteral("lcdNumber9"));
    lcdNumber9->setGeometry(QRect(290, 100, 71, 21));
    lcdNumber9->setDigitCount(4);
    lcdNumber9->setSegmentStyle(QLCDNumber::Flat);
    lcdNumber9->setProperty("intValue", QVariant(0));
    qvtkWidget = new QVTKWidget(tab);
    qvtkWidget->setObjectName(QStringLiteral("qvtkWidget"));
    qvtkWidget->setGeometry(QRect(380, 10, 41, 171));
    qvtkWidget_2 = new QVTKWidget(tab);
    qvtkWidget_2->setObjectName(QStringLiteral("qvtkWidget_2"));
    qvtkWidget_2->setGeometry(QRect(380, 210, 41, 21));
    pushButton_random = new QPushButton(tab);
    pushButton_random->setObjectName(QStringLiteral("pushButton_random"));
    pushButton_random->setGeometry(QRect(120, 180, 131, 31));
    tabWidget->addTab(tab, QString());
    QtViewer->setCentralWidget(centralwidget);

    retranslateUi(QtViewer);
    QObject::connect(horizontalScrollBar1, SIGNAL(valueChanged(int)), lcdNumber1, SLOT(display(int)));
    QObject::connect(horizontalScrollBar0, SIGNAL(valueChanged(int)), lcdNumber0, SLOT(display(int)));
    QObject::connect(horizontalScrollBar4, SIGNAL(valueChanged(int)), lcdNumber4, SLOT(display(int)));
    QObject::connect(horizontalScrollBar5, SIGNAL(valueChanged(int)), lcdNumber5, SLOT(display(int)));
    QObject::connect(horizontalScrollBar3, SIGNAL(valueChanged(int)), lcdNumber3, SLOT(display(int)));
    QObject::connect(horizontalScrollBar2, SIGNAL(valueChanged(int)), lcdNumber2, SLOT(display(int)));
    QObject::connect(horizontalScrollBar6, SIGNAL(valueChanged(int)), lcdNumber6, SLOT(display(int)));
    QObject::connect(horizontalScrollBar7, SIGNAL(valueChanged(int)), lcdNumber7, SLOT(display(int)));
    QObject::connect(horizontalScrollBar8, SIGNAL(valueChanged(int)), lcdNumber8, SLOT(display(int)));
    QObject::connect(horizontalScrollBar9, SIGNAL(valueChanged(int)), lcdNumber9, SLOT(display(int)));
    QObject::connect(horizontalScrollBar10, SIGNAL(valueChanged(int)), lcdNumber10, SLOT(display(int)));
    QObject::connect(horizontalScrollBar11, SIGNAL(valueChanged(int)), lcdNumber11, SLOT(display(int)));

    tabWidget->setCurrentIndex(0);

    QMetaObject::connectSlotsByName(QtViewer);
  }  // setupUi

  void retranslateUi(QMainWindow* QtViewer)
  {
    QtViewer->setWindowTitle(QApplication::translate("QtViewer", "PCLViewer", Q_NULLPTR));
    label0->setText(QApplication::translate("QtViewer", "lateral range", Q_NULLPTR));
    radioButton1->setText(QApplication::translate("QtViewer", "VP Tree", Q_NULLPTR));
    radioButton0->setText(QApplication::translate("QtViewer", "CPU", Q_NULLPTR));
    label1->setText(QApplication::translate("QtViewer", "downsample", Q_NULLPTR));
    label2->setText(QApplication::translate("QtViewer", "dbscan eps", Q_NULLPTR));
    label3->setText(QApplication::translate("QtViewer", "dbscan minpt", Q_NULLPTR));
    label4->setText(QApplication::translate("QtViewer", "reserved", Q_NULLPTR));
    label5->setText(QApplication::translate("QtViewer", "reserved", Q_NULLPTR));
    checkBox1->setText(QApplication::translate("QtViewer", "training mode", Q_NULLPTR));
    radioButton2->setText(QApplication::translate("QtViewer", "CUDA", Q_NULLPTR));
    checkBox3->setText(QApplication::translate("QtViewer", "Enable", Q_NULLPTR));
    lineEdit12->setText(QString());
    label6->setText(QApplication::translate("QtViewer", "Lidar to Ground", Q_NULLPTR));
    tabWidget->setTabText(tabWidget->indexOf(tab0), QApplication::translate("QtViewer", "Objects", Q_NULLPTR));
    lineEdit7->setText(QString());
    label15->setText(QApplication::translate("QtViewer", "intensity i", Q_NULLPTR));
    checkBox2->setText(QApplication::translate("QtViewer", "Enable", Q_NULLPTR));
    lineEdit15->setText(QString());
    lineEdit14->setText(QString());
    label17_3->setText(QApplication::translate("QtViewer", "detection width outer", Q_NULLPTR));
    label17_4->setText(QApplication::translate("QtViewer", "detection length", Q_NULLPTR));
    label16_3->setText(QApplication::translate("QtViewer", "detection width inner", Q_NULLPTR));
    lineEdit16->setText(QString());
    label17_5->setText(QApplication::translate("QtViewer", "ideal width", Q_NULLPTR));
    tabWidget->setTabText(tabWidget->indexOf(tab1), QApplication::translate("QtViewer", "Lane", Q_NULLPTR));
    checkBox0->setText(QApplication::translate("QtViewer", "Enable", Q_NULLPTR));
    label10->setText(QApplication::translate("QtViewer", "intensity i", Q_NULLPTR));
    label11->setText(QApplication::translate("QtViewer", "reserve", Q_NULLPTR));
    label12->setText(QApplication::translate("QtViewer", "parking space degree", Q_NULLPTR));
    label13->setText(QApplication::translate("QtViewer", "parking slot width", Q_NULLPTR));
    label14->setText(QApplication::translate("QtViewer", "parking slot length", Q_NULLPTR));
    tabWidget->setTabText(tabWidget->indexOf(tab2), QApplication::translate("QtViewer", "Parking", Q_NULLPTR));
    label_8->setText(QApplication::translate("QtViewer", "UDP", Q_NULLPTR));
    label_6->setText(QApplication::translate("QtViewer", "IP", Q_NULLPTR));
    lineEdit1->setText(QString());
    label_7->setText(QApplication::translate("QtViewer", "Port", Q_NULLPTR));
    tabWidget->setTabText(tabWidget->indexOf(tab3), QApplication::translate("QtViewer", "UDP", Q_NULLPTR));
    pushButton_random->setText(QApplication::translate("QtViewer", "reserved", Q_NULLPTR));
    tabWidget->setTabText(tabWidget->indexOf(tab), QApplication::translate("QtViewer", "Testing", Q_NULLPTR));
  }  // retranslateUi
};

namespace Ui
{
class QtViewer : public Ui_QtViewer
{
};
}  // namespace Ui

QT_END_NAMESPACE

#endif  // UI_QTVIEWER_H
