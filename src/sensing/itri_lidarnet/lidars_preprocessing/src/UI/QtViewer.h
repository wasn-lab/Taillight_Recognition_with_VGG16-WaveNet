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
  explicit QtViewer(QWidget* parent = 0);
  ~QtViewer();

public slots:
  //------------------------------------------------------------------ TAB 1
  void HorizontalScrollBar0(int value);

  void HorizontalScrollBar1(int value);

  void HorizontalScrollBar2(int value);

  void HorizontalScrollBar3(int value);

  void HorizontalScrollBar4(int value);

  void HorizontalScrollBar5(int value);

  void RadioButton0();

  void RadioButton1();

  void RadioButton2();

  void CheckBox1(int value);

  void CheckBox3(int value);

  void LineEditor12(const QString& str);

  //------------------------------------------------------------------ TAB 2
  void CheckBox2(int value);

  void LineEditor7(const QString& str);

  void LineEditor13(const QString& str);

  void LineEditor14(const QString& str);

  void LineEditor15(const QString& str);

  void LineEditor16(const QString& str);
  //------------------------------------------------------------------ TAB 3
  void CheckBox0(int value);

  void LineEditor2(const QString& str);

  void LineEditor3(const QString& str);

  void LineEditor4(const QString& str);

  void LineEditor5(const QString& str);

  void LineEditor6(const QString& str);

  //------------------------------------------------------------------ TAB 4
  void LineEditor0(const QString& str);

  void LineEditor1(const QString& str);

  void ButtonPressed0();

  //------------------------------------------------------------------ TAB 5
  void HorizontalScrollBar6(int value);

  void HorizontalScrollBar7(int value);

  void HorizontalScrollBar8(int value);

  void HorizontalScrollBar9(int value);

  void HorizontalScrollBar10(int value);

  void HorizontalScrollBar11(int value);

protected:
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;

private:
  Ui::QtViewer* ui;

  boost::property_tree::ptree boost_ptree;
};

#endif  // QTVIEWER_H
