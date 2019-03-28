#ifndef CONFIGWINDOW_H
#define CONFIGWINDOW_H

#include <QMainWindow>
#include <QButtonGroup>

namespace Ui {
class ConfigWindow;
}

class ConfigWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit ConfigWindow(std::vector<int> *_data, QWidget *parent = nullptr);
    ~ConfigWindow();

    void setData(std::vector<int> *data);

private slots:


    void on_btnSave_pressed();

    void on_btnRST_pressed();

    void on_btnBack_pressed();

    // Slide Button Callbacks
    void on_sOHAEN_clicked(bool checked);
    void on_sOCCSOLVE_clicked(bool checked);
    void on_sIDEN_clicked(bool checked);
    void on_sMVOP_clicked(bool checked);
    void on_sPVOP_clicked(bool checked);
    void on_sSFOP_clicked(bool checked);
    void on_sTFOP_clicked(bool checked);


    void on_chbTheta_clicked(bool checked);

    void on_chbConvex_clicked(bool checked);

    void on_chbPerimeter_clicked(bool checked);

    void on_chbEqDiam_clicked(bool checked);

    void on_chbEcc_clicked(bool checked);

    void on_chbCompact_clicked(bool checked);

    void on_ledID_textChanged(const QString &arg1);

    void on_cbxOHA_currentIndexChanged(int index);

    void on_ledNL_textEdited(const QString &arg1);

    void on_ledNMX_textChanged(const QString &arg1);

    void on_cbxTFlow_currentIndexChanged(int index);

private:
    Ui::ConfigWindow *ui;
    std::vector<int> *data;
    int flags;
    bool modified;

    void init(void);
};

#endif // CONFIGWINDOW_H
