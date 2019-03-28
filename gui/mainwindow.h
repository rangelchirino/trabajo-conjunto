#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <videowidget.h>
#include <QPushButton>
#include <QLabel>
#include <QTextEdit>
#include <QMediaPlayer>
#include <QMediaPlaylist>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QFileInfo>
#include <QTime>
#include <QTimer>
#include <QThread>
#include <QProcess>
#include <ConfigWindow.h>
#include <QMessageBox>
#include <fstream>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_btnBackward_pressed();

    void on_btnStop_pressed();

    void on_btnPlay_pressed();

    void on_btnForward_pressed();

    void on_btnOpen_pressed();

    void on_btnProcess_pressed();

    void positionChanged(int pos);

    void on_timer_timeout(void);

    void on_btnConfig_pressed();

private:
    Ui::MainWindow *ui;

    // Private attributes
    ConfigWindow *wcon;
    QTimer *timer;
    std::vector<int> flags;
    QMediaPlayer* mediaPlayer;
    QString filepath;
    QString datapath;
    QString searchpath;

    bool PlayState;
    qint64 TickInc;
    QTime *duration;

    QFlags<Qt::WindowType> WinFlags;

    // Callbacks
    void updateMediaDuration(void);
};

#endif // MAINWINDOW_H
