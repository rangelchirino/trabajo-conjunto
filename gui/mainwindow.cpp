#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <thread>

const std::vector<QString> ohmodels = {"empiric","convex"};

void sdcv_call(QString exepath)
{
    std::system(exepath.toStdString().data());
}

bool fs_exist(QString path)
{
    QFileInfo check_file(path);

    // check if path exists and if yes: Is it really a file and no directory?
    return check_file.exists() && check_file.isFile();
}

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    // Setup Window
    ui->setupUi(this);

    // Set current search path
    searchpath = QDir::currentPath();

    // Widget with black background
    QPalette Pal(palette());
    Pal.setColor(QPalette::Background, Qt::black);
    ui->mediaPlayerWidget->setAutoFillBackground(true);
    ui->mediaPlayerWidget->setPalette(Pal);
    ui->mediaPlayerWidget->show();

    // Set Layout
    ui->centralWidget->setLayout(ui->mainLayer);

    // Initial Enabling States
    ui->lblfilename->setEnabled(false);
    ui->btnPlay->setEnabled(false);
    ui->btnForward->setEnabled(false);
    ui->btnBackward->setEnabled(false);
    ui->btnStop->setEnabled(false);
    ui->btnProcess->setEnabled(false);
    ui->sldvideopos->setEnabled(false);
    ui->sldvideopos->setValue(0);
    ui->mediaPlayerWidget->setEnabled(true);
    ui->btnConfig->setEnabled(false);

    //  Media Player setup
    mediaPlayer = new QMediaPlayer();
    mediaPlayer->setVideoOutput(ui->mediaPlayerWidget);
    duration = new QTime();

    wcon = new ConfigWindow(&flags);

    timer = new QTimer(this);
    connect(timer, SIGNAL(timeout()), this, SLOT(on_timer_timeout()));

    // Connecting Signals
    connect(this->mediaPlayer, &QMediaPlayer::durationChanged, ui->sldvideopos, &QSlider::setMaximum);
    connect(this->mediaPlayer, &QMediaPlayer::positionChanged, this, &MainWindow::positionChanged);
    connect(ui->sldvideopos, &QSlider::sliderMoved, this->mediaPlayer, &QMediaPlayer::setPosition);
}

MainWindow::~MainWindow()
{
    delete ui;
}


/* MEDIA CALLBACKS */
void MainWindow::updateMediaDuration(void)
{
    const qint64 curMediaPos = mediaPlayer->position() / 1000;
    const qint64 totalTimeSec = mediaPlayer->duration() / 1000;
    QString tStr;

    if ( curMediaPos )
    {
        QTime elapsedTime((curMediaPos / 3600) % 60,
                          (curMediaPos / 60) % 60,
                          (curMediaPos % 60),
                          (curMediaPos * 1000) % 1000);

        QTime totalTime((totalTimeSec / 3600) % 60,
                          (totalTimeSec / 60) % 60,
                          (totalTimeSec % 60),
                          (totalTimeSec * 1000) % 1000);

        QString format = "mm:ss";

        if ( totalTime.hour() )
            format = "hh:mm:ss";

        if(TickInc == -1)
        {
            if( totalTime.hour() )
                TickInc = 1000*60*5;
            else if( totalTime.minute() )
                TickInc = 10*1000;
            else
                TickInc = 1000;
        }
        tStr = elapsedTime.toString(format) + " / " + totalTime.toString(format);
    }

    ui->lblvideolen->setText(tStr);
}

/* SLIDER CALLBACKS */
void MainWindow::positionChanged(int pos)
{
    ui->sldvideopos->setValue(pos);
    updateMediaDuration();
}



/* BUTTON CALLBACKS */
void MainWindow::on_btnStop_pressed()
{
    ui->btnPlay->setText("PLAY");
    PlayState=true;

    ui->btnPlay->setEnabled(false);
    ui->btnForward->setEnabled(false);
    ui->btnBackward->setEnabled(false);
    ui->btnStop->setEnabled(false);
    ui->sldvideopos->setEnabled(false);
    ui->sldvideopos->setValue(0);
    ui->btnConfig->setEnabled(false);
    ui->btnProcess->setEnabled(false);

    filepath.clear();

    mediaPlayer->stop();
}

void MainWindow::on_btnPlay_pressed()
{
    if(PlayState)
    {
        ui->btnPlay->setText("PAUSE");
        PlayState=false;

        if (mediaPlayer->state() == QMediaPlayer::PausedState)
        {
            mediaPlayer->play();
            ui->statusBar->showMessage("Playing");
        }
    }
    else
    {
        ui->btnPlay->setText("PLAY");
        PlayState=true;

        if(mediaPlayer->state() == QMediaPlayer::PlayingState)
        {
            mediaPlayer->pause();
            ui->statusBar->showMessage("Paused");
        }
    }
}

void MainWindow::on_btnForward_pressed()
{
    // Update VideoPlayer
    qint64 nextPos = mediaPlayer->position() + TickInc;
    if(nextPos > mediaPlayer->duration()) on_btnStop_pressed();

    mediaPlayer->setPosition(nextPos);
}

void MainWindow::on_btnBackward_pressed()
{
    // Update VideoPlayer
    qint64 nextPos = mediaPlayer->position() - TickInc;
    if(nextPos < 0) nextPos = 0;

    mediaPlayer->setPosition(nextPos);

}

void MainWindow::on_btnOpen_pressed()
{
    filepath = QFileDialog::getOpenFileName(this, "Open a File", searchpath, "Video File (*.avi, *.mpg, *.mp4)");

    if( filepath.length() )
    {
        ui->btnPlay->setEnabled(true);
        ui->btnForward->setEnabled(true);
        ui->btnBackward->setEnabled(true);
        ui->btnStop->setEnabled(true);
        ui->sldvideopos->setEnabled(true);
        ui->btnConfig->setEnabled(true);

        ui->btnPlay->setText("PAUSE");
        PlayState=false;

        ui->lblfilename->setText("Videofile: " + filepath);

        searchpath = QFileInfo(filepath).path();

        mediaPlayer->stop();
        mediaPlayer->setMedia(QUrl::fromLocalFile(filepath));
        mediaPlayer->play();

        TickInc = -1;
    }

}

void MainWindow::on_timer_timeout()
{
    if( !wcon->isVisible() )
    {
        timer->stop();
        wcon->close();
        ui->btnProcess->setEnabled(true);
    }
}

void MainWindow::on_btnProcess_pressed()
{
    // Execute sdcv project
    QString sdcvpath = QDir::currentPath() + "/sdcv/sdcv.exe -filename=\"" + filepath +
            "\" -numlanes=" + QString::number(flags[0]) +
            " -flow="+ (flags[10] ? "rearview" : "frontview") +
            " -flags="+ QString::number(flags[9]) +
            " -nmixtures="+ QString::number(flags[1]) +
            " -ohmodel="+ ohmodels.at(size_t(flags[4])) +
            " -ohandle="+ (flags[3] ? "enable" : "disable") +
            " -osolver="+ (flags[11] ? "enable" : "disable") +
            " -videoproc="+ (flags[5] ? "enable" : "disable") +
            " -videomask="+ (flags[6] ? "enable" : "disable") +
            " -filestat="+ (flags[7] ? "enable" : "disable") +
            " -filetime="+ (flags[8] ? "enable" : "disable") +
            " -id="+ QString::number(flags[2]);


    QMessageBox Msgbox;
        Msgbox.setText(sdcvpath);
        Msgbox.exec();
    //*/

    QProcess *myProcess = new QProcess(this);
    myProcess->start(sdcvpath);
    bool finish=myProcess->waitForFinished();
    qDebug() << "Finished: " << finish;
    //*/

    ui->btnProcess->setEnabled(false);

    QFileInfo info1(filepath);
    datapath = QDir::currentPath() + "sdcv/" + info1.baseName() + "/";
}

void MainWindow::on_btnConfig_pressed()
{
    wcon->show();
    timer->start(100);
}
