#include "ConfigWindow.h"
#include "ui_ConfigWindow.h"

#define BLOB_BASE 771
#define CLEAR_FLAG(X,W)	(X) = (X) & ~(W);
#define SET_FLAG(X,W)	(X) = (X) | (W);

typedef enum
{
    OUTPUT_AREA					= 0x0001,
    OUTPUT_BBOX					= 0x0002,
    OUTPUT_PERIMETER			= 0x0004,
    OUTPUT_COMPACTNESS			= 0x0008,
    OUTPUT_EQDIAM				= 0x0010,
    OUTPUT_ORIENTATION			= 0x0020,
    OUTPUT_ECCENT				= 0x0040,
    OUTPUT_CONVEXITY			= 0x0080,
    OUTPUT_CONVEX_FEATURES		= 0x0100,
    OUTPUT_CENTROID				= 0x0200,
    OUTPUT_VELOCITY				= 0x0400,
} BlobFlag;

ConfigWindow::ConfigWindow(std::vector<int> *_data, QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::ConfigWindow)
{
    ui->setupUi(this);

    // Disable close button on the window
    setWindowFlags(Qt::Window | Qt::WindowMinimizeButtonHint | Qt::WindowMaximizeButtonHint);

    // Initial config values
    data = _data;
    data->clear();
    data->assign(12,0);
    (*data)[0] = 3;             // Number of lanes
    (*data)[1] = 3;             // Number of Gaussian Mixtures
    (*data)[2] = -1;            // Track ID
    (*data)[3] = 0;             // Occlusion Detection Enabler
    (*data)[4] = 0;             // Occlusion Model
    (*data)[5] = 0;             // Output Processed Video
    (*data)[6] = 0;             // Output Mask Video
    (*data)[7] = 0;             // Output Statistic File
    (*data)[8] = 0;             // Output Timing File
    (*data)[9] = BLOB_BASE;     // Blob Flags
    (*data)[10] = 0;            // Traffic Flow
    (*data)[11] = 0;            // Occlusion Solver Enabler
    flags = BLOB_BASE;

    // UI items configuration
    ui->cbxOHA->addItem("Empiric Model Algorithm", 0);
    ui->cbxOHA->addItem("Convex Model Algorithm", 1);

    ui->cbxTFlow->addItem("Front View", 0);
    ui->cbxTFlow->addItem("Rear View", 1);

    QIntValidator *gV = new QIntValidator(1, 2000, this);
    ui->ledNL->setValidator(new QRegularExpressionValidator(QRegularExpression("[1-8]\\d{0,0}"),this));
    ui->ledNMX->setValidator(new QRegularExpressionValidator(QRegularExpression("[3-5]\\d{0,0}"),this));
    ui->ledID->setValidator(gV);


    init();

    //! Signals
    connect(ui->sbtnOHAEN, SIGNAL(clicked(bool)), this, SLOT(on_sOHAEN_clicked(bool)));
    connect(ui->sbtnOCCSOLVE, SIGNAL(clicked(bool)), this, SLOT(on_sOCCSOLVE_clicked(bool)));
    connect(ui->sbtnIDEN, SIGNAL(clicked(bool)), this, SLOT(on_sIDEN_clicked(bool)));
    connect(ui->sbtnMVOP, SIGNAL(clicked(bool)), this, SLOT(on_sMVOP_clicked(bool)));
    connect(ui->sbtnPVOP, SIGNAL(clicked(bool)), this, SLOT(on_sPVOP_clicked(bool)));
    connect(ui->sbtnSFOP, SIGNAL(clicked(bool)), this, SLOT(on_sSFOP_clicked(bool)));
    connect(ui->sbtnTFOP, SIGNAL(clicked(bool)), this, SLOT(on_sTFOP_clicked(bool)));
}

ConfigWindow::~ConfigWindow()
{
    delete ui;
}

void ConfigWindow::setData(std::vector<int> *_data)
{
    data = _data;
    data->clear();
    data->assign(12,0);
    (*data)[0] = 3;             // Number of lanes
    (*data)[1] = 3;             // Number of Gaussian Mixtures
    (*data)[2] = -1;            // Track ID
    (*data)[3] = 0;             // Occlusion Detection Enabler
    (*data)[4] = 0;             // Occlusion Model
    (*data)[5] = 0;             // Output Processed Video
    (*data)[6] = 0;             // Output Mask Video
    (*data)[7] = 0;             // Output Statistic File
    (*data)[8] = 0;             // Output Timing File
    (*data)[9] = BLOB_BASE;     // Blob Flags
    (*data)[10] = 0;            // Traffic Flow
    (*data)[11] = 0;            // Occlusion Solver Enabler
}

void ConfigWindow::init(void)
{
    // Set Font
    ui->lblOHAEN->setFont(QFont("Roboto medium", 12));
    ui->lblOHAEN_2->setFont(QFont("Roboto medium", 12));
    ui->lblPVOP->setFont(QFont("Roboto medium", 12));
    ui->lblIDEN->setFont(QFont("Roboto medium", 12));
    ui->lblMVOP->setFont(QFont("Roboto medium", 12));
    ui->lblSFOP->setFont(QFont("Roboto medium", 12));
    ui->lblTFOP->setFont(QFont("Roboto medium", 12));
    ui->btnRST->setFont(QFont("Roboto medium", 13));
    ui->btnBack->setFont(QFont("Roboto medium", 13));
    ui->btnSave->setFont(QFont("Roboto medium", 13));
    ui->cbxOHA->setFont(QFont("Roboto medium", 13));
    ui->ledID->setFont(QFont("Roboto medium", 13));
    ui->ledNL->setFont(QFont("Roboto medium", 13));
    ui->ledNMX->setFont(QFont("Roboto medium", 13));
    ui->groupBox->setFont(QFont("Roboto medium", 13));
    ui->groupBox_2->setFont(QFont("Roboto medium", 13));
    ui->groupBox_3->setFont(QFont("Roboto medium", 13));
    ui->groupBox_4->setFont(QFont("Roboto medium", 13));
    ui->groupBox_5->setFont(QFont("Roboto medium", 13));
    ui->cbxTFlow->setFont(QFont("Roboto medium", 13));
    ui->lblTFlow->setFont(QFont("Roboto medium", 13));

    // Set Text
    ui->ledID->setPlaceholderText("Track ID");
    ui->ledID->setAlignment(Qt::AlignCenter);
    ui->ledID->setEnabled(false);
    ui->ledNL->setPlaceholderText("Number of Lanes");
    ui->ledNL->setAlignment(Qt::AlignCenter);
    ui->ledNMX->setPlaceholderText("Number of Mixtures (K)");
    ui->ledNMX->setAlignment(Qt::AlignCenter);
    ui->ledID->setText("");
    ui->ledNL->setText("");
    ui->ledNMX->setText("");

    // Set Layout Alignment
    ui->gridLayout_2->setAlignment(Qt::AlignCenter);
    ui->gridLayout_2->setAlignment(ui->sbtnOHAEN, Qt::AlignCenter);
    ui->gridLayout_2->setAlignment(ui->sbtnOCCSOLVE, Qt::AlignCenter);

    ui->verticalLayout_2->setAlignment(Qt::AlignCenter);
    ui->verticalLayout_2->setAlignment(ui->sbtnPVOP, Qt::AlignCenter);

    ui->verticalLayout_3->setAlignment(Qt::AlignCenter);
    ui->verticalLayout_3->setAlignment(ui->sbtnMVOP, Qt::AlignCenter);

    ui->verticalLayout_4->setAlignment(Qt::AlignCenter);
    ui->verticalLayout_4->setAlignment(ui->sbtnSFOP, Qt::AlignCenter);

    ui->verticalLayout_5->setAlignment(Qt::AlignCenter);
    ui->verticalLayout_5->setAlignment(ui->sbtnTFOP, Qt::AlignCenter);

    ui->verticalLayout_6->setAlignment(Qt::AlignCenter);
    ui->verticalLayout_6->setAlignment(ui->sbtnIDEN, Qt::AlignCenter);

    ui->verticalLayout_7->setAlignment(Qt::AlignCenter);
    ui->verticalLayout_7->setAlignment(ui->cbxOHA, Qt::AlignCenter);

    // Set Enable
    ui->cbxOHA->setEnabled(false);
    ui->ledID->setEnabled(false);
    ui->btnSave->setEnabled(false);
    ui->btnRST->setEnabled(false);
    ui->sbtnOCCSOLVE->setEnabled(false);

    // Set Checked
    ui->chbEcc->setChecked(false);
    ui->chbTheta->setChecked(false);
    ui->chbConvex->setChecked(false);
    ui->chbEqDiam->setChecked(false);
    ui->chbPerimeter->setChecked(false);
    ui->chbCompact->setChecked(false);
    //ui->sbtnOCCSOLVE->setChecked(false);

    ui->cbxTFlow->setCurrentIndex(0);

    modified = false;
}

void ConfigWindow::on_btnSave_pressed()
{
    (*data)[0] = (ui->ledNL->text().length() ? ui->ledNL->text().toInt() : 3);
    (*data)[1] = (ui->ledNMX->text().length() ? ui->ledNMX->text().toInt() : 3);
    (*data)[2] = (ui->sbtnIDEN->isChecked() && ui->ledID->text().length() ? ui->ledID->text().toInt() : -1);
    (*data)[3] = ui->sbtnOHAEN->isChecked();
    (*data)[4] = ui->cbxOHA->currentData().toInt();
    (*data)[5] = ui->sbtnPVOP->isChecked();
    (*data)[6] = ui->sbtnMVOP->isChecked();
    (*data)[7] = ui->sbtnSFOP->isChecked();
    (*data)[8] = ui->sbtnTFOP->isChecked();
    (*data)[9] = flags;
    (*data)[10] = ui->cbxTFlow->currentData().toInt();
    (*data)[11] = ui->sbtnOCCSOLVE->isChecked();

    ui->btnSave->setEnabled(false);

    modified = false;
}

void ConfigWindow::on_btnRST_pressed()
{
    init();

    ui->sbtnIDEN->setChecked(false);
    ui->sbtnMVOP->setChecked(false);
    ui->sbtnPVOP->setChecked(false);
    ui->sbtnSFOP->setChecked(false);
    ui->sbtnOHAEN->setChecked(false);
    ui->sbtnTFOP->setChecked(false);
    ui->chbEcc->setChecked(false);
    ui->chbTheta->setChecked(false);
    ui->chbConvex->setChecked(false);
    ui->chbEqDiam->setChecked(false);
    ui->chbPerimeter->setChecked(false);
    ui->chbCompact->setChecked(false);
    ui->cbxTFlow->setCurrentIndex(0);
    ui->sbtnOCCSOLVE->setEnabled(false);
    ui->sbtnOCCSOLVE->setChecked(false);
    ui->btnSave->setEnabled(false);
    ui->btnRST->setEnabled(false);

    (*data)[0] = 3;             // Number of lanes
    (*data)[1] = 3;             // Number of Gaussian Mixtures
    (*data)[2] = -1;            // Track ID
    (*data)[3] = 0;             // Occlusion Detection Enabler
    (*data)[4] = 0;             // Occlusion Model
    (*data)[5] = 0;             // Output Processed Video
    (*data)[6] = 0;             // Output Mask Video
    (*data)[7] = 0;             // Output Statistic File
    (*data)[8] = 0;             // Output Timing File
    (*data)[9] = BLOB_BASE;     // Blob Flags
    (*data)[10] = 0;            // Traffic Flow
    (*data)[11] = 0;            // Occlusion Solver Enabler

    flags = BLOB_BASE;

    modified = false;
}

void ConfigWindow::on_btnBack_pressed()
{
    if( modified )
    {
        QMessageBox::StandardButton reply;

        reply = QMessageBox::question(this, "Save", "Do you want to save your changes?",
            QMessageBox::Save | QMessageBox::Discard | QMessageBox::Cancel);

        if (reply == QMessageBox::Save) {
            // code for saving...
            (*data)[0] = (ui->ledNL->text().length() ? ui->ledNL->text().toInt() : 3);
            (*data)[1] = (ui->ledNMX->text().length() ? ui->ledNMX->text().toInt() : 3);
            (*data)[2] = (ui->sbtnIDEN->isChecked() && ui->ledID->text().length() ? ui->ledID->text().toInt() : -1);
            (*data)[3] = ui->sbtnOHAEN->isChecked();
            (*data)[4] = ui->cbxOHA->currentData().toInt();
            (*data)[5] = ui->sbtnPVOP->isChecked();
            (*data)[6] = ui->sbtnMVOP->isChecked();
            (*data)[7] = ui->sbtnSFOP->isChecked();
            (*data)[8] = ui->sbtnTFOP->isChecked();
            (*data)[9] = flags;
            (*data)[10] = ui->cbxTFlow->currentData().toInt();
            (*data)[11] = ui->sbtnOCCSOLVE->isChecked();

            ui->btnSave->setEnabled(false);
            ui->btnRST->setEnabled(false);

            modified = false;

            this->close();
        }
        if (reply == QMessageBox::Discard)
        {
            ui->btnSave->setEnabled(false);
            ui->btnRST->setEnabled(false);

            modified = false;

            this->close();
        }
        if(reply == QMessageBox::Cancel)
        {
            //toDo
        }
    }
    else
    {
        this->close();
    }
}

void ConfigWindow::on_sOHAEN_clicked(bool checked)
{
    if (checked == Qt::Unchecked)
    {
        ui->cbxOHA->setEnabled(false);
        ui->sbtnOCCSOLVE->setEnabled(false);
        ui->sbtnOCCSOLVE->setChecked(false);
    }
    else
    {
        ui->cbxOHA->setEnabled(true);
        ui->sbtnOCCSOLVE->setEnabled(true);
    }

    ui->btnRST->setEnabled(true);
    ui->btnSave->setEnabled(true);
    modified = true;
}


void ConfigWindow::on_sOCCSOLVE_clicked(bool checked)
{
    (void)checked;
    ui->btnRST->setEnabled(true);
    ui->btnSave->setEnabled(true);
    modified = true;
}

void ConfigWindow::on_sIDEN_clicked(bool checked)
{
    if (checked == Qt::Unchecked)
    {
        ui->ledID->setEnabled(false);
    }
    else
    {
        ui->ledID->setEnabled(true);
    }

    ui->btnRST->setEnabled(true);
    ui->btnSave->setEnabled(true);
    modified = true;
}

void ConfigWindow::on_sMVOP_clicked(bool checked)
{
    (void)checked;
    ui->btnRST->setEnabled(true);
    ui->btnSave->setEnabled(true);
    modified = true;
}

void ConfigWindow::on_sPVOP_clicked(bool checked)
{
    (void)checked;
    ui->btnRST->setEnabled(true);
    ui->btnSave->setEnabled(true);
    modified = true;
}

void ConfigWindow::on_sSFOP_clicked(bool checked)
{
    (void)checked;
    ui->btnRST->setEnabled(true);
    ui->btnSave->setEnabled(true);
    modified = true;
}

void ConfigWindow::on_sTFOP_clicked(bool checked)
{
    (void)checked;
    ui->btnRST->setEnabled(true);
    ui->btnSave->setEnabled(true);
    modified = true;
}






void ConfigWindow::on_chbTheta_clicked(bool checked)
{
    if(checked)
    {
        SET_FLAG(flags, int(OUTPUT_ORIENTATION));
    }
    else
    {
        CLEAR_FLAG(flags, int(OUTPUT_ORIENTATION));
    }

    ui->btnRST->setEnabled(true);
    ui->btnSave->setEnabled(true);
    modified = true;
}

void ConfigWindow::on_chbConvex_clicked(bool checked)
{
    if(checked)
    {
        SET_FLAG(flags, int(OUTPUT_CONVEXITY));
    }
    else
    {
        CLEAR_FLAG(flags, int(OUTPUT_CONVEXITY));
    }

    ui->btnRST->setEnabled(true);
    ui->btnSave->setEnabled(true);
    modified = true;
}

void ConfigWindow::on_chbPerimeter_clicked(bool checked)
{
    if(checked)
    {
        SET_FLAG(flags, int(OUTPUT_PERIMETER));
    }
    else
    {
        CLEAR_FLAG(flags, int(OUTPUT_PERIMETER));
    }

    ui->btnRST->setEnabled(true);
    ui->btnSave->setEnabled(true);
    modified = true;
}

void ConfigWindow::on_chbEqDiam_clicked(bool checked)
{
    if(checked)
    {
        SET_FLAG(flags, int(OUTPUT_EQDIAM));
    }
    else
    {
        CLEAR_FLAG(flags, int(OUTPUT_EQDIAM));
    }

    ui->btnRST->setEnabled(true);
    ui->btnSave->setEnabled(true);
    modified = true;
}

void ConfigWindow::on_chbEcc_clicked(bool checked)
{
    if(checked)
    {
        SET_FLAG(flags, int(OUTPUT_ECCENT));
    }
    else
    {
        CLEAR_FLAG(flags, int(OUTPUT_ECCENT));
    }

    ui->btnRST->setEnabled(true);
    ui->btnSave->setEnabled(true);
    modified = true;
}

void ConfigWindow::on_chbCompact_clicked(bool checked)
{
    if(checked)
    {
        SET_FLAG(flags, int(OUTPUT_COMPACTNESS));
    }
    else
    {
        CLEAR_FLAG(flags, int(OUTPUT_COMPACTNESS));
    }

    ui->btnRST->setEnabled(true);
    ui->btnSave->setEnabled(true);
    modified = true;
}

void ConfigWindow::on_ledID_textChanged(const QString &arg1)
{
    (void)arg1;
    ui->btnRST->setEnabled(true);
    ui->btnSave->setEnabled(true);
    modified = true;
}

void ConfigWindow::on_cbxOHA_currentIndexChanged(int index)
{
    (void)index;
    ui->btnRST->setEnabled(true);
    ui->btnSave->setEnabled(true);
    modified = true;
}

void ConfigWindow::on_ledNL_textEdited(const QString &arg1)
{
    (void)arg1;
    ui->btnRST->setEnabled(true);
    ui->btnSave->setEnabled(true);
    modified = true;
}

void ConfigWindow::on_ledNMX_textChanged(const QString &arg1)
{
    (void)arg1;
    ui->btnRST->setEnabled(true);
    ui->btnSave->setEnabled(true);
    modified = true;
}

void ConfigWindow::on_cbxTFlow_currentIndexChanged(int index)
{
    (void)index;
    ui->btnRST->setEnabled(true);
    ui->btnSave->setEnabled(true);
    modified = true;
}
