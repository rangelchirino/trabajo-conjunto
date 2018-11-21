#include "svmclassifier.hpp"

namespace sdcv {

	/*********************************************************/
	/*  P R I V A T E   P R O T O T Y P E   F U N C T I O N  */
	/*********************************************************/
	static bool csv_filter_rows(const cv::Mat &rc);
	static std::string getSVMTypeString(cv::ml::SVM::Types type);
	static std::string getSVMKernelString(cv::ml::SVM::KernelTypes type);
	/*********************************************************/
	/* 				C L A S S   M E T H O D S 				 */
	/*********************************************************/
	/*!
	* @name	SVMClassifier
	* @brief	SVMClassifier class constructor
	*/
	SVMClassifier::SVMClassifier(cv::ml::SVM::Types type, cv::ml::SVM::KernelTypes kernel, sdcv::ROI roi, int NbClasses, int NbFeatures) {
		auto auxi = roi.getName().find_first_of('-');

		this->type = type;
		this->kernel = kernel;
		this->ModelName = getSVMTypeString(this->type) + "Model" + (auxi  < roi.getName().length() ? roi.getName().substr(0, auxi) : roi.getName()) + "IdM";
		
		//std::vector<std::vector<cv::Point>> Regions = roi.getRegions();
		//std::cout << "ROI Vertex: " << std::endl << cv::Mat(roi.getVertices()) << std::endl << std::endl;
		//std::cout << "ROI Regions: "<< Regions.size() << std::endl << cv::Mat(Regions.at(0)) << std::endl << cv::Mat(Regions.at(1)) << std::endl << cv::Mat(Regions.at(2)) << std::endl << std::endl;
		//std::cin.get();
		this->W2 = (float)(roi.getVertices().at(1).x - roi.getVertices().at(0).x);					// ROI vertex end line width [FIXED]
		this->W1 = (float)(roi.getRegions().back().back().x - roi.getRegions().back().front().x);	// Classification Line Region width
		this->Dx = (W2 - W1) /(float)2.0;
		this->Dy = (float)(roi.getVertices().at(0).y - roi.getRegions().back().front().y); 			// ROIdata.LineArea(2, 2);
		this->RoidLa = (float)(roi.getRegions().back().front().y); 									// ROIdata.LineArea(2,2);
		this->RoiParams.push_back(W1);
		this->RoiParams.push_back(W2);
		this->RoiParams.push_back(Dx);
		this->RoiParams.push_back(Dy);
		this->RoiParams.push_back(RoidLa);

		if(NbClasses > 0 && NbFeatures > 0) {
			counter = cv::Mat::zeros(1, NbClasses, CV_16UC1);
			this->NbClasses = NbClasses;
			this->NbFeatures = NbFeatures;
			this->svm.clear();
			this->svm.reserve(NbClasses);
			
			for(int i = 0; i < NbClasses; i++){
				std::string FileModelName = "DATA/" + roi.getName() + "/Classification/SVM/" + this->ModelName + std::to_string(i) + ".xml";
				std::cout << "Model.Filename: " << FileModelName << std::endl;
				CV_Assert(fs_exist(FileModelName));
				//cv::Ptr<cv::ml::SVM> model;
				//this->svm.at(i) = cv::Algorithm::load<cv::ml::SVM>(this->ModelName + std::to_string(i) + ".xml");
				//model->load(this->ModelName + std::to_string(i) + ".xml");
				cv::Ptr<cv::ml::SVM> model = cv::Algorithm::load<cv::ml::SVM>(FileModelName);
				CV_Assert(model->getSupportVectors().cols == NbFeatures);
				//model = cv::ml::SVM::load(this->ModelName + std::to_string(i) + ".xml");
				this->svm.push_back(model);
				std::cout << "Model.SupportVectors: " << std::endl << this->svm.back()->getSupportVectors() << std::endl << std::endl;
			}
		}

	}

	/*!
	* @name		train
	* @brief	Training step, compute the SVM models
	*
	* @note		NEED TO ADD FEATURES COLUMN INDEX TO SELECT THE FEATURES
	*/
	void SVMClassifier::train(std::string filename, std::vector< std::vector<int> > trainingData, int NbSamples) {
		// Asssert
		CV_Assert(sdcv::fs_exist(filename));
		CV_Assert(!trainingData.empty());
		CV_Assert(NbSamples > 0);
		auto fpart = sdcv::fs_fileparts(filename);
		std::string path = std::get<0>(fpart) + "/Classification/SVM/";
		if (!std::experimental::filesystem::exists(path))
			std::experimental::filesystem::create_directories(path);
		std::ofstream stdout_file(path + "stdout.txt");

		// Parsing CSV file
		std::vector<int> cidx = { 2,7,8,9,4};
		cv::Mat InFullMat = sdcv::fs_read(filename, ',', 1), InFullRecMat, InputSamples;
		sdcv::cv_remove_if(InFullMat, InFullRecMat, csv_filter_rows);	// Row remiving
		sdcv::cv_copy(InFullRecMat, InputSamples, cidx, false);		// Column copying by indexes
		InFullMat.release();
		InFullRecMat.release();

		this->Dimension 	= InputSamples.rows;
		this->NbFeatures 	= InputSamples.cols;
		this->NbClasses 	= (int)trainingData.size();

		stdout_file << "Input sample space" << std::endl << InputSamples << std::endl << std::endl;

		// ------------------------------------------------------------------------------------------
		cv::Mat InSampledMat2(this->Dimension - 1, NbFeatures, CV_32F);
		cv::Mat InSampledVec(this->Dimension - 1, 1, CV_32F);
		int ValidDatas = this->Dimension - 1;

		InputSamples.rowRange(0, ValidDatas).copyTo(InSampledMat2.rowRange(0, ValidDatas));
		InSampledMat2.col(0).copyTo(InSampledVec);
		
		double min, maxId;
		cv::minMaxLoc(InSampledVec, &min, &maxId);

		stdout_file << "SVM" << std::endl << "{" << std::endl;
		stdout_file << "\tDimension: " << InputSamples.rows << std::endl << "\tFeatures: " << InputSamples.cols << std::endl;
		stdout_file << "\tClasses: " << trainingData.size() << std::endl;
		stdout_file << "\tFeatures.Size: " << InSampledMat2.rows << "x" << InSampledMat2.cols << std::endl;
		stdout_file << "\tID: (" << min << ", " << maxId << ")" << std::endl;
		stdout_file << "}" << std::endl << std::endl;
		

		// Select only K samples after the classification line
		cv::Mat InSampledMat3 = cv::Mat::zeros((int)maxId, NbFeatures, CV_32F); //Mean of K samples into the preselected region
		stdout_file << "Mean of each samples: {" << std::endl;
		for (int j = 1; j < maxId; j++)
		{
			bool bMeanFlag = false;
			int SamplerCounter = 0;
			float Vaux[4] = {0.0, 0.0, 0.0, 0.0};

			for (int k = 0; k < ValidDatas; k++)
			{
				/* -------------------------------- As a lamda function -------------------------------- */
				if (InSampledVec.at<float>(k, 0) == (float)j) {
					if (SamplerCounter < NbSamples) {
						Vaux[0] = Vaux[0] + InSampledMat2.at<float>(k, 1); 		// Area
						Vaux[1] = Vaux[1] + InSampledMat2.at<float>(k, 2); 		// Width
						Vaux[2] = Vaux[2] + InSampledMat2.at<float>(k, 3); 		// Height
						Vaux[3] = Vaux[3] + InSampledMat2.at<float>(k, 4); 		// Detected Y
						SamplerCounter++;
					} else {
						InSampledMat3.at<float>(j, 0) = (float)j; 						// Id
						InSampledMat3.at<float>(j, 1) = Vaux[0] / (float)NbSamples; 	// Mean of Area
						InSampledMat3.at<float>(j, 2) = Vaux[1] / (float)NbSamples; 	// Mean of Width
						InSampledMat3.at<float>(j, 3) = Vaux[2] / (float)NbSamples; 	// Mean of Height
						InSampledMat3.at<float>(j, 4) = Vaux[3] / (float)NbSamples; 	// Mean of Detected Y
						bMeanFlag = true;
						break;
					}
				}
				/*-----------------------------------------------------------------------------------------*/
			}
			
			// If there are not K samples
			if (!bMeanFlag)
			{
				stdout_file << "\tSample #" << j << " does not have " << NbSamples << " samples" << std::endl;
				/* -------------------------------- As a lamda function -------------------------------- */
				InSampledMat3.at<float>(j, 0) = (float)j; 								// Id
				InSampledMat3.at<float>(j, 1) = Vaux[0] / (float)(SamplerCounter + 1); 	// Mean of Area
				InSampledMat3.at<float>(j, 2) = Vaux[1] / (float)(SamplerCounter + 1); 	// Mean of width
				InSampledMat3.at<float>(j, 3) = Vaux[2] / (float)(SamplerCounter + 1); 	// Mean of Heigth			
				InSampledMat3.at<float>(j, 4) = Vaux[3] / (float)(SamplerCounter + 1); 	// Mean of detected Y
				/*-----------------------------------------------------------------------------------------*/
			}
			stdout_file << "\tSample.mean #" << j << ":" << InSampledMat3.row(j) << std::endl;
		}
		stdout_file << "}" << std::endl << std::endl;
		
		
		// Move the datas into the classification line	----------------------------------------------------------------
		stdout_file << "ROI Parameters {" << std::endl;
		stdout_file << "Dy: "		<< this->Dy		<< std::endl;
		stdout_file << "Dx: "		<< this->Dx		<< std::endl;
		stdout_file << "W1: "		<< this->W1		<< std::endl;
		stdout_file << "W2: "		<< this->W2		<< std::endl;
		stdout_file << "ROI.LA: " << this->RoidLa << std::endl;
		stdout_file << "}" << std::endl << std::endl;
		
		//float Dy = 94.02;				// Dy = ROIdata.yi(1) - ROIdata.LineArea(2, 2);
		//float W2 = 302.901;			// W2 = ROIdata.xi(2) - ROIdata.xi(1);
		//float W1 = 173.1959;			// W1 = ROIdata.LineArea(2, 3) - ROIdata.LineArea(2, 1);
		//float Dx = (W2 - W1) / 2.0;	// 64.8601 //Dx = (W2 - W1) / 2;
		//float RoidLa = 128.6296;		// ROIdata.LineArea(2,2)

		float dy = 0.0;
		float dx = 0.0;
		float Wx = 0.0;
		for (int j = 1; j < maxId; j++)
		{
			dy = InSampledMat3.at<float>(j, 4) - RoidLa;
			dx = (dy /(float)Dy) * Dx;
			Wx = W1 + (float)2.0 * dx;
			InSampledMat3.at<float>(j, 1) = InSampledMat3.at<float>(j, 1) * ((W1*W1) / (float)(Wx*Wx)); 		// Into class line Mean of Area
			InSampledMat3.at<float>(j, 2) = InSampledMat3.at<float>(j, 2) * (W1 / (float)Wx); 				// Into class line  Mean of width
			InSampledMat3.at<float>(j, 3) = InSampledMat3.at<float>(j, 3) * (W1 / (float)Wx); 				// Into class Mean of Height
			if (InSampledMat3.at<float>(j, 2) != 0.0)
				InSampledMat3.at<float>(j, 3) = InSampledMat3.at<float>(j, 3) / (float)InSampledMat3.at<float>(j, 2); // Into class Mean of Height / Width
			else
				InSampledMat3.at<float>(j, 3) = 0.0;
		}
		stdout_file << std::endl << "SampledMatrix3:" << std::endl << InSampledMat3 << std::endl << std::endl;
		stdout_file << "Moving datas into the classification line DONE" << std::endl;

		// Normalize datas ----------------------------------------------------------------
		float MaxVec[2] = { 0.0,0.0 };
		for (int j = 0; j < 2; j++) {
			InSampledMat2.col(j+1).copyTo(InSampledVec);
			double min, max;
			cv::minMaxLoc(InSampledVec, &min, &max); // Max of Area and Witdh
			MaxVec[j] = (float)max;
		}
		InSampledMat3.col(3).copyTo(InSampledVec);
		double MaxRel;
		cv::minMaxLoc(InSampledVec, &min, &MaxRel); //Max rel

		for (int j = 0; j < maxId; j++) 
		{
			InSampledMat3.at<float>(j,1) = InSampledMat3.at<float>(j, 1) / MaxVec[0];
			InSampledMat3.at<float>(j, 2) = InSampledMat3.at<float>(j, 2) / MaxVec[1];
			InSampledMat3.at<float>(j, 3) = InSampledMat3.at<float>(j, 3) / (float)MaxRel;
		}
		stdout_file << "Normalized Params {" << std::endl;
		stdout_file << "\tArea: " << MaxVec[0] << std::endl;
		stdout_file << "\tWidth: " << MaxVec[1] << std::endl;
		stdout_file << "\tH/W: " << MaxRel << std::endl;
		stdout_file << "}" << std::endl << std::endl;
		stdout_file << "Data.Sampled.Mean.Norm: " << std::endl << InSampledMat3 << std::endl << std::endl;
		stdout_file << "Normalizing datas DONE" << std::endl << std::endl;

		// Get Input Matrix for classification (by Id)	----------------------------------------------------------------
		std::vector<int> IDCount(trainingData.size(), 0);
		std::vector<cv::Mat> MIdX(trainingData.size());
		std::vector<float> Vaux = {0.0, 0.0, 0.0, 0.0, 0.0};
		auto CurrentIDCounter = IDCount.begin();
		auto CurrentMatrix = MIdX.begin();
		for (auto CurrentTrainingClass = trainingData.begin(); CurrentTrainingClass != trainingData.end(); ++CurrentTrainingClass) {
			for (auto IDX = CurrentTrainingClass->begin(); IDX != CurrentTrainingClass->end(); ++IDX) {
				InSampledMat3.rowRange(*IDX, *IDX + 1).copyTo(Vaux);
				if (Vaux[1] == 0.0) (*CurrentIDCounter)++;
			}

			*CurrentMatrix = cv::Mat((int)CurrentTrainingClass->size() - *CurrentIDCounter, 3, CV_32F);
			stdout_file << "IDMatrix.Size: " << CurrentMatrix->rows << "x" << CurrentMatrix->cols << std::endl;
			++CurrentIDCounter;
			++CurrentMatrix;
		}

		std::fill(IDCount.begin(), IDCount.end(), 0);
		CurrentMatrix = MIdX.begin();
		CurrentIDCounter = IDCount.begin();
		for (auto CurrentID = trainingData.begin(); CurrentID != trainingData.end(); ++CurrentID) {
			for (auto ID = CurrentID->begin(); ID != CurrentID->end(); ++ID) {
				Vaux = InSampledMat3.rowRange(*ID, *ID + 1);
				if (Vaux[1] != 0.0) {
					CurrentMatrix->at<float>(*CurrentIDCounter, 0)		= Vaux[1];
					CurrentMatrix->at<float>(*CurrentIDCounter, 1)		= Vaux[2];
					CurrentMatrix->at<float>((*CurrentIDCounter)++, 2)	= Vaux[3];
				}
			}
			stdout_file << "Matrix: " << std::endl << *CurrentMatrix << std::endl << std::endl;
			++CurrentIDCounter;
			++CurrentMatrix;
		}
		stdout_file << "Getting Input Matrix for classification DONE" << std::endl;

		// Save Centroids ----------------------------------------------------------------
		/*cv::FileStorage fs("MIdP.xml", cv::FileStorage::WRITE);
		fs << "MIdP" << MIdP;
		fs << "MIdM" << MIdM;
		fs << "MIdG" << MIdG;
		fs << "InSampledMat3" << InSampledMat3;
		fs.release();//*/
		
		// Get the Models	----------------------------------------------------------------
		int NbModel = 0;
		for (CurrentMatrix = MIdX.begin(); CurrentMatrix != MIdX.end(); ++CurrentMatrix) {
			cv::Ptr<cv::ml::SVM> model;
			cv::Mat labelsOCsvmIdX = cv::Mat::ones(CurrentMatrix->rows, 1, CV_32SC1);

			model = cv::ml::SVM::create();
			model->setType(this->type);
			model->setC(5.00);
			model->setKernel(this->kernel);
			model->setGamma(.000020);
			model->setNu(0.025);
			model->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, (int)1e7, 1e-6));
			model->train(*CurrentMatrix, cv::ml::ROW_SAMPLE, labelsOCsvmIdX);
			model->save(path + this->ModelName + std::to_string(NbModel) + ".xml");
			stdout_file << "Model has been saved in " << this->ModelName + std::to_string(NbModel++) + ".xml" << std::endl;
		}
	}
	
	/*!
	* @name	getDimension
	* @brief	Get the input data dimensionality
	*/
	int SVMClassifier::getDimension(void) { return Dimension; }
	
	/*!
	 * @name	getNumFeatures
	 * @brief	Get the number of features
	 */
	int SVMClassifier::getNumberFeatures(void) { return NbFeatures; }
	
	std::vector<float> SVMClassifier::getNormParam(void) { return this->NormParam; }
	std::vector<float> SVMClassifier::getRoiParams(void) { return this->RoiParams; }
	/*!
	 * @name	getNumFeatures
	 * @brief	Get the number of features
	 */
	int SVMClassifier::getNumberClasses(void) { return NbClasses; }
	
	void SVMClassifier::setNormParam(std::vector<float> NormParam) { this->NormParam = NormParam; }
	/*!
	* @name		classify
	* @brief	Given a vector of sampled features called X, this function classifies it into a class
	*/
	int SVMClassifier::classify(cv::Mat sample) {
		static bool flag = false;
		int Class = -1;

		if (!flag) {
			std::cout << "Model[SMALL].SupportVectors: " << std::endl << this->svm.at(0)->getSupportVectors() << std::endl << std::endl;
			std::cout << "Model[MEDIUM].SupportVectors: " << std::endl << this->svm.at(1)->getSupportVectors() << std::endl << std::endl;
			std::cout << "Model[LARGE].SupportVectors: " << std::endl << this->svm.at(2)->getSupportVectors() << std::endl << std::endl;
			flag = true;
		}

		if (sample.cols == this->NbFeatures) {
			auto Model = this->svm.begin();
			Class = 0;
			while(Model != svm.end()) {
				if ((*Model)->predict(sample) > 0.0) break;
				Class++;
				++Model;
			}

			Class = (Class == this->NbClasses ? -1 : Class);
		}

		return Class;
	}
	
	
	
	/*********************************************************/
	/*  P R I V A T E   R E F E R E N C E   F U N C T I O N  */
	/*********************************************************/
	bool csv_filter_rows(const cv::Mat &rc) { 
		return (rc.at<float>(2) == 0 || rc.at<float>(12) != 2); 
	}

	std::string getSVMKernelString(cv::ml::SVM::KernelTypes type) {
		std::string KernelType;

		switch (type)
		{
			case cv::ml::SVM::CUSTOM:
				KernelType = "CUSTOM";
			break;

			case cv::ml::SVM::LINEAR:
				KernelType = "LINEAR";
			break;

			case cv::ml::SVM::POLY:
				KernelType = "POLY";
			break;

			case cv::ml::SVM::RBF:
				KernelType = "RBF";
			break;

			case cv::ml::SVM::SIGMOID:
				KernelType = "SIGMOID";
			break;

			case cv::ml::SVM::CHI2:
				KernelType = "CHI2";
			break;

			case cv::ml::SVM::INTER:
				KernelType = "INTER";
			break;

			default:
				KernelType = "ERROR";
			break;
		}

		return KernelType;
	}

	std::string getSVMTypeString(cv::ml::SVM::Types type) {
		std::string strType;

		switch(type) {
			case cv::ml::SVM::C_SVC:
				strType = "CSVC";
			break;

			case cv::ml::SVM::NU_SVC:
				strType = "NuSVC";
			break;

			case cv::ml::SVM::ONE_CLASS:
				strType = "OcSVC";
			break;

			case cv::ml::SVM::EPS_SVR:
				strType = "EpsSVR";
			break;

			case cv::ml::SVM::NU_SVR:
				strType = "NuSVR";
			break;

			default:
				strType = "ERROR";
			break;
		}

		return strType;
	}
}

/*! ************** End of file ----------------- CINVESTAV GDL */