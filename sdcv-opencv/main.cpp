/*
* @file: main.cpp
* @authors:
* • Matlab firts release: M.C. Roxana Velazquez
* • Matlab improvements: Dr. Jayro Santiago
* • Opencv and improvements: Ing. Fernando Hermosillo
*
* @date: 28/11/2016
* @history:
* 11/07/2017: Adding ROI Regions in ROI class
* 12/08/2017: Adding SVMClassifier class
* 20/08/2017: Adding VOI method in tracking
*
* @brief:	Sistema de Detección y Clasificacón Vehicular
*			(SDCV) is a software that will be an 
*			Intelligent Transport System solution.
*
* @notes:
* Test Background subtractor with LUV or LAB 
* color space.
* Change YML to XML file format.
*
* copyright: CINVESTAV TELECOM GDL
*/

/*********************************************************/
/*                    I N C L U D E S                    */
/*********************************************************/
#include <iostream>
#include <stdint.h>
#include <algorithm>
#include <string>
#include <vector>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <iterator>
#include <functional>
#include <experimental/filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2\plot.hpp>
#include <sdcv/sdcv.h>
#include <opencv2\tracking.hpp>

/*********************************************************/
/*                     D E F I N E S                     */
/*********************************************************/
#define KEY_ESC		27
#define KEY_SPACE	32
#define KEY_0		49
#define KEY_1		50
#define KEY_2		51


/*********************************************************/
/*                      M A C R O S                      */
/*********************************************************/

/*********************************************************/
/*                  N A M E S P A C E S                  */
/*********************************************************/

/*********************************************************/
/*                     S T R U C T S                     */
/*********************************************************/

/*********************************************************/
/*          P R O T O T Y P E   F U N C T I O N          */
/*********************************************************/
void svm_example(void) {
	// Data for visual representation
	int width = 512, height = 512;
	cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);

	// Set up training data
	int labels[4] = { 1, -1, -1, -1 };			// Labels
	cv::Mat labelsMat(4, 1, CV_32SC1, labels);
	float trainingData[4][2] = { { 501, 10 },{ 255, 10 },{ 501, 255 },{ 10, 501 } };	// Training data
	cv::Mat trainingDataMat(4, 2, CV_32FC1, trainingData);

	// Set up SVM's parameters
	cv::ml::SVM::Types svmtypes = cv::ml::SVM::C_SVC;
	cv::ml::SVM::KernelTypes kernelType = cv::ml::SVM::LINEAR;

	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();		// Create an object of SVM
	svm->setType(svmtypes);									// Set SVM Type (in this case OCSVM)
	svm->setKernel(kernelType);								// Set SVM Kernel type
															//svm->setC(5.00);										// Set the SVM C parameter
															//svm->setGamma(.000020);								// Set SVM gamma parameter
															//svm->setNu(0.025);									// Set the SVM Nu parameter
	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));	// Set the termination criterium

																					// Train the SVM
																					// Train Model sampling by rows (samples, sample_type, labels)
	svm->train(trainingDataMat, cv::ml::ROW_SAMPLE, labelsMat);

	// Show the decision regions given by the SVM
	cv::Vec3b green(0, 255, 0), blue(255, 0, 0);	// define colors
	for (int i = 0; i < image.rows; ++i)
		for (int j = 0; j < image.cols; ++j)
		{
			// The sample
			cv::Mat sampleMat = (cv::Mat_<float>(1, 2) << j, i);

			// Prediction by SVM
			float response = svm->predict(sampleMat);

			if (response == 1)
				image.at<cv::Vec3b>(i, j) = green;
			else if (response == -1)
				image.at<cv::Vec3b>(i, j) = blue;
		}

	// Show the training data
	int thickness = -1;
	int lineType = 8;
	cv::circle(image, cv::Point(501, 10), 5, cv::Scalar(0, 255, 255), thickness, lineType);
	cv::circle(image, cv::Point(255, 10), 5, cv::Scalar(0, 0, 255), thickness, lineType);
	cv::circle(image, cv::Point(501, 255), 5, cv::Scalar(0, 0, 255), thickness, lineType);
	cv::circle(image, cv::Point(10, 501), 5, cv::Scalar(0, 0, 255), thickness, lineType);

	// Show support vectors
	thickness = 2;
	lineType = 8;
	cv::Mat sv = svm->getSupportVectors();
	std::cout << "SV Count: " << sv.rows << std::endl;
	std::cout << "SV[1] = " << sv.row(0) << std::endl;
	for (int i = 0; i < sv.rows; ++i)
	{
		const float* v = sv.ptr<float>(i);
		cv::circle(image, cv::Point((int)v[0], (int)v[1]), 3, cv::Scalar(255, 255, 0), thickness, lineType);
	}

	cv::imwrite("result.png", image);        // save the image*/

	cv::imshow("SVM Simple Example", image); // show it to the user
	cv::waitKey(0);
}

/**********************************************************/
/*            G L O B A L    V A R I A B L E S            */
/**********************************************************/

/*********************************************************/
/*                M A I N   P R O G R A M                */
/*********************************************************/
int main(int argc, const char** argv)
{

	//////////////////////////////////////////////
	//     _____   _____     _____  __      __	//
	//    / ____| |  __ \   / ____| \ \    / /	//
	//   | (___   | |  | | | |       \ \  / /	//
	//    \___ \  | |  | | | |        \ \/ /	//
	//    ____) | | |__| | | |____     \  /		//
	//   |_____/  |_____/   \_____|     \/		//
	//////////////////////////////////////////////
	int key = 0, NbFrame = 0;
	//cv::String videoname = "V16-2016-09-20-1321"; int NumDivLines = 4;
	///cv::String videoname = "V00";
	cv::String videoname = "V01-2016-07-10"; int NumDivLines = 4;
	//cv::String videoname = "V18-2016-10-05-1340";
	//cv::String videoname = "V06-2016-06-15-1838";	int NumDivLines = 4;
	//cv::String videoname = "V07-2016-07-15-1840";
	//cv::String videoname = "video720";
	//cv::String videoname = "20161005_134318";
	///cv::String videoname = "V19-2016-10-05_1343-420x240-25f-19";
	//cv::String videoname = "V07-2016-07-15-1840";
	///std::string videoname = "V18-2016-10-05-1340-420x240-25fv18";

	//std::string videoname = "V25-2016-11-10-420x240-25";	int NumDivLines = 5;
	///std::string videoname = "V4";	int NumDivLines = 5; // 5 lines
	///std::string videoname = "V6";	int NumDivLines = 4; // 4 lines
	///std::string videoname = "V30";	int NumDivLines = 4; // 4 lines
	///std::string videoname = "V31";	int NumDivLines = 4; // 4 lines
	///std::string videoname = "V32";	int NumDivLines = 6; // 6 lines
	//std::string videoname = "V33";	int NumDivLines = 2; // 2
	//std::string videoname = "V34";	int NumDivLines = 3; // 3
	//std::string videoname = "V35";	int NumDivLines = 4; // 4

	// TESTING ENTROPY
	//cv::String videoname = "V01"; int NumDivLines = 4;
	//std::string videoname = "V02"; int NumDivLines = 5;
	//std::string videoname = "V02-R"; int NumDivLines = 5;
	//std::string videoname = "V5"; int NumDivLines = 5;
	//std::string videoname = "V06"; int NumDivLines = 5;
	//std::string videoname = "V09";
	//std::string videoname = "V12";
	//std::string videoname = "V14"; int NumDivLines = 5;
	//std::string videoname = "V14-R"; int NumDivLines = 5;
	//std::string videoname = "V14-2018"; int NumDivLines = 5;
	//std::string videoname = "V15";
	//std::string videoname = "V25-2016-11-10-420x240-25";	int NumDivLines = 5;
	//std::string videoname = "V30";
	//std::string videoname = "V31"; int NumDivLines = 4;
	//std::string videoname = "V32";
	//std::string videoname = "V32-2"; int NumDivLines = 6;
	//std::string videoname = "V33";
	//std::string videoname = "V34";
	//std::string videoname = "V35"; int NumDivLines = 5;
	//std::string videoname = "V39"; int NumDivLines = 5;
	//std::string videoname = "V39-R"; int NumDivLines = 5;
	//std::string videoname = "V55"; int NumDivLines = 4;
	//std::string videoname = "V74-L"; int NumDivLines = 3;
	//std::string videoname = "V74-R"; int NumDivLines = 3;

	/// TESTING OCCLUSION
	//cv::String videoname = "V1"; int NumDivLines = 5;	// OK
	//cv::String videoname = "V2"; int NumDivLines = 5;	// OK
	//cv::String videoname = "V3"; int NumDivLines = 5;	// OK
	//cv::String videoname = "V4"; int NumDivLines = 5;	// OK
	//cv::String videoname = "V5"; int NumDivLines = 5;	// OK
	//cv::String videoname = "V6"; int NumDivLines = 4;	// OK
	//cv::String videoname = "V7"; int NumDivLines = 4;	// OK
	//cv::String videoname = "V8"; int NumDivLines = 4;

	//cv::String videoname = "20161031_123041-25f";	// 5 lines

	/// SYSTEM INIT -----------------------------------------------------------------------------------
	if (!std::experimental::filesystem::exists(std::string("DATA/")))
		std::experimental::filesystem::create_directories(std::string("DATA/" + videoname + "/"));

	if (!std::experimental::filesystem::exists(std::string("DATA/" + videoname)))
		std::experimental::filesystem::create_directory(std::string("DATA/" + videoname));

	if (std::experimental::filesystem::exists(std::string("DATA/" + videoname + "/TEST")))
		std::experimental::filesystem::remove_all(std::string("DATA/" + videoname + "/TEST"));
	std::experimental::filesystem::create_directory(std::string("DATA/" + videoname + "/TEST"));


	cv::String projdir = "DATA/" + videoname + "/";

	cv::VideoCapture video(videoname + ".mp4");
	double fps = video.get(cv::CAP_PROP_FPS);				//
	double videoW = video.get(cv::CAP_PROP_FRAME_WIDTH);	//
	double videoH = video.get(cv::CAP_PROP_FRAME_HEIGHT);	//
	int codec = (int)video.get(cv::CAP_PROP_FOURCC);		// Get Codec Type- Int form
	
	
	/* ROI  */
	sdcv::ROI roi(videoname);
	// 1. Definir ROI en el orden:
	// 1.1. Vertice izquierdo inferior
	// 1.2. Vertice derecho inferior
	// 1.3. Vertice derecho superior
	// 1.4. Vertice izquierdo superior
	// 1.5. Vertice izquierdo inferior
	// 2. Lineas de división
	// 3. Linea de deteccion
	// 4. Linea de fin de ROI
	// 5. Lineas de classificación en el orden:
	// 5.1 Linea superior
	// 5.2 Linea inferior
	// Add a Height/10 extra space for display current ROI definition name {vertex, line detection, ...}
	roi.create(videoname + ".mp4", NumDivLines, 2, true);

	// DETECTOR ---------------------------------------------------------------------------------------
	sdcv::occlusionType occtype = sdcv::OCC_NORM_AREA;			// Algoritmo de oclusion
	sdcv::Detector detector(100, 100000, occtype, true, true);	// Configura clase
	detector.setROI(roi);
	detector.setOclussion(true);								// Habilita algoritmo de oclusion
	detector.setShadowRemoval(false);							// Habilita deteccion de sombras

	// CLASSIFIER -------------------------------------------------------------------------------------
	sdcv::Classifier classifier(4, std::vector<double>({ 0.12, 1.2, 100.0, 0.0 }), std::vector<std::string>({ "S", "M", "G" , "FP" }));

	// TRACKER ----------------------------------------------------------------------------------------
	sdcv::Tracker tracker(8, 12, fps, roi, &classifier);

	std::ofstream file(projdir + "statistics.csv");
	file << "Frame,idAlg,ID,DetectedCentroidX,DetectedCentroidY,EstimatedCentroidX,EstimatedCentroidY,Area,Width,Heigth,Velocity,RelAreaWH,RegionID,Occlusion,ObjId" << std::endl;
	std::ofstream fileRT(projdir + "time.txt");
	fileRT << "Total,Detection,Tracking,Drawing" << std::endl;
	fileRT << 1.0 / fps << std::endl;


	// Video Writer --------------------------------------
	cv::VideoWriter wrvideo(projdir + "video_out.mp4",
		cv::VideoWriter::fourcc('H', '2', '6', '4'),
		fps,
		cv::Size((int)videoW * 2,
		(int)videoH));

	if (!wrvideo.isOpened()) std::cout << "Fail to open video writer with codec: " << codec << std::endl;

	/// SYSTEM RUN  -----------------------------------------------------------------------------------
	cv::destroyAllWindows();
	cv::namedWindow("Track", cv::WINDOW_KEEPRATIO);
	cv::namedWindow("Mask", cv::WINDOW_KEEPRATIO);
	cv::namedWindow("Foreground", cv::WINDOW_KEEPRATIO);
	double t, detectorAvg = 0, trackerAvg = 0, drawAvg = 0;
	while ( true ) {
		cv::Mat frame, Roi, mask, frameTracking, frameForeground;
		double Tdetect, Ttracking, Tdrawing;

		if (!video.read(frame)) break;
		frame.copyTo(frameTracking);
		double time_per_frame;

		t = (double)cv::getTickCount();
		detector.apply(frame, mask);
		t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
		detectorAvg += t;
		Tdetect = t;
		frameTracking.copyTo(frameForeground, mask);

		//cv::Mat detections = frame;
		//detector.draw(detections);

		//
		time_per_frame = t;

		cv::Mat rgbmask;
		// Debug
		cv::cvtColor(mask, rgbmask, CV_GRAY2RGB);
		if (NbFrame > 0) {
			t = (double)cv::getTickCount();
			tracker.track(detector.getBlobs());
			t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
			trackerAvg += t;
			time_per_frame += t;
			Ttracking = t;

			t = (double)cv::getTickCount();
			tracker.draw(frameTracking);
			tracker.draw(rgbmask, false);
			t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
			drawAvg += t;
			Tdrawing = t;

			time_per_frame += t;
		}
		// Measure time
		fileRT << time_per_frame << "," << Tdetect << "," << Ttracking << "," << Tdrawing << std::endl;

		//cv::imshow("DETECTIONS", detections);
		cv::imshow("Track", frameTracking);
		cv::imshow("Mask", rgbmask);
		cv::imshow("Foreground", frameForeground);
		NbFrame++;

		cv::Mat outFrame;
		cv::hconcat(frameTracking, rgbmask, outFrame);
		wrvideo.write(outFrame);
		//cv::imwrite("DATA/RAW/" + std::to_string(NbFrame) + ".bmp", MaskRGB);

		key = cv::waitKey(30);
		if (key == sdcv::VK_SPACE) cv::waitKey();
		else if (key == sdcv::VK_ESC) break;
		else if (key == KEY_0) {
			video.set(cv::CAP_PROP_POS_FRAMES, 0);
			tracker.clear();
		}
	}
	wrvideo.release();
	
	cv::waitKey();
	cv::destroyAllWindows();
	
	return 0;
}

/* ************** E N D   O F   F I L E ----------------- CINVESTAV TELECOM GDL */
