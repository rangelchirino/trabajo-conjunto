/*
* @file: main.cpp
* @authors:
* • Matlab firts release: M.C. Roxana Velazquez
* • Matlab improvements: Dr. Jayro Santiago
* • Opencv and improvements: M.C. Fernando Hermosillo
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
#include <deque>
#include <experimental/filesystem>
#include <opencv2/opencv.hpp>
#include <sdcv/sdcv.h>

#if defined(_WIN32) || defined(_WIN64)
#include <Windows.h>

#elif defined(_UNIX)

#endif

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
#if defined(_WIN32) || defined(_WIN64)
std::string uigetfile(wchar_t filter[] = L"All files\0*.*\0", HWND hWnd = NULL) {
	std::string filepath;

	// OPENFILENAME struct initialization
	TCHAR fbuffer[200];
	OPENFILENAME ofn;
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = hWnd;
	ofn.lpstrFile = fbuffer;
	fbuffer[0] = L'\0';
	ofn.nMaxFile = sizeof(fbuffer);
	ofn.lpstrFilter = filter;
	ofn.nFilterIndex = 1;
	ofn.lpstrFileTitle = NULL;
	ofn.nMaxFileTitle = 0;
	ofn.lpstrInitialDir = NULL;
	ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

	if (GetOpenFileName(&ofn) == TRUE) {
		int i = 0;
		while (fbuffer[i] != L'\0') {
			filepath += (char)fbuffer[i++];
		}
	}
	else {
		std::cout << "\"filediag\" function error" << std::endl;
		exit(-1);
	}

	return filepath;
}

#elif defined(_UNIX)
std::string uigetfile(const wchar_t filter) {

}

#endif

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
	//cv::String videoname = "V16-2016-09-20-1321"; int nDL = 4;
	///cv::String videoname = "V00";
	//cv::String videoname = "V01-2016-07-10"; int nDL = 4;
	//cv::String videoname = "V18-2016-10-05-1340";
	//cv::String videoname = "V06-2016-06-15-1838";	int nDL = 4;
	//cv::String videoname = "V07-2016-07-15-1840";
	//cv::String videoname = "video720";
	//cv::String videoname = "20161005_134318";
	///cv::String videoname = "V19-2016-10-05_1343-420x240-25f-19";
	//cv::String videoname = "V07-2016-07-15-1840";
	///std::string videoname = "V18-2016-10-05-1340-420x240-25fv18";

	//std::string videoname = "V25-2016-11-10-420x240-25";	int nDL = 5;
	///std::string videoname = "V4";	int nDL = 5; // 5 lines
	///std::string videoname = "V6";	int nDL = 4; // 4 lines
	///std::string videoname = "V30";	int nDL = 4; // 4 lines
	///std::string videoname = "V31";	int nDL = 4; // 4 lines
	///std::string videoname = "V32";	int nDL = 6; // 6 lines
	//std::string videoname = "V33";	int nDL = 2; // 2
	//std::string videoname = "V34";	int nDL = 3; // 3
	//std::string videoname = "V35";	int nDL = 4; // 4

	// TESTING ENTROPY
	//cv::String videoname = "V01"; int nDL = 4;
	//std::string videoname = "V02"; int nDL = 5;
	//std::string videoname = "V02-R"; int nDL = 5;
	//std::string videoname = "V5"; int nDL = 5;
	//std::string videoname = "V06"; int nDL = 5;
	//std::string videoname = "V09";
	//std::string videoname = "V12";
	//std::string videoname = "V14"; int nDL = 5;
	//std::string videoname = "V14-R"; int nDL = 5;
	//std::string videoname = "V14-2018"; int nDL = 5;
	//std::string videoname = "V15";
	//std::string videoname = "V25-2016-11-10-420x240-25";	int nDL = 5;
	//std::string videoname = "V30";
	//std::string videoname = "V31"; int nDL = 4;
	//std::string videoname = "V32";
	//std::string videoname = "V32-2"; int nDL = 6;
	//std::string videoname = "V33";
	//std::string videoname = "V34";
	//std::string videoname = "V35"; int nDL = 5;
	//std::string videoname = "V39"; int nDL = 5;
	//std::string videoname = "V39-R"; int nDL = 5;
	//std::string videoname = "V55"; int nDL = 4;
	//std::string videoname = "V74-L"; int nDL = 3;
	//std::string videoname = "V74-R"; int nDL = 3;

	/// TESTING OCCLUSION
	//cv::String videoname = "V1"; int nDL = 5;	// OK
	//cv::String videoname = "V2"; int nDL = 5;	// OK
	//cv::String videoname = "V3"; int nDL = 5;	// OK
	//cv::String videoname = "V4"; int nDL = 5;	// OK
	//cv::String videoname = "V5"; int nDL = 5;	// OK
	//cv::String videoname = "V6"; int nDL = 4;	// OK
	//cv::String videoname = "V7"; int nDL = 4;	// OK
	//cv::String videoname = "V8"; int nDL = 4;

	//cv::String videoname = "20161031_123041-25f";	// 5 lines

	/// SYSTEM INIT -----------------------------------------------------------------------------------
	std::string videofile = uigetfile(L"Video File (mp4) \0*.MP4*\0");
	auto fpart = sdcv::fs_fileparts(videofile);
	std::string videoname = std::get<1>(fpart);

	int nDL = 0;
	if (videofile.length()) {
		std::cout << "nDL: ";
		std::cin >> nDL;
	}

	// Filesystem
	if (!std::experimental::filesystem::exists(std::string("DATA/")))
		std::experimental::filesystem::create_directories(std::string("DATA/" + videoname + "/"));

	if (!std::experimental::filesystem::exists(std::string("DATA/" + videoname)))
		std::experimental::filesystem::create_directory(std::string("DATA/" + videoname));

	if (std::experimental::filesystem::exists(std::string("DATA/" + videoname + "/TEST")))
		std::experimental::filesystem::remove_all(std::string("DATA/" + videoname + "/TEST"));
	std::experimental::filesystem::create_directory(std::string("DATA/" + videoname + "/TEST"));

	cv::String projdir = "DATA/" + videoname + "/";

	// Video Camera/File
	cv::VideoCapture video(videofile);
	double fps = video.get(cv::CAP_PROP_FPS);				//
	double videoW = video.get(cv::CAP_PROP_FRAME_WIDTH);	//
	double videoH = video.get(cv::CAP_PROP_FRAME_HEIGHT);	//
	int codec = (int)video.get(cv::CAP_PROP_FOURCC);		// Get Codec Type- Int form

	/// ROI ------------------------------------------------------------------------------
	sdcv::ROI roi(videoname);
	roi.setup(videofile, nDL, 2, true);

	/// DETECTOR -------------------------------------------------------------------------
	sdcv::occlusionType occtype = sdcv::OCC_DEF_CONTOUR;			// Occlusion Handling Type
	sdcv::Detector detector(100, 100000, occtype, true, true);	// Class Constructor
	detector.setROI(roi);
	detector.setOclussion(true);								// Enable Occlusion Handling
	detector.setShadowRemoval(false);							// Enable Shadow Removal
	cv::Mat fstframe, fstmask;
	video.read(fstframe);
	detector.apply(fstframe, fstmask);

	/// CLASSIFIER -----------------------------------------------------------------------
	sdcv::Classifier classifier(4,
								std::vector<double>({ 0.12, 1.2, 100.0, 0.0 }),
								std::vector<std::string>({ "S", "M", "G" , "FP" }));

	/// TRACKER --------------------------------------------------------------------------
	sdcv::Tracker tracker(8, 12, fps, roi, &classifier);

	/// OUTPUT FILES ---------------------------------------------------------------------
	std::ofstream stafile(projdir + "statistics.csv");
	stafile << "Frame,idAlg,ID,DetectedCentroidX,DetectedCentroidY,EstimatedCentroidX,EstimatedCentroidY,Area,Width,Heigth,Velocity,RelAreaWH,RegionID,Occlusion,ObjId" << std::endl;
	std::ofstream timefile(projdir + "time.csv");
	timefile << "Total,Detection,Tracking,Drawing" << std::endl;
	timefile << 1.0 / fps << std::endl;

	cv::VideoWriter wrvideo(projdir + "video_out.mp4",
							cv::VideoWriter::fourcc('H', '2', '6', '4'),
							fps,
							cv::Size((int)videoW * 2,
							(int)videoH));
	if ( !wrvideo.isOpened() ) std::cout << "Fail to open video writer with codec: " << codec << std::endl;

	/// SYSTEM RUN  ----------------------------------------------------------------------
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
		cv::cvtColor(mask, rgbmask, cv::COLOR_GRAY2RGB);
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

			// Write tracking results in a file
			tracker.writeTracks(stafile);
		}
		// Measure time
		timefile << time_per_frame << "," << Tdetect << "," << Ttracking << "," << Tdrawing << std::endl;

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
		if (key == KEY_SPACE) cv::waitKey();
		else if (key == KEY_ESC) break;
		else if (key == KEY_0) {
			video.set(cv::CAP_PROP_POS_FRAMES, 0);
			tracker.clear();
		}
	}
	
	// Stop the system execution
	cv::waitKey();
	cv::destroyAllWindows();
	wrvideo.release();

	return 0;
}

/* ************** E N D   O F   F I L E ----------------- CINVESTAV TELECOM GDL */
