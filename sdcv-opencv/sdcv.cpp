//////////////////////////////////////////////
//     _____   _____     _____  __      __	//
//    / ____| |  __ \   / ____| \ \    / /	//
//   | (___   | |  | | | |       \ \  / /	//
//    \___ \  | |  | | | |        \ \/ /	//
//    ____) | | |__| | | |____     \  /		//
//   |_____/  |_____/   \_____|     \/		//
//////////////////////////////////////////////

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
#include <sdcv/sdcv.hpp>

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


/**********************************************************/
/*            G L O B A L    V A R I A B L E S            */
/**********************************************************/

/*********************************************************/
/*                M A I N   P R O G R A M                */
/*********************************************************/
int main(int argc, const char** argv)
{
	int key = 0, NbFrame = 2, idTrack = -1;

	/// SYSTEM INIT -----------------------------------------------------------------------------------
	std::string videofile = sdcv::FileDialog::getOpenFilename((wchar_t*)L"Video File (mp4) \0*.MP4*\0");
	
	auto fpart = sdcv::fs_fileparts(videofile);
	std::string videoname = std::get<1>(fpart);

	int nDL = 0;
	if (videofile.length()) {
		std::cout << "nDL: ";
		std::cin >> nDL;

		std::cout << "ID to track: ";
		std::cin >> idTrack;
	}

	// Filesystem
	std::string projdir = "DATA/" + videoname + "/";
	if (!std::experimental::filesystem::exists(std::string("DATA")))
		std::experimental::filesystem::create_directories(std::string("DATA"));

	if (!std::experimental::filesystem::exists(projdir))
		std::experimental::filesystem::create_directory(projdir);
		
	// Video Camera/File
	cv::VideoCapture video(videofile);
	double fps = video.get(cv::CAP_PROP_FPS);				//
	double width = video.get(cv::CAP_PROP_FRAME_WIDTH);		//
	double height = video.get(cv::CAP_PROP_FRAME_HEIGHT);	//
	int codec = (int)video.get(cv::CAP_PROP_FOURCC);		// Get Codec Type- Int form



	/// ROI ------------------------------------------------------------------------------
	sdcv::ROI roi(videoname);
	roi.setup(videofile, nDL, 2, true);



	/// DETECTOR -------------------------------------------------------------------------
	sdcv::OcclusionHandler occhandler = sdcv::OCC_NORM_AREA;		// Occlusion Handling Type
	int nmixtures = 3;
	sdcv::Detector detector(nmixtures, 100, (int)(width*height));	// Class Constructor
	detector.setROI(roi);
	detector.setOcclusion(occhandler);								// Set occlusion handler and enable HOCC
	detector.setFlag(sdcv::HOCC, false);							// Disable HOCC

	cv::Mat fstframe, fstmask;
	video.read(fstframe);
	detector.apply(fstframe, fstmask);



	/// CLASSIFIER -----------------------------------------------------------------------
	sdcv::Classifier classifier(4,
								std::vector<double>({ 0.12, 1.2, 100.0, 0.0 }),
								std::vector<std::string>({ "S", "M", "G" , "FP" }));



	/// TRACKER --------------------------------------------------------------------------
	sdcv::Tracker tracker(8, 12, fps, roi, &classifier);
	if (idTrack > 0) {
		if (std::experimental::filesystem::exists(projdir + "Tensor"))
			std::experimental::filesystem::remove_all(projdir + "Tensor");
		std::experimental::filesystem::create_directory(projdir + "Tensor");
		//tracker.setConvex(true);
		tracker.setID(idTrack);
	}



	/// OUTPUT FILES ---------------------------------------------------------------------
	std::ofstream stafile(projdir + "statistics.csv");
	stafile << "Frame,idAlg,ID,DetectedCentroidX,DetectedCentroidY,EstimatedCentroidX,EstimatedCentroidY,Area,Width,Heigth,Velocity,RelAreaWH,RegionID,Occlusion,ObjId" << std::endl;
	std::ofstream timefile(projdir + "time.csv");
	timefile << "Total,Detection,Tracking,Drawing" << std::endl;
	timefile << 1.0 / fps << std::endl;

	//  Processed Video
	cv::VideoWriter wrvideo(projdir + "video_out.mp4",
							cv::VideoWriter::fourcc('H', '2', '6', '4'),
							fps,
							cv::Size((int)width * 2,
							(int)height));
	if ( !wrvideo.isOpened() ) std::cout << "Fail to open video writer with codec: " << codec << std::endl;

	// Foreground Video
	cv::VideoWriter wrfgvideo(projdir + "fg_mask.mp4",
		cv::VideoWriter::fourcc('H', '2', '6', '4'),
		fps,
		cv::Size((int)width,
		(int)height));
	if (!wrfgvideo.isOpened()) std::cout << "Fail to open video writer with codec: " << codec << std::endl;



	/// SYSTEM RUN  ----------------------------------------------------------------------
	cv::destroyAllWindows();
	cv::namedWindow("Track",		cv::WINDOW_KEEPRATIO);
	cv::namedWindow("Mask",			cv::WINDOW_KEEPRATIO);
	cv::namedWindow("Foreground",	cv::WINDOW_KEEPRATIO);
	double t, detectorAvg = 0.0, trackerAvg = 0.0, drawAvg = 0.0, elapsedTime = 0.0;
	while ( true ) {
		cv::Mat frame, Roi, mask, frameTracking, frameForeground;
		double Tdetect = 0.0, Ttracking = 0.0, Tdrawing = 0.0;

		if (!video.read(frame)) break;
		frame.copyTo(frameTracking);
		

		t = (double)cv::getTickCount();
		detector.apply(frame, mask);
		t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
		detectorAvg += t;
		Tdetect = t;
		frameTracking.copyTo(frameForeground, mask);

		
		double time_per_frame = t;

		cv::Mat rgbmask;
		// Debug
		cv::cvtColor(mask, rgbmask, cv::COLOR_GRAY2RGB);
		if (NbFrame > 0) {
			t = (double)cv::getTickCount();
			tracker.update(detector.getBlobs());
			t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
			trackerAvg += t;
			time_per_frame += t;
			Ttracking = t;
			
			tracker.tensorTrack(frame, projdir);

			t = (double)cv::getTickCount();
			tracker.draw(frameTracking);
			tracker.draw(rgbmask, false);
			t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
			drawAvg += t;
			Tdrawing = t;

			time_per_frame += t;

			// Write tracking results in a file
			tracker.write(stafile);
		}
		elapsedTime += time_per_frame;

		// Measure time
		timefile << time_per_frame << "," << Tdetect << "," << Ttracking << "," << Tdrawing << std::endl;
		if (1.0 / (double)time_per_frame > fps)
			std::cout << "Real-Time Issue: " << time_per_frame << " > " << 1.0 / fps << std::endl;
		//std::cout << time_per_frame << " s" << std::endl;

		//cv::imshow("DETECTIONS", detections);
		cv::imshow("Track", frameTracking);
		cv::imshow("Mask", rgbmask);
		cv::imshow("Foreground", frameForeground);
		NbFrame++;

		cv::Mat outFrame;
		cv::hconcat(frameTracking, rgbmask, outFrame);
		wrvideo.write(outFrame);
		wrfgvideo.write(frameForeground);
		//cv::imwrite("DATA/RAW/" + std::to_string(NbFrame) + ".bmp", MaskRGB);

		key = cv::waitKey(30);
		if (key == KEY_SPACE) cv::waitKey();
		else if (key == KEY_ESC) break;
		else if (key == KEY_0) {
			video.set(cv::CAP_PROP_POS_FRAMES, 0);
			tracker.clear();
		}
	}
	
	// Show Timing Statistics
	std::cout << "Real Time: " << (NbFrame-1) /(double)fps << " s" << std::endl;
	std::cout << "Elapsed Time: " << elapsedTime << " s" << std::endl;

	// Stop the system execution
	cv::waitKey();
	cv::destroyAllWindows();
	wrvideo.release();
	wrfgvideo.release();

	return 0;
}

/* ************** E N D   O F   F I L E ----------------- CINVESTAV TELECOM GDL */
