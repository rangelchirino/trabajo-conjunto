/*!
 * @name		Detector.hpp
 * @author		Fernando Hermosillo.
 * @brief		This file is part of Vehicle Detection and Classification System project.
 * @date		25/11/2016
 *
 * @version		2.0
 *	25/11/2016: Initial version.
 *  12/01/2017: Added convexity defects occlusion handling algorithm (ConDef OHA).
 *  26/10/2017:	Changed erode and dilate structuring elements shape from RECT to ELLIPSE.
 *  10/11/2017: Changed convexity defects occlusion handling algorithm number of iterations to 1.
 *  14/11/2017: Added shadow detection and removal  algorithm.
 *  15/11/2017: Added ConDefs OHA testing.
 *	27/11/2018: Redefinition of the class (ver 2.0).
 *
 * @toDo
 * Add lane width condition in ConDef OHA
 *
 */

#ifndef DETECTOR_HPP
#define DETECTOR_HPP


 /* ---------------------------*/
 /*       Library Include       */
 /* ---------------------------*/
#include <opencv2\opencv.hpp>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <numeric>
#include <sdcv\ROI.hpp>
#include <sdcv\Blob.hpp>
#include <sdcv\BlobList.hpp>
#include <sdcv\sdcv_mat.hpp>
/*!
 * \namespace	sdcv.
 * \brief		Vehicle Detection and Classification System.
 */
namespace sdcv {
	/*!
	 * \class	Detector.
	 * \brief	Used for detecting objects.
	 */
	typedef enum
	{
		OCC_NORM_AREA = 0,
		OCC_DEF_CONTOUR

	} OcclusionHandler;

	typedef enum
	{
		OUTPUT_AREA = 0x0001,		// Output Area Flag
		OUTPUT_BBOX = 0x0002,		// Output Bounding Box Flag
		OUTPUT_CENTROID = 0x0004,	// Output Centroid Flag
		HOCC = 0x0008,				// Occlusion Handling Enable Flag
		SHAREM = 0x0010,			// Shadow Removal Enable Flag
		DRAW_LABEL = 0x0020,		// Drawing Label Flag
		DRAW_BBOX = 0x0040,			// Drawing Bounding Box Flag
		DRAW_CENTROID = 0x0080,		// Drawing Centroid Flag
	} DetectionFlags;

	class Detector {
	public:
		// Constructor
		/*!
		* \name		Detector
		* \brief	Constructor
		*/
		Detector();

		/*!
		* \name		Detector
		* \brief	Constructor
		* \param
		*/
		Detector(int nmixtures = 3,
			int MinBlobArea = 100,
			int MaxBlobArea = std::numeric_limits<int>::infinity(),
			int flags = 0x000F);

		// Get methods
		/*!
		* \name
		* \brief
		* \param
		*/
		std::vector<sdcv::Blob> getBlobs(void);
		int getMinArea(void);
		int getMaxArea(void);
		sdcv::ROI getRoi(void);

		/*!
		* \name
		* \brief
		* \param
		*/
		int getOcclusions(void);

		/*!
		* \name
		* \brief
		* \param
		*/
		void getBackground(cv::OutputArray bg);

		// Set methods
		/*!
		* \name
		* \brief
		* \param
		*/
		void setMOG(cv::Ptr<cv::BackgroundSubtractorMOG2> pMOG);

		/*!
		* \name
		* \brief
		* \param
		*/
		void setErodeKernel(cv::Mat kernel);

		/*!
		* \name
		* \brief
		* \param
		*/
		void setDilateKernel(cv::Mat kernel);

		/*!
		* \name
		* \brief
		* \param
		*/
		void setMinBlobArea(int minArea);

		/*!
		* \name
		* \brief
		* \param
		*/
		void setMaxBlobArea(int maxArea);

		void setROI(sdcv::ROI roi);
		/*!
		* \name
		* \brief
		* \param
		*/
		void setFlags(int flags);

		/*!
		* \name
		* \brief
		* \param
		*/
		void setFlag(DetectionFlags Flag, bool val);

		/*!
		* \name
		* \brief
		* \param
		*/
		void setOcclusion(OcclusionHandler algflag);

		/*!
		* \name
		* \brief
		* \param
		*/
		void setOcclusionParam(std::vector<double> Params); // laneWidthThresC1, normAreaThresC1, laneWidthThresC2, overlapRatio

		// Action methods
		/*!
		* \name
		* \brief
		* \param
		*/
		bool isReady(void);

		/*!
		* \name
		* \brief
		* \param
		*/
		void apply(cv::Mat frame, cv::OutputArray mask);

		/*!
		* \name
		* \brief
		* \param
		*/
		void draw(cv::Mat frame, int drwflags = (int)DRAW_BBOX);

		/*!
		* \name		getBlobs
		* \brief
		* \param
		*/
		void BlobAnalysis(cv::InputArray mask, cv::OutputArray blobMask);

		// Destructor
		/*!
		* \name
		* \brief
		* \param
		*/
		~Detector();


	private:
		cv::Ptr<cv::BackgroundSubtractorMOG2> BGModel;	//!
		double learningRate;

		// Morphologic operations
		cv::Mat erodeKernel;
		cv::Mat dilateKernel;

		//! FLAGS
		int flags;

		// Blob Analysis
		std::vector<sdcv::Blob> blobs;
		int minBlobArea;
		int maxBlobArea;
		int roiArea;
		int NFramesSha;

		// Occlusion handling
		std::vector<double> ocParams;
		OcclusionHandler algOcclusion;

		// Region of interest
		sdcv::ROI roi;

		// Debug
		int NFrame;
		int j;

		// Vehicle Occlusion Index
		int NbOcclusions;


		//  Occlusion handling algorithms
		int normAreasOcclusion(cv::InputArray src, cv::OutputArray mask);
		int convexityDefectsOcclusion(cv::InputArray src, cv::OutputArray mask);
		// int OcclusionAnalysis(cv::InputArray src, cv::OutputArray mask);
	};
};

#endif