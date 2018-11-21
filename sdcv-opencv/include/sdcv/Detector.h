/*!
 * @name		Detector.hpp
 * @author		Fernando Hermosillo.
 * @brief		This file is part of Vehicle Detection and Classification System project.
 * @date		25/11/2016
 *
 * @version
 *	25/11/2016: Initial version.
 *  12/01/2017: Added convexity defects occlusion handling algorithm (ConDef OHA).
 *  26/10/2017:	Changed erode and dilate structuring elements shape from RECT to ELLIPSE.
 *  10/11/2017: Changed convexity defects occlusion handling algorithm number of iterations to 1.
 *  14/11/2017: Added shadow detection and removal  algorithm.
 *  15/11/2017: Added ConDefs OHA testing.
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
#include <sdcv\ROI.h>
#include <sdcv\Blob.h>
#include <sdcv\BlobList.h>
#include <sdcv\sdcv_mat.hpp>
/*!
 * \namespace	sdcv.
 * \brief		Vehicle Detection and Classification System.
 */
namespace sdcv {
	const int MinBlobArea = 100;
	/*!
	 * \class	Detector.
	 * \brief	Used for detecting objects.
	 */
	typedef enum {OCC_NORM_AREA = 1, OCC_DEF_CONTOUR} occlusionType;
	class Detector {
	public:
		// Constructor
		/*!
		* \name		Detector
		* \brief	
		* \param	
		*/
		Detector();

		/*!
		* \name
		* \brief
		* \param
		*/
		Detector(int MinimumBlobArea, int MaximumBlobArea, occlusionType occType = OCC_NORM_AREA, bool bDrawID = false, bool bDrawBBox = false);

		// Get methods
		/*!
		* \name
		* \brief
		* \param
		*/
		std::vector<sdcv::Blob> getBlobs( void );

		int getMinArea(void);
		int getMaxArea(void);
		sdcv::ROI getRoi( void );

		/*!
		* \name
		* \brief
		* \param
		*/
		double getAvg(void);

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
		//inline void setROI(cv::InputArray RoiMask);

		/*!
		* \name
		* \brief
		* \param
		*/
		void setMinimumBlobArea(int MinArea);

		/*!
		* \name
		* \brief
		* \param
		*/
		void setMaximumBlobArea(int MaxArea);

		/*!
		* \name
		* \brief
		* \param
		*/
		void setDrawBBox(bool flag);

		/*!
		* \name
		* \brief
		* \param
		*/
		void setDrawLabel(bool flag);

		/*!
		* \name
		* \brief
		* \param
		*/
		void setAreaOutput(bool flag);

		/*!
		* \name
		* \brief
		* \param
		*/
		void setCentroidOutput(bool flag);

		/*!
		* \name
		* \brief
		* \param
		*/
		void setROI(sdcv::ROI roi);

		/*!
		* \name
		* \brief
		* \param
		*/
		void setOclussion(bool flag);

		/*!
		* \name
		* \brief
		* \param
		*/
		void setOclussionParam(std::vector<double> Params); // laneWidthThresC1, normAreaThresC1, laneWidthThresC2, overlapRatio

		// Action methods
		/*!
		* \name
		* \brief
		* \param
		*/
		bool isReady( void );

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
		void draw( cv::Mat frame );

		/*!
		* \name		getBlobs
		* \brief	
		* \param	
		*/
		void BlobAnalysis(cv::InputArray mask, cv::OutputArray blobMask);

		/*!
		* \name		setShadowRemoval
		* \brief	This function enable/disable shadow detecting and removal algorithm
		*
		* \param flag	Shadow detecting and removal algorithm is enable/disable
		*/
		void setShadowRemoval(bool flag);

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
		bool OutputBBox;
		bool OutputCentroid;
		bool OutputArea;
		bool OcclusionEn;
		bool ShadowEn;
		bool bLabel;
		bool bBbox;
		bool bCentroid;
		bool isFalseLargeBlob;

		// Blob Analysis
		std::vector<sdcv::Blob> blobs;
		int MinimumBlobArea;
		int MaximumBlobArea;
		int RoiArea;
		int NbFrameShadowRemoval;

		// Occlusion handling
		std::vector<double> oclussionParams;
		occlusionType occType;

		// Region of interest
		sdcv::ROI roi;
		
		// Debug
		int NbFrame;
		int j;
		double avg;

		// Vehicle Occlusion Index
		int NbOfOcclusions;


		//  Occlusion handling algorithms
		int normAreasOcclusion(cv::InputArray src, cv::OutputArray mask);
		int convexityDefectsOcclusion(cv::InputArray src, cv::OutputArray mask);
		// int OcclusionAnalysis(cv::InputArray src, cv::OutputArray mask);
	};
};

#endif