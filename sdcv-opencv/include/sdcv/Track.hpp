/*!
 * @name		Track.hpp
 * @author		Fernando Hermosillo.
 * @brief		This file is part of Vehicle Detection and Classification System project.
 * @date		07/12/2016
 *
 * @version
 *	07/12/2016: Initial version.
 *
 */

#ifndef TRACK_HPP
#define TRACK_HPP

/* ---------------------------*/
/*       Library Include       */
/* ---------------------------*/
#include <opencv2\opencv.hpp>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <fstream>
#include <sdcv\Blob.h>
#include <sdcv\ROI.h>
#include <sdcv\Classifier.hpp>


/*!
 * @namespace	sdcv.
 * @brief		Vehicle Detection and Classification System.
 */
namespace sdcv {
	
	/*!
	 * @class	Track
	 * @brief	This class saves each parameter that describes a tracked vehicle.
	*/
	class Track {
	public:
		Track();

		/*!
		 * @name	Track
		 * @brief	Track class constructor
		 *
		 * @param newBlob:	sdcv::Blob	new blob to track
		 */
		Track(sdcv::Blob newBlob);
		
		/*!
		* @name	Track
		* @brief	Track class constructor
		*
		* @param newBlob:	sdcv::Blob	new blob to track
		*/
		Track(sdcv::Blob newBlob, int NbFrame);

		// Action method
		void print(void);						// std::cout
		void print( std::ofstream &file );		// file

		friend std::ostream& operator<<(std::ostream &os, sdcv::Track const &m);
		
		/*!
		 * @name	~Track
		 * @brief	Track class destructor
		 */
		~Track();



		/*!
		 * PUBLIC ATRIBUTES
		 */
		//!< @param	id: integer ID of track
		int id;
		
		//!< @param	bbox: cv::Rect current bounding box of object; used for display
		std::vector<cv::Rect> bbox;
		
		//!< @param	detectedCentroid: cv::Point2f currente detected centroid
		std::vector<cv::Point2f> detectedCentroid;
		
		//!< @param	predictedCentroid: cv::Point2f predicted centroid
		cv::Point2f predictedCentroid;

		//!< @param	estimatedCentroid: cv::Point2f corrected centroid using AKF
		std::vector<cv::Point2f> estimatedCentroid;
		
		//!< @param	AKF: cv::KalmanFilter adaptive Kalman filter object
		cv::KalmanFilter AKF;
		
		//!< @param	area: areas array
		std::vector<double> areas;
		
		//!< @param	normArea: array of normalized areas
		std::vector<double> normAreas;
		
		//!< @param	estimatedArea: estimated area
		double estimatedArea;
		
		//!< @param	velocity: estimated velocity in pixels/s
		double velocity;
		
		//!< @param	consecutiveBackwardDir: ?
		int consecutiveBackwardDir;
		
		//!< @param	totalFrames: number of frames since the track was first detected
		int totalFrames;
		
		//!< @param	totalVisibleFrames: total number of frames in which the track was detected (visible)
		int totalVisibleFrames;
		
		//!< @param	consInvisibleFrames: number of consecutive frames for which the track was not detected (invisible).
		int consInvisibleFrames;

		//!< @param	 lastVisibleFrame: Number of frame where the object was found
		int lastVisibleFrame;

		//!< @param	 NbOfRegion: In which region is the object?
		int NbOfRegion;

		//!< @param	 ClassRegionVisibleCount: How many times this object has been found in the third region
		int ClassRegionVisibleCount;

		//!< @param	 OcclusionRes: This vehicle is a result of an occlusion
		bool OcclusionRes;

		//!< @param	isClassified: The object has already been classified
		bool isClassified;

		int ObjectID;

	private:
		void KalmanFilterInit(void);
	
	};
}

#endif

/* ************** E N D   O F   F I L E ----------------- CINVESTAV GDL */