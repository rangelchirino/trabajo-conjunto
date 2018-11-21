/*!
 * @name		Tracker.hpp
 * @author		Fernando Hermosillo.
 * @brief		This file is part of Vehicle Detection and Classification System project.
 * @date		13/12/2016
 *
 * @version
 *	07/12/2016: Initial version.
 *  13/12/2016: Adding some variables.
 *  xx/05/2017: Tracking erease fixed
 *  07/06/2017: Velocity between lastVisibleFrame and current frame
 *  13/06/2017: assignmentProblemSolver updated, now it allows us to set the cost of non-assigments
 *  14/11/2017: 
 *				- Added Write tracking results to a file / stdout
 *				- Update some methods input parameters
 */

#ifndef TRACKER_HPP
#define TRACKER_HPP

/* ---------------------------*/
/*       Library Include       */
/* ---------------------------*/
#include <opencv2\opencv.hpp>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <numeric>

#include <sdcv\BlobList.h>
#include <sdcv\ROI.h>
#include <sdcv\Track.hpp>
#include <sdcv\Classifier.hpp>
#include <sdcv\munkres.h>
#include <sdcv\Drawing.h>
#include <sdcv\sdcv_files.hpp>
#include <sdcv\sdcv_mat.hpp>

/*!
 * @namespace	sdcv.
 * @brief		Vehicle Detection and Classification System.
 */
namespace sdcv {
	
	/*!
	 * @class	Tracker
	 * @brief	This class has the purpose of managing each blob to track
	*/
	class Tracker {
	public:
		/*!
		 * @name	Track
		 * @brief	Track class constructor
		 * @param minVisibleCount:		M
		 * @param minInvisibleCount:	
		 */
		Tracker(int minVisibleCount, int minInvisibleCount, double fps, sdcv::ROI roi, sdcv::Classifier *classifier);
			
			
		/*!
		 * @name	Track
	 	 * @brief	Track class constructor
		 * @param newBlob:	sdcv::Blob	new blob to track
		 */
		sdcv::Track getTrack(int id);
		int getMinVisibleCount(void);
		int getMinInvisibleCount(void);
		int getNbOfTracks(void);
		int getNbOfOcclusion(int offset);
		
		/*!
		 * @name	setMinVisibleCount
		 * @brief	
		 * @param minVisibleCount:	
		 */
		void setMinVisibleCount(int minVisibleCount);
			
		/*!
		 * @name	setMinInvisibleCount
		 * @brief	
		 * @param minInvisibleCount:	
		 */
		void setMinInvisibleCount(int minInvisibleCount);
		
		/*!
		* @name		setOcclusionTime
		* @brief	This method sets the occlusion time for computing Vehicle Occlusion Index value
		*
		* @param	sec: Interval time in seconds
		*/
		void setOcclusionTime(int sec);

		/*!
		 * @name	run
		 * @brief	
		 * @param detectedBlobs:	std::vector<sdcv::Blob>
		 */
		void track(std::vector<sdcv::Blob> detectedBlobs);
			
		/*!
		 * @name	run
		 * @brief	
		 * @param detectedBlobs:	std::vector<sdcv::Blob>
		 */
		void draw(cv::Mat frame, bool drawRoi = true);
			
		/*!
		 * @name	run
		 * @brief
		 * @param detectedBlobs:	std::vector<sdcv::Blob>
		 */
		void clear(void);

		/*!
		* @name	run
		* @brief
		* @param detectedBlobs:	std::vector<sdcv::Blob>
		*/
		void computeVOI(void);
		
		/*!
		* @name		writeTracks
		* @brief	This function writes tracks in a file
		*
		* @param file	File to write
		*/
		void writeTracks(std::ofstream &file);

		/*!
		* @overload	writeTracks
		* @brief	This function writes tracks in stdout
		*/
		void writeTracks(void);

		/*!
		 * @name	~Track.
		 * @brief	Track class destructor.
		 */
		~Tracker();
			
	private:
		//!< Minimum visible frame count that a blob has to have to add as a track it.
		int minVisibleCount;
		
		//!< Minimum invisible frame count that a blob has to have to delete it.
		int minInvisibleCount;

		//!< Current assigned ID.
		int currentID;
		
		//!< Vector of assigned tracks
		//std::vector<sdcv::Track> assignedTracks;	// Optional
		std::vector<cv::Vec3i> assignedTracks;

		//!< Vector of unassigned tracks
		//std::vector<sdcv::Track> unassignedTracks;	// Optional

		//!< Vector of tracks
		std::vector<sdcv::Track> tracks;			// This or the two above
		
		//!< List of blobs
		sdcv::BlobList blobLst;

		//!< Initial state
		bool init;

		//! < Frames per second
		double fps;

		//! < False positive ID
		std::vector<int> FP_ID;
		std::vector<std::tuple<int,int>> FP_ID_;	// {ID, last}

		//!< ID
		int ID;

		//< Region of interest object
		sdcv::ROI roi;

		//< Classifier object
		sdcv::Classifier *classifier;

		//!< Verbose this class
		bool verbose;

		int FrameCountDebug;

		//!< Store the slope and intersection of a line equation [Region, EndLine, DetectionLine]
		/*double mRegion;
		double bRegion;
		double mEnd;
		double bEnd;
		double mDetection;
		double bDetection;*/

		const cv::Scalar colors[8] = {CV_RGB(255,0,0), CV_RGB(0,255,0), CV_RGB(0,0,255),CV_RGB(255,255,0),CV_RGB(255,0,255),CV_RGB(0,255,255), CV_RGB(125,0,125), CV_RGB(0,125,125) };

		//
		// Vehicle Occlusion Index
		int OcclusionTime;
		int Frame0;		//
		int ID_k_m_1;	//	ID(k-1)
		std::vector<cv::Point3d> VOI;		//

		// 
		int AssignedIdCounter;
		/*!
		* @name	getAssigmentCostMatrix
		* @brief	This method compute the cost matrix for each detected centroid and each tracked centroid.
		* @param none.
		*/
		cv::Mat getAssignmentCostMatrix(void);

		/*!
		* @name	assignamentProblemSolver
		* @brief	This method solves the assignament problem.
		* @param CostMatrix:			Input,
		* @param assignments:			Output,
		* @param unassignedTracks:		Output,
		* @param unassignedDetections:	Output,
		*/
		void assignmentProblemSolver(cv::Mat CostMatrix, std::vector<cv::Point> &assignments, std::vector<int> &unassignedTracks, std::vector<int> &unassignedDetections, int costOfNonAssigment);

		/*!
		* @name	distance
		* @brief	This method computes a distance between the location of a detected object and a predicted location.
		*			It takes into account the covariance of the predicted state and the process noise.
		* @param AKF:			Input, Kalman filter object to get the distance
		* @param centroids:	Input, List of detected centroids.
		* @param CostMatrix:	Output, The cost matrix.
		*/
		void distance(cv::KalmanFilter AKF, cv::Mat centroids, cv::OutputArray CostMatrix);

		/*!
		* @name	add
		* @brief	Track class constructor
		* @param newBlob:	sdcv::Blob	new blob to track
		*/
		void add(std::vector<sdcv::Blob> detectedBlobs, std::vector<int> unassignedDetections);
			
			
		/*!
		* @name	update.
		* @brief	Update tracks.
		* @param assignedBlobs:	std::vector<sdcv::Blob>	a vector of assigned blobs.
		*/
		void update(std::vector<sdcv::Blob> detectedBlobs, std::vector<cv::Point> assignments, std::vector<int> unassignedTracks);
		
		
		/*!
		 * @name	erease.
		 * @brief	Deletes tracks and increase the vehicle count according to its class.
		 * @param classifier:	sdcv::Classifier 
		 */
		void erease(void);
		
		
	};
}

#endif