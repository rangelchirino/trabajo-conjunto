/*!
 * @name		Blob.hpp
 * @author		Fernando Hermosillo.
 * @brief		This file is part of Vehicle Detection and Classification System project.
 * @date		25/11/2016
 *
 * @version
 *	25/11/2016: Initial version.
 *  26/11/2016:
 */

#ifndef BLOB_HPP
#define BLOB_HPP

/* ---------------------------*/
/*       Library Include       */
/* ---------------------------*/
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <opencv2\opencv.hpp>

/*!
 * @namespace	sdcv.
 * @brief		Vehicle Detection and Classification System.
 */
namespace sdcv {
	/*! 
	 * @class	Blob.
	 * @brief	Used for getting blob features.
	 */
	class Blob 
	{
	public:
		// Constructor
		Blob(std::vector< cv::Point > contour, int id=0, int area = -1);

		// Get methods
		cv::Point2f getCentroid( void );
		cv::Rect getBBox( void );
		int	getArea( void );
		double getNormArea(void);
		std::vector<cv::Point> getContour(void);
		bool getOccluded(void);
		int getParentId(void);
		double getEccentricity(void);

		// Set methods
		void setBlob(std::vector< cv::Point > contour);
		void setNormArea(double normArea);
		void setLanePosition(int position);
		void setOcclusion(bool value);
		void setParentId(int id);

		// Action methods
		void print();
		void print( std::ofstream &file );

		/*
		 * @name	match
		 * @brief	This function matches blobs by computing the euclidean distance between theirs centroids.
		 *
		 * @param	src1	Previous blobs' centroids in matricial form.
		 * @param	src2	Current blobs' centroids in matricial form.
		 * @param	dst		Stores a nx2 matrix with the Previous blob index as the first column and 
		 *					Current blob index as the last column.
		 */
		static void match(std::vector<cv::Point2f> src1, std::vector<cv::Point2f> src2, std::vector<cv::Point> &dst, double maxDistance);

		/*
		 * @overloaded
		 */
		cv::Point match(std::vector<cv::Point2f> src, double maxDistance);

		// Destructor
		~Blob(void);

	private:
		//!< @param	 id:	Stores the relative blob id, may change each frame
		int id;
		
		//!< @param	 id:	Stores the blob id that created this blob
		int parentId;

		//!< @param	 centroid:	a 2D Point that store the blob's centroid (X, Y)
		cv::Point2f centroid;
		
		//!< @param	 bbox:	Store the blob's bounding box 
		cv::Rect bbox;
		
		//!< @param	 contour:	Store the blob's contour
		std::vector<cv::Point> contour;
		
		//!< @param	 area:	Store the blob's area
		int area;
		
		//!< @param	 normArea:	Store the blob's normalized area with the lane width
		double normArea;

		//!< @param	 lanePosition: Number of divition line in which the object is in
		int lanePosition;

		//!< @param	 occluded:	Tells wheather this blob was occluded
		bool occluded;

		//!< @param	 eccentricity:	Blob eccentricity
		double eccentricity;
	};
}

#endif