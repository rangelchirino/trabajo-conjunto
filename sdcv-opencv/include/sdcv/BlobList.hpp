/*!
 * @name		BlobList.hpp
 * @author		Fernando Hermosillo.
 * @brief		This file is part of Vehicle Detection and Classification System project.
 * @date		25/11/2016
 *
 * @version
 *	25/11/2016: Initial version.
 *  26/11/2016:
 */

#ifndef BLOB_LIST_HPP
#define BLOB_LIST_HPP


/* ---------------------------*/
/*       Library Include       */
/* ---------------------------*/
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>

#include <opencv2\opencv.hpp>
#include <sdcv\Blob.hpp>


/*!
 * @namespace	sdcv.
 * @brief		Vehicle Detection and Classification System.
 */
namespace sdcv {
	/*! 
	 * @class	BlobList.
	 * @brief	Used for creating a blob list.
	 */
	class BlobList {
	public:
		// Constructor
		BlobList(void);
		BlobList(std::vector<sdcv::Blob> blobs);
		
		// Get methods
		std::vector<int> getAreas( void );
		std::vector<cv::Point2f> getCentroids( void );
		std::vector<cv::Rect > getBBoxes( void );
		int getLen( void );

		static std::vector<int>	getAreas(std::vector<sdcv::Blob> blobs);
		static std::vector<cv::Point2f> getCentroids(std::vector<sdcv::Blob> blobs);
		static std::vector<cv::Rect > getBBoxes(std::vector<sdcv::Blob> blobs);

		// Set methods
		
		
		// Action methods
		void add(sdcv::Blob blob);
		void clear( void );
		
		// Destructor
		~BlobList();


	private:
		std::vector<int>		 areas;
		std::vector<cv::Point2f> centroids;
		std::vector<cv::Rect >	 bboxes;
		int len;
	};

};

#endif