/**
* \file \sdcv_mat.hpp
* \author \Fernando Hermosillo
* \date \12/07/2017 09:00
*
* \brief This file define the SVM Classifier class and is part of
*		 Vehicle Detection and Classification System project
*
* \version Revision: 1.3
*/
 
#ifndef SDCV_MAT_HPP
#define SDCV_MAT_HPP
	
	/*********************************************************/
	/*                    I N C L U D E S                    */
	/*********************************************************/
	#include <iostream>
	#include <string>
	#include <fstream>
	#include <iterator>
	#include <functional>
	#include <algorithm>
	#include <vector>
	#include <opencv2\opencv.hpp>
	#include <sdcv\Track.hpp>

	/*********************************************************/
	/*                     D E F I N E S                     */
	/*********************************************************/
	
	/*********************************************************/
	/*                      M A C R O S                      */
	/*********************************************************/

	/*********************************************************/
	/* 			U S I N G   N A M E S P A C E S 			 */
	/*********************************************************/
	
	/*********************************************************/
	/*                       E N U M S                       */
	/*********************************************************/
	
	/*********************************************************/
	/*                     S T R U C T S                     */
	/*********************************************************/

	/*********************************************************/
	/*          P R O T O T Y P E   F U N C T I O N          */
	/*********************************************************/
	
	/*********************************************************/
	/*                       C L A S S                       */
	/*********************************************************/
	namespace sdcv {

		typedef enum {
			TOP_DOWN = 0,
			BOTTOM_UP,
			LEFT_2_RIGHT,
			RIGHT_2_LEFT

		} eOrientationLine;


		/*!
		 * @name	cv_remove_if
		 * @brief	Remove columns/rows if lamda is equals to parameter polarity.
		 */
		void cv_remove_if(const cv::Mat &src, cv::Mat &dst, std::function<bool(cv::Mat src)> lamda, bool row_column = true, bool polarity = true);
		
		/*!
		 * @name	cv_remove
		 * @brief	Remove columns/rows given a vector of indexes from the src array
		 */
		void cv_remove(cv::InputArray src, cv::OutputArray dst, std::vector<int> ridx, bool row_column = true);
		
		/*!
		 * @name	cv_copy
		 * @brief	Copy columns/rows given a vector of indexes from the src array
		 */
		void cv_copy(cv::InputArray src, cv::OutputArray dst, std::vector<int> idx, bool row_column = true);

		/*!
		* @name	norm
		* @brief	Normalize data for svm classification
		*/
		void norm(sdcv::Track track, cv::OutputArray sample, int NbSamples, std::vector<float> params);

		/*!
		* @name	norm
		* @brief	Normalize data for svm classification
		*/
		bool find(std::vector<cv::Point> v, int value);


		/*!
		* @name	norm
		* @brief	Normalize data for svm classification
		*/
		double pointLineTest(cv::Point begin, cv::Point end, cv::Point2d pt, bool retpoint = false);


		/*******************************************************************************************************
		 * V 1.2
		 *******************************************************************************************************/
		/*!
		* @name		distanceToLine
		* @brief	Compute the distance between a point and a line with slope 'slope' and intersection 'b'.
		*			Additional can specify the direction, that is top-down and bottom-up.
		*/
		template<class _Tp>
		double distanceToLine(cv::Point_<_Tp> pt, double m, double b, sdcv::eOrientationLine orientation) {
			double scale = 1.0;

			if (orientation == BOTTOM_UP || orientation == RIGHT_2_LEFT)	scale = -1.0;

			return ((double)((double)pt.x*m + b - (double)pt.y))*scale;
		}

		/*******************************************************************************************************
		* V 1.3
		*******************************************************************************************************/
		/*!
		* @name		euclidean
		* @brief	Computes euclidean distance between two points
		*/
		template<class _Tp>
		double euclidean(cv::Point_<_Tp> p1, cv::Point_<_Tp> p2) {
			cv::Point_<_Tp> pdiff = p2 - p1;

			return std::sqrt((double)(pdiff.x*pdiff.x) + (double)(pdiff.y*pdiff.y));
		}
	}
	
#endif /* SDCV_MAT_HPP */

/*! ************** End of file ----------------- CINVESTAV GDL */