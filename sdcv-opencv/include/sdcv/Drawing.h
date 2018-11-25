/*!
 * @name		Drawing.hpp
 * @author		Fernando Hermosillo.
 * @brief		This file is part of Vehicle Detection and Classification System project.
 * @date		10/12/2016
 *
 * @version
 * @history
 * 26/11/2016: Initial version.
 * 10/12/2016: Added ItoS function.
 * 05/07/2017: Added imRegion function and the function imLine has been change its struct (drawing the current line to be gotten).
 * 23/11/2018: Added a region of interest to set lines and polygons
 */

#ifndef DRAWING_HPP
#define DRAWING_HPP


/* ---------------------------*/
/*       Library Include       */
/* ---------------------------*/
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <opencv2\opencv.hpp>


/*!
 * @namespace 	sdcv.
 * @brief 		Vehicle Detection and Classification System.
 */
namespace sdcv {
	const int VK_BACK	= 8;
	const int VK_ENTER	= 13;
	const int VK_ESC	= 27;
	const int VK_SPACE	= 32;

	typedef enum {
		E_SHAPE_RECT,
		E_SHAPE_CIRCLE
	}eAnnotationShape;
	/*!
	 * @name	imLine.
	 * @version	1.0
	 * @brief	This function returns a set of points for a line that you will set in an image.
	 * @param	frame (Input) - Frame where line is gonna be gotten.
	 * @param	WindowName (Input) - A string that contains the name of the window in case of line is gonna be dram
	 * @param	Cinit (Input) - Initial Conditions
	 * @param	roi (Input) - A region of interest where a line can be choosen
	 *
	 * @return 	A set of points
	 */
	std::vector<cv::Point> imLine(cv::OutputArray frame, cv::String WindowName = "IMLINE", cv::Point Cinit = cv::Point(-1, -1), cv::Rect roi = cv::Rect());

	/*!
	* @name	imPoint.
	* @version	1.0
	* @brief	This function returns a set of points for a line that you will set in an image.
	* @param	frame (Input) - Frame where line is gonna be gotten.
	* @param	WindowName (Input) - A string that contains the name of the window in case of line is gonna be dram
	*
	* @return 	A point
	*/
	cv::Point imPoint(cv::InputArray frame, cv::String WindowName = "IMPOINT");
	
	/*!
	* @name		imRegions.
	* @version	1.3
	* @brief	This function returns a set of points for a line that you will set 
	*			in an image.
	* @param	frame (Input) - Frame where line is gonna be taken
	* @param	wname (Input) - A string containing the name of the window
	* @param	NbRegions (Input) - Number of regions
	* @param	roi (Input) - A region of interest where the regions can be choosen
	*
	* @return 	A set of points
	*/
	std::vector< std::vector<cv::Point> > imRegions(cv::OutputArray frame, 
													cv::String wname = "RegionWindow", 
													int NbRegions = 2, 
													cv::Rect roi = cv::Rect());

	/*!
	 * @name	vertices2polygon.
	 * @version	1.0.
	 * @brief	This function converts a set of points into a polygon.
	 * @param	vertices: As Input is a set of points.
	 * @param	frameSize: As Input is the frame's size to be generated.
	 * @param	polygon: As Output is the frame object to be generated.
	 *
	 * @return None.
	 */
	void vertices2polygon(std::vector<cv::Point> vertices, cv::Size frameSize, cv::OutputArray polygon);


	/*!
	 * @name	ItoS.
	 * @brief	This function converts an integer number to string.
	 * @param	Number: Integer number.
	 * @return std::string.
	 */
	std::string ItoS(int Number);

	/*!
	* @name		euclideanContour
	* @brief	
	* @param	
	* @param	
	* @return	std::string.
	*/
	std::vector<cv::Point> euclideanContour(std::vector<cv::Point> contour, cv::Point point);

	/*!
	* @name		drawPoints.
	* @brief	
	* @param	
	* @param	
	* @param	
	* @return	void
	*/
	void drawPoints(cv::Mat &img, std::vector<cv::Point> vector, cv::Scalar color);
	void drawPoints(cv::Mat &img, std::vector<cv::Point2f> vector, cv::Scalar color);

	/*!
	* @name		drawPoints.
	* @brief	
	* @param	
	* @return	void
	*/
	void plot2d(std::string name, cv::Size Axes, std::vector<int> xData, std::vector<int> yData);

	/*!
	* @name		insertObjectAnnotation.
	* @brief	
	*
	* @param	dst			source/destination image
	* @param	position	Position in which the annotations is being setted
	* @param	label		A string that is being 
	* @param	shape		Annotation shape {rectangular, circular}
	* @param	color		Annotation color
	*
	*/
	void insertObjectAnnotation(cv::InputOutputArray I, cv::Rect position, std::string label, eAnnotationShape shape = E_SHAPE_RECT, cv::Scalar color = CV_RGB(255,255,0));

	/*!
	 * @name	roipoly
	 * @brief	This function creates a polygon ROI (Region Of Interest) in an image.
	 *
	 * @param frame	Source image
	 * @param poly	A std vector of points that stores the polygon vertices.
	 * @param mask	An U8 BW image that store the polygon ROI mask
	*/
	void roipoly(cv::InputArray frame, std::vector<cv::Point> &poly, cv::OutputArray mask);
	void roipoly(cv::InputArray frame, std::vector<cv::Point> &poly);
	void roipoly(cv::InputArray frame, cv::OutputArray mask);
};

#endif