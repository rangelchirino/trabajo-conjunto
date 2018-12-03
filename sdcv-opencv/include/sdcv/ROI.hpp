/*!
 * @name		ROI.hpp
 * @author		Fernando Hermosillo.
 * @brief		This file is part of Vehicle Detection and Classification System project.
 * @date		25/11/2016
 *
 * @version
 *	25/11/2016: Initial version.
 *  26/11/2016:
 *  05/07/2017:	Added NbRegions and the vector of lines for the regions.
 *	23/11/2018: Added a region of interest to set lines and ROI, also display 
 *				the current line to setup
 */

#ifndef ROI_HPP
#define ROI_HPP


/* ---------------------------*/
/*       Library Include       */
/* ---------------------------*/
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>
#include <vector>
#include <opencv2\opencv.hpp>
#include <sdcv\Drawing.hpp>
#include <sdcv\Blob.hpp>




/*!
 * @namespace 	sdcv.
 * @brief 		Vehicle Detection and Classification System.
 */
namespace sdcv {
	typedef struct {
		int NbLineLanes;
		cv::Mat lineLane; 					// set of division lines for each lane
		std::vector< double > m_lineLanes; 	// slope for each lane's division line
		std::vector< double > b_lineLanes; 	// intersection for each lane's division line
		std::vector< cv::Mat > DL_mask;
	} DivLaneLine_t;

	/*!
	 * @class 	ROI.
	 * @brief	Used for setting and characterizing the region of interest (ROI).
	 */
	class ROI {
	private:
		//
		cv::String name;
		
		// Region of Interest
		std::vector<cv::Point> vertices;
		cv::Rect bbox;
		int area;
		cv::Mat mask;

		// Dividing lanes' lines
		int NbLineLanes;
		cv::Mat lineLane; 					// set of division lines for each lane
		std::vector< double > m_lineLanes; 	// slope for each lane's division line
		std::vector< double > b_lineLanes; 	// intersection for each lane's division line
		std::vector< cv::Mat > DL_mask;
		std::vector< cv::Mat > DL_green;
		
		// Line detection
		std::vector< cv::Point > lineDetection;
		cv::Point2f cLineDetection;
		cv::Mat LineArea;
		
		// End Line
		std::vector< cv::Point > EndLine;
		cv::Point2d EndLineEquation;		// x: slope, y: b
		
		// Regions
		int NbRegions;
		std::vector< std::vector<cv::Point> > Regions;
		std::vector< cv::Point2d > RegionLinesEquation;		// xi: slopei, yi: bi

		// Private methods
		void getRoiPoly( cv::OutputArray frame, std::vector<cv::Point> vertex = std::vector<cv::Point>());
		void save( void );
		bool load( void );


	public:
		// Constructor
		ROI();
		ROI(cv::String name);

		// Get methods
		cv::Rect getBbox(void);
		std::vector<cv::Point> getVertices(void);
		cv::Mat getMask(void);
		int getArea(void);

		DivLaneLine_t getLaneData( void );
		int getNumLanes( void );
		cv::Mat getDivLineLane( void );
		std::vector< double > getSlopeLane( void );
		std::vector< double > getIntersecLane( void );
		cv::Mat getDivLaneMask(int Nb);
		std::vector< cv::Mat > getDivLaneMask( void );
		std::string getName(void);

		std::vector<cv::Point> getEndLine(void);
		cv::Point2d getEndLineEq( void );

		std::vector< cv::Point > getLineDetection( void );
		cv::Point2f getCenterLineDetection( void );

		int getNbRegions(void);
		std::vector< std::vector<cv::Point> > getRegions(void);
		std::vector< cv::Point2d > getRegionLinesEquation();

		// Set methods
		void setName(cv::String name);
		void setVertices(cv::Mat frame);
		void setDetectionLine(cv::Mat frame);
		void setEndingLine(cv::Mat frame);
		void setRegions(cv::Mat frame, int NbRegions, cv::Point vertexIdx);

		// Action methods
		void setup(cv::String videonamePath, int NbDivLines = 4, int NbRegions = 0, bool bLoad = false);
		void apply(cv::InputArray frame, cv::OutputArray image);
		void draw(cv::Mat &frame, bool roi = true, bool lineDetection = true, bool endLine = false, bool bDrawRegions = false);

		// Destructor
		~ROI();
		
	};
};

#endif /* ROI_HPP */

/*! ************** End of file ----------------- CINVESTAV GDL */