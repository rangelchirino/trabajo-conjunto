/*!
 * @name		Classifier.hpp
 * @author		Fernando Hermosillo.
 * @brief		This file is part of Vehicle Detection and Classification System project.
 * @date		07/12/2016
 *
 * @version
 *	07/12/2016: Initial version.
 *
 */

#ifndef CLASSIFIER_HPP
#define CLASSIFIER_HPP

/* ---------------------------*/
/*       Library Include       */
/* ---------------------------*/
#include <opencv2\opencv.hpp>
#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <sdcv\Blob.hpp>
#include <sdcv\ROI.hpp>


/*!
 * @namespace	sdcv.
 * @brief		Vehicle Detection and Classification System.
 */
namespace sdcv {
	
	/*!
	 * @class	Classifier
	 * @brief	This class has the purpose of managing each blob to track
	*/
	class Classifier {
	public:
		Classifier();
		Classifier(int NbClasses, std::vector<double> thresholdValue, std::vector<std::string> labels);

		void setNumClasses(int NbClasses);
		void setLabels(int idx, std::string name);
		void setClassThresVal(int idx, double ThresVal);

		int getNumClasses( void );
		std::string getLabels(int idx);
		int getCount(int idx);
		
		int update(int idx);
		int update(double value);
		
	private:
		std::vector<std::string> labels;
		std::vector<int> count;
		std::vector<double> thresholdValue;
		int NbOfClasses;

		//void save( void );
		//void load( void );
	};
}

#endif /* CLASSIFIER_HPP */

/*! ************** End of file ----------------- CINVESTAV GDL */