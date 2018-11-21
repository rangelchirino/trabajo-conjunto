/*!
 * @name		Classifier.cpp
 * @author		Fernando Hermosillo.
 * @brief		This file is part of Vehicle Detection and Classification System project.
 * @date		07/12/2016
 *
 * @version
 *	07/12/2016: Initial version.
 *
 */
 
#include "Classifier.hpp"

namespace sdcv {
	/**/
	Classifier::Classifier() {}
	
	Classifier::Classifier(int NbClasses, std::vector<double> thresholdValue, std::vector<std::string> labels) {
		CV_Assert(NbClasses == thresholdValue.size() && NbClasses == labels.size());
		
		this->NbOfClasses = NbClasses;
		this->thresholdValue = thresholdValue;
		this->count = std::vector<int>(NbClasses, 0);
		this->labels = labels;
	}

	void Classifier::setNumClasses(int NumClasses) { 
		this->NbOfClasses = NumClasses;
		labels.clear();
		count.clear();
	}
	
	void Classifier::setLabels(int idx, std::string className) {
		if( idx < count.size() ) this->labels.at(idx) = className;
	}
	
	void Classifier::setClassThresVal(int idx, double ThresVal) {
		if( idx < count.size() ) this->thresholdValue.at(idx) = ThresVal;
	}
	
	int Classifier::getNumClasses( void ) { return NbOfClasses; }
	
	std::string Classifier::getLabels(int idx) {
		if( idx < count.size() ) return this->labels.at(idx);
		else return "";
	}
	
	int Classifier::getCount(int idx) {
		if( idx < count.size() ) return this->count.at(idx);
	    return -1;
	}

	int Classifier::update(int idx) {
		if( idx < count.size() ) this->count.at(idx)++;

		return idx;
	}
	
	int Classifier::update(double value) {
		int Class = 0;

		std::vector<int>::iterator cntIt = count.begin();

		for(std::vector<double>::iterator it = thresholdValue.begin(); it != thresholdValue.end() - 1; ++it, ++cntIt) {
			if (value < *it) {
				(*cntIt)++;
				return Class;
			}
			Class++;
		}

		return Class;
	}

	void save( void ) {
		
	}
	
	void load( void ) {
		
	}
}