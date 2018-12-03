/**
* \file \svmlassifier.hpp
* \author \Fernando Hermosillo
* \date \12/07/2017 09:00
*
* \brief This file define the SVM Classifier class and is part of
*		 Vehicle Detection and Classification System project
*
* \version Revision: 0.1
*/
 
#ifndef SVM_CLASSIFIER_HPP
#define SVM_CLASSIFIER_HPP
	
	/*********************************************************/
	/*                    I N C L U D E S                    */
	/*********************************************************/
	#include <sdcv/sdcv_mat.hpp>
	#include <sdcv/sdcv_files.hpp>
	#include <opencv2/opencv.hpp>
	#include <sdcv\ROI.hpp>
	
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
		/*!
		* \class 	Classifier.
		* \brief	Used for classifying  vehicles with the SVM algorithm
		*/
		class SVMClassifier {
			private:
				float NbSamples;							//!< @param	NbSamples: Number of samples
				int NbClasses;								//!< @param	NbClasses: Number of Classes
				int NbFeatures;								//!< @param	NbFeatures: Number of features
				int Dimension; 								//!< @param	Dimension: Data dimensionality
				std::vector<cv::Ptr<cv::ml::SVM>> svm;		//!< @param	svm:: for more than one class
				std::string ModelName;
				cv::Mat counter;
				cv::ml::SVM::Types type;					//!< @param	type: SVM algorithm type
				cv::ml::SVM::KernelTypes kernel;			//!< @param	kernel: SVM kernel type
				float Dx;									//!< @param	Dx: 
				float Dy;									//!< @param	Dy: 
				float W1;									//!< @param	W1: 
				float W2;									//!< @param	W2: 
				float RoidLa;								//!< @param	RoidLa: 
				std::vector<float> NormParam;
				std::vector<float> RoiParams;
				
			public:
				/*!
				 * @name	SVMClassifier
				 * @brief	SVMClassifier class constructor
				 */
				SVMClassifier(cv::ml::SVM::Types type, cv::ml::SVM::KernelTypes kernel, sdcv::ROI roi, int NbClasses = -1, int NbFeatures = -1);
				
				/*!
				 * @name	train
				 * @brief	Training step, compute the SVM models
				 * \param	trainingData: Vector of input training data
				 */
				void train(std::string filename, std::vector< std::vector<int> > trainingData, int NbSamples);
				
				/*!
				 * @name	getDimension
				 * @brief	Get the input data dimensionality
				 */
				int getDimension(void);
				
				/*!
				 * @name	getNumFeatures
				 * @brief	Get the number of features
				 */
				int getNumberFeatures(void);
				
				/*!
				 * @name	getNumFeatures
				 * @brief	Get the number of features
				 */
				int getNumberClasses(void);
				std::vector<float> getNormParam( void );
				std::vector<float> getRoiParams(void);

				void setNormParam(std::vector<float> NormParam);

				/*!
				 * @name	classify
				 * @brief	Given a vector of sampled features called X, 
				 *			this function classifies it and return the class 
				 *			at this sample belongs.
				 */
				int classify(cv::Mat sample);
		};
	}
	
#endif	/* SVM_CLASSIFIER_HPP */

/*! ************** End of file ----------------- CINVESTAV GDL */