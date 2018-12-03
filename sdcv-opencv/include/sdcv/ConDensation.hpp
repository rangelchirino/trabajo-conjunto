/*!
 * @name		ConDensation.hpp
 * @author		Fernando Hermosillo.
 * @brief		This file is part of Vehicle Detection and Classification System project.
 * @date		20/10/2017
 *
 * @version
 * 10/11/2017: Initial version.
 *
 */
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>


/*!
 * This class implements the Conditional Density Propagation algorithm.
 * It's based on ConDensation [1] opencv 2.4 algorithm for opencv 3.xx C++ interface compatibility.
 * Some fixes has been added.
 *
 * The problem of tracking curves in dense visual clutter is challenging. Kalman filtering is inadequate
 * because it is based on Gaussian densities which, being unimodal, cannot represent simultaneous alternative hypotheses.
 * The Condensation algorithm uses “factored sampling”, previously applied to the interpretation of static
 * images, in which the probability distribution of possible interpretations is represented by a randomly generated set.
 * Condensation uses learned dynamical models, together with visual observations, to propagate the random set
 * over time. The result is highly robust tracking of agile motion. Notwithstanding the use of stochastic methods, the
 * algorithm runs in near real-time [1].
 *
 *
 * [1] 	Isard, M.; Blake, A (August 1998). "CONDENSATION-- conditional density propagation of visual tracking". 
 * 		International Journal of Computer Vision. 29 (1): 5–28. doi:10.1023/A:1008078328650.
 */
namespace sdcv {
	namespace bayesian {
		
		class RandState             //! Rand state
		{
			public:
				RandState();
				~RandState();
				
				//! Set random state and range
				void setRange(int64 state = 0, float a = 0.0f, float b = 1.0f);
				
				//! Returns uniformly distributed floating random number from [low,high) range
				float getUniformSample();
			
			private:
				cv::RNG rng;
				float lowerBound;
				float higherBound;
		};
		
		
		class ConDensation 
		{
		public:
			//
			//! All Matrices interfaced here are expected to have dp cols, and are float.
			//
			ConDensation();
			ConDensation(int dp, int numSamples, float flocking = 0.9f);

			//! Reset. call at least once before correct()
			void initSampleSet(cv::Mat lowerBound, cv::Mat upperBound, cv::Mat dynam = cv::Mat());

			//! Update the state and return prediction.
			cv::Mat correct(cv::Mat measurement);

			//! Access single samples(read only).
			int   sampleCount(void);
			float sample(int j, int i);

		private:
			int DP;                     	//! Sample dimension
			int numSamples;             	//! Number of the Samples                 
			float flocking;             	//! flocking/congealing factor
			cv::Mat range;          //! Scaling factor for correction, the upper bound from the orig. samples
			cv::Mat dynamMatr;      //! Matrix of the linear dynamic system  
			cv::Mat samples;        //! Arr of the Sample Vectors             
			cv::Mat newSamples;     //! Temporary array of the Sample Vectors 
			cv::Mat confidence;     //! Confidence for each Sample            
			cv::Mat cumulative;     //! Cumulative confidence                 
			cv::Mat randomSample;   //! RandomVector to update sample set     
			cv::Mat state;          //! Predicted state vector
			cv::Mat mean;           //! Internal mean vector
			cv::Mat measure;        //! Cached measurement vector
			std::vector<RandState> rng;		//! One rng for each dimension.
			
			//! Performing Time Update routine for ConDensation algorithm
			void updateByTime(void);
		};
	}
}