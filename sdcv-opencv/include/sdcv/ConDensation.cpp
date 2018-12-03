#include "ConDensation.hpp"

namespace sdcv {
	namespace bayesian {
		//! RandState class
		RandState::RandState() 
		{
		}
		
		RandState::~RandState()
		{
		}
		
		//! Set random state and range
		void RandState::setRange(int64 state, float a, float b) {
			rng.state = state;
			lowerBound = a; 
			higherBound = b;
		}
		
		//! Returns uniformly distributed floating random number from [low,high) range
		float RandState::getUniformSample() { 
			return rng.uniform(lowerBound, higherBound);
		}
		
		
		//! ConDensation Class
		ConDensation::ConDensation(void)
			: DP(0)
			, numSamples(0)
			, flocking(0.0F)
		{
		}

		ConDensation::ConDensation(int dp, int numSamples, float flocking)
			: DP(dp)
			, numSamples(numSamples)
			, flocking(flocking)
			, rng(DP)
		{
		}

		//! Access single samples(read only).
		int   ConDensation::sampleCount()       { return samples.rows; }
		float ConDensation::sample(int j,int i) { return samples.at<float>(j,i); }

		//!
		void ConDensation::initSampleSet(cv::Mat lower, cv::Mat upper, cv::Mat dynam)
		{
			CV_Assert( (lower.type() == CV_32FC1) && (upper.type() == CV_32FC1) );
			CV_Assert( (lower.cols == DP) && (upper.cols == DP) );
			CV_Assert( (lower.rows == 1)  && (upper.rows == 1) );
			cv::Mat lowerBound = lower;
			cv::Mat upperBound = upper;
			upper.copyTo(range); // cache for reuse in correct()
			/*std::cout << "RANGE.INIT: " << range << std::endl;*/

			// dynamics might be empty (then we'll use an identity matrix), or a DP x DP x float transformation mat
			CV_Assert( dynam.empty() || ((dynam.rows == DP)  && (dynam.type() == CV_32FC1)) );
			dynamMatr    = dynam.empty() ? cv::Mat_<float>::eye(DP, DP) : dynam;
			/*std::cout << "Dynamics.INIT: " << std::endl << dynamMatr << std::endl;*/

			cumulative   = cv::Mat::zeros(numSamples, 1, CV_32FC1);
			samples      = cv::Mat::zeros(numSamples, DP, CV_32FC1);
			newSamples   = cv::Mat::zeros(numSamples, DP, CV_32FC1);
			randomSample = cv::Mat::zeros(1, DP, CV_32FC1);
			state        = cv::Mat::zeros(1, DP, CV_32FC1);
			mean         = cv::Mat::zeros(1, DP, CV_32FC1);
			confidence	 = cv::Mat(numSamples, 1, CV_32FC1);
			confidence.setTo(cv::Scalar::all((float)(1.0f / numSamples)));
			/*std::cout << "CONFIDENCE.INIT: " << confidence.t() << std::endl << std::endl;
			std::cout << "RING.SIZE: " << rng.size() << std::endl;*/

			for(int d = 0; d < DP; d++)
				rng[d].setRange(cv::getTickCount(), lowerBound.at<float>(d), upperBound.at<float>(d));
			
			// Generating the samples 
			for(int s = 0; s < numSamples; s++)
				for(int d = 0; d < DP; d++)
					samples.at<float>(s, d) = rng[d].getUniformSample();

			/*std::cout << "SAMPLES.INIT: " << std::endl << samples.t() << std::endl << std::endl;*/
		}


		//!
		void ConDensation::updateByTime()
		{
			// Calculating the Mean 
			mean.setTo(0);
			float sum = 0.0f;
			for(int s = 0; s < numSamples; s++)
			{
				state = samples.row(s) * confidence.at<float>(s);
				mean += state;
				sum  += confidence.at<float>(s);
				cumulative.at<float>(s) = sum;
			}

			// Taking the new state vector from transformation of mean by dynamics matrix 
			mean /= sum;
			state = mean * dynamMatr;
			//sum  /= numSamples;
			
			// Initialize the random number generator.
			cv::RNG rngUp(cv::getTickCount());

			// We want a record of the span of the particle distribution. 
			// The resampled distribution is dependent on this quantity.
			std::vector<float> sampleMax(DP,FLT_MIN), sampleMin(DP,FLT_MAX);
			// Updating the set of random samples 
			// The algorithm of the original code always picked the last
			// sample, so was not really a weighted random re-sample.  It
			// wasn't really random, either, due to careless seeding of the
			// random number generation.

			// This version resamples according to the weights calculated by
			// the calling program and tries to be more consistent about
			// seeding the random number generator more carefully.
			for(int s = 0; s < numSamples; s++)
			{
				// Choose a random number between 0 and the sum of the particles' weights.
				float randNumber = rngUp.uniform(0.0f, sum);

				// Use that random number to choose one of the particles.
				int j = 0;
				while( (cumulative.at<float>(j) <= randNumber) && (j < numSamples-1)) j++;
				//while( (cumulative(j) <= (float) s * sum) && (j<numSamples-1)) j++;
					

				// Keep track of the max and min of the sample particles.
				// We'll use that to calculate the size of the distribution.
				for(int d = 0; d < DP; d++) 
				{
					newSamples.at<float>(s,d) = samples.at<float>(j,d);
					sampleMax[d] = cv::max(sampleMax[d], newSamples.at<float>(s,d));
					sampleMin[d] = cv::min(sampleMin[d], newSamples.at<float>(s,d));
				}
			}

			// Reinitializes the structures to update samples randomly 
			for(int d = 0; d < DP; d++)
			{
				float diff = flocking * (sampleMax[d] - sampleMin[d]);

				if ( 0 )
				{
					// This line may not be strictly necessary, but it prevents
					// the particles from congealing into a single particle in the
					// event of a poor choice of fitness (weighting) function.
					diff = cv::max(diff, 0.02f * newSamples.at<float>(0,d));
				} else {
					// Rule 1 : reaching the target is the goal here, right ? 
					// * if we lost it         : swarm out  
					// * if target was reached : hog it .
					diff = cv::min(diff, flocking * (measure.at<float>(d) - newSamples.at<float>(0,d)));
				}

				// Re-seed and set the limits to the geometric extent of the distribution.
				rng[d].setRange(cv::getTickCount()+d,-diff,diff);
				
				// extra spin on the electronic roulette.(sic)
				rng[d].getUniformSample();
			}
			
			// Adding the random-generated vector to every projected vector in sample set
			for( int s=0; s<numSamples; s++ )
			{
				cv::Mat r = newSamples.row(s) * dynamMatr;
				for( int d=0; d<DP; d++ )
				{
					samples.at<float>(s,d) = r.at<float>(d)+ rng[d].getUniformSample();
				}
			}
		}

		//
		//! Adjust confidence based on euclidean distance and return predicted state
		//
		cv::Mat ConDensation::correct(cv::Mat measurement) 
		{
			/*std::cout << "Dynamics.CORRECT(K-1): " << std::endl << dynamMatr << std::endl;
			std::cout << "CONFIDENCE.CORRECT(K-1): " << confidence.t() << std::endl;
			std::cout << "RANGE.CORRECT(K-1): " << range << std::endl << std::endl;
			std::cout << "SAMPLES.CORRECT(K-1): " << std::endl << samples.t() << std::endl << std::endl;*/

			measure = measurement;
			for (int s = 0; s < numSamples; s++) 
			{
				double dist = 0;
				for(int d=0; d < DP; d++)
				{
					float diff = (measure.at<float>(d) - samples.at<float>(s,d))/range.at<float>(d);
					dist += diff*diff;
					/*std::cout << "Diff(s,d): (" << measure.at<float>(d) << " - " << samples.at<float>(s, d) << ")/" << range.at<float>(d) << " = " << diff << std::endl;*/
				}
				/*std::cout << "dist(s) = " << dist << std::endl;*/
				confidence.at<float>(s) = float(cv::exp(-100.0f * cv::sqrt((double)dist/(DP*DP))));
			}

			/*std::cout << std::endl << std::endl << "CONFIDENCE.CORRECT(K): " << confidence.t() << std::endl;
			std::cout << "RANGE.CORRECT(K): " << confidence.t() << std::endl << std::endl;*/

			updateByTime();

			return state;
		}
	}
}