/* ----------------------------*/
/*       Library Include       */
/* ----------------------------*/
#include "Track.hpp"

namespace sdcv {
	/*--------------------------------*/
	/*     Class reference method     */
	/*--------------------------------*/
	Track::Track()
	{

	}
	

	Track::Track(sdcv::Blob blob, int NumFrame)
	{
		// TRACKING VARIABLES
		id = 0;													// Initial track ID
		ObjectID = 0;											// New object added ID
		isDone = false;											//
		bbox.push_back(blob.getBBox());							//
		detectedCentroid.push_back(blob.getCentroid());			//
		predictedCentroid = cv::Point2f(0.0f, 0.0f);			//
		estimatedCentroid.push_back(cv::Point2f(0.0f, 0.0f));	//
		velocity = 0.0;											//
		consecutiveBackwardDir = 0;								//
		totalFrames = 1;										//
		totalVisibleFrames = 1;									//
		consInvisibleFrames = 0;								//
		lastVisibleFrame = NumFrame;							// Get current frame
		contour = blob.getContour();							// Blob Contour

		// CLASSIFICATION VARIABLES
		areas.push_back(blob.getArea());						//
		normAreas.push_back(blob.getNormArea());				//
		estimatedArea = 0.0;									//
		NbOfRegion = 0;											//
		ClassRegionVisibleCount = 0;							//
		isClassified = false;									//

		// VEHICLE OCCLUSION INDEX VARIABLES
		OcclusionRes = blob.getOccluded();						//

		// STATE ESTIMATOR INIT
		KalmanFilterInit();
	}

	// Action method
	void Track::print(void) 
	{
		std::cout << "---------------- [TRACK  INFO] ----------------"	<< std::endl;
		std::cout << "- ID                  : "		<< id << std::endl;
		std::cout << "- Bounding box        : "		<< bbox.back() << std::endl;
		std::cout << "- Detected Centroid   : "		<< detectedCentroid.back() << std::endl;
		std::cout << "- Predicted Centroid  : "		<< predictedCentroid << std::endl;
		std::cout << "- Estimated Centroid  : "		<< estimatedCentroid.back() << std::endl;
		std::cout << "- Area                : "		<< areas.back() << std::endl;
		std::cout << "- Normalized Area     : "		<< normAreas.back() << std::endl;
		std::cout << "- Estimated Area      : "		<< estimatedArea << std::endl;
		std::cout << "- Velocity            : "		<< velocity << std::endl;
		std::cout << "- Total Frames        : "		<< totalFrames << std::endl;
		std::cout << "- Last Visble Frame   : "		<< lastVisibleFrame << std::endl;
		std::cout << "- Visible Frames      : "		<< totalVisibleFrames << std::endl;
		std::cout << "- Invisible Frames    : "		<< consInvisibleFrames << std::endl;
		std::cout << "- Backward Direction  : "		<< consecutiveBackwardDir << std::endl;
		std::cout << "- Number of Region	: "		<< NbOfRegion << std::endl;
		std::cout << "- Class frame visible : "		<< ClassRegionVisibleCount << std::endl;
		std::cout << "-----------------------------------------------" << std::endl << std::endl;
	}

	void Track::print( std::ofstream &file  ) 
	{
		file << "----------------------- [Track Info] -----------------------:"	<< std::endl;
		file << "- Bounding box: "			<< cv::format(bbox, cv::Formatter::FMT_DEFAULT) << std::endl;
		file << "- Detected Centroids: \n"	<< cv::format(detectedCentroid, cv::Formatter::FMT_DEFAULT) << std::endl;
		file << "- Predicted Centroids: "	<< predictedCentroid << std::endl;
		file << "- Estimated Centroids: \n"	<< cv::format(estimatedCentroid, cv::Formatter::FMT_DEFAULT) << std::endl;
		file << "- Kalman Filter: \n"		<< std::endl;
		file << "-- X_hat:\n"				<< AKF.statePost << std::endl;
		file << "-- P:\n"					<< AKF.errorCovPost << std::endl;
		file << "-- K:\n"					<< AKF.gain << std::endl;
		file << "-- R:\n"					<< AKF.measurementNoiseCov << std::endl;
		file << "-- Q:\n"					<< AKF.processNoiseCov << std::endl;
		file << "- Areas: "					<< cv::format(areas, cv::Formatter::FMT_DEFAULT) << std::endl;
		file << "- Normalized Area: "		<< cv::format(normAreas, cv::Formatter::FMT_DEFAULT) << std::endl;
		file << "- Estimated Area"			<< estimatedArea << std::endl;
		file << "- Velocity: "				<< velocity << std::endl;
		file << "- Total Frames: "			<< totalFrames << std::endl;
		file << "- Visible Frames: "		<< totalVisibleFrames << std::endl;
		file << "- Invisible Frames: "		<< consInvisibleFrames << std::endl;
		file << "-------------------------------------------------------------" << std::endl << std::endl;
	}
	



	std::ostream& operator<<(std::ostream &os, sdcv::Track const &track)
	{
		return os << track.id << "," << track.detectedCentroid.back().x << "," << track.detectedCentroid.back().y << "," << track.estimatedCentroid.back().x << "," << track.estimatedCentroid.back().y << "," << track.areas.back() << "," << track.bbox.back().width << "," << track.bbox.back().height << "," << track.velocity << "," << (double)(track.bbox.back().width / (double)track.bbox.back().height) << "," << track.NbOfRegion << "," << track.OcclusionRes << "," << track.ObjectID;
	}


	void Track::KalmanFilterInit(void)
	{
		float dt = 1; // 1/fps
		AKF.init(4, 2, 0);

		AKF.transitionMatrix = (cv::Mat_<float>(4, 4) << 1.0f,dt,0.0f,0.0f,   0.0f,1.0f,0.0f,0.0f,   0.0f,0.0f,1.0f,dt,   0.0f,0.0f,0.0f,1.0f);		// [X	Vx	Y	Vy]
		AKF.measurementMatrix = (cv::Mat_<float>(2, 4) << 1.0f,0.0f,0.0f,0.0f,   0.0f,0.0f,1.0f,0.0f);												// [X	Vx	Y	Vy]

		cv::setIdentity(AKF.processNoiseCov, cv::Scalar::all(0.9f));		// Q
		cv::setIdentity(AKF.measurementNoiseCov, cv::Scalar::all(0.3f));	// R
		cv::setIdentity(AKF.errorCovPost, cv::Scalar::all(1.0f));			// P(k)+ (Post)

		AKF.statePost.at<float>(0) = detectedCentroid.back().x;				// Post
		AKF.statePost.at<float>(1) = 0.0f;									// Post
		AKF.statePost.at<float>(2) = detectedCentroid.back().y;				// Post
		AKF.statePost.at<float>(3) = 0.0f;									// Post
	}

	
	Track::~Track() 
	{

	}
}

/* ************** E N D   O F   F I L E ----------------- CINVESTAV GDL */