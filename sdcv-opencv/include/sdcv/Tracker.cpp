
/*!
 * @name		Tracker.cpp
 * @author		Fernando Hermosillo.
 * @brief		This file is part of Vehicle Detection and Classification System project.
 * @date		13/12/2016
 *
 * @version
 *	07/12/2016: Initial version.
 *  13/12/2016: Fixed std::
 *
 * For SVM: V = (area, width, h/w)
 *
 *
 *
 */

/* ---------------------------*/
/*       Library Include       */
/* ---------------------------*/
#include "Tracker.hpp"
#include <math.h>
#include <fstream>


//std::ofstream outfile("tracking.txt");
//#define out std::cout

// Traffic control flow direction selection
static sdcv::eOrientationLine trafficOrientation = sdcv::eOrientationLine::TOP_DOWN;


// For debug
int detected_id = 14;


namespace sdcv {
	/*-------------------------------------*/
	/*		Private Prototype Function		*/
	/*-------------------------------------*/

	/*--------------------------------*/
	/*     Class reference method     */
	/*--------------------------------*/
	// Constructor ------------------------------------------------------------------------------------
	Tracker::Tracker(int minVisibleCount, int minInvisibleCount, double fps, sdcv::ROI roi, sdcv::Classifier *classifier) {
		this->minVisibleCount = minVisibleCount;
		this->minInvisibleCount = minInvisibleCount;
		this->fps = fps;
		this->roi = roi;
		ID = 0;
		FP_ID.push_back(0);
		init = false;
		this->verbose = verbose;

		FrameCountDebug = 0;
		Frame0 = -1;
		ID_k_m_1 = 0;
		OcclusionTime = 10;

		AssignedIdCounter = 0;

		this->classifier = classifier;


		// Contour Extraction
		convexcount = 1;
		convexflag = false;
		convexID = -1;
	}
			
	// Get methods ------------------------------------------------------------------------------------
	sdcv::Track Tracker::getTrack(int id) {
		if( id < (int)tracks.size() ) return tracks.at(id);
		else return sdcv::Track();
	}
	int Tracker::getMinVisibleCount(void) { return minVisibleCount; }
	int Tracker::getMinInvisibleCount(void) { return minInvisibleCount; }
	int Tracker::getNbOfTracks(void) { return this->ID; }
	
	int Tracker::getNbOfOcclusion(int offset) {
		int NbOfTracks = 0;
		for (auto track : assignedTracks) {
			if (track[0] > offset && track[1]) {
				NbOfTracks++;
			}
		}

		return NbOfTracks;
	}





	void Tracker::setID(int id)
	{
		convexID = id;
	}

	void Tracker::setConvex(bool cvxen)
	{
		convexflag = cvxen;
	}
	
	void Tracker::tensorTrack(cv::Mat frame, std::string homedir) {
		if (convexID > 0) {
			for (auto Tk : tracks)
			{
				if (Tk.id == convexID)
				{
					// Get Mask
					std::vector<std::vector<cv::Point>> contours;
					cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
					if (convexflag)
					{
						// Compute Convex Hull
						std::vector<cv::Point> hull;
						cv::convexHull(Tk.contour, hull);
						contours = { hull };
					}
					else
					{
						contours = { Tk.contour };
					}

					// Draw Contour
					cv::drawContours(mask, contours, 0, CV_RGB(255, 255, 255), cv::FILLED);

					// RGB
					cv::Mat rgbmask;
					frame.copyTo(rgbmask, mask);

					cv::imwrite(homedir + "Tensor/T" + std::to_string(convexID) + "-" + std::to_string(convexcount) + ".png", rgbmask);
					convexcount++;
					break;
				}
			}
		}
	}






	// Set methods ------------------------------------------------------------------------------------
	void Tracker::setMinVisibleCount(int minVisibleCount)
	{
		this->minVisibleCount = minVisibleCount;
	}
	
	void Tracker::setMinInvisibleCount(int minInvisibleCount) 
	{
		this->minInvisibleCount = minInvisibleCount;
	}

	void Tracker::setOcclusionTime(int sec)
	{
		OcclusionTime = sec;
	}
	
	// Action methods ---------------------------------------------------------------------------------
	void Tracker::clear(void) {
		tracks.clear();
		assignedTracks.clear();
		currentID = 0;
		ID = 0;
		FP_ID.clear();
		FP_ID.push_back(0);
		init = false;

		FrameCountDebug = 0;
		Frame0 = -1;
		ID_k_m_1 = 0;
		OcclusionTime = 10;
	}

	void Tracker::computeVOI(void) {
		if (Frame0 == -1)
		{
			Frame0 = FrameCountDebug;
		}
		else if (FrameCountDebug - Frame0 == int(fps*this->OcclusionTime))
		{
			int Vni = ID - ID_k_m_1;
			int Vno = getNbOfOcclusion(ID_k_m_1);
			VOI.push_back(cv::Point3d((double)Vni, (double)Vno, (double)(Vno / (double)Vni)));

			Frame0 = -1;
			ID_k_m_1 = ID;
		}
	}


	void Tracker::write(std::ofstream &file)
	{
		for (auto track : tracks)
			file << FrameCountDebug << "," << 0 << "," << track << std::endl;
	}

	void Tracker::write(void)
	{
		for (auto track : tracks)
			std::cout << FrameCountDebug << "," << 0 << "," << track << std::endl;
	}


	//#define SDCV_TRACKING_DEBUG
	void Tracker::update(std::vector<sdcv::Blob> detections) {
		FrameCountDebug++;

		blobLst = sdcv::BlobList(detections);
		
		int NbDetections = (int)detections.size();
		int NbTracks = (int)tracks.size();
		
		if( NbTracks == 0 ) init = false;	// No tracks
		if( NbDetections == 0 ) return;		// Nothing to track
		
		if( init == false ) {
			for(auto detection : detections) {
				tracks.push_back(sdcv::Track(detection, FrameCountDebug));

				// Check for Regions and Visible blob classify region
				std::vector< cv::Point2d > regions = roi.getRegionLinesEquation();
				/// CAMBIO DE FLUJO VEHICULAR -----------------------------------------------------------------------------
				int regionIndex = (trafficOrientation == sdcv::eOrientationLine::TOP_DOWN ? 0 : 2);
				/// CAMBIO DE FLUJO VEHICULAR -----------------------------------------------------------------------------
				for (auto iterator = regions.begin()+1; iterator != regions.end(); ++iterator) {
					if (sdcv::distanceToLine(detection.getCentroid(), iterator->x, iterator->y, trafficOrientation) < 0.0)
						tracks.back().NbOfRegion = regionIndex;
					/// CAMBIO DE FLUJO VEHICULAR -----------------------------------------------------------------------------
					if(trafficOrientation == sdcv::eOrientationLine::TOP_DOWN) regionIndex++;	// TOP_DOWN
					else regionIndex--;															// BOTTOM_UP
					/// CAMBIO DE FLUJO VEHICULAR -----------------------------------------------------------------------------
					
				}
				// CAREFULL WITH THIS CONDITION
				if (tracks.back().NbOfRegion == 0) tracks.back().ClassRegionVisibleCount++;
			}

			init = true;
		}
		else {
			cv::Mat CostMatrix = getAssignmentCostMatrix();

			/* Solve the assignment problem --------------------------------------------------------*/
			std::vector<cv::Point> assignments;
			std::vector<int> unassignedTracks, unassignedDetections;
			assignmentProblemSolver(CostMatrix, assignments, unassignedTracks, unassignedDetections, 10);

			/* Update tracks -----------------------------------------------------------------------*/
			updateTracks(detections, assignments, unassignedTracks);

			/* Delete tracks -----------------------------------------------------------------------*/
			remove();

			/* Create new tracks -------------------------------------------------------------------*/
			add(detections, unassignedDetections);

			/* Clear blobs list before exiting */
			blobLst.clear();
		}

		/* Save data into file */
		//if (verbose) {
		//	for (auto it = tracks.begin(); it != tracks.end(); it++) {
		//		//it->print();
		//		// Write this flag #####################################################
		//		/// matchedTrack->isFP
		//		// #####################################################################
		//		file << FrameCountDebug << "," << 0 << "," << *it << std::endl;
		//	}
		//}

		// VEHICLE OCCLUSION INDEX ----------------------------------------------------------------------------------------
		this->computeVOI();
	}
	
	void Tracker::draw(cv::Mat frame, bool drawRoi) {
		
		// Draw ROI
		if (drawRoi) {
			cv::line(frame, roi.getVertices().at(0), roi.getVertices().at(1), CV_RGB(0, 255, 0), 4);
			cv::line(frame, roi.getVertices().at(1), roi.getVertices().at(2), CV_RGB(0, 255, 0), 4);
			cv::line(frame, roi.getVertices().at(2), roi.getVertices().at(3), CV_RGB(0, 255, 0), 4);
			cv::line(frame, roi.getVertices().at(3), roi.getVertices().at(0), CV_RGB(0, 255, 0), 4);
			
			// Draw Detection Line
			cv::line(frame, roi.getLineDetection().front(), roi.getLineDetection().back(), CV_RGB(0, 0, 255), 2);

			// Draw End Line
			//cv::line(frame, roi.getEndLine().front(), roi.getEndLine().back(), CV_RGB(255, 255, 0), 2);

			// Draw Regions
			auto regions = roi.getRegions();
			int idx = 0;
			for (auto region : regions) {
				if(idx++)
					cv::line(frame, region.front(), region.back(), CV_RGB(255, 0, 255), 2);
			}

			// Draw vehicle count
			cv::Mat rec2 = frame(cv::Rect(0, 0, frame.cols, 15));
			cv::Mat colorClass(rec2.size(), rec2.type(), CV_RGB(0,0,0));
			cv::addWeighted(colorClass, 0.5, rec2, 0.5, 0.0, rec2);
			for (int i = 0; i < classifier->getNumClasses() - 1; i++)
				cv::putText(frame, classifier->getLabels(i) + ": " + std::to_string(classifier->getCount(i)), cv::Point(100 * i, 10), cv::FONT_ITALIC, 0.4, CV_RGB(255,255,255));
		}
		
		// Display reliable tracks that have been visible for more than a minimum number of frames
		for(std::vector<sdcv::Track>::iterator it = tracks.begin(); it != tracks.end(); ++it) {
			if( it->id != 0 && it->totalVisibleFrames > this->minVisibleCount) {
					cv::Rect bbox = it->bbox.back() & cv::Rect(0, 0, frame.cols, frame.rows);

					cv::Scalar color = (!it->OcclusionRes ? CV_RGB(255,255,0) : CV_RGB(0,255,255));

					cv::rectangle(frame, bbox, color, 1);
					cv::rectangle(frame, cv::Rect(bbox.x, bbox.y-20, bbox.width, 20) & cv::Rect(0, 0, frame.cols, frame.rows), color, 1);

					cv::Mat rec = frame(cv::Rect(bbox.x, bbox.y-20, bbox.width, 20) & cv::Rect(0, 0, frame.cols, frame.rows));
					cv::Mat colorArray(rec.size(), rec.type(), color);
					cv::addWeighted(colorArray, 0.7, rec, 1.0, 0.0, rec);

					
					//cv::putText(frame, ItoS(it->id), cv::Point(bbox.x, bbox.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,0));
					cv::putText(frame, ItoS(it->id) + "," + ItoS((int)it->isClassified), cv::Point(bbox.x, bbox.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
					//cv::putText(frame, ItoS(it->id) + "," + ItoS((int)it->NbOfRegion), cv::Point(bbox.x, bbox.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
			}
		}
	}

	// Private methods --------------------------------------------------------------------------------
	void Tracker::add(std::vector<sdcv::Blob> detectedBlobs, std::vector<int> unassignedDetections) {
		
		#ifdef SDCV_TRACKING_DEBUG
		out << "[Add New Tracks Begin] -------------------" << std::endl;
		out << "Number of new tracks: " << unassignedDetections.size() << std::endl << std::endl;
		#endif
		

		// Add new tracks
		for(std::vector<int>::iterator it = unassignedDetections.begin(); it != unassignedDetections.end(); ++it) {
			///tracks.push_back( sdcv::Track(detectedBlobs.at(*it)) );
			// Add this flag #######################################################
			/// matchedTrack->isFP = 0;
			// #####################################################################
			tracks.push_back( sdcv::Track(detectedBlobs.at(*it), FrameCountDebug) );
			// Check for Regions and Visible blob classify region
			std::vector< cv::Point2d > regions = roi.getRegionLinesEquation();
			
			/// CAMBIO DE FLUJO VEHICULAR -----------------------------------------------------------------------------
			int regionIndex = (trafficOrientation == sdcv::eOrientationLine::TOP_DOWN ? 0 : 2);
			/// CAMBIO DE FLUJO VEHICULAR -----------------------------------------------------------------------------
			for (auto iterator = regions.begin()+1; iterator != regions.end(); ++iterator) {
				if (sdcv::distanceToLine(detectedBlobs.at(*it).getCentroid(), iterator->x, iterator->y, trafficOrientation) < 0)
					tracks.back().NbOfRegion = regionIndex;

				/// CAMBIO DE FLUJO VEHICULAR -----------------------------------------------------------------------------
				if(trafficOrientation == sdcv::eOrientationLine::TOP_DOWN) regionIndex++;	// TOP_DOWN
				else regionIndex--;															// BOTTOM_U
				/// CAMBIO DE FLUJO VEHICULAR -----------------------------------------------------------------------------
			}
			/// CAREFULL WITH THIS !
			if (tracks.back().NbOfRegion == 0) tracks.back().ClassRegionVisibleCount++;
			
			
			#ifdef SDCV_TRACKING_DEBUG
			out << "Unassigned detection [" << *it << "] has been added to track!" << std::endl;
			detectedBlobs.at(*it).print();
			out << std::endl;
			#endif
		}

		#ifdef SDCV_TRACKING_DEBUG
		out << "[Add New Tracks End] -------------------" << std::endl;
		#endif
	}


	void Tracker::updateTracks(std::vector<sdcv::Blob> detectedBlobs, std::vector<cv::Point> assignments, std::vector<int> unassignedTracks) {
		int trackIdx, detectionIdx; // Idexes for matched track and detection.
		double Th_OR = 0.3;
		bool test;

		#ifdef SDCV_TRACKING_DEBUG
		out << "[Update Tracks Begin] --------------------" << std::endl;
		out << "Number of assigments detected: " << assignments.size() << std::endl;
		out << "Number of assigments not detected: " << unassignedTracks.size() << std::endl << std::endl;
		#endif

		///std::cout << "Number of assigments detected: " << assignments.size() << std::endl;
		///std::cout << "Number of assigments not detected: " << unassignedTracks.size() << std::endl << std::endl;
		double m = (roi.getLineDetection().at(1).y - roi.getLineDetection().at(0).y) / (roi.getLineDetection().at(1).x - roi.getLineDetection().at(0).x);
		double b = roi.getLineDetection().at(1).y - m*roi.getLineDetection().at(1).x;
		std::vector< cv::Point2d > regions = roi.getRegionLinesEquation();

		///std::cout << "[Update Assigned Tracks] -----------------------------------" << std::endl;
		/* Update assigned tracks -----------------------------------------------------------------------*/
		for(auto it = assignments.begin(); it != assignments.end(); ++it) {
			double occlusionRate = 0.001;

			// Set the current blob and track that were matched.
			trackIdx	 = it->x;
			detectionIdx = it->y;
			auto matchedBlob = std::next(detectedBlobs.begin(), detectionIdx);
			auto matchedTrack = std::next(tracks.begin(), trackIdx);
			
			// FOR RGB EXTRACTION ---------------------------------------************************************************************
			matchedTrack->contour = matchedBlob->getContour();
			// FOR RGB EXTRACTION ---------------------------------------************************************************************

			/// CHECK VEHICLE REGION FOR SVM CLASSIFICATION
			/// CAMBIO DE FLUJO VEHICULAR -----------------------------------------------------------------------------
			int regionIndex = (trafficOrientation == sdcv::eOrientationLine::TOP_DOWN ? 0 : 2);
			/// CAMBIO DE FLUJO VEHICULAR -----------------------------------------------------------------------------
			for (auto iterator = regions.begin(); iterator != regions.end(); ++iterator) {
				if (sdcv::distanceToLine(matchedBlob->getCentroid(), iterator->x, iterator->y, trafficOrientation) < 0)
					matchedTrack->NbOfRegion = regionIndex;
				/// CAMBIO DE FLUJO VEHICULAR -----------------------------------------------------------------------------
				if(trafficOrientation == sdcv::eOrientationLine::TOP_DOWN) regionIndex++;	// TOP_DOWN
				else regionIndex--;															// BOTTOM_U
				/// CAMBIO DE FLUJO VEHICULAR -----------------------------------------------------------------------------
			}
			/// CAREFULL WITH THIS !
			if(matchedTrack->NbOfRegion == 0) matchedTrack->ClassRegionVisibleCount++;

			// Classification
			if (matchedTrack->ClassRegionVisibleCount >= 3 && matchedTrack->id) matchedTrack->isClassified = true;

			//
			if(matchedTrack->id != 0 ) {
				double prevNormArea = matchedTrack->normAreas.back();
				if(prevNormArea < matchedBlob->getNormArea()) occlusionRate = 1 - (prevNormArea/(double)matchedBlob->getNormArea());
				
				if(occlusionRate < Th_OR) {
					// the measurement result will be trusted more than predicted one
					cv::setIdentity(matchedTrack->AKF.processNoiseCov, cv::Scalar::all((1 - occlusionRate + 1e-50)));
					cv::setIdentity(matchedTrack->AKF.measurementNoiseCov, cv::Scalar::all((occlusionRate + 1e-50)));
				} else {
					// The system will trust the predicted result completely
					cv::setIdentity(matchedTrack->AKF.processNoiseCov, cv::Scalar::all((1e-50)));
					cv::setIdentity(matchedTrack->AKF.measurementNoiseCov, cv::Scalar::all((100)));
				}

			}

			// FOR OCCLUSION INDEX -------------------------------------------------------------------------
			///matchedTrack->OcclusionRes = blob->getOccluded();
			// FOR OCCLUSION INDEX END ---------------------------------------------------------------------

			// Correct the estimate of the object's location using the new detection.
			cv::Mat correctedCentroidMat = matchedTrack->AKF.correct( cv::Mat(matchedBlob->getCentroid()) );
			cv::Point2f correctedCentroid(correctedCentroidMat.at<float>(0,0), correctedCentroidMat.at<float>(2,0));

			// Update detected centroid.
			matchedTrack->detectedCentroid.push_back(matchedBlob->getCentroid() );

			// Update total frames.
			matchedTrack->totalFrames++;

			if( occlusionRate < Th_OR ) {
				/// CAMBIO DE SENTIDO -------------------------------------------------------------------------------------------
				cv::Point2f point = (trafficOrientation == sdcv::eOrientationLine::TOP_DOWN ? *(matchedTrack->estimatedCentroid.end() - 2) - correctedCentroid
					: matchedTrack->detectedCentroid.back() - *(matchedTrack->detectedCentroid.end() - 2)
					);
				/// CAMBIO DE SENTIDO -------------------------------------------------------------------------------------------
				
				double velocity = (double)cv::sqrt(point.x*point.x + point.y*point.y) * (double)(fps /(double)(FrameCountDebug - matchedTrack->lastVisibleFrame));

				if (point.y < 0.0f) {
					matchedTrack->velocity = velocity;
					matchedTrack->consecutiveBackwardDir = 0;
				}
				else {
					matchedTrack->velocity = velocity*(-1);
					matchedTrack->consecutiveBackwardDir++;
				}

				// Replace predicted bounding box with detected bounding box
				matchedTrack->bbox.push_back(matchedBlob->getBBox());

				// Replace predicted centroid with corrected centroid
				matchedTrack->estimatedCentroid.at(matchedTrack->totalVisibleFrames) = correctedCentroid;

				// Update Track's area
				matchedTrack->areas.push_back(matchedBlob->getArea() );
				if(matchedBlob->getNormArea() > 0.06) {
					if( matchedTrack->normAreas.back() == 0.00 ) matchedTrack->normAreas.back() = matchedBlob->getNormArea();
					else matchedTrack->normAreas.push_back(matchedBlob->getNormArea() );
				}

				if( (matchedTrack->id == 0) && (matchedTrack->consecutiveBackwardDir != 0) ) test = true;
				else {
					// Update visibility.
					matchedTrack->totalVisibleFrames++;
					matchedTrack->consInvisibleFrames = 0;
				}

			} /* occlusionRate < Th_OR */

			// Compute estimated area (Implicit in blob?)
			// Test if the object is before line detection
			bool in = (bool)(sdcv::distanceToLine(matchedBlob->getCentroid(), m, b, trafficOrientation) > 0 ? true : false);

			if( in ) 
			{
				if( (matchedTrack->id != 0) ) {
					// Check whether the matched track had had a normalized area or not.
					if (matchedTrack->estimatedArea == 0) 
					{
						matchedTrack->estimatedArea = matchedTrack->normAreas.back();
					} 
					else  
					{
						// This sum could be performance if it has a variable/vector that contains it in each track
						double sumNormAreas = std::accumulate(matchedTrack->normAreas.begin(), matchedTrack->normAreas.end(), 0.0);
						matchedTrack->estimatedArea = sumNormAreas/(double)matchedTrack->normAreas.size();
					}
				}
			}
			else {
				///classifier.update(lostTrack->estimatedArea);		/// ADD CLASSIFICATION HERE
			}

			// Check overlapping between matched track and all the tracks
			bool overlapFlag = false;
			for(auto it = tracks.begin(); it != tracks.end(); it++) {
				if(it != matchedTrack) {
					double bbox_A_area = matchedTrack->bbox.back().area();
					double bbox_B_area = it->bbox.back().area();
					double unionArea = cv::Rect(matchedTrack->bbox.back() & it->bbox.back()).area();
            		double overlapRatio = unionArea/(double)(bbox_A_area + bbox_B_area);
             		double minRatio = unionArea/(double)(std::min(bbox_A_area, bbox_B_area));
					
					if( (overlapRatio > 0.57) || (minRatio == 1) ) {
						overlapFlag = true;

						if (it->id == 0) 
						{
							it->consInvisibleFrames = 100;
						}
						else if (matchedTrack->id == 0) 
						{
							matchedTrack->consInvisibleFrames = 100;
						}
						else 
						{
							if (matchedTrack->totalVisibleFrames > it->totalVisibleFrames) 
							{
								it->consInvisibleFrames = 100;
							}
							else 
							{
								matchedTrack->consInvisibleFrames = 100;
							}
						}

                   		break;
					}
				}
			}

			//  Check if the object is a large vehicle created after line detection
			bool isLarge = false;
			if( (matchedTrack->id == 0) && (matchedTrack->normAreas.back() > 1.01) && !in ) {
				matchedTrack->consInvisibleFrames = 100;
				isLarge = true;
			}

			// Increment the next id
			if( matchedTrack->id == 0 && matchedTrack->totalVisibleFrames >= minVisibleCount && !isLarge && !overlapFlag) {
				if( FP_ID.back() != 0 ) {
					/// matchedTrack->isFP = 1;
					this->AssignedIdCounter++;

					matchedTrack->id = FP_ID.back();
					matchedTrack->ObjectID = this->AssignedIdCounter;

					if((int)FP_ID.size() == 1 )  FP_ID.back() = 0;
					else FP_ID.pop_back();
				} else {
					this->ID++;
					this->AssignedIdCounter++;

					matchedTrack->id = this->ID;
					matchedTrack->ObjectID = this->AssignedIdCounter;	/// NEW VARIABLE ADDED

					// VEHICLE OCCLUSION INDEX ----------------------------------------------------------------------------------------
					assignedTracks.push_back(cv::Vec3i(this->ID, matchedTrack->OcclusionRes, -1));
					// ----------------------------------------------------------------------------------------------------------------
				}
			}

			matchedTrack->lastVisibleFrame = FrameCountDebug;
		} /* END FOR */



		/* Update unassigned tracks ----------------------------------------------------------------------*/
		for(int i = 0; i < (int)unassignedTracks.size(); i++) {
			int unassignedInds = unassignedTracks.at(i);
			std::vector<sdcv::Track>::iterator unmatchedTrack = std::next(tracks.begin(), unassignedInds);

			unmatchedTrack->totalFrames++;

			if( unmatchedTrack->velocity < 0 ) unmatchedTrack->consecutiveBackwardDir++;

			// Find the indices of 'new' tracks with totalFrames equal to 2 and distance < 25
			std::vector<double> distances;
			for(std::vector<sdcv::Track>::iterator it = tracks.begin(); it != tracks.end(); ++it) {
				/// CAMBIO DE SENTIDO -------------------------------------------------------------------------------------------
				cv::Point2f point = (trafficOrientation == sdcv::eOrientationLine::TOP_DOWN ? it->predictedCentroid - unmatchedTrack->estimatedCentroid.back()
					: it->detectedCentroid.back() - *(it->detectedCentroid.end() - 2)
					);
				/// CAMBIO DE SENTIDO -------------------------------------------------------------------------------------------
				double dist = std::sqrt(point.x*point.x + point.y*point.y);
				if( dist == 0.0 ) dist = 100.0;
				distances.push_back( dist );
			}

			auto minElementIt = std::min_element(distances.begin(), distances.end());
			int ind = (int)std::distance(distances.begin(), minElementIt);
			std::vector<sdcv::Track>::iterator minTrack = std::next(tracks.begin(), ind);

			cv::Mat distancesMat(distances), IdMat, totalFramesMat;
			for(auto it = tracks.begin(); it != tracks.end(); ++it) {
				// IDs
				IdMat.push_back(it->id);

				// totalFrames
				totalFramesMat.push_back(it->totalFrames);
			}

			cv::Mat newInds = (IdMat == 0) & (totalFramesMat == 2) & (distancesMat <= 25);
			int nonZeroCount = cv::countNonZero( newInds );
			bool noAssignedID = minTrack->id == 0;

			if (nonZeroCount == 1 && noAssignedID)
			{
				minTrack->consInvisibleFrames = 100;
			}
			else if(nonZeroCount == 2 && noAssignedID ) 
			{
				// THIS CONDITION REASSIGN AN ID FROM A VEHICLE TO OTHER (minTrack->normAreas.back() > 0.8*unmatchedTrack->normAreas.back())
				unmatchedTrack->bbox = minTrack->bbox;
				unmatchedTrack->totalVisibleFrames++;
				unmatchedTrack->areas.push_back(minTrack->areas.at(0));

				if( minTrack->normAreas.back() > .06) unmatchedTrack->normAreas.push_back(minTrack->normAreas.back());	

				unmatchedTrack->estimatedCentroid.push_back(minTrack->estimatedCentroid.back());
				unmatchedTrack->detectedCentroid.push_back(minTrack->detectedCentroid.back());
				unmatchedTrack->consInvisibleFrames = 0;
				minTrack->consInvisibleFrames = 100;

			} 
			else
			{
				unmatchedTrack->consInvisibleFrames++;
			}
		}
	}


	/*!
	 * @name	erease
	 * @brief	Deletes tracks that are not anymore in the frame and false tracks.
	 */
	void Tracker::remove(void) {
		if( tracks.empty() )
			return;


		// Compute the fraction of the track's age for which it was visible.
		cv::Mat IdMat, consecutiveBackwardDirMat, VelocityMat, consInvisibleFramesMat, lostIndsROI, VisibilityMat;
		int NbTracks = (int)tracks.size();
		for(auto it = tracks.begin(); it != tracks.end(); ++it) {
			double visibilityElement = it->totalVisibleFrames / (double)it->totalFrames;
			VisibilityMat.push_back(visibilityElement);

			// IDs
			IdMat.push_back(it->id);

			// consecutiveBackwardDir
			consecutiveBackwardDirMat.push_back(it->consecutiveBackwardDir);

			// velocity
			VelocityMat.push_back(it->velocity);

			// consInvisibleFrames
			/// CAMBIO DE SENTIDO -------------------------------------------------------------------------------------------
			if (it->lastVisibleFrame != this->FrameCountDebug && it->isClassified && sdcv::distanceToLine(it->estimatedCentroid.back(), roi.getEndLineEq().x, roi.getEndLineEq().y, trafficOrientation) < 0.0)
				it->consInvisibleFrames = 100;

			cv::Point2f ptest = (trafficOrientation == sdcv::eOrientationLine::TOP_DOWN ? it->estimatedCentroid.back() : it->detectedCentroid.back());
			int v = int(sdcv::distanceToLine(ptest, roi.getEndLineEq().x, roi.getEndLineEq().y, trafficOrientation) < 0 ? 100 : it->consInvisibleFrames);
			/// CAMBIO DE SENTIDO -------------------------------------------------------------------------------------------
			
			consInvisibleFramesMat.push_back(v);
		}

		// Check backward direction -----------------------------------------------------------------
		cv::Mat lostIndsBackDir = ( (IdMat == 0) & ((consecutiveBackwardDirMat == 1) | ((VelocityMat >= -3) & (VelocityMat <= 3))));				// >= -2 && <= 2

		lostIndsBackDir = (IdMat != 0)	&	(( (VelocityMat < -1000)	|	(VelocityMat > 1000)	|	
						  ((VelocityMat >= -1)	&	(VelocityMat <= 1)))	|
						  (consecutiveBackwardDirMat == 5))	|	lostIndsBackDir;

		// Find the indices of 'lost' tracks  ----------------------------------------------------------
		cv::Mat lostInds = lostIndsBackDir | ((IdMat == 0) & (VisibilityMat < 0.65)) | (consInvisibleFramesMat >= this->minInvisibleCount) ;
		/* ------------------------------------------------------------------------------*/

		// SEARCH IN THIS CODE THE VALUE OF "lostTrack = tracks.begin()" SEEMS FISHY TO ME
		int nonZero = cv::countNonZero( lostInds );
		if( nonZero > 0) 
		{
			double m = (roi.getLineDetection().at(1).y - roi.getLineDetection().at(0).y) / (roi.getLineDetection().at(1).x - roi.getLineDetection().at(0).x);
			double b = roi.getLineDetection().at(1).y - m*roi.getLineDetection().at(1).x;

			auto lostTrack = tracks.begin();
			for(int idx = 0; idx < lostInds.rows; idx++, lostTrack++) {
				if (lostInds.at<unsigned char>(idx)) lostTrack->isDone = true;
				// See if the object is after ending line -----------------------------------------------------------------------------------
				/// CAMBIO DE SENTIDO -------------------------------------------------------------------------------------------
				cv::Point2f ptest = (trafficOrientation == sdcv::eOrientationLine::TOP_DOWN ? lostTrack->estimatedCentroid.back() : lostTrack->detectedCentroid.back());
				double value = sdcv::distanceToLine(ptest, roi.getEndLineEq().x, roi.getEndLineEq().y, trafficOrientation);
				/// CAMBIO DE SENTIDO -------------------------------------------------------------------------------------------

				if (value < 0)  lostInds.at<unsigned char>(idx) = 255;

				bool flag = false;
				bool flagId = ((unsigned int)lostInds.at<unsigned char>(idx)) && (lostTrack->id);

				if( flagId ) {
					if (lostTrack->estimatedArea == 0.0) {
						// compute estimated area only if the object is created after line detection
						double sumNormAreas = std::accumulate(lostTrack->normAreas.begin(), lostTrack->normAreas.end(), 0.0);
						lostTrack->estimatedArea = sumNormAreas/(double)lostTrack->normAreas.size();
					}
				}


				// See if the object is before line detection -------------------------------------------
				/// CAMBIO DE SENTIDO -------------------------------------------------------------------------------------------
				bool inPoly = (bool)(sdcv::distanceToLine(ptest, m, b, trafficOrientation) > 0 ? true : false);
				/// CAMBIO DE SENTIDO -------------------------------------------------------------------------------------------

				if( flagId && (inPoly || (lostTrack->totalVisibleFrames == minVisibleCount)) && !lostTrack->isClassified ) 
				{
					flag = true;
					classifier->update( 3 );
					if( FP_ID.back() == 0 ) {
						FP_ID.clear();
						FP_ID.push_back( lostTrack->id );
					} else FP_ID.push_back( lostTrack->id );
				}
				
				// CLASSIFICATION --------------------------------------------------------------------------------------------
				//		VEHICLE OCCLUSION INDEX ------------------------------------------------------------------------------
				if( flagId && !flag ) 
					assignedTracks.at(lostTrack->id - 1)[2] = classifier->update(lostTrack->estimatedArea);
				// -----------------------------------------------------------------------------------------------------------

			}
		} /* nonZero > 0 */

		// Delete lost tracks ------------------------------------------------------------------
		tracks.erase( std::remove_if(tracks.begin(), tracks.end(), [](sdcv::Track const& obj) { return obj.isDone; }), tracks.end());
		/*
		std::vector<sdcv::Track> tmp;
		std::vector<sdcv::Track>::iterator itTrack = tracks.begin();
		for(int i = 0; i < lostInds.cols, itTrack != tracks.end(); i++, ++itTrack) 
		{
			if( !lostInds.at<unsigned char>(i) ) 
			{
				tmp.push_back(*itTrack);
			}
			
		}
		tracks = tmp;
		*/
	}


	cv::Mat Tracker::getAssignmentCostMatrix( void ) {
		cv::Mat MeasurementsCentroid = cv::Mat(blobLst.getCentroids(), true);
		cv::Mat CostMatrix;
		
		for(std::vector<Track>::iterator it = tracks.begin(); it != tracks.end(); ++it) {
			cv::Rect bbox = it->bbox.back();

			// Predict the current location of the track
			cv::Mat predictedState = it->AKF.predict();
			it->predictedCentroid = cv::Point2f(predictedState.at<float>(0,0), predictedState.at<float>(2,0));
			
			/*std::cout << "" << it->predictedCentroid << std::endl;*/
			
			it->estimatedCentroid.push_back(it->predictedCentroid);

			// Shift the bounding box so that its center is at the predicted location
			cv::Point2f predictBBox((float)bbox.width, (float)bbox.height);
			predictBBox = it->predictedCentroid - (predictBBox * 0.5);
			it->bbox.back() = cv::Rect((int)predictBBox.x, (int)predictBBox.y, bbox.width, bbox.height);

			// Compute the cost of assigning each detection to each track
			distance(it->AKF, MeasurementsCentroid, CostMatrix);
		}

		// Convert cost matrix to <int>.
		CostMatrix.convertTo(CostMatrix, CV_16S);

		return CostMatrix;
	}

	/*
	 * Use the distance method to find the best matches. 
	 * The computed distance values describe how a set of measurements matches the Kalman filter. 
	 *
	 * You can thus select a measurement that best fits the filter. This strategy can be used for matching object
	 * detections against object tracks in a multiobject tracking problem. 
	 *
	 * This distance computation takes into account the covariance of the predicted state and the process noise.
	 * The distance method can only be called after the predict method.
	 *
	 * d = distance(obj, z_matrix) computes a distance between the location of a detected object and the predicted
	 * location by the Kalman filter object. Each row of the N-column z_matrix input matrix contains a measurement
	 * vector. The distance method returns a row vector where each distance element corresponds to the measurement
	 * input.
	 */
	void Tracker::distance(cv::KalmanFilter AKF, cv::Mat centroids, cv::OutputArray CostMatrix) {
		CV_Assert(blobLst.getLen() > 0);
		
		int NbTracks = (int)tracks.size();
		int NbDetections = blobLst.getLen();
		cv::Mat Tmp;
		cv::Mat costRow = cv::Mat::zeros(cv::Size(1, NbDetections), CV_32F);
		CostMatrix.copyTo(Tmp);

		// Calculating distance for each row of blobLst.getCentroids()
		//S = HPH.T + R
		//d(z) = (z - H*x).T * S.inv * (z- H*x) + ln( det(S) );
		cv::Mat Sigma = (AKF.measurementMatrix * AKF.errorCovPost * AKF.measurementMatrix.t()) + (AKF.measurementNoiseCov);
		cv::Mat centroidPred = cv::Mat(AKF.measurementMatrix * AKF.statePost).t();

		std::vector<float> distances;
		for(int i = 0; i < centroids.rows; i++) {
			float xo = centroids.row(i).at<float>(0,0) - centroidPred.at<float>(0,0);
			float yo = centroids.row(i).at<float>(0,1) - centroidPred.at<float>(0,1);

			cv::Mat aux = (cv::Mat_<float>(1,2) << xo,yo);
			cv::Mat distance = aux * Sigma.inv() * aux.t() + log10( cv::determinant(Sigma) );

			distances.push_back(distance.at<float>(0,0));
		}
		
		// Add a new row
		Tmp.push_back( cv::Mat(distances).t() );
		Tmp.copyTo( CostMatrix );
	}


	/*!
	 * @name
	 * @brief
	 * @param CostMatrix:				Cost matrix (Number of Tracks, Number of Detections)
	 * @param assignments:				(Track Index, Detection Index)
	 * @param unassignedTracks:			(Track Index)
	 * @param unassignedDetections:
	 */
	void Tracker::assignmentProblemSolver(cv::Mat CostMatrix, std::vector<cv::Point> &assignments, std::vector<int> &unassignedTracks, std::vector<int> &unassignedDetections, int costOfNonAssigment) {
		costOfNonAssigment = costOfNonAssigment * 2;

		cv::Mat_<int> solvedMatrix( CostMatrix );
		Munkres solver;
		solver.diag( false );

		int find_for_id = 30, numtrack = 0;
		bool find = false;
		sdcv::Track *trackF = NULL;
		for (auto track = this->tracks.begin(); track != this->tracks.end(); ++track) {
			if (this->ID >= find_for_id && track->id == find_for_id) {
				find = true;
				trackF = &(*track);
				break;
			}
			numtrack++;
		}

		solver.solve( solvedMatrix );
		

		// Return the assigned tracks
		for(int i = 0; i < solvedMatrix.rows; i++) {
			
			std::vector<int> rowVector;
			solvedMatrix.row(i).copyTo( rowVector );

			std::vector<int>::iterator it = std::find(rowVector.begin(), rowVector.end(), 0);
			int idx = (int)std::distance(rowVector.begin(), it);

			if (it != rowVector.end() && CostMatrix.at<short>(i, idx) <= costOfNonAssigment) assignments.push_back(cv::Point(i, idx));
			else unassignedTracks.push_back( i );
		}

		// Return the unassignedDetections
		for(int i = 0; i < solvedMatrix.cols; i++) {
			if(cv::countNonZero(solvedMatrix.col(i)) == solvedMatrix.rows )
				unassignedDetections.push_back(i);
		}
	}
	
	// Destructor -------------------------------------------------------------------------------------
	Tracker::~Tracker() { 
		// Classification results
		sdcv::fs_write("DATA/" + this->roi.getName() + "/class.csv", cv::Mat(this->assignedTracks), sdcv::SDCV_FS_WRITE, "ID,OCC,CLASS");

		// Vehicle Occlusion handling
		sdcv::fs_write("DATA/" + this->roi.getName() + "/voi.csv", cv::Mat(this->VOI), sdcv::SDCV_FS_WRITE, "Vin,Vocc,VOI");
	}
	


	// ------------------------------------------------------------------------------------------------

}

/* ************** E N D   O F   F I L E ----------------- CINVESTAV GDL */