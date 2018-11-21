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
		this->ID = 0;
		this->FP_ID.push_back(0);
		this->init = false;
		this->verbose = verbose;

		this->FrameCountDebug = 0;
		this->Frame0 = -1;
		this->ID_k_m_1 = 0;
		this->OcclusionTime = 10;

		this->AssignedIdCounter = 0;

		this->classifier = classifier;
	}
			
	// Get methods ------------------------------------------------------------------------------------
	sdcv::Track Tracker::getTrack(int id) {
		if( id < (int)this->tracks.size() ) return tracks.at(id);
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

	// Set methods ------------------------------------------------------------------------------------
	void Tracker::setMinVisibleCount(int minVisibleCount) { this->minVisibleCount = minVisibleCount; }
	void Tracker::setMinInvisibleCount(int minInvisibleCount) { this->minInvisibleCount = minInvisibleCount; }
	void Tracker::setOcclusionTime(int sec) { this->OcclusionTime = sec; }
	
	// Action methods ---------------------------------------------------------------------------------
	void Tracker::clear(void) {
		this->tracks.clear();
		this->assignedTracks.clear();
		this->currentID = 0;
		this->ID = 0;
		this->FP_ID.clear();
		this->FP_ID.push_back(0);
		this->init = false;

		this->FrameCountDebug = 0;
		this->Frame0 = -1;
		this->ID_k_m_1 = 0;
		this->OcclusionTime = 10;
	}

	void Tracker::computeVOI(void) {
		if (Frame0 == -1) Frame0 = FrameCountDebug;
		else if (FrameCountDebug - Frame0 == int(fps*this->OcclusionTime)) {
			int Vni = ID - ID_k_m_1;
			int Vno = getNbOfOcclusion(ID_k_m_1);
			VOI.push_back(cv::Point3d((double)Vni, (double)Vno, (double)(Vno / (double)Vni)));

			/* ---------------------------------------------- Evaluation ---------------------------------------------- */
			/*std::cout << "VEHICLE OCCLUSION INDEX { " << std::endl;
			std::cout << "\tInterval [" << Frame0 << ", " << FrameCountDebug << "]: " << std::endl;
			std::cout << "\tVehicle(in): " << Vni << ", Vehicle(occ): " << Vno << std::endl;
			std::cout << "\tLast ID: " << std::to_string(ID_k_m_1) << ", Current ID: " << std::to_string(ID) << std::endl;
			std::cout << "\tVOI: " << std::to_string(VOI) << std::endl;
			std::cout << "}" << std::endl << std::endl;*/
			/* -------------------------------------------------------------------------------------------------------- */
			Frame0 = -1;
			ID_k_m_1 = ID;
			//	Frame0 = FrameCountDebug + 1;
		}
	}


	void Tracker::writeTracks(std::ofstream &file) {
		for (auto track : tracks)
			file << FrameCountDebug << "," << 0 << "," << track << std::endl;
	}

	void Tracker::writeTracks(void) {
		for (auto track : tracks)
			std::cout << FrameCountDebug << "," << 0 << "," << track << std::endl;
	}


	//#define SDCV_TRACKING_DEBUG
	void Tracker::track(std::vector<sdcv::Blob> detections) {
		#ifdef SDCV_TRACKING_DEBUG
		out << std::endl << std::endl << "[TRACKING CLASS BEGIN]-------------------------------------------" << std::endl;
		out << "Frame: " << FrameCountDebug << std::endl;
		#endif
		FrameCountDebug++;

		///std::cout << "\n\nFrame: " << FrameCountDebug << std::endl;
		blobLst = sdcv::BlobList(detections);
		
		int NbDetections = (int)detections.size();
		int NbTracks = (int)tracks.size();
		
		if( NbTracks == 0 ) init = false;	// No tracks
		if( NbDetections == 0 ) return;		// Nothing to track

		#ifdef SDCV_TRACKING_DEBUG
		out << "Number of tracks: " << NbTracks << std::endl;
		out << "Number of detections: " << NbDetections << std::endl;
		#endif
		
		if( init == false ) {
			///std::cout << "Detected centroids" << std::endl << blobLst.getCentroids() << std::endl;
			for(auto detection = detections.begin(); detection != detections.end(); ++detection) {
				tracks.push_back(sdcv::Track(*detection, FrameCountDebug));
				// Check for Regions and Visible blob classify region
				std::vector< cv::Point2d > regions = roi.getRegionLinesEquation();
				/// CAMBIO DE FLUJO VEHICULAR -----------------------------------------------------------------------------
				int regionIndex = (trafficOrientation == sdcv::eOrientationLine::TOP_DOWN ? 0 : 2);
				/// CAMBIO DE FLUJO VEHICULAR -----------------------------------------------------------------------------
				for (auto iterator = regions.begin()+1; iterator != regions.end(); ++iterator) {
					if (sdcv::distanceToLine(detection->getCentroid(), iterator->x, iterator->y, trafficOrientation) < 0.0)
						tracks.back().NbOfRegion = regionIndex;
					/// CAMBIO DE FLUJO VEHICULAR -----------------------------------------------------------------------------
					if(trafficOrientation == sdcv::eOrientationLine::TOP_DOWN) regionIndex++;	// TOP_DOWN
					else regionIndex--;															// BOTTOM_UP
					/// CAMBIO DE FLUJO VEHICULAR -----------------------------------------------------------------------------
					
				}
				// CAREFULL WITH THIS CONDITION
				if (tracks.back().NbOfRegion == 0) tracks.back().ClassRegionVisibleCount++;

				
				#ifdef SDCV_TRACKING_DEBUG
				out << "New track added: " << it->getCentroid() << std::endl;
				#endif
			}

			init = true;


			#ifdef SDCV_TRACKING_DEBUG
			FrameCountDebug++;
			out << "[TRACKING CLASS END]-------------------------------------------" << std::endl;
			//cv::waitKey();
			#endif
		}
		else {

			cv::Mat CostMatrix = getAssignmentCostMatrix();

			/* Solve the assignment problem --------------------------------------------------------*/
			std::vector<cv::Point> assignments;
			std::vector<int> unassignedTracks, unassignedDetections;
			assignmentProblemSolver(CostMatrix, assignments, unassignedTracks, unassignedDetections, 10);

			/* Update tracks -----------------------------------------------------------------------*/
			update(detections, assignments, unassignedTracks);

			/* Delete tracks -----------------------------------------------------------------------*/
			erease();

			/* Create new tracks -------------------------------------------------------------------*/
			add(detections, unassignedDetections);

			/* Clear blobs list before exiting */
			blobLst.clear();

#ifdef SDCV_TRACKING_DEBUG
			out << "[TRACKING CLASS END]-------------------------------------------" << std::endl;
			FrameCountDebug++;
			cv::waitKey();
#endif
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


	void Tracker::update(std::vector<sdcv::Blob> detectedBlobs, std::vector<cv::Point> assignments, std::vector<int> unassignedTracks) {
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
			
			/// DEBUG
			/*
			if (matchedTrack->id) {
				std::cout << "Track: " << matchedTrack->id << std::endl;
			}
			// */

			//if (std::abs(matchedTrack->detectedCentroid.back().x - blob->getCentroid().x) > 15 && std::abs(matchedTrack->detectedCentroid.back().y - blob->getCentroid().y) > 15) continue;
			#ifdef SDCV_TRACKING_DEBUG
			out << "Blob matched: " << detectionIdx << std::endl;
			matchedBlob->print( );
			out << std::endl << "Track matched: " << trackIdx << std::endl;
			matchedTrack->print( );
			#endif
			///std::cout << "Blob matched: " << detectionIdx << std::endl;
			///blob->print();
			///std::cout << std::endl << "Track matched: " << trackIdx << std::endl;
			///matchedTrack->print();
			
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

				#ifdef SDCV_TRACKING_DEBUG
				out << std::endl << "Occlusion Rate: " << occlusionRate << std::endl;
				out << "Kalman Filter: Noise Covariances have been updated on condition: ";
				#endif
				///std::cout << std::endl << "Occlusion Rate: " << occlusionRate << " < 0.3 ?" << std::endl;
				if(occlusionRate < Th_OR) {
					// the measurement result will be trusted more than predicted one
					cv::setIdentity(matchedTrack->AKF.processNoiseCov, cv::Scalar::all((1 - occlusionRate + 1e-50)));
					cv::setIdentity(matchedTrack->AKF.measurementNoiseCov, cv::Scalar::all((occlusionRate + 1e-50)));
					
					#ifdef SDCV_TRACKING_DEBUG
					out << "1" << std::endl;
					#endif
				} else {
					// The system will trust the predicted result completely
					cv::setIdentity(matchedTrack->AKF.processNoiseCov, cv::Scalar::all((1e-50)));
					cv::setIdentity(matchedTrack->AKF.measurementNoiseCov, cv::Scalar::all((100)));

					#ifdef SDCV_TRACKING_DEBUG
					out << "2" << std::endl;
					#endif
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
			#ifdef SDCV_TRACKING_DEBUG
			out << std::endl << "Corrected Centroid: " << correctedCentroid << std::endl;
			#endif

			// Update total frames.
			matchedTrack->totalFrames++;
			#ifdef SDCV_TRACKING_DEBUG
			out << "Total frames: " << matchedTrack->totalFrames << std::endl;
			#endif

			if( occlusionRate < Th_OR ) {
				// Update velocity
				/*int endIdx = (int)matchedTrack->estimatedCentroid.size() - 1;
				float x1 = matchedTrack->estimatedCentroid.at(endIdx - 1).x;
				float y1 = matchedTrack->estimatedCentroid.at(endIdx - 1).y;
				float x2 = correctedCentroid.x;
				float y2 = correctedCentroid.y;
				double velocity = cv::sqrt(cv::pow((x1 - x2),2)+ cv::pow((y1 - y2),2))*(fps/(FrameCountDebug - matchedTrack->lastVisibleFrame));
				
				if( y2 > y1 ) {
					matchedTrack->velocity = velocity;
					matchedTrack->consecutiveBackwardDir = 0;
				} else {
					matchedTrack->velocity = velocity*(-1);
					matchedTrack->consecutiveBackwardDir++;
				//std::cout << "Backward with centroid: " << correctedCentroid << "\t" << matchedTrack->estimatedCentroid.at(endIdx - 2) << std::endl;
				} */ // OLD METHOD

				/// CAMBIO DE SENTIDO -------------------------------------------------------------------------------------------
				cv::Point2f point = (trafficOrientation == sdcv::eOrientationLine::TOP_DOWN ? *(matchedTrack->estimatedCentroid.end() - 2) - correctedCentroid
					: matchedTrack->detectedCentroid.back() - *(matchedTrack->detectedCentroid.end() - 2)
					);
				/// CAMBIO DE SENTIDO -------------------------------------------------------------------------------------------
				
				double velocity = (double)cv::sqrt(point.x*point.x + point.y*point.y) * (double)(fps /(double)(FrameCountDebug - matchedTrack->lastVisibleFrame));
				// y1 = matchedTrack->estimatedCentroid - y2 = correctedCentroid.y
				// point.y > 0 if matchedTrack->estimatedCentroid > correctedCentroid
				// point.y < 0 if correctedCentroid > matchedTrack->estimatedCentroid
				if (point.y < 0.0f) {
					matchedTrack->velocity = velocity;
					matchedTrack->consecutiveBackwardDir = 0;
				}
				else {
					matchedTrack->velocity = velocity*(-1);
					matchedTrack->consecutiveBackwardDir++;
				}
				//std::cout << "Match " << matchedTrack->detectedCentroid.back() << ": " << matchedTrack->consecutiveBackwardDir << std::endl;
				/// CHANGE THIS CONDITION TOO FOR DIFFERENT ORIENTATION


				/*int endIdx = matchedTrack->detectedCentroid.size() - 1;
				float x1 = matchedTrack->detectedCentroid.at(endIdx - 1).x;
				float y1 = matchedTrack->detectedCentroid.at(endIdx - 1).y;
				float x2 = matchedTrack->detectedCentroid.back().x;
				float y2 = matchedTrack->detectedCentroid.back().y;*/
				//float velocity = sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2))*fps; // sqrt(...)/dt -> dt = 1/fps

				/*std::cout << "Current frame: " << FrameCountDebug << std::endl;
				std::cout << "Last frame visible: " << matchedTrack->lastVisibleFrame << std::endl;
				std::cout << "Point 1: " << matchedTrack->estimatedCentroid.at(endIdx - 1) << std::endl;
				std::cout << "Point 2: " << correctedCentroid << std::endl;
				std::cout << "Velocity: " << velocity << std::endl << std::endl;*/
				
				///if(matchedTrack->totalVisibleFrames > 2) {
				
				
				///}
				// Replace predicted bounding box with detected bounding box
				matchedTrack->bbox.push_back(matchedBlob->getBBox());

				// Replace predicted centroid with corrected centroid
				matchedTrack->estimatedCentroid.at(matchedTrack->totalVisibleFrames) = correctedCentroid;

				// Update Track's area
				matchedTrack->areas.push_back(matchedBlob->getArea() );
				if(matchedBlob->getNormArea() > 0.06) {					////////////////////////////////////// CHECK THIS VALUE
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

			// Compute estimated area 
			#ifdef SDCV_TRACKING_DEBUG
			out << std::endl << ">>>>> Compute estimated area for blob: " << detectionIdx << std::endl;
			out << "Centroid: " << matchedBlob->getCentroid() << std::endl << "Line detection: " <<  roi.getCenterLineDetection() << std::endl;
			#endif

			// Test if the object is before line detection
			//bool in = (blob->getCentroid().y < roi.getCenterLineDetection().y);
			bool in = (bool)(sdcv::distanceToLine(matchedBlob->getCentroid(), m, b, trafficOrientation) > 0 ? true : false);

			if( in ) {
				#ifdef SDCV_TRACKING_DEBUG
				out << "It is before line detection" << std::endl;
				#endif

				if( (matchedTrack->id != 0) ) {
					// Check whether the matched track had had a normalized area or not.
					if (matchedTrack->estimatedArea == 0) {
						matchedTrack->estimatedArea = matchedTrack->normAreas.back();
						#ifdef SDCV_TRACKING_DEBUG
						out << "New estimated area: " << matchedTrack->estimatedArea << std::endl;
						#endif
					} else  {
						// This sum could be performance if it has a variable/vector that contains it in each track
						double sumNormAreas = std::accumulate(matchedTrack->normAreas.begin(), matchedTrack->normAreas.end(), 0.0);
						matchedTrack->estimatedArea = sumNormAreas/(double)matchedTrack->normAreas.size();

						#ifdef SDCV_TRACKING_DEBUG
						out << "Norm areas: " << cv::format(matchedTrack->normAreas,0) << std::endl;
						out << "Sum of Norm areas: " << sumNormAreas << std::endl;
						out << "Estimated area: " << matchedTrack->estimatedArea << std::endl;
						#endif
					}
				}
			}
			else {
				///classifier.update(lostTrack->estimatedArea);		/// ADD CLASSIFICATION HERE
			#ifdef SDCV_TRACKING_DEBUG
				out << "It is after line detection" << std::endl;
			#endif
			}
			#ifdef SDCV_TRACKING_DEBUG
			out << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl << std::endl;
			#endif

			// Check overlapping between matched track and all the tracks
			bool overlapFlag = false;

			#ifdef SDCV_TRACKING_DEBUG
			out << std::endl << ">>>>> Check overlapping" << std::endl;
			int idxTrackDebug = 0;
			#endif

			for(auto it = tracks.begin(); it != tracks.end(); it++) {
				if(it != matchedTrack) {
					double bbox_A_area = matchedTrack->bbox.back().area();
					double bbox_B_area = it->bbox.back().area();
					double unionArea = cv::Rect(matchedTrack->bbox.back() & it->bbox.back()).area();
            		double overlapRatio = unionArea/(double)(bbox_A_area + bbox_B_area);
             		double minRatio = unionArea/(double)(std::min(bbox_A_area, bbox_B_area));

					#ifdef SDCV_TRACKING_DEBUG
					out << "Minimum ratio [" << trackIdx << ", " << idxTrackDebug << "]: " << minRatio << std::endl;
					#endif

					
					if( (overlapRatio > 0.57) || (minRatio == 1) ) {
						overlapFlag = true;

						#ifdef SDCV_TRACKING_DEBUG
						out << "There was an overlaping between matched track and: " << idxTrackDebug << std::endl;
						#endif

						if (it->id == 0) {
							it->consInvisibleFrames = 100;
							///std::cout << "C1: " << it->detectedCentroid.back() << std::endl;
						}
						else if (matchedTrack->id == 0) {
							matchedTrack->consInvisibleFrames = 100;
							///std::cout << "C2:" << it->detectedCentroid.back() << std::endl;
						}
						else {
							if (matchedTrack->totalVisibleFrames > it->totalVisibleFrames) {
								it->consInvisibleFrames = 100;
								///std::cout << "C3:" << it->detectedCentroid.back() << std::endl;
							}
							else {
								matchedTrack->consInvisibleFrames = 100;
								///std::cout << "C4:" << it->detectedCentroid.back() << std::endl;
							}
						}

                   		break;
					}
				}

				#ifdef SDCV_TRACKING_DEBUG
				idxTrackDebug++;
				#endif
			}
			
			#ifdef SDCV_TRACKING_DEBUG
			out << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl << std::endl;
			#endif

			//  Check if the object is a large vehicle created after line detection
			#ifdef SDCV_TRACKING_DEBUG
			out << std::endl << ">>>>> Check for a large vehicle created after line detection" << std::endl;
			out << "Is vehicle large? ";
			#endif

			bool isLarge = false;
			if( (matchedTrack->id == 0) && (matchedTrack->normAreas.back() > 1.01) && !in ) {
				matchedTrack->consInvisibleFrames = 100;
				///std::cout << "C5:" << matchedTrack->detectedCentroid.back() << std::endl;
				isLarge = true;
			}
			#ifdef SDCV_TRACKING_DEBUG
			out << (isLarge ? "Yes" : "No") << std::endl;
			out << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl << std::endl;
			#endif

			// Increment the next id
			#ifdef SDCV_TRACKING_DEBUG
			out << std::endl << ">>>>> Increment the next id" << std::endl;
			#endif

			if( matchedTrack->id == 0 && matchedTrack->totalVisibleFrames >= minVisibleCount && !isLarge && !overlapFlag) {
				if( FP_ID.back() != 0 ) {
					// Add this flag #######################################################
					/// matchedTrack->isFP = 1;
					this->AssignedIdCounter++;	/// NEW VARIABLE ADDED
					// #####################################################################

					matchedTrack->id = FP_ID.back();
					matchedTrack->ObjectID = this->AssignedIdCounter;		/// NEW VARIABLE ADDED

					#ifdef SDCV_TRACKING_DEBUG
					out << "Matched ID was setted by FP" << std::endl;
					out << "Track #" << trackIdx << " has new ID [" << FP_ID.back() << "]" << std::endl;
					#endif

					if((int)FP_ID.size() == 1 )  FP_ID.back() = 0;
					else FP_ID.pop_back();
				} else {
					this->ID++;
					this->AssignedIdCounter++;	/// NEW VARIABLE ADDED
					#ifdef SDCV_TRACKING_DEBUG
					out << "Track #" << trackIdx << " has new ID [" << this->ID << "]" << std::endl;
					#endif

					matchedTrack->id = this->ID;
					matchedTrack->ObjectID = this->AssignedIdCounter;	/// NEW VARIABLE ADDED

					// VEHICLE OCCLUSION INDEX ----------------------------------------------------------------------------------------
					assignedTracks.push_back(cv::Vec3i(this->ID, matchedTrack->OcclusionRes, -1));
					// ----------------------------------------------------------------------------------------------------------------
				}
			}

			matchedTrack->lastVisibleFrame = FrameCountDebug;

			#ifdef SDCV_TRACKING_DEBUG
			out << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl << std::endl;
			#endif
			///std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
		} /* END FOR */



		/* Update unassigned tracks ----------------------------------------------------------------------*/
		#ifdef SDCV_TRACKING_DEBUG
		out << std::endl << ">>>>> Update unassigned tracks" << std::endl;
		#endif
		///std::cout << "[Update Unassigned Tracks] -----------------------------------" << std::endl;
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

			#ifdef SDCV_TRACKING_DEBUG
			out << "Min element: " << *minElementIt << " at " << ind << std::endl;
			#endif

			cv::Mat distancesMat(distances), IdMat, totalFramesMat;
			for(auto it = tracks.begin(); it != tracks.end(); ++it) {
				// IDs
				IdMat.push_back(it->id);

				// totalFrames
				totalFramesMat.push_back(it->totalFrames);
			}
			
			#ifdef SDCV_TRACKING_DEBUG
			out << "distances: " << std::endl << distancesMat << std::endl << std::endl;
			out << "ID Tracks: " << std::endl << IdMat << std::endl << std::endl;
			out << "Total Frames Tracks: " << std::endl << totalFramesMat << std::endl << std::endl;
			#endif

			cv::Mat newInds = (IdMat == 0) & (totalFramesMat == 2) & (distancesMat <= 25);
			int nonZeroCount = cv::countNonZero( newInds );
			bool noAssignedID = minTrack->id == 0;

			#ifdef SDCV_TRACKING_DEBUG
			out << "New Vehicles: " << std::endl << newInds << std::endl << std::endl;
			#endif
			/*if (unmatchedTrack->id == detected_id) {
				std::cout << "last detected Centroid: " << unmatchedTrack->detectedCentroid.back() << std::endl;
				std::cout << "Estimated Centroid: " << unmatchedTrack->estimatedCentroid.back() << std::endl;
				//std::cout << "BackwardDir: " << unmatchedTrack->consecutiveBackwardDir << std::endl;
			}*/
			if (nonZeroCount == 1 && noAssignedID) {
				minTrack->consInvisibleFrames = 100;
				///std::cout << "C6:" << unmatchedTrack->detectedCentroid.back() << std::endl;
			}
			else if(nonZeroCount == 2 && noAssignedID ) {	/// THIS CONDITION REASSIGN AN ID FROM A VEHICLE TO OTHER (minTrack->normAreas.back() > 0.8*unmatchedTrack->normAreas.back())
				unmatchedTrack->bbox = minTrack->bbox;
				unmatchedTrack->totalVisibleFrames++;
				unmatchedTrack->areas.push_back(minTrack->areas.at(0));

				if( minTrack->normAreas.back() > .06) unmatchedTrack->normAreas.push_back(minTrack->normAreas.back());	

				unmatchedTrack->estimatedCentroid.push_back(minTrack->estimatedCentroid.back());
				unmatchedTrack->detectedCentroid.push_back(minTrack->detectedCentroid.back());
				unmatchedTrack->consInvisibleFrames = 0;
				minTrack->consInvisibleFrames = 100;
				///std::cout << "C7:" << unmatchedTrack->detectedCentroid.back() << std::endl;

			} else unmatchedTrack->consInvisibleFrames++;

			///std::cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl;
		}
		#ifdef SDCV_TRACKING_DEBUG
		out << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" << std::endl << std::endl;
		out << "[Update Tracks End] --------------------" << std::endl;
		#endif
	}


	/*!
	 * @name	erease
	 * @brief	Deletes tracks that are not anymore in the frame and false tracks.
	 */
	void Tracker::erease(void) {
		#ifdef SDCV_TRACKING_DEBUG
		out << "[Delete Tracks Begin] --------------------" << std::endl;
		#endif


		if( tracks.empty() ) {
			#ifdef SDCV_TRACKING_DEBUG
			out << "[Delete Tracks End] --------------------" << std::endl;
			#endif
			return;
		}


		// Compute the fraction of the track's age for which it was visible.
		int NbTracks = (int)tracks.size();
		cv::Mat IdMat, consecutiveBackwardDirMat, VelocityMat, consInvisibleFramesMat, lostIndsROI, VisibilityMat;
		
		#ifdef SDCV_TRACKING_DEBUG
		out << "<<<<<<<<<<<<< Visibility" << std::endl;
		out << "Contour: " << roi.getVertices() << std::endl;
		#endif
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
			/*else if (it->lastVisibleFrame != this->FrameCountDebug && sdcv::euclidean(it->estimatedCentroid.back(), *(it->detectedCentroid.end()-3)) > 25.0) {
				it->consInvisibleFrames = 100;
			}*/

			cv::Point2f ptest = (trafficOrientation == sdcv::eOrientationLine::TOP_DOWN ? it->estimatedCentroid.back()
				: it->detectedCentroid.back()
				);
			int v = int(sdcv::distanceToLine(ptest, roi.getEndLineEq().x, roi.getEndLineEq().y, trafficOrientation) < 0 ? 100 : it->consInvisibleFrames);
			/// CAMBIO DE SENTIDO -------------------------------------------------------------------------------------------
			
			consInvisibleFramesMat.push_back(v);
		}

		#ifdef SDCV_TRACKING_DEBUG
		out << "Tracks visibility: "			<< std::endl << VisibilityMat << std::endl;
		out << "Tracks ID: "					<< std::endl << IdMat << std::endl;
		out << "Tracks BackwardDir: "			<< std::endl << consecutiveBackwardDirMat << std::endl;
		out << "Tracks Velocity: "			<< std::endl << VelocityMat << std::endl;
		out << "Tracks consInvisibleFrames: " << std::endl << consInvisibleFramesMat << std::endl;
		out << "lost Indexes in ROI: "		<< std::endl << lostIndsROI << std::endl;
		out << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl << std::endl;
		#endif
		/*std::cout << "Tracks visibility: " << std::endl << VisibilityMat << std::endl;
		std::cout << "Tracks ID: " << std::endl << IdMat << std::endl;
		std::cout << "Tracks BackwardDir: " << std::endl << consecutiveBackwardDirMat << std::endl;
		std::cout << "Tracks Velocity: " << std::endl << VelocityMat << std::endl;
		std::cout << "Tracks consInvisibleFrames: " << std::endl << consInvisibleFramesMat << std::endl; //*/

		// Check backward direction -----------------------------------------------------------------
		cv::Mat lostIndsBackDir = ( (IdMat == 0) & ((consecutiveBackwardDirMat == 1) | 
								  ((VelocityMat >= -3) & (VelocityMat <= 3))));				// >= -2 && <= 2

		/*std::cout << std::endl << "id == 0: " << std::endl << (IdMat == 0) << std::endl;
		std::cout << "backwarddir == 1: " << std::endl << (consecutiveBackwardDirMat == 1) << std::endl;
		std::cout << "velocity >= -2: " << std::endl << (VelocityMat >= -2) << std::endl;
		std::cout << "velocity <= 2: " << std::endl << (VelocityMat <= 2) << std::endl;
		std::cout << "lost indexes backward dir 1:" << std::endl << lostIndsBackDir << std::endl << std::endl; //*/

		lostIndsBackDir = (IdMat != 0)	&	(( (VelocityMat < -1000)	|	(VelocityMat > 1000)	|	
						  ((VelocityMat >= -1)	&	(VelocityMat <= 1)))	|
						  (consecutiveBackwardDirMat == 5))	|	lostIndsBackDir;
		
		/*std::cout << std::endl << "ID != 0: "			<< std::endl << (IdMat != 0)						<< std::endl;
		std::cout << "Velocity < -1000: "				<< std::endl << (VelocityMat < -1000)				<< std::endl;
		std::cout << "Velocity > 1000: "				<< std::endl << (VelocityMat > 1000)				<< std::endl;
		std::cout << "Velocity >= -1: "					<< std::endl << (VelocityMat >= -1)					<< std::endl;
		std::cout << "Velocity <= 1: "					<< std::endl << (VelocityMat <= 1)					<< std::endl;
		std::cout << "consecutiveBackwardDirMat == 5: " << std::endl << (consecutiveBackwardDirMat == 5)	<< std::endl;
		std::cout << "lost Indexes Backward Dir 2:"		<< std::endl << lostIndsBackDir						<< std::endl << std::endl; //*/


		#ifdef SDCV_TRACKING_DEBUG
		out << std::endl << "<<<< Check backward direction" << std::endl;
		out << "Visibility: " << std::endl << VisibilityMat << std::endl << std::endl;
		out << "ID: " << std::endl << IdMat << std::endl;
		out << "Velocity: " << std::endl << VelocityMat << std::endl;
		out << "Consecutive Backward Dir: " << std::endl << consecutiveBackwardDirMat << std::endl;
		out << "lost Indexes Backward Dir:" << std::endl << lostIndsBackDir << std::endl << std::endl;
		out << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl;
		#endif
		
		/*std::cout << "Visibility: " << std::endl << VisibilityMat << std::endl << std::endl;
		std::cout << "ID: " << std::endl << IdMat << std::endl;
		std::cout << "Velocity: " << std::endl << VelocityMat << std::endl;
		std::cout << "Consecutive Backward Dir: " << std::endl << consecutiveBackwardDirMat << std::endl;
		std::cout << "lost Indexes Backward Dir:" << std::endl << lostIndsBackDir << std::endl << std::endl;*/

		// Find the indices of 'lost' tracks  ----------------------------------------------------------
		cv::Mat lostInds = lostIndsBackDir | ((IdMat == 0) & (VisibilityMat < 0.65)) | 
						   (consInvisibleFramesMat >= this->minInvisibleCount) ;

		#ifdef SDCV_TRACKING_DEBUG
		out << std::endl << "<<<< Find the indices of 'lost' tracks" << std::endl;
		out << "Min Invisible Frames: " << this->minInvisibleCount << std::endl;
		out << "Invisible Frames: " << std::endl << consInvisibleFramesMat << std::endl;
		out << "lost Indexes:" << std::endl << lostInds << std::endl << std::endl;
		out << "<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl << std::endl;
		#endif
		/*std::cout << "Min Invisible Frames: " << this->minInvisibleCount << std::endl;
		std::cout << "Invisible Frames: " << std::endl << consInvisibleFramesMat << std::endl;*/
		/* ------------------------------------------------------------------------------*/
		



		// SEARCH IN THIS CODE THE VALUE OF "lostTrack = tracks.begin()" SEEMS FISHY TO ME
		int nonZero = cv::countNonZero( lostInds );
		/*std::cout << "There are " << nonZero << " lost indexes : \n" << std::endl;
		std::cout << "Lost Indexes:" << std::endl << lostInds << std::endl;*/
		
		#ifdef SDCV_TRACKING_DEBUG
		out << "Lost Indexes non zero values: \n" << nonZero << std::endl;
		#endif
		if( nonZero > 0) {
			/*std::vector<cv::Point> ContourPt;
			ContourPt.push_back(roi.getLineDetection().at(0));	// First point
			ContourPt.push_back(cv::Point(roi.getDivLineLane().at<short>(0, 2), roi.getDivLineLane().at<short>(0, 3)));	// Second point
			int NbDvLnIdx = roi.getNumLanes() - 1;
			ContourPt.push_back(cv::Point(roi.getDivLineLane().at<short>(NbDvLnIdx, 2), roi.getDivLineLane().at<short>(NbDvLnIdx, 3))); // Third point
			ContourPt.push_back(roi.getLineDetection().at(1));	// Fourth point*/
			double m = (roi.getLineDetection().at(1).y - roi.getLineDetection().at(0).y) / (roi.getLineDetection().at(1).x - roi.getLineDetection().at(0).x);
			double b = roi.getLineDetection().at(1).y - m*roi.getLineDetection().at(1).x;

			std::vector<sdcv::Track>::iterator lostTrack = tracks.begin();
			for(int idx = 0; idx < lostInds.rows; idx++, lostTrack++) {

				// See if the object is after ending line -----------------------------------------------------------------------------------
				/// CAMBIO DE SENTIDO -------------------------------------------------------------------------------------------
				cv::Point2f ptest = (trafficOrientation == sdcv::eOrientationLine::TOP_DOWN ? lostTrack->estimatedCentroid.back()
					: lostTrack->detectedCentroid.back()
					);
				double value = sdcv::distanceToLine(ptest, roi.getEndLineEq().x, roi.getEndLineEq().y, trafficOrientation);
				/// CAMBIO DE SENTIDO -------------------------------------------------------------------------------------------

				//std::cout << "lostInds:" << std::endl << lostInds << std::endl;
				if (value < 0)  lostInds.at<unsigned char>(idx) = 255;
				//std::cout << "lostIndsNew:" << std::endl << lostInds << std::endl;
				///std::cout << "Track with id [" << lostTrack->id << "], centroid " << lostTrack->estimatedCentroid.back();
				///std::cout << " is after ending line [" << (value < 0 ? "YES" : "NO") << "]" << std::endl;

				bool flag = false;
				bool flagId = ((unsigned int)lostInds.at<unsigned char>(idx)) && (lostTrack->id);

				if( flagId ) {
					if (lostTrack->estimatedArea == 0) {
						// compute estimated area only if the object is created after line detection
						double sumNormAreas = std::accumulate(lostTrack->normAreas.begin(), lostTrack->normAreas.end(), 0.0);
						lostTrack->estimatedArea = sumNormAreas/(double)lostTrack->normAreas.size();
					}
				}


				// See if the object is before line detection -------------------------------------------
				/// CAMBIO DE SENTIDO -------------------------------------------------------------------------------------------
				bool inPoly = (bool)(sdcv::distanceToLine(ptest, m, b, trafficOrientation) > 0 ? true : false);
				/// CAMBIO DE SENTIDO -------------------------------------------------------------------------------------------
				
				///std::cout << "Track with id [" << lostTrack->id << "] and centroid " << lostTrack->estimatedCentroid.back();
				///std::cout << " is before line detection [" << inPoly << "]" << std::endl;
				#ifdef SDCV_TRACKING_DEBUG
				out << "object is before line detection: " << (inPoly ? "Yes" : "No") << std::endl;
				#endif
				
				if( flagId && (inPoly || (lostTrack->totalVisibleFrames == minVisibleCount)) && !lostTrack->isClassified ) {
					flag = true;
					classifier->update( 3 );
					if( FP_ID.back() == 0 ) {
						FP_ID.clear();
						FP_ID.push_back( lostTrack->id );
					} else FP_ID.push_back( lostTrack->id );
				}
				
				// CLASSIFICATION --------------------------------------------------------------------------------------------
				//		VEHICLE OCCLUSION INDEX ------------------------------------------------------------------------------
				if( flagId && !flag ) assignedTracks.at(lostTrack->id - 1)[2] = classifier->update(lostTrack->estimatedArea);
				// -----------------------------------------------------------------------------------------------------------

			}
		} /* nonZero > 0 */

		// Delete lost tracks ------------------------------------------------------------------
		#ifdef SDCV_TRACKING_DEBUG
		out << std::endl << "<<<<<<<< Delete lost tracks" << std::endl;
		#endif

		/*
			tracks.erase( std::remove_if(tracks.begin(), tracks.end(), [](sdcv::Track const& obj) { return obj.isDelete(); }), tracks.end());
		*/
		std::vector<sdcv::Track> tmp;
		std::vector<sdcv::Track>::iterator itTrack = tracks.begin();
		for(int i = 0; i < lostInds.cols, itTrack != tracks.end(); i++, ++itTrack) {
			if( !lostInds.at<unsigned char>(i) ) {
				tmp.push_back(*itTrack);
				#ifdef SDCV_TRACKING_DEBUG
				out << "Track #" << i << " wasn't delete!" << std::endl;
				itTrack->print();
				out << std::endl;
				#endif
			}
			else {
				if (itTrack->id != 0) {
					/*
					// lostIndsBackDir
					bool someC = false;
					std::cout << "Track {" << itTrack->id << "} deleted by: ";
					if (itTrack->velocity < -1000 && itTrack->velocity > 1000) {
						std::cout << itTrack->velocity << " < -1000 | > 1000" << std::endl;
						someC = true;
					}
					if (itTrack->velocity >= -1 && itTrack->velocity <= 1) {
						std::cout << itTrack->velocity << " >= -1 -1000 | <= 1" << std::endl;
						someC = true;
					}
					if (itTrack->consecutiveBackwardDir == 5) {
						std::cout << itTrack->consecutiveBackwardDir << " == 5" << std::endl;
						someC = true;
					}
					// Visibility
					if (itTrack->consInvisibleFrames >= this->minInvisibleCount) {
						std::cout << "Invisibility: " << itTrack->consInvisibleFrames << " >= " << this->minInvisibleCount << std::endl << std::endl;
						someC = true;
					}
					// End of line
					if (!someC) std::cout << "End of line" << std::endl;
					std::cin.get();
					// */
					/*std::cout << "Track.ID: " << itTrack->id << std::endl;
					std::cout << "Detected.Size: " << itTrack->detectedCentroid.size() << std::endl;
					std::cout << "Estimated.Size: " << itTrack->estimatedCentroid.size() << std::endl << std::endl;*/
					/*std::string fname = "DATA/" + this->roi.getName() + "/TEST/Track" + std::to_string(itTrack->id) + ".csv";
					int file_counter = 1;
					while (sdcv::fs_exist(fname)) {
						fname = "DATA/" + this->roi.getName() + "/TEST/Track" + std::to_string(itTrack->id) + "_" + std::to_string(file_counter++) + ".csv";
					}
					std::string header = "detectedCentroidX,detectedCentroidY";
					sdcv::fs_write(fname, cv::Mat(itTrack->detectedCentroid), sdcv::SDCV_FS_WRITE, header);*/
				}

			#ifdef SDCV_TRACKING_DEBUG
				out << "Track #" << i << " was delete!" << std::endl;

				itTrack->print();
				out << std::endl;
			#endif
			}
			
		}
		tracks = tmp;

		#ifdef SDCV_TRACKING_DEBUG
		out << std::endl << "<<<<<<<<<<<<<<<<<<<<<<<<<" << std::endl << std::endl;
		out << "[Delete Tracks End] --------------------" << std::endl;
		#endif
	}


	cv::Mat Tracker::getAssignmentCostMatrix( void ) {
		cv::Mat MeasurementsCentroid = cv::Mat(blobLst.getCentroids(), true);
		cv::Mat CostMatrix;
		
		// DEBUG
		#ifdef SDCV_TRACKING_DEBUG
		out << "------------------ Assigment Problem Solver ------------------" << std::endl;
		out << "Detected Centroids:" << std::endl << MeasurementsCentroid << std::endl << std::endl;
		#endif
		/*std::cout << "------------------ Assigment Problem Solver ------------------" << std::endl;
		std::cout << "Detected Centroids:" << std::endl << MeasurementsCentroid << std::endl << std::endl;
		std::cout << "Predicted Centroid Kalman:" << std::endl;*/
		// END DEBUG
		
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

		// DEBUG
		#ifdef SDCV_TRACKING_DEBUG
		out << "Cost Matrix:" << std::endl << CostMatrix << std::endl << std::endl;
		#endif
		/*std::cout << std::endl << "Cost Matrix:" << std::endl << CostMatrix << std::endl << std::endl;*/
		// END DEBUG

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

			///std::cout << CostMatrix.at<short>(i, idx) << " <= " << costOfNonAssigment << std::endl;
			if (it != rowVector.end() && CostMatrix.at<short>(i, idx) <= costOfNonAssigment) assignments.push_back(cv::Point(i, idx));
			else unassignedTracks.push_back( i );
		}

		/*if (find) {
			std::cout << "Track: " << find_for_id  << ", " << trackF->detectedCentroid.back() << std::endl;
			std::cout << "Cost matrix at {" << this->FrameCountDebug << "} frame" << std::endl << CostMatrix << std::endl;
			std::cout << "Solved cost matrix at {" << this->FrameCountDebug << "} frame" << std::endl << solvedMatrix << std::endl;
			for()
		}*/

		// Return the unassignedDetections
		for(int i = 0; i < solvedMatrix.cols; i++) {
			if(cv::countNonZero(solvedMatrix.col(i)) == solvedMatrix.rows )
				unassignedDetections.push_back(i);
		}

		// Show results
		#ifdef SDCV_TRACKING_DEBUG
		out << "Solved matrix = " << std::endl << solvedMatrix << std::endl << std::endl;
		out << "{Results}" << std::endl;
		out << "Matches = "				<< std::endl << cv::Mat(assignments) << std::endl << std::endl;
		out << "unassigned Tracks = "		<< std::endl << cv::Mat(unassignedTracks) << std::endl << std::endl;
		out << "unassigned Detections = " << std::endl << cv::Mat(unassignedDetections) << std::endl << std::endl;
		out << "[Assignament Problem Solver End]----------------" << std::endl << std::endl << std::endl;
		#endif

		/// DEBUG --------------------------------------------------------------------------------------------------
		/*std::cout << "Solved matrix:" << std::endl << solvedMatrix << std::endl << std::endl;
		std::cout << "{Results}" << std::endl;
		std::cout << "Matches (Track, Detection):" << std::endl << cv::Mat(assignments) << std::endl << std::endl;
		std::cout << "unassigned Tracks:" << std::endl << cv::Mat(unassignedTracks) << std::endl << std::endl;
		std::cout << "unassigned Detections:" << std::endl << cv::Mat(unassignedDetections) << std::endl << std::endl;
		std::cout << "---------- Assignament Problem Solver End ----------" << std::endl << std::endl << std::endl;*/
		/// DEBUG --------------------------------------------------------------------------------------------------
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