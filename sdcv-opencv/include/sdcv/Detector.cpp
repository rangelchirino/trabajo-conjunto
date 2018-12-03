/* ---------------------------*/
/*       Library Include       */
/* ---------------------------*/
#include "Detector.hpp"

static sdcv::eOrientationLine trafficOrientation = sdcv::eOrientationLine::BOTTOM_UP; 

/*!
 * @namespace	sdcv.
 * @brief		Vehicle Detection and Classification System.
 */
namespace sdcv {
	typedef struct {
		double normArea;	// !< An area that was normalized with the lane width
		double laneWidth;	// !< Lane With
		double x_DL1;		// !< First lane x coordinated
		double x_DL2;		// !< Last lane x coordinated
		int i;				// !< Number of lane in which an object was found

	} NormArea_t;

	typedef std::tuple<double, double, int, double, double> laneDescriptor;

	static NormArea_t normalizeArea(DivLaneLine_t DivLaneLine, sdcv::Blob &blob);
	static double inLine(double slope, double b, cv::Point point);






	// ----------------------------------------------------
	Detector::Detector() {
		BGModel = cv::createBackgroundSubtractorMOG2(500, 16.0, false);
		BGModel->setNMixtures(3);

		flags = (int)OUTPUT_AREA | (int)OUTPUT_BBOX | (int)OUTPUT_CENTROID;

		minBlobArea = 100;
		maxBlobArea = std::numeric_limits<int>::infinity();

		erodeKernel = cv::Mat::ones(cv::Size(2, 2), CV_8U);
		dilateKernel = cv::Mat::ones(cv::Size(3, 3), CV_8U);

		ocParams = { 1.12, 1.1, 2.27, 0.002 };

		NbOcclusions = 0;

		NFramesSha = 0;
	}

	Detector::Detector(int nmixtures, int minBlobArea, int maxBlobArea, int flags) {
		if (algOcclusion == OCC_DEF_CONTOUR) BGModel = cv::createBackgroundSubtractorMOG2(300, 32.0, false);
		else BGModel = cv::createBackgroundSubtractorMOG2();
		BGModel->setNMixtures(nmixtures);

		this->minBlobArea = minBlobArea;
		this->maxBlobArea = maxBlobArea;

		ocParams = { 1.012, 1.01, 2.27, 0.06 };
		algOcclusion = OCC_NORM_AREA;

		erodeKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2));
		dilateKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));

		NFrame = 0;
		j = 0;

		NbOcclusions = 0;

		NFramesSha = 0;
	}


	// Get methods
	std::vector<sdcv::Blob> Detector::getBlobs(void)
	{
		return blobs;
	}

	int Detector::getMinArea(void)
	{
		return minBlobArea;
	}

	int Detector::getMaxArea(void)
	{
		return maxBlobArea;
	}

	sdcv::ROI Detector::getRoi(void)
	{
		return roi;
	}

	int Detector::getOcclusions(void)
	{
		return NbOcclusions;
	}

	void Detector::getBackground(cv::OutputArray bg)
	{
		BGModel->getBackgroundImage(bg);
	}

	// Set methods
	void Detector::setMOG(cv::Ptr<cv::BackgroundSubtractorMOG2> pMOG)
	{
		BGModel = pMOG;
	}

	void Detector::setErodeKernel(cv::Mat kernel)
	{
		erodeKernel = kernel;
	}

	void Detector::setDilateKernel(cv::Mat kernel)
	{
		dilateKernel = kernel;
	}

	void Detector::setMinBlobArea(int MinArea)
	{
		minBlobArea = MinArea;
	}

	void Detector::setMaxBlobArea(int MaxArea)
	{
		maxBlobArea = MaxArea;
	}

	void Detector::setFlags(int flags)
	{
		this->flags = flags;
	}

	void Detector::setFlag(DetectionFlags Flag, bool val)
	{
		if (val) flags |= (int)Flag;
		else flags &= ~((int)Flag);

		if (Flag == (int)SHAREM)	BGModel->setDetectShadows(val);
	}

	void Detector::setOcclusion(OcclusionHandler algflag)
	{
		algOcclusion = algflag;
		setFlag(HOCC, true);
	}

	void Detector::setROI(sdcv::ROI roi) {
		this->roi = roi;
		roiArea = (int)cv::contourArea(roi.getVertices());
	}

	void Detector::setOcclusionParam(std::vector<double> Params) {
		CV_Assert(Params.size() == 4);
		ocParams = Params;
	}


	// Action methods
	bool Detector::isReady(void) {

		return bool((!this->BGModel->empty()) & (!this->dilateKernel.empty()) & (!this->erodeKernel.empty()));
	}





	void Detector::apply(cv::Mat frame, cv::OutputArray mask) {
		cv::Mat foreground, frameRoi, frameSmooth;

		NbOcclusions = 0;

		CV_Assert(!frame.empty());

		/* Apply Mixture of Gaussians Model */
		BGModel->apply(frame, frameRoi);

		/* Se aplica una mascara de la region de interes */
		CV_Assert(!roi.getMask().empty());
		roi.apply(frameRoi, foreground);

		// Remove shadows
		if (flags & (int)SHAREM)
			cv::threshold(foreground, foreground, 128, 255, cv::THRESH_BINARY);



		/* Postfiltrado */
		if ((flags & (int)HOCC) && algOcclusion == OCC_DEF_CONTOUR)
		{
			cv::morphologyEx(foreground, foreground, cv::MORPH_CLOSE, erodeKernel);
		}
		else
		{
			cv::erode(foreground, foreground, erodeKernel);
			cv::dilate(foreground, foreground, dilateKernel);
		}

		/* Get Blobs */
		BlobAnalysis(foreground, mask);
		CV_Assert(!mask.empty());

		/* Occlusion Handling */
		if (flags & (int)HOCC) {
			// Copy Blob vector
			std::vector<sdcv::Blob> PrevBlobs = blobs;
			cv::Mat OclussionMask;

			if (algOcclusion == OCC_DEF_CONTOUR) NbOcclusions = convexityDefectsOcclusion(mask, OclussionMask);
			else NbOcclusions = normAreasOcclusion(mask, OclussionMask);

			OclussionMask.copyTo(mask);

			// Search for the blobs which will be created by the OH algorithm
			if (blobs.size() > PrevBlobs.size()) {
				auto newAreas = sdcv::BlobList::getAreas(blobs);
				auto oldAreas = sdcv::BlobList::getAreas(PrevBlobs);
				auto blob = blobs.begin();
				for (auto area : newAreas) {
					if (std::find(oldAreas.begin(), oldAreas.end(), area) == oldAreas.end())
						blob->setOcclusion(true);

					++blob;
				}
			}

		} /* Occlusion Handling */

		/* END OF DETECTOR */
		CV_Assert(!mask.empty());
		NFrame++;
	}

	void Detector::draw(cv::Mat frame, int drwflags) {
		// Draw ROI parameters
		cv::line(frame, roi.getLineDetection().at(0), roi.getLineDetection().at(1), cv::Scalar(0, 255, 0), 2);
		cv::line(frame, roi.getEndLine().at(0), roi.getEndLine().at(1), cv::Scalar(0, 0, 255), 2);
		cv::circle(frame, roi.getCenterLineDetection(), 2, cv::Scalar(255, 0, 0), -1);

		// Draw features (BBox, Label, etc)
		if (drwflags & (int)DRAW_BBOX) {
			for (auto blob : blobs) {
				cv::rectangle(frame, blob.getBBox(), cv::Scalar(0, 255, 255), 1);
				cv::Mat rec = frame(blob.getBBox());
				cv::Mat color(rec.size(), CV_8UC3, cv::Scalar(0, 255, 255));
				cv::addWeighted(color, 0.2, rec, 1.0, 0.0, rec);
				if (drwflags & (int)DRAW_CENTROID) cv::circle(frame, cv::Point(blob.getCentroid()), 2, cv::Scalar(255, 0, 0));
			}
		}
	}


	// Destructor
	Detector::~Detector() {}


	void Detector::BlobAnalysis(cv::InputArray mask, cv::OutputArray blobMask) {
		std::vector< std::vector<cv::Point> > contours;
		cv::Mat maskTmp;

		blobs.clear();
		mask.copyTo(maskTmp);

		/* Analisis de componentes conectados */
		cv::findContours(maskTmp, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

		cv::Mat ForegroundMask = cv::Mat::zeros(maskTmp.size(), CV_8UC1);

		for (int i = 0; i < (int)contours.size(); i++)
		{
			int area = (int)cv::contourArea(contours[i]);
			if (area > minBlobArea && area < maxBlobArea) {
				cv::drawContours(ForegroundMask, contours, (int)i, CV_RGB(255, 255, 255), cv::FILLED);
				if (!(flags & OUTPUT_AREA)) area = -1;
				blobs.push_back(sdcv::Blob(contours[i], (int)i, area));
			}
		}
		ForegroundMask.copyTo(blobMask);
	}

	int Detector::convexityDefectsOcclusion(cv::InputArray mask, cv::OutputArray maskOclussion) {
		int NbOfOcclusions = 0;
		bool hasOcclusion = false;
		cv::Mat tmp;

		mask.copyTo(tmp);

		int NumberOfBlob = 0;	// EVALUATION & TESTING VARIABLE

		for (auto &blob : blobs)
		{
			normalizeArea(roi.getLaneData(), blob);

			std::vector<cv::Point> hull;

			/* Convex hull */
			cv::convexHull(blob.getContour(), hull);
			double hullArea = cv::contourArea(hull);
			double solidity = blob.getArea() / (double)hullArea;

			/* First condition:
			 * If a blob has a solidity less than 0.9 is a candidate of occlusion.
			 * This threshold is based on that non occluded blobs has its solidity almost equals to 1.0
			 */
			if (solidity < 0.90) {
				// Approximate contour using Dougglas .... [X] for simplifying occlusion detection.
				std::vector<cv::Point> approxContour;
				double perimeter = cv::arcLength(blob.getContour(), true);
				cv::approxPolyDP(blob.getContour(), approxContour, 0.04*perimeter, true);
				if (approxContour.size() < 5) approxContour = blob.getContour();

				// Ellipse fitting needs at least 5 points.
				if (approxContour.size() > 5) {
					cv::RotatedRect rt = cv::fitEllipse(approxContour);

					// Height over width ratio.
					// Some times ellipse fitting triggers to an extreme point, in that case is better to take the blob height
					//double H = ((double)(rt.size.height / (double)blob->getBBox().height) > 2.0 ? (double)blob->getBBox().height : (double)rt.size.height);
					double eccentricity = (double)(rt.size.height / (double)rt.size.width);

					/* Second condition:
					 * If the blob orientation or angle is less than lane line division orientation there is an occlusion
					 * For camera angle sometimes some angles > 90 are consider between 0 ~ 5.
					 *
					 * Third conditions:
					 * If height over width ratio is greater than 2.0 and its solidity is less than 0.6 there is an occlusion
					 * This conditions prevents large and small vehicles like motocycles and trailers that theirs relation heigth/width
					 * is greater than 2.0, will be divided.
					 */
					 //if (eccentricity > 1.8 && solidity < 0.70 && rt.angle < 10.0) rt.angle = 90 - rt.angle;
					if ((rt.angle > 5.0 && rt.angle < 90.0 && (eccentricity > 1.4 || solidity < 0.65)) || (eccentricity > 1.8 && solidity < 0.65)) {
						// Concave curve expansion analysis [REF], for convexity defects.
						std::vector<int> contoursHullIndex;
						std::vector<cv::Vec4i> defects;
						cv::convexHull(approxContour, contoursHullIndex);
						cv::convexityDefects(approxContour, contoursHullIndex, defects);

						// If there are more than a defect point select the one that is near centroid
						if (defects.size() > 1) {
							cv::Point minDistDefPt, minDistDefProj;
							float minDistDef = std::numeric_limits<float>::infinity();
							cv::Rect bbox;
							for (auto defect : defects)
							{
								hasOcclusion = true;

								cv::Point defectPoint = approxContour[defect[2]], nPt;
								double defLen = (double)(defect[3] / (double)256.0);

								cv::Point2f df = (cv::Point2f)defectPoint - blob.getCentroid();
								float defDist = std::sqrt(df.x*df.x + df.y*df.y);
								if (defDist < minDistDef) {
									cv::Point middlePt = (approxContour[defect[0]] + approxContour[defect[1]]) / 2;
									int distancePt = ((int)(defLen)+1) * 4;
									if (middlePt.x > defectPoint.x && middlePt.y > defectPoint.y)		nPt = cv::Point(-distancePt, -distancePt);
									else if (middlePt.x > defectPoint.x && middlePt.y < defectPoint.y)	nPt = cv::Point(-distancePt, distancePt);
									else if (middlePt.x < defectPoint.x && middlePt.y > defectPoint.y)	nPt = cv::Point(distancePt, -distancePt);
									else if (middlePt.x < defectPoint.x && middlePt.y < defectPoint.y)	nPt = cv::Point(distancePt, distancePt);
									nPt = nPt + defectPoint;

									// Update lens
									minDistDef = defDist;
									minDistDefPt = defectPoint;
									minDistDefProj = nPt;
									bbox = blob.getBBox();
								}
							}	// END FOR
							cv::clipLine(bbox, minDistDefPt, minDistDefProj);
							cv::line(tmp, minDistDefPt, minDistDefProj, CV_RGB(0, 0, 0), 2);
						}
						// Otherwise
						else if (defects.size()) {
							auto defect = defects.at(0);
							NbOfOcclusions++;
							hasOcclusion = true;

							cv::Point defectPoint = approxContour[defect[2]], nPt;
							double defLen = (double)(defect[3] / (double)256.0);

							cv::Point middlePt = (approxContour[defect[0]] + approxContour[defect[1]]) / 2;
							int distancePt = (int)(defLen * 4.0);
							if (middlePt.x > defectPoint.x && middlePt.y > defectPoint.y)		nPt = cv::Point(-distancePt, -distancePt);
							else if (middlePt.x > defectPoint.x && middlePt.y < defectPoint.y)	nPt = cv::Point(-distancePt, distancePt);
							else if (middlePt.x < defectPoint.x && middlePt.y > defectPoint.y)	nPt = cv::Point(distancePt, -distancePt);
							else if (middlePt.x < defectPoint.x && middlePt.y < defectPoint.y)	nPt = cv::Point(distancePt, distancePt);
							nPt = nPt + defectPoint;
							cv::clipLine(blob.getBBox(), defectPoint, nPt);
							cv::line(tmp, defectPoint, nPt, CV_RGB(0, 0, 0), 2);

						}				// END ELSE-IF
					}			// END IF Occlusion Condition 2 and 3
				}		// END IF approxContour.size() > 5
			}	// END IF solidity < 0.90

			NumberOfBlob++;
		}	// END FOR

		// Test if there was an occlusion
		if (hasOcclusion) {
			BlobAnalysis(tmp, maskOclussion);
			for (auto &blob : blobs) {
				sdcv::normalizeArea(roi.getLaneData(), blob);
			}
		}
		else tmp.copyTo(maskOclussion);

		return NbOfOcclusions;
	}

	int Detector::normAreasOcclusion(cv::InputArray mask, cv::OutputArray maskOclussion) {
		int NbOfOcclusions = 0;
		int NbBlobs = (int)blobs.size();

		DivLaneLine_t LaneData = this->roi.getLaneData();
		cv::Mat foreground;
		mask.copyTo(foreground);

		/* Linea de detección (ecuación de la linea recta) */
		double m = (roi.getLineDetection().at(1).y - roi.getLineDetection().at(0).y) / (roi.getLineDetection().at(1).x - roi.getLineDetection().at(0).x);
		double b = roi.getLineDetection().at(1).y - m * roi.getLineDetection().at(1).x;

		for (auto blob = blobs.begin(); blob != blobs.end(); ++blob) {
			double overlapRatio;

			/* Extrae area normalizada y el carril donde se ubica */
			NormArea_t NormArea = normalizeArea(LaneData, *blob);
			if (NbBlobs > 1 && blob < blobs.end() - 1) {
				auto blobNext = blob + 1;

				int totalArea = blob->getBBox().area() + blobNext->getBBox().area();
				overlapRatio = (double)cv::Rect(blob->getBBox() & blobNext->getBBox()).area() / (double)totalArea;

			}
			else  overlapRatio = 0.0;

			// Test whether the vehicle is before detection line
			bool in2 = (bool)(sdcv::distanceToLine(blob->getCentroid(), m, b, trafficOrientation) > 0 ? true : false);

			// Comprueba si la relación ancho del blob con respecto al ancho del carril es maypr que cierto umbral
			bool c1 = ((blob->getBBox().width / (double)NormArea.laneWidth) > ocParams.at(0)) && (NormArea.normArea < ocParams.at(1)); ///&& in1;
			bool c2 = in2 && ((double)(blob->getBBox().width / (double)NormArea.laneWidth) > ocParams.at(2));

			// Evalua las condiones
			if ((c1 || c2 || (overlapRatio > ocParams.at(3)))) {
				cv::Point2d dist(blob->getCentroid().x - NormArea.x_DL1, NormArea.x_DL2 - blob->getCentroid().x);
				cv::Mat foregroundOclussion;

				if ((dist.x < dist.y) && (NormArea.i != 0)) {
					NbOfOcclusions++;
					cv::Point pt1(LaneData.lineLane.at<unsigned short>(NormArea.i, 0), LaneData.lineLane.at<unsigned short>(NormArea.i, 1));
					cv::Point pt2(LaneData.lineLane.at<unsigned short>(NormArea.i, 2), LaneData.lineLane.at<unsigned short>(NormArea.i, 3));
					cv::clipLine(blob->getBBox(), pt1, pt2);
					cv::line(foreground, pt1, pt2, CV_RGB(0, 0, 0), 2);

				}
				else if ((dist.x > dist.y) && ((NormArea.i + 1) < roi.getNumLanes() - 1)) {
					NbOfOcclusions++;
					cv::Point pt1(LaneData.lineLane.at<unsigned short>(NormArea.i + 1, 0), LaneData.lineLane.at<unsigned short>(NormArea.i + 1, 1));
					cv::Point pt2(LaneData.lineLane.at<unsigned short>(NormArea.i + 1, 2), LaneData.lineLane.at<unsigned short>(NormArea.i + 1, 3));
					cv::clipLine(blob->getBBox(), pt1, pt2);
					cv::line(foreground, pt1, pt2, CV_RGB(0, 0, 0), 2);
				}

			}

		}

		// Update Blobs
		if (NbOfOcclusions > 0) {
			cv::Mat ForegroundMask;
			BlobAnalysis(foreground, maskOclussion);

			/* Update Normalized Area */
			for (auto &blob : blobs) {
				normalizeArea(LaneData, blob);
			}
		}
		else foreground.copyTo(maskOclussion);

		return NbOfOcclusions;
	}



	// --------------------------------------------------------------------------
	NormArea_t normalizeArea(DivLaneLine_t DivLaneLine, sdcv::Blob &blob) {
		NormArea_t NormArea;

		NormArea.laneWidth = std::numeric_limits<double>::infinity();
		NormArea.i = DivLaneLine.NbLineLanes - 1;

		for (int i = 0; i < DivLaneLine.NbLineLanes - 1; i++) {
			NormArea.x_DL1 = (blob.getCentroid().y - DivLaneLine.b_lineLanes.at(i)) / (double)DivLaneLine.m_lineLanes.at(i);
			NormArea.x_DL2 = (blob.getCentroid().y - DivLaneLine.b_lineLanes.at(i + 1)) / (double)DivLaneLine.m_lineLanes.at(i + 1);

			if (NormArea.x_DL2 > (double)blob.getCentroid().x) {
				NormArea.laneWidth = abs(NormArea.x_DL1 - NormArea.x_DL2);
				NormArea.i = i;
				break;
			}
		}

		NormArea.normArea = blob.getArea() / (double)(NormArea.laneWidth * NormArea.laneWidth);
		blob.setNormArea(NormArea.normArea);

		return(NormArea);
	}




	double inLine(double slope, double b, cv::Point point) {
		return (point.x*slope + b - point.y);
	}
};
