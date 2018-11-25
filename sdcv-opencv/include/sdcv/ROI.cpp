#include "ROI.h"

namespace sdcv {
	/* PUBLIC METHODS *************************************************************** */
	// Constructor & Destructor
	ROI::ROI() {
		this->name = "ROIdata";
		this->NbLineLanes = 0;
		this->NbRegions = 0;
	}

	ROI::ROI(cv::String name) {
		this->name = name;
		this->NbLineLanes = 0;
		this->NbRegions = 0;
	}

	ROI::~ROI() {

	}


	// Get methods
	cv::Rect ROI::getBbox(void) { return bbox; }
	std::vector<cv::Point> ROI::getVertices(void) { return vertices; }
	cv::Mat ROI::getMask(void) { return mask; }
	int ROI::getArea(void) { return area; }
	std::string ROI::getName(void) {
		return name;
	}
	
	DivLaneLine_t ROI::getLaneData(void) {
		DivLaneLine_t DivLaneLines;

		DivLaneLines.lineLane = lineLane;
		DivLaneLines.b_lineLanes = b_lineLanes;
		DivLaneLines.m_lineLanes = m_lineLanes;
		DivLaneLines.DL_mask = DL_mask;
		DivLaneLines.NbLineLanes = NbLineLanes;

		return DivLaneLines;
	}

	int ROI::getNumLanes(void) 
	{ 
		return NbLineLanes; 
	}

	cv::Mat ROI::getDivLineLane(void) 
	{
		return lineLane;
	}
	
	std::vector< double > ROI::getSlopeLane(void) 
	{ 
		return m_lineLanes; 
	}

	std::vector< double > ROI::getIntersecLane(void) 
	{ 
		return b_lineLanes; 
	}

	cv::Mat ROI::getDivLaneMask(int Nb) {
		CV_Assert(Nb < (int)DL_mask.size());
		return DL_mask.at(Nb);
	}

	std::vector< cv::Mat > ROI::getDivLaneMask(void) 
	{ 
		return DL_mask; 
	}

	std::vector< cv::Point > ROI::getLineDetection(void) 
	{ 
		return lineDetection; 
	}
	
	cv::Point2f ROI::getCenterLineDetection(void) 
	{ 
		return cLineDetection; 
	}

	std::vector<cv::Point> ROI::getEndLine(void) 
	{ 
		return EndLine; 
	}
	cv::Point2d ROI::getEndLineEq(void) 
	{ 
		return EndLineEquation;
	}

	int ROI::getNbRegions(void) 
	{
		return NbRegions;
	}
	std::vector< std::vector<cv::Point> > ROI::getRegions(void) 
	{
		return Regions; 
	}
	std::vector< cv::Point2d > ROI::getRegionLinesEquation() 
	{ 
		return RegionLinesEquation; 
	}

	// Set methods
	void ROI::setName(cv::String name) 
	{
		this->name = name;
	}

	void ROI::setVertices(cv::Mat frame) {
		cv::namedWindow("ROI2", cv::WINDOW_KEEPRATIO);
		cv::imshow("ROI2", frame);
		
		// Setting ROI coordinates
		sdcv::roipoly(frame, vertices, mask);
		CV_Assert(vertices.size() == 4);

		std::vector< std::vector<cv::Point> > contours;
		cv::Mat roimask;
		mask.copyTo(roimask);
		cv::findContours(roimask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		CV_Assert(contours.size() == 1);

		// Ramer-Douglas-Peucker algorithm
		std::vector<cv::Point> approxPolygon;
		cv::approxPolyDP(contours[0], approxPolygon, 1.0, true);
		CV_Assert(approxPolygon.size() == 4 || approxPolygon.size() == 5);

		Blob blob(contours[0], 0);
		area = blob.getArea();
		bbox = blob.getBBox();

		// Save data
		save();
	}

	void ROI::setDetectionLine(cv::Mat frame) {
		cv::namedWindow("ROI2", cv::WINDOW_KEEPRATIO);
		cv::imshow("ROI2", frame);

		lineDetection = sdcv::imLine(frame, "ROI2", cv::Point(-1, -1));
		cv::line(frame, lineDetection.front(), lineDetection.back(), cv::Scalar(255, 0, 0), 2);
		cv::imshow("ROI2", frame);
		cv::waitKey();

		// x_minor to x_major
		if (lineDetection.at(0).x > lineDetection.at(1).x) {
			cv::Point pt = lineDetection.at(0);

			lineDetection.at(0) = lineDetection.at(1);
			lineDetection.at(1) = pt;
		}

		cLineDetection = cv::Point2f((lineDetection.front().x + lineDetection.back().x) / (float)2, (lineDetection.front().y + lineDetection.back().y) / (float)2);
		std::cout << "detection line: {" << lineDetection.front() << " | " << lineDetection.back() << "}" << std::endl << std::endl;

		// Save data
		save();
	}

	void ROI::setEndingLine(cv::Mat frame) {
		cv::namedWindow("ROI2", cv::WINDOW_KEEPRATIO);
		cv::imshow("ROI2", frame);

		EndLine = imLine(frame, "ROI2");
		cv::line(frame, EndLine.front(), EndLine.back(), CV_RGB(255, 255, 0), 2);
		cv::imshow("ROI2", frame);

		double slope, b;
		slope = (EndLine.back().y - EndLine.front().y) / (double)(EndLine.back().x - EndLine.front().x);
		b = EndLine.back().y - slope*EndLine.back().x;
		EndLineEquation = cv::Point2d(slope, b);

		std::cout << "End Line: {" << std::endl << "\t" << EndLine.front() << "	| " << EndLine.back() << std::endl;
		std::cout << "\tEquation: Y = " << EndLineEquation.x << "X + " << EndLineEquation.y << std::endl << "}" << std::endl;

		// Save data
		save();
	}

	void ROI::setRegions(cv::Mat frame, int NbRegions, cv::Point vertexIdx) {
		cv::namedWindow("ROI2", cv::WINDOW_KEEPRATIO);
		cv::imshow("ROI2", frame);

		CV_Assert(NbRegions >= 0);
		this->NbRegions = NbRegions + 1;
		std::vector<cv::Point> tmp;
		tmp.push_back(vertices.at(vertexIdx.x));
		tmp.push_back(vertices.at(vertexIdx.y));
		Regions = sdcv::imRegions(frame, "ROI2", NbRegions);
		Regions.insert(Regions.begin(), tmp);

		double slope, b;
		for (int i = 0; i < NbRegions; i++) {
			slope = (Regions.at(i).back().y - Regions.at(i).front().y) / (double)(Regions.at(i).back().x - Regions.at(i).front().x);
			b = Regions.at(i).back().y - slope*Regions.at(i).back().x;
			RegionLinesEquation.push_back(cv::Point2d(slope, b));
			std::cout << "Region #" << i << ": {" << std::endl << "\t" << this->Regions.at(i).front() << " | " << Regions.at(i).back() << std::endl;
			std::cout << "\tEquation: Y = " << RegionLinesEquation.at(i).x << "X + " << RegionLinesEquation.at(i).y << std::endl << "}" << std::endl;
		}

		// Save data
		save();
	}


	// Action methods
	void ROI::setup(cv::String videonamePath, int NbDivLines, int NbRegions, bool bLoad) {
		if (bLoad && load()) return;

		// Video objects define
		cv::VideoCapture video(videonamePath);
		cv::Mat frame;
		cv::String str;

		// Load a frame from video
		CV_Assert(video.isOpened());
		video.read(frame);
		video.release();
		cv::namedWindow("ROI2", cv::WINDOW_KEEPRATIO);
		cv::imshow("ROI2", frame);

		// Setting ROI coordinates
		cv::Rect r = cv::Rect(0, 0, frame.cols, frame.rows);
		getRoiPoly(frame);
		CV_Assert(!mask.empty());
		

		// Setting lanes's division lines ---------------------------------------------
		
		// To display the legend
		int baseline = 0;
		frame.push_back(cv::Mat::zeros(20, frame.cols, frame.type()));
		str = "Lane Division Lines (1/" +
			std::to_string(NbDivLines) + ")";
		cv::Size textSize = cv::getTextSize(str, cv::FONT_ITALIC, 0.4,
			1, &baseline);
		baseline += 1;
		cv::Point textOrg((frame.cols - textSize.width) / 2,
			(frame.rows - textSize.height + 2));
		cv::putText(frame, str, textOrg, cv::FONT_ITALIC, 0.4, CV_RGB(255, 255, 255), 1);
		// End

		this->NbLineLanes = NbDivLines;
		cv::Mat laneTmp = cv::Mat::zeros(cv::Size(4, NbDivLines), CV_64F);

		for (int i = 0; i < NbDivLines; i++) {
			std::vector<cv::Point> lineTmp = sdcv::imLine(frame, "ROI2", cv::Point(-1, -1), r);
			cv::line(frame, lineTmp.at(0), lineTmp.at(1), cv::Scalar(0, 0, 255), 2); // Draw line

			// To display the legend
			frame.rowRange(cv::Range(r.height, frame.rows)) = cv::Mat::zeros(cv::Size(r.width, 20), frame.type());
			str = "Region of Interest (" +
				std::to_string(i+2) + "/" +
				std::to_string(NbDivLines) + ")";

			cv::Size textSize = cv::getTextSize(str, cv::FONT_ITALIC, 0.4,
				1, &baseline);
			baseline += 1;
			cv::Point textOrg((frame.cols - textSize.width) / 2,
				(frame.rows - textSize.height + 2));
			cv::putText(frame, str, textOrg, cv::FONT_ITALIC, 0.4, CV_RGB(255, 255, 255), 1);
			// End 

			cv::imshow("ROI2", frame);

			// y_major to y_minor
			if (lineTmp.at(0).y < lineTmp.at(1).y) {
				cv::Point pt = lineTmp.at(0);
				lineTmp.at(0) = lineTmp.at(1);
				lineTmp.at(1) = pt;
			}

			laneTmp.at<double>(i, 0) = lineTmp.at(0).x;
			laneTmp.at<double>(i, 1) = lineTmp.at(0).y;
			laneTmp.at<double>(i, 2) = lineTmp.at(1).x;
			laneTmp.at<double>(i, 3) = lineTmp.at(1).y;

			double dblAuxVar = (lineTmp.at(0).y - lineTmp.at(1).y) / (double)(lineTmp.at(0).x - lineTmp.at(1).x); // m = (y2 - y1)/(x2 - x1)
			m_lineLanes.push_back(dblAuxVar);

			dblAuxVar = (lineTmp.at(0).y - (double)(dblAuxVar * lineTmp.at(0).x)); // b = y1 - m*x1
			b_lineLanes.push_back(dblAuxVar);
		}
		laneTmp.convertTo(lineLane, CV_16S);

		std::cout << "lanePosition = " << std::endl;
		std::cout << lineLane << std::endl;
		std::cout << "m_lineLanes = " << std::endl;
		std::cout << cv::format(m_lineLanes, cv::Formatter::FMT_DEFAULT) << std::endl;
		std::cout << "b_lineLanes = " << std::endl;
		std::cout << cv::format(b_lineLanes, cv::Formatter::FMT_DEFAULT) << std::endl;



		
		// Setting detection line -----------------------------------------------------
		// To display the legend
		frame.rowRange(cv::Range(r.height, frame.rows)) = cv::Mat::zeros(cv::Size(r.width, 20), frame.type());
		textSize = cv::getTextSize("Detection Line", cv::FONT_ITALIC, 0.4,
			1, &baseline);
		baseline += 1;
		textOrg = cv::Point((frame.cols - textSize.width) / 2,
			(frame.rows - textSize.height + 2));
		cv::putText(frame, "Detection Line", textOrg, cv::FONT_ITALIC, 0.4, CV_RGB(255, 255, 255), 1);
		// End

		std::vector<cv::Point> linePts = sdcv::imLine(frame, "ROI2", cv::Point(-1, -1), r);
		cv::line(frame, linePts.at(0), linePts.at(1), cv::Scalar(255, 0, 0), 2);
		cv::imshow("ROI2", frame);
		lineDetection = linePts;

		// x_minor to x_major
		if (lineDetection.at(0).x > lineDetection.at(1).x) {
			cv::Point pt = lineDetection.at(0);

			lineDetection.at(0) = lineDetection.at(1);
			lineDetection.at(1) = pt;
		}

		cLineDetection = cv::Point2f((lineDetection.at(0).x + lineDetection.at(1).x) / (float)2, (lineDetection.at(0).y + lineDetection.at(1).y) / (float)2);
		std::cout << "linePosition = [" << lineDetection.at(0).x << ", " << lineDetection.at(0).y << ", " << lineDetection.at(1).x << ", " << lineDetection.at(1).y << "]" << std::endl << std::endl;

		// End line -------------------------------------------------------------------
		// To display the legend
		frame.rowRange(cv::Range(r.height, frame.rows)) = cv::Mat::zeros(cv::Size(r.width, 20), frame.type());
		textSize = cv::getTextSize("End Line", cv::FONT_ITALIC, 0.4,
			1, &baseline);
		baseline += 1;
		textOrg = cv::Point((frame.cols - textSize.width) / 2,
							(frame.rows - textSize.height + 2));
		cv::putText(frame, "End Line", textOrg, cv::FONT_ITALIC, 0.4, CV_RGB(255, 255, 255), 1);
		// End

		EndLine = sdcv::imLine(frame, "ROI2", cv::Point(-1, -1), r);
		cv::line(frame, EndLine.at(0), EndLine.at(1), CV_RGB(255, 255, 0), 2);
		cv::imshow("ROI2", frame);

		double slope, b;
		slope = (EndLine.at(1).y - EndLine.at(0).y) / (double)(EndLine.at(1).x - EndLine.at(0).x);
		b = EndLine.at(1).y - slope*EndLine.at(1).x;
		EndLineEquation = cv::Point2d(slope, b);

		std::cout << "End Line: {" << std::endl << "\t" << EndLine.at(0) << " --> " << EndLine.at(1) << std::endl;
		std::cout << "\tEquation: Y = " << EndLineEquation.x << "X + " << EndLineEquation.y << std::endl << "}" << std::endl;

		// Region Line ----------------------------------------------------------------
		// To display the legend
		frame.rowRange(cv::Range(r.height, frame.rows)) = cv::Mat::zeros(cv::Size(r.width, 20), frame.type());
		str = "Classification Lines (" + std::to_string(NbRegions) + ")";
		textSize = cv::getTextSize(str, cv::FONT_ITALIC, 0.4,
			1, &baseline);
		baseline += 1;
		textOrg = cv::Point((frame.cols - textSize.width) / 2,
							(frame.rows - textSize.height + 2));
		cv::putText(frame, str, textOrg, cv::FONT_ITALIC, 0.4, CV_RGB(255, 255, 255), 1);
		// End

		CV_Assert(NbRegions >= 0);
		this->NbRegions = NbRegions + 1;
		std::vector<cv::Point> tmp;
		tmp.push_back(vertices.at(0));
		tmp.push_back(vertices.at(1));
		Regions = sdcv::imRegions(frame, "ROI2", NbRegions);
		Regions.insert(Regions.begin(), tmp);

		for (int i = 0; i < NbRegions; i++) {
			slope = (Regions.at(i).back().y - Regions.at(i).front().y) / (double)(Regions.at(i).back().x - Regions.at(i).front().x);
			b = Regions.at(i).back().y - slope*Regions.at(i).back().x;
			RegionLinesEquation.push_back(cv::Point2d(slope, b));
			std::cout << "Region " << i << ": {" << std::endl << "\t" << Regions.at(i).front() << " --> " << Regions.at(i).back() << std::endl;
			std::cout << "\tEquation: Y = " << RegionLinesEquation.at(i).x << "X + " << RegionLinesEquation.at(i).y << std::endl << "}" << std::endl;
		}

		// Save 
		// To display the legend
		frame.rowRange(cv::Range(r.height, frame.rows)) = cv::Mat::zeros(cv::Size(r.width, 20), frame.type());
		str = "Press \"Enter\" to save the ROI parameters";
		textSize = cv::getTextSize(str, cv::FONT_ITALIC, 0.4,
			1, &baseline);
		baseline += 1;
		textOrg = cv::Point((frame.cols - textSize.width) / 2,
			(frame.rows - textSize.height + 2));
		cv::putText(frame, str, textOrg, cv::FONT_ITALIC, 0.4, CV_RGB(255, 255, 255), 1);
		cv::imshow("ROI2", frame);
		// End
		cv::waitKey(0);
		cv::destroyAllWindows();

		save();
	}

	void ROI::draw(cv::Mat &frame, bool bDrawRoi, bool bDrawLineDetection, bool bDrawEndLine, bool bDrawRegions) {
		// Draw ROI
		if (bDrawRoi) {
			cv::line(frame, vertices.at(0), vertices.at(1), CV_RGB(0, 255, 0), 4);
			cv::line(frame, vertices.at(1), vertices.at(2), CV_RGB(0, 255, 0), 4);
			cv::line(frame, vertices.at(2), vertices.at(3), CV_RGB(0, 255, 0), 4);
			cv::line(frame, vertices.at(3), vertices.at(0), CV_RGB(0, 255, 0), 4);
		}

		// Draw detection line
		if (bDrawLineDetection) cv::line(frame, lineDetection.front(), lineDetection.back(), CV_RGB(0, 0, 255), 2);

		// Ending line
		if (bDrawEndLine) cv::line(frame, EndLine.front(), EndLine.back(), CV_RGB(0, 0, 255), 2);

		if (bDrawRegions) {
			for (auto region : Regions)
				cv::line(frame, region.front(), region.back(), CV_RGB(0, 255, 255), 2);
		}
	}

	void ROI::save(void) {
		std::string filename = "DATA/" + name + "/config.yml";

		cv::FileStorage fs(filename, cv::FileStorage::WRITE);

		fs << "vertices" << vertices;
		fs << "area" << area;
		fs << "bbox" << bbox;
		fs << "mask" << mask;

		fs << "NbDivLines" << NbLineLanes;
		fs << "lanePosition" << lineLane;
		fs << "slopeDivLines" << m_lineLanes;
		fs << "interceptDivLines" << b_lineLanes;

		fs << "linePosition" << lineDetection;
		fs << "cDivLines" << cLineDetection;
		//fs << "LineArea" << LineArea;

		fs << "EndLine" << EndLine;
		fs << "EndLineEquation" << EndLineEquation;

		fs << "NbRegions" << NbRegions;
		fs << "Regions" << Regions;
		fs << "RegionsEquationLine" << RegionLinesEquation;
	}

	bool ROI::load(void) {
		cv::FileStorage fs("DATA/" + name + "/config.yml", cv::FileStorage::READ);
		bool exitSuccess = false;

		if (fs.isOpened()) {
			exitSuccess = true;
			fs["vertices"] >> vertices;
			fs["area"] >> area;
			fs["bbox"] >> bbox;
			fs["mask"] >> mask;

			fs["NbDivLines"] >> NbLineLanes;
			fs["lanePosition"] >> lineLane;
			fs["slopeDivLines"] >> m_lineLanes;
			fs["interceptDivLines"] >> b_lineLanes;
			//fs["LaneMask"] >> DL_mask;

			fs["linePosition"] >> lineDetection;
			fs["cDivLines"] >> cLineDetection;
			//fs["LineArea"] >> LineArea;

			fs["EndLine"] >> EndLine;
			fs["EndLineEquation"] >> EndLineEquation;

			fs["NbRegions"] >> NbRegions;
			fs["Regions"] >> Regions;
			fs["RegionsEquationLine"] >> RegionLinesEquation;
		}

		return(exitSuccess);
	}


	void ROI::apply(cv::InputArray src, cv::OutputArray dst) {
		CV_Assert(!mask.empty());
		CV_Assert(!src.empty());

		src.copyTo(dst, mask);
	}

	// PROTECTED METHODS **************************************************************
	void ROI::getRoiPoly(cv::OutputArray frame, std::vector<cv::Point> vertex) {
		unsigned int NbLines = 4, idx = 0;
		cv::Point Cinit(-1, -1);
		cv::Mat tmp, aux;

		// Get ROI vertices
		frame.copyTo(tmp);
		frame.copyTo(aux);
		cv::Rect roi_r = cv::Rect(0, 0, tmp.cols, tmp.rows);

		// To display the legend
		cv::String str = "Region of Interest (1/" +
			std::to_string(NbLines) + ")";
		tmp.push_back(cv::Mat::zeros(20, tmp.cols, tmp.type()));
		int baseline = 0;
		cv::Size textSize = cv::getTextSize(str, cv::FONT_ITALIC, 0.4,
			1, &baseline);
		baseline += 1;
		cv::Point textOrg(	(tmp.cols - textSize.width) / 2,
							(tmp.rows - textSize.height+2));
		cv::putText(tmp, str, textOrg, cv::FONT_ITALIC, 0.4, CV_RGB(255, 255, 255), 1);
		// End

		int N = NbLines;
		if (vertex.size() == 0) {
			while (NbLines) {
				//bool validPoint = true;
				std::vector<cv::Point> pt = sdcv::imLine(tmp, "ROI2", Cinit, roi_r);
				cv::Rect r(pt.at(0) - cv::Point(15, 15), cv::Size(30, 30));

				//if( pt.at(1).inside(r) ) validPoint = false;
				//std::cout << "Rec = " << r << std::endl;
				//std::cout << "Point: " << pt.at(1) << std::endl;

				if (!pt.back().inside(r)) {
					std::cout << "Valid line" << std::endl;
					cv::line(tmp, pt.front(), pt.back(), cv::Scalar(0, 255, 0, 0.8), 2); // Draw line																							  //cv::rectangle(frameTmp, cv::Point(pt.front().x - 5, pt.front().y - 5), cv::Point(pt.front().x + 5, pt.front().y + 5), cv::Scalar(255, 0, 30, 0.8));
					cv::line(aux, pt.front(), pt.back(), cv::Scalar(0, 255, 0, 0.8), 2); // Draw line																							  //cv::rectangle(frameTmp, cv::Point(pt.front().x - 5, pt.front().y - 5), cv::Point(pt.front().x + 5, pt.front().y + 5), cv::Scalar(255, 0, 30, 0.8));

					vertices.push_back(pt.front());

					NbLines--;

					// To display the legend
					tmp.rowRange(cv::Range(roi_r.height, tmp.rows)) = cv::Mat::zeros(cv::Size(roi_r.width, 20), tmp.type());
					str = "Region of Interest (" +
						std::to_string(N - NbLines + 1) + "/" +
						std::to_string(N) + ")";
					textSize = cv::getTextSize(str, cv::FONT_ITALIC, 0.4, 1, &baseline);
					baseline += 1;
					textOrg = cv::Point((tmp.cols - textSize.width) / 2,
										(tmp.rows - textSize.height + 2));
					cv::putText(tmp, str, textOrg, cv::FONT_ITALIC, 0.4, 
								CV_RGB(255, 255, 255), 1);
					// End

					Cinit = pt.back();

					cv::imshow("ROI2", tmp);
				}
				else {
					std::cout << "Invalid line" << std::endl;
					cv::Mat tmp2;
					tmp.copyTo(tmp2);
					cv::line(tmp2, pt.front(), pt.back(), cv::Scalar(0, 0, 255), 2); // Draw line
					cv::imshow("ROI2", tmp2);
					cv::waitKey(1000);
					cv::imshow("ROI2", tmp);
				}

			}
			aux.copyTo(frame);
		}
		else {
			vertices = vertex;
		}

		// Vertices to ROI
		CV_Assert(vertices.size() == 4);
		sdcv::vertices2polygon(vertices, frame.size(), mask);

		std::vector< std::vector<cv::Point> > contours;
		cv::Mat RoiMask;
		mask.copyTo(RoiMask);
		cv::findContours(RoiMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		CV_Assert(contours.size() == 1);

		std::vector<cv::Point> approxPolygon;
		cv::approxPolyDP(contours[0], approxPolygon, 1.0, true); // Ramer-Douglas-Peucker algorithm

		CV_Assert(approxPolygon.size() == 4 || approxPolygon.size() == 5);

		frame.copyTo(tmp);
		cv::Mat img, imgtmp;
		frame.copyTo(img);
		tmp.copyTo(imgtmp, mask);
		cv::drawContours(imgtmp, contours, 0, cv::Scalar(0, 255, 0), cv::FILLED);
		cv::addWeighted(imgtmp, 0.1, img, 1, 0, img, -1);
		cv::imshow("ROI2", img);

		Blob blob(contours[0], 0);
		area = blob.getArea();
		bbox = blob.getBBox();

		img.copyTo(frame);
	}
};