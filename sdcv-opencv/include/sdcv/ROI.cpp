#include "ROI.h"

namespace sdcv {
	/* PUBLIC METHODS */
	// Constructor
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


	// Get methods
	cv::Rect ROI::getBbox(void) { return bbox; }
	std::vector<cv::Point> ROI::getVertices(void) { return vertices; }
	cv::Mat ROI::getMask(void) { return(this->mask); }
	int ROI::getArea(void) { return area; }
	std::string ROI::getName(void) {
		return(this->name);
	}
	//cv::Mat ROI::getLanesLine( void ) { return this->lineLane; }
	//int ROI::getNumberLineLanes( void ) { return this->NbLineLanes; }
	DivLaneLine_t ROI::getLaneData(void) {
		DivLaneLine_t DivLaneLines;

		DivLaneLines.lineLane = this->lineLane;
		DivLaneLines.b_lineLanes = this->b_lineLanes;
		DivLaneLines.m_lineLanes = this->m_lineLanes;
		DivLaneLines.DL_mask = this->DL_mask;
		DivLaneLines.NbLineLanes = this->NbLineLanes;

		return DivLaneLines;
	}
	int ROI::getNumLanes(void) { return(this->NbLineLanes); }
	cv::Mat ROI::getDivLineLane(void) { return(this->lineLane); }
	std::vector< double > ROI::getSlopeLane(void) { return(this->m_lineLanes); }
	std::vector< double > ROI::getIntersecLane(void) { return(this->b_lineLanes); }
	cv::Mat ROI::getDivLaneMask(int Nb) {
		CV_Assert(Nb < (int)this->DL_mask.size());
		return(this->DL_mask.at(Nb));
	}
	std::vector< cv::Mat > ROI::getDivLaneMask(void) { return(this->DL_mask); }

	std::vector< cv::Point > ROI::getLineDetection(void) { return(this->lineDetection); }
	cv::Point2f ROI::getCenterLineDetection(void) { return(this->cLineDetection); }

	std::vector<cv::Point> ROI::getEndLine(void) { return(this->EndLine); }
	cv::Point2d ROI::getEndLineEq(void) { return(this->EndLineEquation); }

	int ROI::getNbRegions(void) { return this->NbRegions; }
	std::vector< std::vector<cv::Point> > ROI::getRegions(void) { return this->Regions; }
	std::vector< cv::Point2d > ROI::getRegionLinesEquation() { return this->RegionLinesEquation; }

	// Set methods
	void ROI::setName(cv::String name) { this->name = name; }
	void ROI::setVertices(cv::Mat frame) {
		cv::namedWindow("ROI2", cv::WINDOW_KEEPRATIO);
		cv::imshow("ROI2", frame);

		cv::Mat roimask;
		std::vector<cv::Point> polygon;
		// Setting ROI coordinates
		sdcv::roipoly(frame, polygon, roimask);
		this->vertices = polygon;

		// Vertices to ROI
		CV_Assert(this->vertices.size() == 4);
		this->mask = roimask;

		std::vector< std::vector<cv::Point> > contours;
		this->mask.copyTo(roimask);
		cv::findContours(roimask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		CV_Assert(contours.size() == 1);

		std::vector<cv::Point> approxPolygon;
		cv::approxPolyDP(contours[0], approxPolygon, 1.0, true); // Ramer-Douglas-Peucker algorithm

		CV_Assert(approxPolygon.size() == 4 || approxPolygon.size() == 5);

		Blob blob(contours[0], 0);
		this->area = blob.getArea();
		this->bbox = blob.getBBox();

		this->save();
	}

	void ROI::setDetectionLine(cv::Mat frame) {
		cv::namedWindow("ROI2", cv::WINDOW_KEEPRATIO);
		cv::imshow("ROI2", frame);

		this->lineDetection = imLine(frame, "ROI2", cv::Point(-1, -1));
		cv::line(frame, lineDetection.front(), lineDetection.back(), cv::Scalar(255, 0, 0), 2);
		cv::imshow("ROI2", frame);
		cv::waitKey();

		// x_minor to x_major
		if (lineDetection.at(0).x > lineDetection.at(1).x) {
			cv::Point pt = this->lineDetection.at(0);

			this->lineDetection.at(0) = this->lineDetection.at(1);
			this->lineDetection.at(1) = pt;
		}

		this->cLineDetection = cv::Point2f((this->lineDetection.front().x + this->lineDetection.back().x) / (float)2, (this->lineDetection.front().y + this->lineDetection.back().y) / (float)2);
		std::cout << "detection line: {" << lineDetection.front() << " | " << lineDetection.back() << "}" << std::endl << std::endl;

		this->save();
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

		this->save();
	}

	void ROI::setRegions(cv::Mat frame, int NbRegions, cv::Point vertexIdx) {
		cv::namedWindow("ROI2", cv::WINDOW_KEEPRATIO);
		cv::imshow("ROI2", frame);

		CV_Assert(NbRegions >= 0);
		this->NbRegions = NbRegions + 1;
		std::vector<cv::Point> tmp;
		tmp.push_back(vertices.at(vertexIdx.x));
		tmp.push_back(vertices.at(vertexIdx.y));
		this->Regions = sdcv::imRegions(frame, NbRegions, "ROI2");
		this->Regions.insert(this->Regions.begin(), tmp);

		double slope, b;
		for (int i = 0; i < this->NbRegions; i++) {
			slope = (this->Regions.at(i).back().y - this->Regions.at(i).front().y) / (double)(this->Regions.at(i).back().x - this->Regions.at(i).front().x);
			b = this->Regions.at(i).back().y - slope*this->Regions.at(i).back().x;
			RegionLinesEquation.push_back(cv::Point2d(slope, b));
			std::cout << "Region #" << i << ": {" << std::endl << "\t" << this->Regions.at(i).front() << " | " << this->Regions.at(i).back() << std::endl;
			std::cout << "\tEquation: Y = " << RegionLinesEquation.at(i).x << "X + " << RegionLinesEquation.at(i).y << std::endl << "}" << std::endl;
		}

		this->save();
	}


	// Action methods
	void ROI::create(cv::String videonamePath, int NbDivLines, int NbRegions, bool bLoad) {
		if (bLoad && load()) return;

		// Video objects define
		cv::VideoCapture video(videonamePath);
		cv::Mat frame;

		// Load a frame from video
		CV_Assert(video.isOpened());
		video.read(frame);
		video.release();
		cv::namedWindow("ROI2", cv::WINDOW_KEEPRATIO);
		cv::imshow("ROI2", frame);

		// Setting ROI coordinates
		getRoiPoly(frame);
		CV_Assert(!this->mask.empty());


		// Setting lanes's division lines
		this->NbLineLanes = NbDivLines;
		cv::Mat laneTmp = cv::Mat::zeros(cv::Size(NbDivLines, 4), CV_64F);

		for (int i = 0; i < NbDivLines; i++) {
			std::vector<cv::Point> lineTmp = imLine(frame, "ROI2", cv::Point(-1, -1));
			cv::line(frame, lineTmp.at(0), lineTmp.at(1), cv::Scalar(0, 0, 255), 2); // Draw line
			cv::imshow("ROI2", frame);

			// y_major to y_minor
			if (lineTmp.at(0).y < lineTmp.at(1).y) {
				cv::Point pointTemp = lineTmp.at(0);
				lineTmp.at(0) = lineTmp.at(1);
				lineTmp.at(1) = pointTemp;
			}

			laneTmp.at<double>(i, 0) = lineTmp.at(0).x;
			laneTmp.at<double>(i, 1) = lineTmp.at(0).y;
			laneTmp.at<double>(i, 2) = lineTmp.at(1).x;
			laneTmp.at<double>(i, 3) = lineTmp.at(1).y;

			double dblAuxVar = (lineTmp.at(0).y - lineTmp.at(1).y) / (double)(lineTmp.at(0).x - lineTmp.at(1).x); // m = (y2 - y1)/(x2 - x1)
			this->m_lineLanes.push_back(dblAuxVar);

			dblAuxVar = (lineTmp.at(0).y - (double)(dblAuxVar * lineTmp.at(0).x)); // b = y1 - m*x1
			this->b_lineLanes.push_back(dblAuxVar);
		}
		laneTmp.convertTo(this->lineLane, CV_16S);

		// Get the mask
		for (int i = 0; i < this->lineLane.rows; i++) {
			cv::Mat laneMask = cv::Mat::zeros(this->mask.size(), this->mask.type());

			laneMask.setTo(cv::Scalar(255, 255, 255));

			cv::Point pt1 = cv::Point(this->lineLane.at<short>(i, 0), this->lineLane.at<short>(i, 1));
			cv::Point pt2 = cv::Point(this->lineLane.at<short>(i, 2), this->lineLane.at<short>(i, 3));
			cv::line(laneMask, pt1, pt2, cv::Scalar(0, 0, 0), 2);

			DL_mask.push_back(laneMask);
		}

		std::cout << "lanePosition = " << std::endl;
		std::cout << this->lineLane << std::endl;
		std::cout << "m_lineLanes = " << std::endl;
		std::cout << cv::format(this->m_lineLanes, cv::Formatter::FMT_DEFAULT) << std::endl;
		std::cout << "b_lineLanes = " << std::endl;
		std::cout << cv::format(this->b_lineLanes, cv::Formatter::FMT_DEFAULT) << std::endl;

		/* Setting line detection */
		std::vector<cv::Point> linePts = imLine(frame, "ROI2", cv::Point(-1, -1));
		cv::line(frame, linePts.at(0), linePts.at(1), cv::Scalar(255, 0, 0), 2);
		cv::imshow("ROI2", frame);
		this->lineDetection = linePts;

		// x_minor to x_major
		if (lineDetection.at(0).x > lineDetection.at(1).x) {
			cv::Point pt = this->lineDetection.at(0);

			this->lineDetection.at(0) = this->lineDetection.at(1);
			this->lineDetection.at(1) = pt;
		}

		this->cLineDetection = cv::Point2f((this->lineDetection.at(0).x + this->lineDetection.at(1).x) / (float)2, (this->lineDetection.at(0).y + this->lineDetection.at(1).y) / (float)2);
		std::cout << "linePosition = [" << this->lineDetection.at(0).x << ", " << this->lineDetection.at(0).y << ", " << this->lineDetection.at(1).x << ", " << this->lineDetection.at(1).y << "]" << std::endl << std::endl;

		/* End line */
		EndLine = imLine(frame, "ROI2", cv::Point(-1, -1));
		cv::line(frame, EndLine.at(0), EndLine.at(1), CV_RGB(255, 255, 0), 2);
		cv::imshow("ROI2", frame);

		double slope, b;
		slope = (EndLine.at(1).y - EndLine.at(0).y) / (double)(EndLine.at(1).x - EndLine.at(0).x);
		b = EndLine.at(1).y - slope*EndLine.at(1).x;
		EndLineEquation = cv::Point2d(slope, b);

		std::cout << "End Line: {" << std::endl << "\t" << EndLine.at(0) << " --> " << EndLine.at(1) << std::endl;
		std::cout << "\tEquation: Y = " << EndLineEquation.x << "X + " << EndLineEquation.y << std::endl << "}" << std::endl;

		/* Region Line */
		CV_Assert(NbRegions >= 0);
		this->NbRegions = NbRegions + 1;
		std::vector<cv::Point> tmp;
		tmp.push_back(this->vertices.at(0));
		tmp.push_back(this->vertices.at(1));
		this->Regions = sdcv::imRegions(frame, NbRegions, "ROI2");
		this->Regions.insert(this->Regions.begin(), tmp);

		for (int i = 0; i < this->NbRegions; i++) {
			slope = (this->Regions.at(i).back().y - this->Regions.at(i).front().y) / (double)(this->Regions.at(i).back().x - this->Regions.at(i).front().x);
			b = this->Regions.at(i).back().y - slope*this->Regions.at(i).back().x;
			RegionLinesEquation.push_back(cv::Point2d(slope, b));
			std::cout << "Region " << i << ": {" << std::endl << "\t" << this->Regions.at(i).front() << " --> " << this->Regions.at(i).back() << std::endl;
			std::cout << "\tEquation: Y = " << RegionLinesEquation.at(i).x << "X + " << RegionLinesEquation.at(i).y << std::endl << "}" << std::endl;
		}

		// Debugging
		cv::imwrite(name + ".bmp", this->mask);
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
			for (auto it = this->Regions.begin(); it != this->Regions.end(); ++it) {
				cv::line(frame, it->front(), it->back(), CV_RGB(0, 255, 255), 2);
			}
		}
	}

	void ROI::save(void) {
		std::string filename = "DATA/" + this->name + "/" + this->name + ".yml";

		cv::FileStorage fs(filename, cv::FileStorage::WRITE);

		fs << "vertices" << this->vertices;
		fs << "area" << this->area;
		fs << "bbox" << this->bbox;
		fs << "mask" << this->mask;

		fs << "NbDivLines" << this->NbLineLanes;
		fs << "lanePosition" << this->lineLane;
		fs << "slopeDivLines" << this->m_lineLanes;
		fs << "interceptDivLines" << this->b_lineLanes;
		fs << "LaneMask" << this->DL_mask;

		fs << "linePosition" << this->lineDetection;
		fs << "cDivLines" << this->cLineDetection;
		fs << "LineArea" << this->LineArea;

		fs << "EndLine" << this->EndLine;
		fs << "EndLineEquation" << this->EndLineEquation;

		fs << "NbRegions" << this->NbRegions;
		fs << "Regions" << this->Regions;
		fs << "RegionsEquationLine" << this->RegionLinesEquation;
	}

	bool ROI::load(void) {
		cv::FileStorage fs("DATA/" + this->name + "/" + this->name + ".yml", cv::FileStorage::READ);
		bool exitSuccess = false;

		if (fs.isOpened()) {
			exitSuccess = true;
			fs["vertices"] >> this->vertices;
			fs["area"] >> this->area;
			fs["bbox"] >> this->bbox;
			fs["mask"] >> this->mask;

			fs["NbDivLines"] >> this->NbLineLanes;
			fs["lanePosition"] >> this->lineLane;
			fs["slopeDivLines"] >> this->m_lineLanes;
			fs["interceptDivLines"] >> this->b_lineLanes;
			fs["LaneMask"] >> this->DL_mask;

			fs["linePosition"] >> this->lineDetection;
			fs["cDivLines"] >> this->cLineDetection;
			fs["LineArea"] >> this->LineArea;

			fs["EndLine"] >> this->EndLine;
			fs["EndLineEquation"] >> this->EndLineEquation;

			fs["NbRegions"] >> this->NbRegions;
			fs["Regions"] >> this->Regions;
			fs["RegionsEquationLine"] >> this->RegionLinesEquation;

		}

		return(exitSuccess);
	}


	void ROI::apply(cv::InputArray frame, cv::OutputArray image) {
		CV_Assert(!this->mask.empty());
		CV_Assert(!frame.empty());

		frame.copyTo(image, this->mask);
	}

	// Testing methods
#include <windows.h>
	void ROI::testing(eROItest en2Test, cv::Mat frame, std::vector<cv::Point> input) {
		cv::Mat aux, aux1;
		std::vector< std::vector<cv::Point> > contours;
		std::vector<cv::Point> approxPolygon;
		int NbLanes;
		bool bInit = true;
		std::cout << std::endl << std::endl << "[TESTING]: ROI {" << std::endl << std::endl;
		std::cout << "\tModule: ";

		switch (en2Test) {
		case ROI_TEST_VERTEX:
			CV_Assert(input.size() == 4);
			std::cout << "Vertices" << std::endl;

			vertices2polygon(input, frame.size(), aux);
			cv::findContours(aux, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
			cv::approxPolyDP(contours[0], approxPolygon, 1.0, true); // Ramer-Douglas-Peucker algorithm

			SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_GREEN);
			std::cout << "[\tRUN\t]\t";
			SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 15);
			std::cout << "Trapezoidal region" << std::endl;

			if (approxPolygon.size() > 5 || approxPolygon.size() < 4) {
				SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_RED);
				std::cout << "[\tFAILED\t]\t";
			}
			else {
				SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_GREEN);
				std::cout << "[\tPASSED\t]\t";
				this->getRoiPoly(frame, input);
			}
			SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 15);
			std::cout << "Size:" << approxPolygon.size() << std::endl;
			break;

		case ROI_TEST_DIV_LINES:
			CV_Assert(input.size() > 0);
			NbLanes = input.at(0).x;
			std::cout << "Divided Lane's Line" << std::endl;


			if (input.size() - 1 != NbLanes) bInit = false;
			SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_GREEN);
			std::cout << "[\tRUN\t]\t";
			SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 15);
			std::cout << "Lines" << std::endl;
			SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), bInit ? FOREGROUND_GREEN : FOREGROUND_RED);
			if (!bInit) std::cout << "[\tFAILED\t]\t";
			else std::cout << "[\tPASSED\t]\t";
			SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 15);
			std::cout << input.size() - 1 << "\\" << NbLanes << std::endl;


			break;

		case ROI_TEST_LINE_DETECTION:

			break;
		}

		std::cout << std::endl << "}" << std::endl << std::endl;
		std::cin.get();

	}


	// Destructor
	ROI::~ROI(void) {

	}

	/* PROTECTED METHODS */
	void ROI::getRoiPoly(cv::OutputArray frame, std::vector<cv::Point> vertex) {
		unsigned int NbLines = 4, idx = 0;
		cv::Point initialCondition(-1, -1);
		cv::Mat frameTmp;

		// Get ROI vertices
		frame.copyTo(frameTmp);
		if (vertex.size() == 0) {
			while (NbLines) {
				//bool validPoint = true;
				std::vector<cv::Point> pt = imLine(frameTmp, "ROI2", initialCondition);
				cv::Rect r(pt.at(0) - cv::Point(15, 15), cv::Size(30, 30));

				//if( pt.at(1).inside(r) ) validPoint = false;
				//std::cout << "Rec = " << r << std::endl;
				//std::cout << "Point: " << pt.at(1) << std::endl;

				if (!pt.back().inside(r)) {
					std::cout << "Valid line" << std::endl;
					cv::line(frameTmp, pt.front(), pt.back(), cv::Scalar(0, 255, 0, 0.8), 2); // Draw line
																							  //cv::rectangle(frameTmp, cv::Point(pt.front().x - 5, pt.front().y - 5), cv::Point(pt.front().x + 5, pt.front().y + 5), cv::Scalar(255, 0, 30, 0.8));

					this->vertices.push_back(pt.front());

					NbLines--;

					initialCondition = pt.back();

					cv::imshow("ROI2", frameTmp);
				}
				else {
					std::cout << "Invalid line" << std::endl;
					cv::Mat tmp;
					frameTmp.copyTo(tmp);
					cv::line(tmp, pt.front(), pt.back(), cv::Scalar(0, 0, 255), 2); // Draw line
					cv::imshow("ROI2", tmp);
					cv::waitKey(1000);
					cv::imshow("ROI2", frameTmp);
				}

			}
			frameTmp.copyTo(frame);
		}
		else {
			this->vertices = vertex;
		}

		// Vertices to ROI
		CV_Assert(this->vertices.size() == 4);
		vertices2polygon(this->vertices, frame.size(), this->mask);

		std::vector< std::vector<cv::Point> > contours;
		cv::Mat RoiMask;
		this->mask.copyTo(RoiMask);
		cv::findContours(RoiMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		CV_Assert(contours.size() == 1);

		std::vector<cv::Point> approxPolygon;
		cv::approxPolyDP(contours[0], approxPolygon, 1.0, true); // Ramer-Douglas-Peucker algorithm

		CV_Assert(approxPolygon.size() == 4 || approxPolygon.size() == 5);

		frame.copyTo(frameTmp);
		cv::Mat img, imgtmp;
		frame.copyTo(img);
		frameTmp.copyTo(imgtmp, this->mask);
		cv::drawContours(imgtmp, contours, 0, cv::Scalar(0, 255, 0), cv::FILLED);
		cv::addWeighted(imgtmp, 0.1, img, 1, 0, img, -1);
		cv::imshow("ROI2", img);

		Blob blob(contours[0], 0);
		this->area = blob.getArea();
		this->bbox = blob.getBBox();

		img.copyTo(frame);
	}
};