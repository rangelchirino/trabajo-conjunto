#include "Drawing.hpp"


namespace sdcv {
	typedef std::tuple<std::vector<cv::Point>, cv::Mat, std::string, cv::Rect> lineOnMouseData_t;
	static void lineOnMouseCallback(int event, int X, int Y, int flags, void* pvUserDat);
	static void pointOnMouseCallback(int event, int X, int Y, int flags, void* pvUserDat);
	
	std::string ItoS(int Number) {
		std::string str;          // string which will contain the result
		std::ostringstream bridge;   // stream used for the conversion
		bridge << Number;      // insert the textual representation of 'Number' in the characters in the stream
		str = bridge.str(); // set 'Result' to the contents of the stream

		return( str );
	}

	cv::Point imPoint(cv::InputArray frame, cv::String WindowName) {
		cv::Point pt = cv::Point(-1, -1);
		
		cv::namedWindow(WindowName, cv::WINDOW_KEEPRATIO);
		cv::imshow(WindowName, frame.getMat());

		cv::setMouseCallback(WindowName, pointOnMouseCallback, &pt);

		while (cv::getWindowProperty(WindowName, 0) >= 0 && pt == cv::Point(-1, -1)) cv::waitKey(30);
		std::cout << "End of Point" << std::endl;

		// Test if window has been closed
		if (cv::getWindowProperty(WindowName, 0) < 0) {
			std::cerr << "Window [" << WindowName << "] has been closed!" << std::endl;
			std::cout << "Press a key to exit..." << std::endl;
			std::cin.get();
			exit(-1);
		}

		// Unset mouse callback
		cv::setMouseCallback(WindowName, NULL, NULL);
		cv::destroyWindow(WindowName);

		return pt;
	}

	std::vector<cv::Point> imLine(cv::OutputArray frame, cv::String WindowName, cv::Point Cinit, cv::Rect roi) {
			std::vector<cv::Point> lpt;
			cv::Mat tmp;
			frame.copyTo(tmp);

			// Initial conditions
			if (Cinit.x >= 0 || Cinit.y >= 0) 
				lpt.push_back(Cinit);
			std::cout << "Initial Conditions: " << Cinit << std::endl;

			// Region of interest
			if (!roi.width || !roi.height)
				roi = cv::Rect(0, 0, frame.size().width, frame.size().height);
			std::cout << "Roi: " << roi << std::endl;
			
			cv::imshow(WindowName, tmp);

			// Set mouse callback
			lineOnMouseData_t onMouseTuple(lpt, tmp, WindowName, roi);
			cv::setMouseCallback(WindowName, lineOnMouseCallback, &onMouseTuple);
			
			// Wait until line be setted or window is finished
			while (cv::getWindowProperty(WindowName, 0) >= 0 && std::get<0>(onMouseTuple).size() < 2) {
				if (cv::waitKey(30) == sdcv::VK_ESC)
					std::get<0>(onMouseTuple).clear();
			}

			// Test if window has been closed
			if (cv::getWindowProperty(WindowName, 0) < 0) {
				std::cerr << "Window [" << WindowName << "] has been closed!" << std::endl;
				std::cout << "Press a key to exit..." << std::endl;
				std::cin.get();
				exit(-1);
			}

			// Unset mouse callback
			cv::setMouseCallback(WindowName, NULL, NULL);

			return std::get<0>(onMouseTuple);
	}
	
	std::vector< std::vector<cv::Point> > imRegions(cv::OutputArray frame,
													cv::String wname,
													int NbRegions,
													cv::Rect roi) 
	{
		cv::Mat frameTmp;
		std::vector< std::vector<cv::Point> > Regions;
		int idx = 0;

		frame.copyTo(frameTmp);

		while (NbRegions--) {
			Regions.push_back( sdcv::imLine(frameTmp, wname, cv::Point(-1,-1), roi) );
			cv::line(frameTmp, Regions.at(idx).front(), Regions.at(idx).back(), CV_RGB(255, 0, 255), 2);
			cv::imshow(wname, frameTmp);
			idx++;
		}
		
		frameTmp.copyTo(frame);

		return Regions;
	}

	std::vector<cv::Point> euclideanContour(std::vector<cv::Point> contour, cv::Point point) {
		std::vector<cv::Point> mindist;
		cv::Mat sortedIdx, dis;

		for (auto it = contour.begin(); it != contour.end(); it++) {
			cv::Point diff = *it - point;
			double e = cv::sqrt(diff.x*diff.x + diff.y*diff.y);
			dis.push_back(e);
		}

		cv::sortIdx(dis, sortedIdx, cv::SORT_EVERY_COLUMN | cv::SORT_ASCENDING);

		mindist.push_back(contour.at(sortedIdx.at<int>(0)));
		mindist.push_back(contour.at(sortedIdx.at<int>(1)));


		return mindist;
	}

	void vertices2polygon(std::vector<cv::Point> vertices, cv::Size sz, cv::OutputArray polygon) {
		cv::Mat tmp = cv::Mat::zeros(sz, CV_8UC1);
		polygon.release();

		// Create a polygon from vertices
		std::vector<cv::Point> approxPolygon;
		cv::approxPolyDP(vertices, approxPolygon, 1.0, true); // Ramer-Douglas-Peucker algorithm
		cv::fillConvexPoly(tmp, &approxPolygon[0], (int)approxPolygon.size(), 255, 8, 0); // Fill polygon white
		tmp.copyTo(polygon);
	}
	
	
	
	static void pointOnMouseCallback(int event, int X, int Y, int flags, void* pvUserDat) {
		cv::Point *pt = (cv::Point *)pvUserDat;

		if (event == cv::EVENT_LBUTTONDOWN)
			*pt = cv::Point(X, Y);
	}

	static void lineOnMouseCallback(int event, int X, int Y, int flags, void* pvUserDat) {
		lineOnMouseData_t *onMouseTuple = (lineOnMouseData_t *)pvUserDat;
		std::string WinName = std::get<2>(*onMouseTuple);
		cv::Rect r = std::get<3>(*onMouseTuple);
		

		if( std::get<0>(*onMouseTuple).size() ) {
			cv::Point pt = std::get<0>(*onMouseTuple).front();
			cv::Mat img;
			
			if (r.contains(cv::Point(X, Y))) {
				std::get<1>(*onMouseTuple).copyTo(img);
				cv::line(img, pt, cv::Point(X, Y), CV_RGB(0, 0, 255), 2);
				cv::circle(img, pt, 4, CV_RGB(0, 0, 0));
				cv::imshow(WinName, img);
				cv::waitKey(30);
			}
		}

		if (event == cv::EVENT_LBUTTONDOWN && r.contains(cv::Point(X,Y)) )
			std::get<0>(*onMouseTuple).push_back(cv::Point(X, Y));
			
	}

	void drawPoints(cv::Mat &img, std::vector<cv::Point> vector, cv::Scalar color) {
		for (auto it = vector.begin(); it != vector.end(); it++) {
			auto itNext = it + 1;

			if (itNext != vector.end())
				cv::line(img, *it, *itNext, color);
		}
	}

	void drawPoints(cv::Mat &img, std::vector<cv::Point2f> vector, cv::Scalar color) {
		for (auto it = vector.begin(); it != vector.end(); it++) {
			auto itNext = it + 1;

			if (itNext != vector.end())
				cv::line(img, cv::Point( (int)it->x, (int)it->y), cv::Point((int)itNext->x, (int)itNext->y), color);
		}
	}


	void plot2d(std::string name, cv::Size Axes, std::vector<int> xData, std::vector<int> yData) {
		cv::namedWindow(name, cv::WINDOW_KEEPRATIO);
		
		cv::Mat graph = cv::Mat::zeros(Axes, CV_8UC3);



		graph.release();
	}

	void insertObjectAnnotation(cv::InputOutputArray I, cv::Rect position, std::string label, eAnnotationShape shape, cv::Scalar color) {
		cv::Mat tmp = I.getMat();
		cv::Point labelPosBottom;
		int baseline;
		cv::Size textSize;

		if(label.length() > 0)
			textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseline);
		
		if (shape == E_SHAPE_RECT) {
			cv::rectangle(tmp, position, color, 1);
			labelPosBottom = cv::Point(position.x + position.width/2, position.y);
		}
		else {
			cv::circle(tmp, cv::Point(position.x, position.y), position.width, color, 1);
			labelPosBottom = cv::Point(position.x, position.y - position.width);
		}


		if (label.length() > 0) {
			cv::Rect labelPos = cv::Rect(labelPosBottom.x - textSize.width/2, labelPosBottom.y - textSize.height - textSize.height / 2, textSize.width, textSize.height + textSize.height/2);
			cv::Mat rec = tmp(labelPos);
			cv::Mat colorArray(rec.size(), rec.type(), color);
			cv::addWeighted(colorArray, 0.7, rec, 1.0, 0.0, rec);

			cv::putText(tmp, label, cv::Point(labelPos.x, labelPos.y + textSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0));
		}

		tmp.copyTo(I);
	}

	void roipoly(cv::InputArray frame, std::vector<cv::Point> &poly, cv::OutputArray mask) {
		bool finish = false;
		cv::Point ic(-1, -1);
		cv::Mat img = frame.getMat();
		poly.clear();

		cv::namedWindow("ROI TOOL", cv::WINDOW_KEEPRATIO);
		
		// Get ROI vertices
		while (true) {
			std::vector<cv::Point> pt = sdcv::imLine(img, "ROI TOOL", ic);
			cv::Rect r(pt.at(0) - cv::Point(15, 15), cv::Size(30, 30));

			if (!pt.back().inside(r)) {
				cv::line(img, pt.front(), pt.back(), cv::Scalar(0, 255, 0, 0.8), 2); // Draw line
				cv::circle(img, pt.front(), 4, CV_RGB(0, 0, 0));

				poly.push_back(pt.front());
				ic = pt.back();
				cv::imshow("ROI TOOL", img);
				
				if (poly.size() > 2) {
					cv::Point diff(poly.front() - pt.back());
					double distance = std::sqrt(diff.x*diff.x + diff.y*diff.y);
					if (distance < 5.0) break;
				}

			}
			else {
				cv::Mat imgTmp;
				img.copyTo(imgTmp);
				cv::line(imgTmp, pt.front(), pt.back(), cv::Scalar(0, 0, 255), 2); // Draw line
				cv::imshow("ROI TOOL", imgTmp);
				cv::waitKey(1000);
				cv::imshow("ROI TOOL", img);
			}

		}

		cv::destroyWindow("ROI TOOL");

		// Vertices to ROI
		sdcv::vertices2polygon(poly, frame.size(), mask);
	}

	void roipoly(cv::InputArray frame, std::vector<cv::Point> &poly) {
		cv::Mat mask;
		roipoly(frame, poly, mask);
	}


	void roipoly(cv::InputArray frame, cv::OutputArray mask) {
		std::vector<cv::Point> poly;
		roipoly(frame, poly, mask);
	}
}