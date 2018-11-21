#include "Blob.h"


/*!
 * @namespace	sdcv.
 * @brief		Vehicle Detection and Classification System.
 */
namespace sdcv {
	// Constructor
	Blob::Blob(std::vector< cv::Point > contour, int id, int area) {
		this->contour = contour;

		this->id = id;
		this->creatorId = -1;

		cv::Moments m = moments(contour);
		this->centroid = cv::Point2d((float)(m.m10/(double)m.m00), (float)(m.m01/(double)m.m00));
		
		if( area < 0 ) this->area = (int)cv::contourArea(contour);
		else this->area = area;
		
		this->bbox = cv::boundingRect(contour);
		this->normArea = 0.0;

		this->lanePosition = -1;

		this->occluded = false;
	}

	// Get methods
	cv::Point2f Blob::getCentroid( void ) { return this->centroid; }
	int Blob::getArea(void) { return this->area; }
	double Blob::getNormArea(void) { return normArea; }
	cv::Rect Blob::getBBox(void) { return this->bbox; }
	std::vector<cv::Point> Blob::getContour(void) { return this->contour; }
	bool Blob::getOccluded(void) { return(this->occluded); }
	int Blob::getParentId(void) { return this->creatorId; }

	// Set methods
	void Blob::setBlob(std::vector< cv::Point > contour) {
		cv::Moments m = moments(contour);
		this->centroid = cv::Point2f((float)(m.m10/(double)m.m00), (float)(m.m01/(double)m.m00));
		this->area = (int)cv::contourArea(contour);
		this->bbox = cv::boundingRect(contour);
		this->contour = contour;
		this->occluded = false;
	}

	void Blob::setNormArea(double normArea) { this->normArea = normArea; }
	void Blob::setLanePosition(int position) { this->lanePosition = position; }
	void Blob::setOcclusion(bool value) { this->occluded = value; }
	void Blob::setParentId(int parentId) { this->creatorId = parentId;  }

	// Action methods
	void Blob::print( std::ofstream &file  ) {
		file << "[Blob Info]:"			<< std::endl;
		file << "- Centroid: "			<< centroid << std::endl;
		file << "- Bounding box: "		<< bbox << std::endl;
		file << "- Area: "				<< area << std::endl;
		file << "- normalized Area: "	<< normArea << std::endl;
		file << "-----------------------------------" << std::endl << std::endl;
	}

	void Blob::print( void  ) {
		std::cout << "----------- [Blob Info]: ----------"			<< std::endl;
		std::cout << "- Centroid        : "	<< centroid << std::endl;
		std::cout << "- Bounding box    : "	<< bbox << std::endl;
		std::cout << "- Area            : "	<< area << std::endl;
		std::cout << "- normalized Area : "	<< normArea << std::endl;
		std::cout << "-----------------------------------" << std::endl << std::endl;
	}

	
	void Blob::match(std::vector<cv::Point2f> src1, std::vector<cv::Point2f> src2, std::vector<cv::Point> &dst, double maxDistance) {
		dst.clear();
		
		auto src1It = src1.begin();
		for (int i = 0; i < src1.size(); i++) {
			double minNorm = std::numeric_limits<double>::infinity();
			int index = -1;
			auto src2It = src2.begin();
			for (int j = 0; j < src2.size(); j++) {
				double norm = std::sqrt( std::pow(src2It->x- src1It->x, 2) + std::pow(src2It->y - src1It->y, 2));
				if (norm < minNorm) {
					minNorm = norm;
					index = j;
				}
				src2It++;
			}
			if(minNorm <= maxDistance) dst.push_back(cv::Point(i, index));
			src1It++;
		}
	}

	// Overloaded
	cv::Point Blob::match(std::vector<cv::Point2f> src, double maxDistance) {
		std::vector<cv::Point> dst;
		std::vector<cv::Point2f> src1 = { this->centroid };
		match(src1, src, dst, maxDistance);
		if (dst.front().x == this->id) this->occluded = true;

		return(dst.front());
	}

	// Destructor
	Blob::~Blob(void) { }
}