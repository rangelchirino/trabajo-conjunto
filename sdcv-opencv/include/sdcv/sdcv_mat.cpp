#include "sdcv_mat.hpp"

namespace sdcv {
	
	/*!
	 * @name	cv_remove_if
	 * @brief	Remove columns/rows if lamda is equals to parameter polarity.
	 */
	void cv_remove_if(const cv::Mat &src, cv::Mat &dst, std::function<bool(cv::Mat src)> lamda, bool RowOrColumn, bool polarity) {
		int n = RowOrColumn ? src.rows : src.cols;

		dst.release();
		for (int i = 0; i < n; i++)
		{
			cv::Mat rc = RowOrColumn ? src.row(i) : src.col(i);
			if ( lamda(rc) == polarity ) continue; // remove element
			
			if (dst.empty()) dst = rc;
			else
			{
				if (RowOrColumn)
					cv::vconcat(dst, rc, dst);
				else
					cv::hconcat(dst, rc, dst);
			}
		}
	}
	
	/*!
	 * @name	cv_remove
	 * @brief	Remove columns/rows given a vector of indexes from the src array
	 */
	void cv_remove(cv::InputArray src, cv::OutputArray dst, std::vector<int> ridx, bool RowOrColumn) {
		cv::Mat tmp;
		src.copyTo(tmp);
		dst.release();

		int endRIdx = (RowOrColumn ? tmp.rows : tmp.cols);
		auto it = ridx.begin();
		for (int i = 0; i < endRIdx; i++) {
			if (i != *it || it == ridx.end()) {
				if (dst.empty()) (RowOrColumn ? tmp.row(i).copyTo(dst) : tmp.col(i).copyTo(dst));
				else (RowOrColumn ? cv::vconcat(dst, tmp.row(i), dst) : cv::hconcat(dst, tmp.col(i), dst));
			}
			else if (it != ridx.end()) it++;
		}
	}
	
	/*!
	 * @name	cv_copy
	 * @brief	Copy columns/rows given a vector of indexes from the src array
	 */
	void cv_copy(cv::InputArray src, cv::OutputArray dst, std::vector<int> indexes, bool RowOrColumn) {
		cv::Mat tmp;
		
		src.copyTo(tmp);
		dst.release();

		int UpperLimit = (RowOrColumn ? tmp.rows : tmp.cols);

		for (auto it = indexes.begin(); it != indexes.end(); ++it) {
			if (*it >= 0 && *it < UpperLimit) {
				for (int i = 0; i < UpperLimit; i++) {
					if (i == *it) {
						if (dst.empty())
							(RowOrColumn ? tmp.row(i).copyTo(dst) : tmp.col(i).copyTo(dst));
						else
							(RowOrColumn ? cv::vconcat(dst, tmp.row(i), dst) : cv::hconcat(dst, tmp.col(i), dst));

						break;
					}
				}
			}
		}

	}


	void norm(sdcv::Track track, cv::OutputArray sample, int NbSamples, std::vector<float> params) {
		// params:
		// W1, W2, DX, DY, RoiLa, NArea, NWidth, NHeight

		// Mean of samples ------------------------------------------------------------------
		cv::Mat tmp = cv::Mat::zeros(1,3,CV_32F);
		int end = (int)track.areas.size() - 1;
		if (track.NbOfRegion < 2 || track.ClassRegionVisibleCount < NbSamples || !track.id || track.consInvisibleFrames) {
			std::cout << "Not pass!" << std::endl;
			return;
		}

		for (int i = 0; i < NbSamples; i++) {
				tmp.at<float>(0) += (float)track.areas.at(end - i);
		}
		
		tmp.at<float>(0) = tmp.at<float>(0) / (float)NbSamples;

		end = (int)track.bbox.size() - 1;
		for (int i = 0; i < NbSamples; i++) {
			tmp.at<float>(1) += (float)track.bbox.at(end - i).width;
			tmp.at<float>(2) += (float)track.bbox.at(end - i).height;
		}
		tmp.at<float>(1) = tmp.at<float>(1) / (float)NbSamples;
		tmp.at<float>(2) = tmp.at<float>(2) / (float)NbSamples;

		// Move the datas into the classification line ------------------------------------------------------------------
		float W1 = params.at(0);
		float W2 = params.at(1);
		float Dx = params.at(2);
		float Dy = params.at(3);
		float RoidLa = params.at(4);
		float dy = track.detectedCentroid.back().y - RoidLa;
		float dx = (dy / (float)Dy) * Dx;
		float Wx = W1 + (float)2.0 * dx;

		tmp.at<float>(0) = tmp.at<float>(0) * ((W1*W1) / (float)(Wx*Wx)); 	// Into class line Mean of Area
		tmp.at<float>(1) = tmp.at<float>(1) * (W1 / (float)Wx); 			// Into class line  Mean of width
		tmp.at<float>(2) = tmp.at<float>(2) * (W1 / (float)Wx); 			// Into class Mean of Height
		if (tmp.at<float>(1) != 0.0)
			tmp.at<float>(2) = tmp.at<float>(2) / (float)tmp.at<float>(1);	// Into class Mean of Height / Width
		else
			tmp.at<float>(2) = 0.0;

		// Normalize datas ------------------------------------------------------------------
		float NArea = params.at(5);
		float NWidth = params.at(6);
		float NHeight = params.at(7);
		tmp.at<float>(0) = tmp.at<float>(0) / NArea;
		tmp.at<float>(1) = tmp.at<float>(1) / NWidth;
		tmp.at<float>(2) = tmp.at<float>(2) / NHeight;

		// Sample ------------------------------------------------------------------
		tmp.copyTo(sample);
	}
	
	bool find(std::vector<cv::Point> v, int value) {
		for (auto it = v.begin(); it != v.end(); ++it) {
			if (it->y == value) return true;
		}

		return false;
	}

	double pointLineTest(cv::Point begin, cv::Point end, cv::Point2d pt, bool retpoint) {
		cv::Point diff = end - begin;
		double slope = diff.y / (double)diff.x;
		double b = (double)end.y - (double)slope*end.x;

		double distance2line = pt.y - slope*pt.x - b;

		return retpoint ? distance2line : (double)(distance2line > 0);
	}

	
	double distanceToLine(cv::Point pt, double slope, double b, eOrientationLine orientation) {
		double scale;
		if (orientation == TOP_DOWN || orientation == LEFT_2_RIGHT)
			scale = 1.0;
		else if (orientation == BOTTOM_UP || orientation == RIGHT_2_LEFT)
			scale = -1.0;

		return ((double)((double)pt.x*slope + b - (double)pt.y))*scale;
	}

	double distanceToLine(cv::Point2f pt, double slope, double b, sdcv::eOrientationLine orientation) {
		double scale;
		if (orientation == TOP_DOWN || orientation == LEFT_2_RIGHT)
			scale = 1.0;
		else if (orientation == BOTTOM_UP || orientation == RIGHT_2_LEFT)
			scale = -1.0;

		return ((double)((double)pt.x*slope + b - (double)pt.y))*scale;
	}

	double distanceToLine(cv::Point2d pt, double slope, double b, sdcv::eOrientationLine orientation) {
		double scale;
		if (orientation == TOP_DOWN || orientation == LEFT_2_RIGHT)
			scale = 1.0;
		else if (orientation == BOTTOM_UP || orientation == RIGHT_2_LEFT)
			scale = -1.0;

		return ((double)((double)pt.x*slope + b - (double)pt.y))*scale;
	}

	double euclidean(cv::Point a, cv::Point b) {
		cv::Point2d diff = b - a;

		return std::sqrt(diff.x*diff.x + diff.y*diff.y);
	}

	double euclidean(cv::Point2f a, cv::Point2f b) {
		cv::Point2d diff = b - a;

		return std::sqrt(diff.x*diff.x + diff.y*diff.y);
	}

	double euclidean(cv::Point2d a, cv::Point2d b) {
		cv::Point2d diff = b - a;

		return std::sqrt(diff.x*diff.x + diff.y*diff.y);
	}
}

/*! ************** End of file ----------------- CINVESTAV GDL */