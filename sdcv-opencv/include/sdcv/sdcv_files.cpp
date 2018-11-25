#include "sdcv_files.hpp"

#define isDigit(C) ((C) >= '0' && (C) <= '9' || (C) == '.')

namespace sdcv {
	static std::vector<float> fsread__callback(std::string line, char charSeparated);
	
	template <typename T, typename U>
	void copy(T first, T last, U &dst) {
		while (first != last) {
			dst += *first;
			first++;
		}
	}

	/*!
	 * @name	csvread
	 * @brief	Parsing a xsv file into a cv::Mat
	 */
	cv::Mat fs_read(const std::string& filename, char charSeparated, int RowOffset, int ColOffset, int RowLast, int ColLast) {
		std::ifstream file(filename);	
		if (!file) throw std::exception();

		std::string line;
		cv::Mat matrix, fsMatrix;

		while (RowOffset > 0 && std::getline(file, line))
			RowOffset--;

		// For each line in the given file:
		while (std::getline(file, line))
			matrix.push_back(cv::Mat(fsread__callback(line, charSeparated)).t());

		ColOffset = (ColOffset >= 0 && ColOffset <= matrix.cols ? ColOffset : 0);
		RowLast = (RowLast > 0 && RowLast < matrix.rows ? RowLast : matrix.rows);
		ColLast = (ColLast > 0 && ColLast < matrix.cols ? ColLast : matrix.cols);

		matrix(cv::Range(0, RowLast), cv::Range(ColOffset, ColLast)).copyTo(fsMatrix);

		return fsMatrix;
	}
	
	/*!
	 * @name	csvwrite
	 * @brief	Write a cv::Mat into a csv file
	 */
	void fs_write(const std::string filename, char charSeparated, cv::Mat data) {
		std::ofstream file(filename);

		file << cv::format(data, cv::Formatter::FMT_CSV);
		
		file.close();
	}

	void fs_write(const std::string filename, cv::Mat data, int flag, std::string header) {
		std::ofstream file;
		
		if ( flag ) file.open(filename, std::ofstream::out | std::ofstream::app);
		else {
			file.open(filename);

			if (!header.empty()) 
				file << header << std::endl;
		}

		file << cv::format(data, cv::Formatter::FMT_CSV);

		file.close();
	}
		
	/*!
	 * @name	fs_exist
	 * @brief	Test if a file exist
	 */
	bool fs_exist(std::string filename) {
		
		std::ifstream file(filename);
		
		bool exist = file.is_open();
		
		file.close();

		return exist;
	}
	
	/*!
	 * @name	fs_split_name
	 * @brief	Split the file path with name into three fields {DIRECTORY, NAME, EXTENSION}
	 * \example	
	 *			std::string filename = "C:/Documents/file.ext";
	 *			std::vector<std::string> splitted;
	 *			fs_split_name(filename, splitted);
	 *			splitted.at(0) --> "C:/Documents/"
	 *			splitted.at(1) --> "file"
	 *			splitted.at(2) --> "ext"
	 */
	void fs_split_path(std::string filename, std::vector<std::string> &splitted) {
		std::vector<char> findChar = {'/', '.', '\0'};
		std::string::iterator it_begin = filename.begin();
		bool inc = true;

		for(auto find_for = findChar.begin(); find_for != findChar.end(); ++find_for) {
			std::string descriptor;
			
			std::string::iterator it = find_last(it_begin, filename.end(), *find_for);
			if(*it == *find_for) {
				if (inc) it++;
				sdcv::copy(it_begin, it, descriptor);
				it_begin = it++;
				inc = inc ^ true;
			}

			splitted.push_back(descriptor);
		}
		
		if( !splitted.back().empty() ) splitted.back().erase(splitted.back().begin());
	}	
	
	std::tuple<std::string, std::string, std::string> fs_fileparts(std::string filename) {
		std::string path, name, ext;
		
		size_t slash = filename.find_last_of('/');
		size_t dot = filename.find_last_of('.');

		bool f1 = slash < filename.length();
		bool f2 = dot < filename.length();

		// Try with '\'
		if (!f1) {
			slash = filename.find_last_of('\\');
			f1 = slash < filename.length();
		}
		

		if (f1 && f2) {
			path = filename.substr(0, slash + 1);
			name = filename.substr(slash + 1, dot - slash - 1);
			ext = filename.substr(dot);
			
		} else if (f1 && !f2) {
			path = filename.substr(0, slash + 1);
			name = filename.substr(slash + 1);
			ext = "";
		} else if (!f1 && f2) {
			path = "";
			name = filename.substr(0, dot);
			ext = filename.substr(dot);
		} else {
			path = "";
			name = filename;
			ext = "";
		}

		return std::make_tuple(path, name, ext);
	}

	template <typename T, typename U> T find_last(T first, T last, U value2find) {
		T it = first, it_last = last;

		while (it != last) {
			if (*it == value2find) it_last = it;
			it++;
		}

		return it_last;
	}

	std::vector<float> fsread__callback(std::string line, char charSeparated) {
		std::vector<float> values;
		std::string value;

		line.erase(std::remove(line.begin(), line.end(), ' '), line.end());
		
		for (auto it = line.begin(); it != line.end(); ++it) {

			if (*it != charSeparated) {
				value += *it;
			} 
			else {
				values.push_back(std::stof(value));
				value.clear();
			}

		}

		if (value.length() > 0) values.push_back(std::stof(value));

		return values;
	}
}