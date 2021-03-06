/**
* \file \sdcv_files.hpp
* \author \Fernando Hermosillo
* \date \12/07/2017 09:00
*
* \brief This file define the SVM Classifier class and is part of
*		 Vehicle Detection and Classification System project
*
* \version Revision: 1.2
*
* \note		A XSV file is a x-separated values file, x is a char
*/
 
#ifndef SDCV_FILES_HPP
#define SDCV_FILES_HPP
	
	/*********************************************************/
	/*                    I N C L U D E S                    */
	/*********************************************************/
	#include <iostream>
	#include <string>
	#include <fstream>
	#include <iterator>
	#include <functional>
	#include <algorithm>
	#include <vector>
	#include <experimental\filesystem>
	#include <Windows.h>
	#include <opencv2\opencv.hpp>
	
	/*********************************************************/
	/*                     D E F I N E S                     */
	/*********************************************************/
	
	/*********************************************************/
	/*                      M A C R O S                      */
	/*********************************************************/

	/*********************************************************/
	/* 			U S I N G   N A M E S P A C E S 			 */
	/*********************************************************/
	
	/*********************************************************/
	/*                       E N U M S                       */
	/*********************************************************/
	
	/*********************************************************/
	/*                     S T R U C T S                     */
	/*********************************************************/

	/*********************************************************/
	/*          P R O T O T Y P E   F U N C T I O N          */
	/*********************************************************/
	
	/*********************************************************/
	/*                       C L A S S                       */
	/*********************************************************/
	namespace sdcv {
		const int SDCV_FS_WRITE = 0;
		const int SDCV_FS_APPEND = 1;



		typedef struct {
			int RowOffset;
			int ColOffset;
			int RowLast;
			int ColLast;

			void operator()(void)
			{
				RowOffset = -1;
				ColOffset = -1;
				RowLast = 0;
				ColLast = 0;
			}

		} csvparam_t;


		template <typename T, typename U> 
		T find_last(T first, T last, U value2find);

		/*!
		 * @name	csvread
		 * @brief	Parsing a xsv file into a cv::Mat
		 */
		cv::Mat fs_read(const std::string& filename, char charSeparated = ',', int RowOffset = 0, int ColOffset = 0, int RowLast = -1, int ColLast = -1);
		
		/*!
		 * @name	csvwrite
		 * @brief	Write a cv::Mat into a xsv file
		 */
		void fs_write(const std::string filename, char charSeparated, cv::Mat data);
		void fs_write(const std::string filename, cv::Mat data, int flag = 0, std::string header = "");
		
		/*!
		 * @name	fs_exist
		 * @brief	Test if a file exist
		 */
		bool fs_exist(std::string filename);
		
		/*!
		 * @name	fs_split_name
		 * @brief	Split the file path with name into three fields {DIRECTORY, NAME, EXTENSION}
		 */
		void fs_split_path(std::string filename, std::vector<std::string> &splitted);
		std::tuple<std::string, std::string, std::string> fs_fileparts(std::string filename);


		class FileDialog {	
			public:
				inline static bool exist(std::string filename);
				inline static void csvread(const std::string filename, cv::OutputArray dst, csvparam_t param);
				inline static void csvwrite(const std::string filename, cv::InputArray src, int mode = SDCV_FS_WRITE, std::string header = "");

				
				static std::string getOpenFilename(wchar_t filter[] = (wchar_t *)L"All files\0*.*\0", HWND hWnd = NULL);
				static std::string getSaveFilename(wchar_t filter[] = (wchar_t *)L"All files\0*.*\0", HWND hWnd = NULL);

			private:

		};

		
		
	}
	
#endif /* SDCV_FILES_HPP */

/*! ************** End of file ----------------- CINVESTAV GDL */