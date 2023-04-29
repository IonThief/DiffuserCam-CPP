#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/hal/interface.h>
#include "helpers.hpp"
#include "configs.hpp"

int main(int argc, char** argv ){
  //------------------------//
  // Load Data
  // Read image data, 2D array and one channel Gray image
	// Correct
  //------------------------//
  cv:: Mat psf, data;

  cv::Mat img_1  = cv::imread( psfname, cv::IMREAD_UNCHANGED); //Read Point Spread Function
  img_1.convertTo(psf, CV_32F);

  cv::Mat img_2 = cv::imread( dataname, cv::IMREAD_UNCHANGED); //Read sensor data
  img_2.convertTo(data, CV_32F);


  //------------------------//
  // Remove Non-trivial background
	// Correct
  //------------------------//
  cv::Rect roi(5, 5, 10, 10);
  cv::Scalar bg = cv::mean(psf(roi));
  psf  -= bg[0];
  data -= bg[0];


	std::remove("timePerResolution.csv");
	std::remove("time.csv");

	class csvfile file("timePerResolution.csv");
	file << "Resolution" << "Time (ms)" << endrow;

	// I should've used a function to find the common factor for decrementing image size
	//for (int i =0; psf.size().width >= 400; i++){
	for (int i =0; i < 1; i++){
  //------------------------//
  // Resize input images
	// Correct
  //------------------------//
	cv::resize(psf, psf, cv::Size(psf.size().width-40.0, psf.size().height-30.0));
	cv::resize(data, data, cv::Size(data.size().width-40.0, data.size().height-30.0));


  //------------------------//
  // Normalize input images
	// Correct
  //------------------------//
	cv::normalize(psf, psf, 1, 0 , cv::NORM_L2, -1);
	cv::normalize(data, data, 1, 0 , cv::NORM_L2, -1);


	//------------------------//
	// get the size of the input images
	//------------------------//
	sensor_size[0] = psf.rows;
	sensor_size[1] = psf.cols;

	cv::Mat image = runADMM(&psf, &data);

	file << std::to_string(sensor_size[0]) + "x" + std::to_string(sensor_size[1]) << averageStepTimes() << endrow;

	/*****************
	** // NOTICE: image should be converted to another type to show
	** image.convertTo(image, CV_32F, 2000.0, 0.0); // alpha value increases the contrast of the image, beta value is the brightness

	** // Save the image
	** cv::FileStorage file("constructed_image.ext", cv::FileStorage::WRITE);
	** file << "result image" << image;

	** // Show the image
  ** namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
  ** imshow("Display Image", image);
  ** cv::waitKey(0);
	*****************/

	class csvfile file2("time.csv");
	file2 << "Run #" + std::to_string(i) << endrow;
	printTimings(); // print timing information

	// IT WORKS FINE AND THE HAND APPEARS
	std::cout << "Finished run " << i << std::endl;
	std::cout << "-------------------------" << std::endl;
	}

  return 0;
}
