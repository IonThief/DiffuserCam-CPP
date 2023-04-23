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


  //------------------------//
  // Resize input images
	// Correct
  //------------------------//
	cv::resize(psf, psf, cv::Size(), f, f, cv::INTER_AREA);
	cv::resize(data, data, cv::Size(), f, f, cv::INTER_AREA);


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

	// NOTICE: image should be converted to another type to show
	image.convertTo(image, CV_32F, 1000.0, 0.0); // alpha value increases the contrast of the image, beta value is the brightness

	cv::imwrite("output.png", image);
  namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
  imshow("Display Image", image);
  cv::waitKey(0);

	printTimings(); // print timing information

	// IT WORKS FINE AND THE HAND APPEARS
	std::cout << "Done" << std::endl;

  return 0;
}
