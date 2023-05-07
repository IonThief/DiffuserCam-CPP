#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/hal/interface.h>
#include "helpers.hpp"
#include "configs.hpp"
#include "Csvfile.hpp"
#include "pylon/PylonIncludes.h"
#include "pylon/usb/BaslerUsbInstantCamera.h"

int main(int argc, char** argv ){

	Pylon::PylonInitialize();

	Pylon::CBaslerUsbInstantCamera camera(Pylon::CTlFactory::GetInstance().CreateFirstDevice());
	std::cout << "Using device " << camera.GetDeviceInfo().GetModelName() << std::endl;


  camera.RegisterConfiguration( new Pylon::CSoftwareTriggerConfiguration,
																Pylon::RegistrationMode_ReplaceAll,
																Pylon::Cleanup_Delete );
	Pylon::CGrabResultPtr ptrGrabResult;
	cv::Mat sensor_data;

	camera.StartGrabbing(1);
	while(camera.IsGrabbing()){

    camera.WaitForFrameTriggerReady( 1000, Pylon::TimeoutHandling_ThrowException );
    camera.ExecuteSoftwareTrigger();
    camera.RetrieveResult( 5000, ptrGrabResult, Pylon::TimeoutHandling_ThrowException );

		if (ptrGrabResult->GrabSucceeded()){
			sensor_data = cv::Mat(ptrGrabResult->GetHeight(),
														ptrGrabResult->GetWidth(),
														CV_8UC1, (uint8_t *)ptrGrabResult->GetBuffer());

			cv::imshow("image", sensor_data);
			cv::waitKey(1000);
		}
	}



  cv:: Mat psf, data;

  cv::Mat img_1  = cv::imread( psfname, cv::IMREAD_UNCHANGED); //Read Point Spread Function
  img_1.convertTo(psf, CV_32F);

  sensor_data.convertTo(data, CV_32FC1);


  cv::Rect roi(5, 5, 10, 10);
  cv::Scalar bg = cv::mean(psf(roi));
  psf  -= bg[0];
  data -= bg[0];


	std::remove("timePerResolution.csv");
	std::remove("time.csv");


  //------------------------//
  // Resize input images
	// Correct
  //------------------------//
	cv::resize(psf, psf, cv::Size(400, 300));
	cv::resize(data, data, cv::Size(400, 300));


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

	// print the size of the input images and their types
	std::cout << "psf size: " << psf.size() << std::endl;
	std::cout << "psf type: " << psf.type() << std::endl;
	std::cout << "data size: " << data.size() << std::endl;
	std::cout << "data type: " << data.type() << std::endl;



	cv::Mat image = runADMM(&psf, &data);


	// NOTICE: image should be converted to another type to show
	image.convertTo(image, CV_32F, 9000.0, 0.0); // alpha value increases the contrast of the image, beta value is the brightness


	// Show the image
  namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
  imshow("Display Image", image);
	cv::waitKey(0);
	cv::destroyAllWindows();


	// IT WORKS FINE AND THE HAND APPEARS
	std::cout << "-------------------------" << std::endl;

	camera.Close();
  return 0;
}
