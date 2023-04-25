#ifndef __HELPERS__
#define __HELPERS__

/* ADMM helper functions */
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/hal/interface.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <numeric>
#include "classes.hpp"
#include "configs.hpp"


std::vector<std::chrono::microseconds> ADMM_Step_times;
std::vector<std::chrono::microseconds> C_times;
std::vector<std::chrono::microseconds> CT_times;
std::vector<std::chrono::microseconds> M_times;
std::vector<std::chrono::microseconds> MT_times;
std::vector<std::chrono::microseconds> Psi_times;
std::vector<std::chrono::microseconds> PsiT_times;
std::vector<std::chrono::microseconds> U_update_times;
std::vector<std::chrono::microseconds> V_update_times;
std::vector<std::chrono::microseconds> W_update_times;
std::vector<std::chrono::microseconds> X_update_times;
std::vector<std::chrono::microseconds> eta_update_times;
std::vector<std::chrono::microseconds> fftshift_times;
std::vector<std::chrono::microseconds> ifftshift_times;
std::vector<std::chrono::microseconds> init_Matrices_times;
std::vector<std::chrono::microseconds> precompute_H_fft_times;
std::vector<std::chrono::microseconds> precompute_PsiTPsi_times;
std::vector<std::chrono::microseconds> precompute_R_divmat_times;
std::vector<std::chrono::microseconds> precompute_X_divmat_times;
std::vector<std::chrono::microseconds> r_calc_times;
std::vector<std::chrono::microseconds> rho_update_times;
std::vector<std::chrono::microseconds> roll_times;
std::vector<std::chrono::microseconds> runADMM_times;
std::vector<std::chrono::microseconds> softThresh_times;
std::vector<std::chrono::microseconds> xi_update_times;
std::vector<std::chrono::microseconds> removeColumn_times;
std::vector<std::chrono::microseconds> removeRow_times;


//------------------------//
/* Soft threshold function */
// DONE
// src is two channel matrix, tau is a float value
// output is two channel matrix same size as src
//------------------------//
void softThresh(cv::Mat *dest, cv::Mat *src, float tau){
	class Timer t(&softThresh_times); // start timer
  struct Operator {
    float tau;
    cv::Mat *modMat;
    void operator ()(float &pixel, const int * position) const {
      if(pixel>0){
        modMat->at<float>(position[0], position[1]) = std::max(0.0f, pixel - tau);
      } else {
        modMat->at<float>(position[0], position[1]) = -(std::max(0.0f, -pixel - tau));
      }
    }
  };
  Operator thresh;
  thresh.tau = tau;
  thresh.modMat = dest;
  (*src).forEach<float>(thresh);
}


//------------------------//
/* Roll function */
// DONE
//------------------------//
/* Roll function similar to np.roll in python */
void roll(cv::Mat *dest, cv::Mat *src, int shift, int axis){
	class Timer t(&roll_times); // start timer
  src->copyTo(*dest);

  if (axis==0) {
    for(size_t i=0; i<shift; i++){
      cv::Mat temp(0, dest->cols, dest->type());
      dest->row((dest->rows)-1).copyTo(temp);
      dest->pop_back(1);
      cv::vconcat(temp, *dest, *dest);
    }
  }
  else if (axis==1) {
    cv::rotate(*dest, *dest, cv::ROTATE_90_CLOCKWISE);
    for(size_t i=0; i<shift; i++){
      cv::Mat temp(0, dest->cols, dest->type());
      dest->row((dest->rows)-1).copyTo(temp);
      dest->pop_back(1);
      cv::vconcat(temp, *dest, *dest);
    }
    cv::rotate(*dest, *dest, cv::ROTATE_90_COUNTERCLOCKWISE);
  }
}

//------------------------//
/* Psi function */
// DONE
//------------------------//
void Psi(cv::Mat *dest, cv::Mat *src){
	class Timer t(&Psi_times); // start timer
  cv::Mat v0, v1;
  roll(&v0, src, 1, 0);
  cv::subtract(v0, *src, v0);
  roll(&v1, src, 1, 1);
  cv::subtract(v1, *src, v1);
	cv::merge(std::vector<cv::Mat>{v0, v1}, *dest);
}


//------------------------//
/* U_update function */
// DONE
//------------------------//
void U_update(cv::Mat *dest, cv::Mat *eta, cv::Mat *image_est, float tau){
	class Timer t(&U_update_times); // start timer
	cv::Mat p1;
	Psi(&p1, image_est);
	p1 = p1 + (*eta)/mu2;
	cv::Mat p1_plane[p1.channels()];
	cv::split(p1, p1_plane);

	cv::Mat dest_plane[p1.channels()];
	cv::split(*dest, dest_plane);

	dest_plane[0] = cv::Mat::zeros(p1.rows, p1.cols, CV_32F);
	softThresh(dest_plane, &p1_plane[0], (tau/mu2));

	dest_plane[1] = cv::Mat::zeros(p1.rows, p1.cols, CV_32F);
	softThresh(&dest_plane[1], &p1_plane[1], (tau/mu2));

	cv::merge(dest_plane, p1.channels(), *dest);
}


//------------------------//
/* M function */
// DONE
//------------------------//
void removeColumn(cv::Mat& mat, int colIndex){
	class Timer t(&removeColumn_times); // start timer
  if (colIndex < 0 || colIndex >= mat.cols) {
      throw std::out_of_range("Invalid column index");
  }

  // Create a new matrix with one less column
  cv::Mat newMat(mat.rows, mat.cols - 1, mat.type());

  // Copy the columns up to the one to be removed
  mat(cv::Rect(0, 0, colIndex, mat.rows)).copyTo(newMat(cv::Rect(0, 0, colIndex, mat.rows)));

  // Copy the columns after the one to be removed
  mat(cv::Rect(colIndex + 1, 0, mat.cols - colIndex - 1, mat.rows)).copyTo(newMat(cv::Rect(colIndex, 0, mat.cols - colIndex - 1, mat.rows)));

  // Update the original matrix with the new one
  mat = newMat;
}

void removeRow(cv::Mat& mat, int rowIndex){
	class Timer t(&removeRow_times); // start timer
  if (rowIndex < 0 || rowIndex >= mat.rows) {
      throw std::out_of_range("Invalid row index");
  }

  // Create a new matrix with one less row
  cv::Mat newMat(mat.rows - 1, mat.cols, mat.type());

  // Copy the rows up to the one to be removed
  mat(cv::Rect(0, 0, mat.cols, rowIndex)).copyTo(newMat(cv::Rect(0, 0, mat.cols, rowIndex)));

  // Copy the rows after the one to be removed
  mat(cv::Rect(0, rowIndex + 1, mat.cols, mat.rows - rowIndex - 1)).copyTo(newMat(cv::Rect(0, rowIndex, mat.cols, mat.rows - rowIndex - 1)));

  // Update the original matrix with the new one
  mat = newMat;
}

// fftshift implementation
void fftshift(cv::Mat *dest, cv::Mat *src){
	class Timer t(&fftshift_times); // start timer
  int bottom = 0;
  int right  = 0;

  // Check Matrix dimensions
  if (src->cols == src->rows){
    if ((src->cols)%2){
      bottom = 1;
      right  = 1;}
  }
  else{
    if (src->cols > src->rows){
      bottom = (src->rows % 2 ? 1 : 0);
      right  = (src->cols % 2 ? 1 : 0);}
    else{
      bottom = (src->rows % 2 ? 1 : 0);
      right  = (src->cols % 2 ? 1 : 0);}
  }

  cv::copyMakeBorder(*src, *dest, 0, bottom, 0, right, cv::BORDER_CONSTANT, 0);

	int cy = dest->rows/2;
	int cx = dest->cols/2;

  cv::Mat q0(*dest, cv::Rect(0, 0, cx, cy));   // Top-Left - Create an ROI per quadrant
	cv::Mat q1(*dest, cv::Rect(cx, 0, cx, cy));  // Top-Right
	cv::Mat q2(*dest, cv::Rect(0, cy, cx, cy));  // Bottom-Left
	cv::Mat q3(*dest, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

	cv::Mat tmp;       // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);

  int colToRemove = (right  == 0 ? 0 : dest->cols/2-1);
  int rowToRemove = (bottom == 0 ? 0 : dest->rows/2-1);

  if(rowToRemove) removeRow(*dest, rowToRemove);
  if(colToRemove) removeColumn(*dest, colToRemove);
}


// ifftshift implementation
void ifftshift(cv::Mat *dest, cv::Mat *src){
	class Timer t(&ifftshift_times); // start timer
  int top  = 0;
  int left = 0;

  // Check Matrix dimensions
  if (src->cols == src->rows){
    if ((src->cols)%2){
      top = 1;
      left  = 1;}
  }
  else{
    if (src->cols > src->rows){
      top = (src->rows % 2 ? 1 : 0);
      left  = (src->cols % 2 ? 1 : 0);}
    else{
      top = (src->rows % 2 ? 1 : 0);
      left  = (src->cols % 2 ? 1 : 0);}
  }

  cv::copyMakeBorder(*src, *dest, top, 0, left, 0, cv::BORDER_CONSTANT, 0);

	int cy = dest->rows/2;
	int cx = dest->cols/2;

  cv::Mat q0(*dest, cv::Rect(cx, cy, cx, cy));   // Top-Left - Create an ROI per quadrant
	cv::Mat q1(*dest, cv::Rect(0, cy, cx, cy));  // Top-Right
	cv::Mat q2(*dest, cv::Rect(cx, 0, cx, cy));  // Bottom-Left
	cv::Mat q3(*dest, cv::Rect(0, 0, cx, cy)); // Bottom-Right

	cv::Mat tmp;       // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);

  int colToRemove = (left == 0 ? 0 : dest->cols/2);
  int rowToRemove = (top  == 0 ? 0 : dest->rows/2);

  if(rowToRemove) removeRow(*dest, rowToRemove);
  if(colToRemove) removeColumn(*dest, colToRemove);
}

// M function implementation
void M(cv::Mat *dest, cv::Mat *vk, cv::Mat *H_fft){
	class Timer t(&M_times); // start timer
	ifftshift(dest, vk);
	cv::dft(*dest, *dest, cv::DFT_COMPLEX_OUTPUT);
	cv::mulSpectrums(*dest, *H_fft, *dest, 0, false);
	cv::dft(*dest, *dest, cv::DFT_INVERSE | cv::DFT_SCALE);
	fftshift(dest, dest);
	cv::Mat planes[2];
	cv::split(*dest, planes);
	*dest = planes[0];
}


//------------------------//
/* C function */
// DONE
//------------------------//
void C(cv::Mat *dest, cv::Mat *src){
	class Timer t(&C_times); // start timer
	int full_size[2];

	full_size[0] = sensor_size[0] * 2;
	full_size[1] = sensor_size[1] * 2;

	int top 		= (full_size[0] - sensor_size[0])/2;
	int bottom 	= (full_size[0] + sensor_size[0])/2;
	int left 		= (full_size[1] - sensor_size[1])/2;
	int right 	= (full_size[1] + sensor_size[1])/2;

	// imagine ROI as a vector, first two parameters are the starting point, last two are the end point
  *dest = cv::Mat(*src, cv::Rect(left, top, right-left, bottom-top));
}


//------------------------//
/* CT function */
// DONE
//------------------------//
void CT(cv::Mat *dest, cv::Mat *src){
	class Timer t(&CT_times); // start timer
	int full_size[2];

	full_size[0] = sensor_size[0] * 2;
	full_size[1] = sensor_size[1] * 2;

	int v_pad = (full_size[0] - sensor_size[0])/2;
	int h_pad = (full_size[1] - sensor_size[1])/2;
	cv::copyMakeBorder(*src, *dest, v_pad, v_pad, h_pad, h_pad, cv::BORDER_CONSTANT, 0);
}


//------------------------//
/* precompute_X_divmat function */
// DONE
//------------------------//
void precompute_X_divmat(cv::Mat *dest){
	class Timer t(&precompute_X_divmat_times); // start timer
	cv::Mat tmp = cv::Mat::ones(sensor_size[0], sensor_size[1] , CV_32F);
	CT(&tmp, &tmp);
	*dest = 1.0f/(tmp + mu1);
}

//------------------------//
/* W_update function */
// DONE
//------------------------//
void W_update(cv::Mat *dest, cv::Mat *rho, cv::Mat *image_est){
	class Timer t(&W_update_times); // start timer
	*dest = cv::max((((*rho)/mu3) + (*image_est)), 0.0f);
}


//------------------------//
/* PsiT function */
// DONE 100%
//------------------------//
// src should be a 2 channel matrix
void PsiT(cv::Mat *dest, cv::Mat *src){
	class Timer t(&PsiT_times); // start timer
	cv::Mat frames[src->channels()];
	cv::Mat diffs[src->channels()];

	cv::split(*src, frames);

	roll(&diffs[0], &frames[0], (frames[0].rows - 1), 0);
	diffs[0] = diffs[0] - frames[0];

	roll(&diffs[1], &frames[1], (frames[1].cols - 1), 1);
	diffs[1] = diffs[1] - frames[1];

	*dest = diffs[0] + diffs[1];
}

//------------------------//
/* MT function */
// DONE 100%
//------------------------//
void MT(cv::Mat *dest, cv::Mat *x, cv::Mat *H_fft){
	class Timer t(&MT_times); // start timer
	cv::Mat x_planes[x->channels()];

	cv::split(*x, x_planes);
	for(int i = 0; i < x->channels(); i++){
		ifftshift(&x_planes[i], &x_planes[i]);
	}

	cv::Mat x_zeroed;
	cv::merge(x_planes, x->channels() , x_zeroed);

	cv::dft(x_zeroed , x_zeroed, cv::DFT_COMPLEX_OUTPUT);

	cv::mulSpectrums(x_zeroed, *H_fft, *dest, 0, true);

	cv::dft(*dest, *dest, cv::DFT_INVERSE | cv::DFT_SCALE);
	cv::Mat planes[2];
	cv::split(*dest, planes);
	*dest = planes[0];
	fftshift(dest, dest);
}


//------------------------//
/* r_calc function */
// DONE 100%
//------------------------//
void r_calc(cv::Mat *dest, cv ::Mat *w, cv::Mat *rho, cv::Mat *u, cv::Mat *eta, cv::Mat *x, cv::Mat *xi, cv::Mat *H_fft){
	class Timer t(&r_calc_times); // start timer

	cv::Mat p1;
	p1 = w->mul(mu3);
	cv::subtract(p1, *rho, p1);

	cv::Mat p2;
	p2 = u->mul(mu2);
	cv::subtract(p2, *eta, p2);
	PsiT(&p2, &p2);

	cv::Mat p3;
	p3 = x->mul(mu1);
	cv::subtract(p3, *xi, p3);
	MT(&p3, &p3, H_fft);

	*dest = p1 + p2 + p3;
}


//------------------------//
/* X_update function */
// DONE 100%
//------------------------//
void X_update(cv::Mat *dest, cv::Mat *xi, cv::Mat *image_est, cv::Mat *H_fft, cv::Mat *sensor_reading, cv::Mat *X_divmat){
	class Timer t(&X_update_times); // start timer
	cv::Mat p2;
	M(&p2, image_est, H_fft);
	cv::multiply(p2, mu1, p2);

	cv::Mat p3;
	CT(&p3, sensor_reading);

	*dest = *xi + p2 + p3;
	cv::multiply(*dest, *X_divmat, *dest);
}


//------------------------//
/* V_update function */
// Done 100%
//------------------------//
void V_update(cv::Mat *dest, cv::Mat *w, cv::Mat *rho, cv::Mat *u, cv::Mat *eta, cv::Mat *x, cv::Mat *xi, cv::Mat *H_fft, cv::Mat *R_divmat){
	class Timer t(&V_update_times); // start timer
	cv::Mat p1;
	r_calc(&p1, w, rho, u, eta, x, xi, H_fft);
	ifftshift(&p1, &p1);
	cv::dft(p1, p1, cv::DFT_COMPLEX_OUTPUT);

	cv::Mat freq_space_result;
	cv::Mat imaginary = cv::Mat::zeros(R_divmat->size(), CV_32F);
	cv::Mat R_divmat_planes[] = {*R_divmat , imaginary};
	cv::merge(R_divmat_planes, 2, freq_space_result);

	cv::mulSpectrums(p1, freq_space_result , freq_space_result, 0, false);

	cv::dft(freq_space_result, freq_space_result, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT);
	cv::Mat planes[2];
	cv::split(freq_space_result, planes);
	fftshift(dest, &planes[0]);
}


//------------------------//
/* precompute_PsiTPsi function */
// DONE
//------------------------//
void precompute_PsiTPsi(cv::Mat *dest){
	class Timer t(&precompute_PsiTPsi_times); // start timer
	int full_size[2];
	full_size[0] = sensor_size[0] * 2;
	full_size[1] = sensor_size[1] * 2;
	*dest = cv::Mat::zeros(full_size[0], full_size[1], CV_32F);
	dest->at<float>(0, 0) = 4;
	dest->at<float>(0, 1) = -1;
	dest->at<float>(1, 0) = -1;
	dest->at<float>(0, full_size[1] - 1) = -1;
	dest->at<float>(full_size[0] - 1, 0) = -1;
	cv::dft(*dest, *dest, cv::DFT_COMPLEX_OUTPUT);
}


//------------------------//
/* precompute_R_divmat function */
// DONE
//------------------------//
void precompute_R_divmat(cv::Mat *dest, cv::Mat *H_fft, cv::Mat *PsiTPsi){
	class Timer t(&precompute_R_divmat_times); // start timer

	cv::Mat MTM_component;
	cv::Mat H_fft_result;
	cv::mulSpectrums(*H_fft, *H_fft, H_fft_result, cv::DFT_COMPLEX_OUTPUT, true);
	cv::Mat H_fft_planes[H_fft_result.channels()];
	cv::split(H_fft_result, H_fft_planes);
	cv::magnitude(H_fft_planes[0], H_fft_planes[1], H_fft_result);
	MTM_component = H_fft_result.mul(mu1);

	cv::Mat PsiTPsi_component;
	cv::Mat PsiTPsi_planes[PsiTPsi->channels()];
	cv::split(*PsiTPsi, PsiTPsi_planes);
	cv::magnitude(PsiTPsi_planes[0], PsiTPsi_planes[1], PsiTPsi_component);
	PsiTPsi_component = PsiTPsi_component.mul(mu2);

	float id_component = mu3;

	// calculate the final result
	*dest = 1.0f/(MTM_component + PsiTPsi_component + id_component);
}


//------------------------//
/* xi_update function */
//
//------------------------//
void xi_update(cv::Mat *xi, cv::Mat *V, cv::Mat *H_fft, cv::Mat *X){
	class Timer t(&xi_update_times); // start timer
	cv::Mat dest;
	M(&dest, V, H_fft);
	cv::subtract(dest, *X, dest);
	dest = dest.mul(mu1);
	cv::add(dest, *xi, dest);
	dest.copyTo(*xi);
}


//------------------------//
/* eta_update function */
//
//------------------------//
void eta_update(cv::Mat *eta, cv::Mat *V, cv::Mat *U){
	class Timer t(&eta_update_times); // start timer
	cv::Mat dest;
	Psi(&dest, V);
	cv::subtract(dest, *U, dest);
	dest = dest.mul(mu2);
	cv::add(dest, *eta, dest);
	dest.copyTo(*eta);
}


//------------------------//
/* rho_update function */
//
//------------------------//
void rho_update(cv::Mat *rho, cv::Mat *V, cv::Mat *W){
	class Timer t(&rho_update_times); // start timer
	cv::Mat dest;
	dest = ((*V - *W) * mu3) + *rho;
	dest.copyTo(*rho);
}


//------------------------//
/* init_Matrices function */
//
//------------------------//
void init_Matrices(cv::Mat *X, cv::Mat *U, cv::Mat *V, cv::Mat *W, cv::Mat *xi, cv::Mat *eta, cv::Mat *rho, cv::Mat *H_fft){
	class Timer t(&init_Matrices_times); // start timer
	int full_size[2];
	full_size[0] = sensor_size[0] * 2;
	full_size[1] = sensor_size[1] * 2;
	// initialize X, U, V, W, xi, eta, rho
	*X		= cv::Mat::zeros(full_size[0], full_size[1], CV_32F);
	*U		= cv::Mat::zeros(full_size[0], full_size[1], CV_32FC2);
	*V		= cv::Mat::zeros(full_size[0], full_size[1], CV_32F);
	*W 		= cv::Mat::zeros(full_size[0], full_size[1], CV_32F);
	*xi		= cv::Mat::zeros(full_size[0], full_size[1], CV_32F);
	*eta	= cv::Mat::zeros(full_size[0], full_size[1], CV_32FC2);
	*rho	= cv::Mat::zeros(full_size[0], full_size[1], CV_32F);
}


//------------------------//
/* precompute_H_fft function */
// DONE 100%
//------------------------//
void precompute_H_fft(cv::Mat *dest, cv::Mat *psf){
	class Timer t(&precompute_H_fft_times); // start timer
	CT(dest, psf);
	ifftshift(dest, dest);
	cv::dft(*dest, *dest, cv::DFT_COMPLEX_OUTPUT);
}


//------------------------//
/* ADMM_Step function */
//
//------------------------//
void ADMM_Step(cv::Mat *X, cv::Mat *U, cv::Mat *V, cv::Mat *W, cv::Mat *xi, cv::Mat *eta, cv::Mat *rho, cv::Mat *H_fft, cv::Mat *data, cv::Mat *X_divmat, cv::Mat *R_divmat){
	class Timer t(&ADMM_Step_times); // start timer
	U_update(U, eta, V, tau);
	X_update(X, xi, V, H_fft, data, X_divmat);
	V_update(V, W, rho, U, eta, X, xi, H_fft, R_divmat);
	W_update(W, rho, V);
	xi_update(xi, V, H_fft, X);
	eta_update(eta, V, U);
	rho_update(rho, V, W);
}


//------------------------//
/* runADMM function */
//
//------------------------//
cv::Mat runADMM(cv::Mat *psf, cv::Mat *data){
	class Timer t(&runADMM_times); // start timer

	cv::Mat H_fft;
	precompute_H_fft(&H_fft, psf);

	cv::Mat X, U, V, W, xi, eta, rho;
	init_Matrices(&X, &U, &V, &W, &xi, &eta, &rho, &H_fft);

	cv::Mat X_divmat;
	precompute_X_divmat(&X_divmat);

	cv::Mat PsiTPsi;
	precompute_PsiTPsi(&PsiTPsi);

	cv::Mat R_divmat;
	precompute_R_divmat(&R_divmat, &H_fft, &PsiTPsi);

	cv::Mat dest;
	for (int i = 0; i < iters; i++){
		std::cout << "ADMM step " << i+1 << "/" << iters << std::endl;
		ADMM_Step(&X, &U, &V, &W, &xi, &eta, &rho, &H_fft, data, &X_divmat, &R_divmat);
	}

	cv::Mat image;
	C(&image, &V);
	cv::threshold(image, image, 0.0, 0.0, cv::THRESH_TOZERO);

	return image;
}



void printFunctionTiming(std::vector<std::chrono::microseconds> durations, std::string functionName){
	class csvfile file(CSV_FILE);
	double sum = 0;

	file << functionName;
	for (auto i : durations){
		file << i.count();
		sum += i.count();}
	file << endrow;

	file << "size of" << functionName;
	file << durations.size() << endrow;
	file << "mean of" << functionName;
	file << sum/(durations.size()) << endrow;

	double std_dev = 0;
	for (auto i : durations)
		std_dev += pow(((double)i.count() - sum/(durations.size())),2);
	file << "std dev of" << functionName;
	file << sqrt(std_dev/(durations.size())) << endrow;
	file << endrow;
}


// 27 functions to time
void printTimings(){
	std::remove(CSV_FILE);  //remove the file if it exists
	class csvfile file(CSV_FILE);
	file << "++++++++++++++++++++++++++++++++++++++++++++++++++++" << endrow;

	printFunctionTiming(ADMM_Step_times, "ADMM_Step");

	printFunctionTiming(C_times, "C");

	printFunctionTiming(CT_times, "CT");

	printFunctionTiming(M_times, "M");

	printFunctionTiming(MT_times, "MT");

	printFunctionTiming(Psi_times, "Psi");

	printFunctionTiming(PsiT_times, "PsiT");

	printFunctionTiming(U_update_times, "U_update");

	printFunctionTiming(V_update_times, "V_update");

	printFunctionTiming(W_update_times, "W_update");

	printFunctionTiming(X_update_times, "X_update");

	printFunctionTiming(eta_update_times, "eta_update");

	printFunctionTiming(fftshift_times, "fftshift");

	printFunctionTiming(ifftshift_times, "ifftshift");

	printFunctionTiming(init_Matrices_times, "init_Matrices");

	printFunctionTiming(precompute_H_fft_times, "precompute_H_fft");

	printFunctionTiming(precompute_PsiTPsi_times, "precompute_PsiTPsi");

	printFunctionTiming(precompute_R_divmat_times, "precompute_R_divmat");

	printFunctionTiming(precompute_X_divmat_times, "precompute_X_divmat");

	printFunctionTiming(r_calc_times, "r_calc");

	printFunctionTiming(rho_update_times, "rho_update");

	printFunctionTiming(roll_times, "roll");

	printFunctionTiming(runADMM_times, "runADMM");

	printFunctionTiming(softThresh_times, "softThresh");

	printFunctionTiming(xi_update_times, "xi_update");

	printFunctionTiming(removeColumn_times, "removeColumn");

	printFunctionTiming(removeRow_times, "removeRow");

	file << "-----------------------------------------------------" << endrow;
}

#endif
