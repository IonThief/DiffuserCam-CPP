#ifndef __CONFIGS__
#define __CONFIGS__

/* ADMM config */
#include <iostream>

std::string psfname = "../images/psf_sample.tif";
std::string dataname = "../images/rawdata_hand_sample.tif";
float f = 0.25;
float mu1 = 1.0e-6;
float mu2 = 1.0e-5;
float mu3 = 4.0e-5;
float tau = 0.0001;
int iters = 15;
// sensor size is initialized in main.cpp
// it will be global to all files, later i will try to find a better way
int sensor_size[2];

#endif
