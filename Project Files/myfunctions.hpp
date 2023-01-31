// Project 2 of CS 5330: Pattern Recognition and Computer Vision
// Created by Dhruvil Parikh

#ifndef myfunctions_hpp
#define myfunctions_hpp

#include <iostream>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "myfunctions.cpp"

using namespace std;
using namespace cv;

bool sortPairedVector(const pair<string, float> &a, const pair<string, float> &b);

void MatchingResult(vector<pair<string, float>> &distance_metric, int N);

// ##############################################################################################################################

cv::Vec3s applyHorizontalSobelX(const cv::Mat &src, int i, int j);

cv::Vec3s applyVerticalSobelX(const cv::Mat &src, int i, int j);

int sobelX3x3(const cv::Mat &src, cv::Mat &dst);

// SOBELY3x3
cv::Vec3s applyHorizontalSobelY(const cv::Mat &src, int i, int j);

cv::Vec3s applyVerticalSobelY(const cv::Mat &src, int i, int j);

int sobelY3x3(const cv::Mat &src, cv::Mat &dst);

// MAGNITUDE GRADIENT IMAGE
cv::Vec3b computeMagnitude (cv::Mat &sx, cv::Mat &sy, int i, int j); 

int magnitude (cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);

// ##############################################################################################################################

void getFeature(Mat &src, Mat &feature);

float sumSquaredDiff(Mat &square1, Mat &square2);

void baselineMatching(Mat &feature_src, Mat &temp, float &distance);
// ##############################################################################################################################

void getRGHistFeature(Mat &src, int nBins, float **featureHist);

void getRGHistFeature2D(Mat &src, int nBins, float *featureHist);

void getRGBHistFeature3D(Mat &src, int nBins, float *featureHist);

float histogramIntersection(float **featureHist1, float **featureHist2, int Hsize, float *distanceArray);

float histogramIntersection2D(float *featureHist1, float *featureHist2, int Hsize);

float histogramIntersection3D(float *featureHist1, float *featureHist2, int Hsize);

float sumSquaredDiff3D(float *featureHist1, float *featureHist2, int Hsize);

void histogramMatching(float **feature_src_2DHist, float **feature_temp_2DHist, Mat &temp, float &distance, int Hsize, float *distanceArray);

void histogramMatching3D(float *feature_src_3DHist, float *feature_temp_3DHist, Mat &temp, float &distance, int Hsize);

// #######################################################################################################################################################

void splitImage(Mat &src, Mat &dstTop, Mat &dstBottom, int split_row);

void multiHistogramMatching3D(Mat &temp, Mat &temp_top, Mat &temp_bottom, float *feature_src_3DHist_top, float *feature_src_3DHist_bottom,
                              float *feature_temp_3DHist_top, float *feature_temp_3DHist_bottom, float &distance, int Hsize, int split_row);

// #######################################################################################################################################################

void readFiles(char *dirname, string target_image, int flag_task, int N, int Hsize);

#endif