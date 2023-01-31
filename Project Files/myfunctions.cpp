// Project 2 of CS 5330: Pattern Recognition and Computer Vision
// Created by Dhruvil Parikh

#include <iostream>

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <dirent.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

bool sortPairedVector(const pair<string, float> &a, const pair<string, float> &b)
{
    return (a.second < b.second);
}

void MatchingResult(vector<pair<string, float>> &distance_metric, int N)
{

    Mat result_images;

    sort(distance_metric.begin(), distance_metric.end(), sortPairedVector);

    for (int i = 0; i < N; i++)
    {
        cout << distance_metric[i].first << " " << distance_metric[i].second << endl;
        result_images = imread(distance_metric[i].first, IMREAD_COLOR);
        namedWindow("Image" + to_string(i), 1);
        imshow("Image" + to_string(i), result_images);
    }

    waitKey();

    destroyAllWindows();
}

// ##############################################################################################################################

cv::Vec3s applyHorizontalSobelX(const cv::Mat &src, int i, int j)
{
    cv::Vec3s sum = {0, 0, 0};
    int c = 0;
    int filter_rows = 3, filter_cols = 3;

    for (c = 0; c < 3; c++)
    {

        sum[c] = (-1) * src.at<cv::Vec3b>(i, j - 1)[c] + 0 * src.at<cv::Vec3b>(i, j)[c] + 1 * src.at<cv::Vec3b>(i, j + 1)[c];
    }

    return (sum);
}

cv::Vec3s applyVerticalSobelX(const cv::Mat &src, int i, int j)
{
    cv::Vec3s sum = {0, 0, 0};
    int c = 0;
    int filter_rows = 3, filter_cols = 3;

    for (c = 0; c < 3; c++)
    {
        sum[c] = 1 * src.at<cv::Vec3s>(i - 1, j)[c] + 2 * src.at<cv::Vec3s>(i, j)[c] + 1 * src.at<cv::Vec3s>(i + 1, j)[c];
        sum[c] /= 4;
    }

    return (sum);
}

int sobelX3x3(const cv::Mat &src, cv::Mat &dst)
{

    int i = 0, j = 0;

    Mat temp;
    temp = Mat::zeros(src.rows, src.cols, CV_16SC3);

    for (i = 0; i < src.rows; i++)
    {
        for (j = 1; j < src.cols - 1; j++)
        {
            temp.at<cv::Vec3s>(i, j) = applyHorizontalSobelX(src, i, j);
        }
    }

    for (i = 1; i < src.rows - 1; i++)
    {
        for (j = 0; j < src.cols; j++)
        {
            dst.at<cv::Vec3s>(i, j) = applyVerticalSobelX(temp, i, j);
        }
    }
    
    return 0;
}

// SOBELY3x3
cv::Vec3s applyHorizontalSobelY(const cv::Mat &src, int i, int j)
{
    cv::Vec3s sum = {0, 0, 0};
    int c = 0;
    int filter_rows = 3, filter_cols = 3;

    for (c = 0; c < 3; c++)
    {

        sum[c] = 1 * src.at<cv::Vec3b>(i, j - 1)[c] + 2 * src.at<cv::Vec3b>(i, j)[c] + 1 * src.at<cv::Vec3b>(i, j + 1)[c];
        sum[c] /= 4;
    }

    return (sum);
}

cv::Vec3s applyVerticalSobelY(const cv::Mat &src, int i, int j)
{
    cv::Vec3s sum = {0, 0, 0};
    int c = 0;
    int filter_rows = 3, filter_cols = 3;

    for (c = 0; c < 3; c++)
    {
        sum[c] = 1 * src.at<cv::Vec3s>(i - 1, j)[c] + 0 * src.at<cv::Vec3s>(i, j)[c] + (-1) * src.at<cv::Vec3s>(i + 1, j)[c];
    }

    return (sum);
}

int sobelY3x3(const cv::Mat &src, cv::Mat &dst)
{

    int i = 0, j = 0;

    Mat temp;
    temp = Mat::zeros(src.rows, src.cols, CV_16SC3);

    for (i = 0; i < src.rows; i++)
    {
        for (j = 1; j < src.cols - 1; j++)
        {
            temp.at<cv::Vec3s>(i, j) = applyHorizontalSobelY(src, i, j);
        }
    }

    for (i = 1; i < src.rows - 1; i++)
    {
        for (j = 0; j < src.cols; j++)
        {
            dst.at<cv::Vec3s>(i, j) = applyVerticalSobelY(temp, i, j);
        }
    }

    return 0;
}

// MAGNITUDE GRADIENT IMAGE
cv::Vec3b computeMagnitude (cv::Mat &sx, cv::Mat &sy, int i, int j) {

    cv::Vec3b mag = {0, 0, 0};
    int c = 0;
    int sx_squared = 0, sy_squared = 0;

    for (c = 0; c < 3; c++ ) {
        
        mag[c] = sqrt(sx.at<cv::Vec3s>(i, j)[c] * sx.at<cv::Vec3s>(i, j)[c] + sy.at<cv::Vec3s>(i, j)[c] * sy.at<cv::Vec3s>(i, j)[c]);

    }

    return(mag);

}

int magnitude (cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {

    int i = 0, j = 0;

    for (i = 0; i < dst.rows ; i++) {
        for (j = 0; j < dst.cols; j++) {

            dst.at<cv::Vec3s>(i, j) = computeMagnitude(sx, sy, i, j);

        }
    }

    cv::convertScaleAbs(dst, dst);

    return 0;

}

// ##############################################################################################################################

void getFeature(Mat &src, Mat &feature)
{

    int x = 0, y = 0;

    for (int i = src.rows / 2 - 4; i < src.rows / 2 + 5; i++)
    {
        y = 0;
        for (int j = src.cols / 2 - 4; j < src.cols / 2 + 5; j++)
        {
            for (int c = 0; c < 3; c++)
            {
                feature.at<Vec3b>(x, y)[c] = src.at<Vec3b>(i, j)[c];
            }
            y++;
        }
        x++;
    }
}

float sumSquaredDiff(Mat &square1, Mat &square2)
{

    int diff = 0;
    int square = 0;
    int sum = 0;
    float result = 0;

    for (int i = 0; i < square1.rows; i++)
    {
        for (int j = 0; j < square1.cols; j++)
        {
            for (int c = 0; c < 3; c++)
            {
                diff = square1.at<Vec3b>(i, j)[c] - square2.at<Vec3b>(i, j)[c];
                square = diff * diff;
                sum += square;
            }
        }
    }

    result = sum / 243;

    return result;
}

void baselineMatching(Mat &feature_src, Mat &temp, float &distance)
{

    Mat feature_temp(9, 9, CV_8UC3, Scalar(0, 0, 0));

    // Get the middle 9x9 square feature from the Image:
    getFeature(temp, feature_temp);

    // Compute the Sum Squared Difference between two feature vectors:
    distance = sumSquaredDiff(feature_src, feature_temp);
}

// ##############################################################################################################################

void getRGHistFeature(Mat &src, int nBins, float **featureHist)
{

    int r = 0, g = 0, i = 0, j = 0;
    float coeff = src.rows * src.cols;
    float sum = 0;

    for (i = 0; i < src.rows; i++)
    {
        for (j = 0; j < src.cols; j++)
        {

            r = (nBins * src.at<Vec3b>(i, j)[2]) / (src.at<Vec3b>(i, j)[2] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[0] + 1);
            g = (nBins * src.at<Vec3b>(i, j)[1]) / (src.at<Vec3b>(i, j)[2] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[0] + 1);
            featureHist[r][g]++;
        }
    }

    for (i = 0; i < nBins; i++)
    {
        for (j = 0; j < nBins; j++)
        {
            featureHist[i][j] = featureHist[i][j] / coeff;
            // sum += featureHist[i][j];
        }
    }
}

void getRGHistFeature2D(Mat &src, int nBins, float *featureHist)
{

    int r = 0, g = 0, i = 0, j = 0;
    float coeff = src.rows * src.cols;
    float sum = 0;

    for (i = 0; i < src.rows; i++)
    {
        for (j = 0; j < src.cols; j++)
        {

            r = (nBins * src.at<Vec3b>(i, j)[2]) / (src.at<Vec3b>(i, j)[2] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[0] + 1);
            g = (nBins * src.at<Vec3b>(i, j)[1]) / (src.at<Vec3b>(i, j)[2] + src.at<Vec3b>(i, j)[1] + src.at<Vec3b>(i, j)[0] + 1);
            featureHist[r*nBins + g]++;
        }
    }

    for (i = 0; i < nBins*nBins; i++)
    {
            featureHist[i] = featureHist[i] / coeff;
    }
}

void getRGBHistFeature3D(Mat &src, int nBins, float *featureHist)
{

    int r = 0, g = 0, b = 0, i = 0, j = 0;
    int divisor = 256 / nBins;
    float coeff = src.rows * src.cols;
    float sum = 0;

    for (i = 0; i < src.rows; i++)
    {
        for (j = 0; j < src.cols; j++)
        {

            r = src.at<Vec3b>(i, j)[2] / divisor;
            g = src.at<Vec3b>(i, j)[1] / divisor;
            b = src.at<Vec3b>(i, j)[0] / divisor;
            featureHist[r * nBins * nBins + g * nBins + b]++;
        }
    }

    for (i = 0; i < nBins * nBins * nBins; i++)
    {
        featureHist[i] = featureHist[i] / coeff;
    }
}

float histogramIntersection(float **featureHist1, float **featureHist2, int Hsize, float *distanceArray)
{

    int i = 0, j = 0, k = 0;
    float distance = 0;

    for (i = 0; i < Hsize; i++)
    {
        for (j = 0; j < Hsize; j++)
        {
            distanceArray[k] = ((featureHist1[i][j] > featureHist2[i][j]) ? featureHist2[i][j] : featureHist1[i][j]);
            k++;
        }
    }

    for (i = 0; i < Hsize * Hsize; i++)
    {
        distance += distanceArray[i];
    }

    distance = 1 - distance;

    return distance;
}

float histogramIntersection2D(float *featureHist1, float *featureHist2, int Hsize)
{

    int i = 0, j = 0, k = 0;
    float sum = 0;
    float distance = 0;

    for (i = 0; i < Hsize; i++)
    {
        for (j = 0; j < Hsize; j++)
        {
            sum = ((featureHist1[i*Hsize + j] > featureHist2[i*Hsize + j]) ? featureHist2[i*Hsize + j] : featureHist1[i*Hsize + j]);
            distance += sum;
        }
    }

    distance = 1 - distance;

    return distance;
}

float histogramIntersection3D(float *featureHist1, float *featureHist2, int Hsize)
{

    int i = 0, j = 0;
    float sum = 0;
    float distance = 0;

    for (i = 0; i < Hsize * Hsize * Hsize; i++)
    {
        sum = (featureHist1[i] > featureHist2[i] ? featureHist2[i] : featureHist1[i]);
        distance += sum;
    }

    distance = 1 - distance;

    return distance;
}

float sumSquaredDiff3D(float *featureHist1, float *featureHist2, int Hsize)
{

    int i = 0, j = 0;
    float diff = 0;
    float sum = 0;
    float square = 0;
    float distance = 0;

    for (i = 0; i < Hsize * Hsize * Hsize; i++)
    {
        diff = featureHist1[i] - featureHist2[i];
        square = diff*diff;
        sum += square;
    }

    distance = sum;

    return distance;
}

void histogramMatching(float **feature_src_2DHist, float **feature_temp_2DHist, Mat &temp, float &distance, int Hsize, float *distanceArray)
{

    getRGHistFeature(temp, Hsize, feature_temp_2DHist);

    distance = histogramIntersection(feature_src_2DHist, feature_temp_2DHist, Hsize, distanceArray);
}

void histogramMatching3D(float *feature_src_3DHist, float *feature_temp_3DHist, Mat &temp, float &distance, int Hsize)
{

    getRGBHistFeature3D(temp, Hsize, feature_temp_3DHist);

    distance = histogramIntersection3D(feature_src_3DHist, feature_temp_3DHist, Hsize);
}

// #######################################################################################################################################################

void splitImage(Mat &src, Mat &dstTop, Mat &dstBottom, int split_row)
{

    // Mat dstTop(row, src.cols, CV_8UC3);
    // Mat dstBottom(src.rows - row, src.cols, CV_8UC3);

    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            for (int c = 0; c < 3; c++)
            {

                if (i < split_row)
                {
                    dstTop.at<Vec3b>(i, j)[c] = src.at<Vec3b>(i, j)[c];
                }
                else
                {
                    dstBottom.at<Vec3b>(i - split_row, j)[c] = src.at<Vec3b>(i, j)[c];
                }
            }
        }
    }
}

void multiHistogramMatching3D(Mat &temp, Mat &temp_top, Mat &temp_bottom, float *feature_src_3DHist_top, float *feature_src_3DHist_bottom,
                              float *feature_temp_3DHist_top, float *feature_temp_3DHist_bottom, float &distance, int Hsize, int split_row)
{

    float distance1 = 0;
    float distance2 = 0;

    splitImage(temp, temp_top, temp_bottom, split_row);

    getRGBHistFeature3D(temp_top, Hsize, feature_temp_3DHist_top);
    getRGBHistFeature3D(temp_bottom, Hsize, feature_temp_3DHist_bottom);

    distance1 = histogramIntersection3D(feature_src_3DHist_top, feature_temp_3DHist_top, Hsize);
    distance2 = histogramIntersection3D(feature_src_3DHist_bottom, feature_temp_3DHist_bottom, Hsize);

    distance = (distance1 + distance2) / 2;
}

// #######################################################################################################################################################

void readFiles(char *dirname, string target_image, int flag_task, int N, int Hsize)
{

    char buffer[256];
    FILE *fp;
    DIR *dirp;
    struct dirent *dp;
    int i;

    // Open the Directory.
    dirp = opendir(dirname);
    if (dirp == NULL)
    {
        printf("Cannot Open Directory %s\n", dirname);
        exit(-1);
    }

    // Common Initializations:
    Mat src;
    Mat temp;
    src = imread(target_image, IMREAD_COLOR);
    vector<pair<string, float>> distance_metric;
    float distance = 0;

    // Initializations for Task 1: ################################################################################################################
    Mat feature_src(9, 9, CV_8UC3, Scalar(0, 0, 0));
    getFeature(src, feature_src);

    // Initializations for Task 2: ################################################################################################################

    // Initialize an array to store distance_metric:
    float *distance_array = new float[Hsize * Hsize];
    // Initialize the distance_array data to all zeros:
    for (int i = 0; i < Hsize * Hsize; i++)
    {
        distance_array[i] = 0;
    }

    // Initialize 2D array for computing 2D rg Histogram for target image:
    // Allocate an array of int pointers:
    float **feature_src_2DHist = new float *[Hsize];
    // Allocating the actual data:
    feature_src_2DHist[0] = new float[Hsize * Hsize];
    // Initializing the row pointers:
    for (int i = 0; i < Hsize; i++)
    {
        feature_src_2DHist[i] = &(feature_src_2DHist[0][i * Hsize]);
    }
    // Initializing the data to all zeros:
    for (int i = 0; i < Hsize; i++)
    {
        for (int j = 0; j < Hsize; j++)
        {
            feature_src_2DHist[i][j] = 0;
        }
    }

    // Initialize 2D array for computing 2D rg Histogram for temporary image:
    // Allocate an array of int pointers:
    float **feature_temp_2DHist = new float *[Hsize];
    // Allocating the actual data:
    feature_temp_2DHist[0] = new float[Hsize * Hsize];
    // Initailizing the row pointers:
    for (int i = 0; i < Hsize; i++)
    {
        feature_temp_2DHist[i] = &(feature_temp_2DHist[0][i * Hsize]);
    }
    // Initalizing the temp feature 2D array data to all zeros:
    for (int i = 0; i < Hsize; i++)
    {
        for (int j = 0; j < Hsize; j++)
        {
            feature_temp_2DHist[i][j] = 0;
        }
    }

    // Initialize 3D array for computing 3D RGB Histogram for target image:
    float *feature_src_3DHist = new float[Hsize * Hsize * Hsize];
    // Initializing the data to all zeros:
    for (int i = 0; i < Hsize * Hsize * Hsize; i++)
    {
        feature_src_3DHist[i] = 0;
    }

    // Initialize 3D array for computing 3D RGB Histogram for temporary image:
    float *feature_temp_3DHist = new float[Hsize * Hsize * Hsize];
    // Initializing the data to all zeros:
    for (int i = 0; i < Hsize * Hsize * Hsize; i++)
    {
        feature_temp_3DHist[i] = 0;
    }

    // Get rg histogram for source image:
    getRGHistFeature(src, Hsize, feature_src_2DHist);

    // Get rgb histogram for source image:
    getRGBHistFeature3D(src, Hsize, feature_src_3DHist);

    // Initializations for Task 3: #############################################################################################################

    int split_row = src.rows / 2;

    Mat src_top(split_row, src.cols, CV_8UC3);
    Mat src_bottom(src.rows - split_row, src.cols, CV_8UC3);
    Mat temp_top(split_row, src.cols, CV_8UC3);
    Mat temp_bottom(src.rows - split_row, src.cols, CV_8UC3);

    // Initialize 3D array for computing 3D RGB Histogram for target image:
    float *feature_src_3DHist_top = new float[Hsize * Hsize * Hsize];
    // Initializing the data to all zeros:
    for (int i = 0; i < Hsize * Hsize * Hsize; i++)
    {
        feature_src_3DHist_top[i] = 0;
    }

    // Initialize 3D array for computing 3D RGB Histogram for temporary image:
    float *feature_temp_3DHist_top = new float[Hsize * Hsize * Hsize];
    // Initializing the data to all zeros:
    for (int i = 0; i < Hsize * Hsize * Hsize; i++)
    {
        feature_temp_3DHist_top[i] = 0;
    }

    // Initialize 3D array for computing 3D RGB Histogram for target image:
    float *feature_src_3DHist_bottom = new float[Hsize * Hsize * Hsize];
    // Initializing the data to all zeros:
    for (int i = 0; i < Hsize * Hsize * Hsize; i++)
    {
        feature_src_3DHist_bottom[i] = 0;
    }

    // Initialize 3D array for computing 3D RGB Histogram for temporary image:
    float *feature_temp_3DHist_bottom = new float[Hsize * Hsize * Hsize];
    // Initializing the data to all zeros:
    for (int i = 0; i < Hsize * Hsize * Hsize; i++)
    {
        feature_temp_3DHist_bottom[i] = 0;
    }

    // Split source image into two
    splitImage(src, src_top, src_bottom, split_row);

    // Get rgb histogram for source image:
    getRGBHistFeature3D(src_top, Hsize, feature_src_3DHist_top);
    getRGBHistFeature3D(src_bottom, Hsize, feature_src_3DHist_bottom);

    // Initializations for Task 4: ###############################################################################################################

    Mat src_Sobel_X;
    Mat src_Sobel_Y;
    Mat src_grad;
    src_Sobel_X = Mat::zeros(src.rows, src.cols, CV_16SC3);
    src_Sobel_Y = Mat::zeros(src.rows, src.cols, CV_16SC3);
    src_grad = Mat::zeros(src.rows, src.cols, CV_8UC3);
    // src.copyTo(src_grad);

    Mat temp_Sobel_X;
    Mat temp_Sobel_Y;
    Mat temp_grad;
    temp_Sobel_X = Mat::zeros(src.rows, src.cols, CV_16SC3);
    temp_Sobel_Y = Mat::zeros(src.rows, src.cols, CV_16SC3);
    temp_grad = Mat::zeros(src.rows, src.cols, CV_8UC3);

    float distance_color;
    float distance_grad;

    // Initialize 3D array for computing 3D RGB Histogram for target image:
    float *feature_src_3DHist_color = new float [Hsize*Hsize*Hsize];
    // Initializing the data to all zeros:
    for(int i = 0; i < Hsize*Hsize*Hsize; i++) {
        feature_src_3DHist_color[i] = 0;
    }

    // Initialize 3D array for computing 3D RGB Histogram for temporary image:
    float *feature_temp_3DHist_color = new float [Hsize*Hsize*Hsize];
    // Initializing the data to all zeros:
    for(int i = 0; i < Hsize*Hsize*Hsize; i++) {
        feature_temp_3DHist_color[i] = 0;
    }

    // Initialize 3D array for computing 3D RGB Histogram for target image:
    float *feature_src_3DHist_grad = new float [Hsize*Hsize*Hsize];
    // Initializing the data to all zeros:
    for(int i = 0; i < Hsize*Hsize*Hsize; i++) {
        feature_src_3DHist_grad[i] = 0;
    }

    // Initialize 3D array for computing 3D RGB Histogram for temporary image:
    float *feature_temp_3DHist_grad = new float [Hsize*Hsize*Hsize];
    // Initializing the data to all zeros:
    for(int i = 0; i < Hsize*Hsize*Hsize; i++) {
        feature_temp_3DHist_grad[i] = 0;
    }


    sobelX3x3(src, src_Sobel_X);
    sobelY3x3(src, src_Sobel_Y);
    magnitude(src_Sobel_X, src_Sobel_Y, src_grad);

    // Get rgb histogram for source image:
    getRGBHistFeature3D(src, Hsize, feature_src_3DHist_color);
    getRGBHistFeature3D(src_grad, Hsize, feature_src_3DHist_grad);




    // Loop over all the files in the image file listing. #########################################################################################
    while ((dp = readdir(dirp)) != NULL)
    {

        // Check if the file is an image.
        if (strstr(dp->d_name, ".jpg") || strstr(dp->d_name, ".png") || strstr(dp->d_name, ".ppm") || strstr(dp->d_name, ".tif"))
        {

            // printf("Processing Image File: %s\n", dp->d_name);

            // Build the overall filename.
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);

            // MY CODE STARTS FROM HERE

            temp = imread(buffer, IMREAD_COLOR);

            switch (flag_task)
            {

            case 1:
                baselineMatching(feature_src, temp, distance);
                break;

            case 2:
                histogramMatching(feature_src_2DHist, feature_temp_2DHist, temp, distance, Hsize, distance_array);
                // histogramMatching3D(feature_src_3DHist, feature_temp_3DHist, temp, distance, Hsize);
                break;

            case 3:
                multiHistogramMatching3D(temp, temp_top, temp_bottom, feature_src_3DHist_top, feature_src_3DHist_bottom,
                                         feature_temp_3DHist_top, feature_temp_3DHist_bottom, distance, Hsize, split_row);
                break;

            case 4:
                sobelX3x3(temp, temp_Sobel_X);
                sobelY3x3(temp, temp_Sobel_Y);
                magnitude(temp_Sobel_X, temp_Sobel_Y, temp_grad);

                // Get the rgb chromaticity histogram feature from the Image:
                getRGBHistFeature3D(temp, Hsize, feature_temp_3DHist_color);
                getRGBHistFeature3D(temp_grad, Hsize, feature_temp_3DHist_grad);

                // Compute the Histogram Intersection between two feature vectors:
                distance_color = histogramIntersection3D(feature_src_3DHist_color, feature_temp_3DHist_color, Hsize);

                // Compute the Histogram Intersection between two feature vectors:
                distance_grad = histogramIntersection3D(feature_src_3DHist_grad, feature_temp_3DHist_grad, Hsize);
                distance = (distance_color + distance_grad) / 2;


                break;
            }

            distance_metric.push_back({buffer, distance});

            // MY CODE ENDS HERE

            // printf("Full Path Name: %s\n", buffer);
            // break;
        }
    }

    MatchingResult(distance_metric, N);

    destroyAllWindows();
}