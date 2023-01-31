#include "myfunctions.hpp"

// Laws Filter R5L5:
cv::Vec3s applyHorizontalLawsL5(const cv::Mat &src, int i, int j)
{
    cv::Vec3s sum = {0, 0, 0};
    int c = 0;
    int filter_rows = 3, filter_cols = 3;

    for (c = 0; c < 3; c++)
    {

        sum[c] = (1 * src.at<cv::Vec3b>(i, j - 2)[c] + 4 * src.at<cv::Vec3b>(i, j - 1)[c] + 6 * src.at<cv::Vec3b>(i, j)[c] + 4 * src.at<cv::Vec3b>(i, j + 1)[c] + 1 * src.at<cv::Vec3b>(i, j + 2)[c])/16;
    }

    return (sum);
}

cv::Vec3s applyVerticalLawsR5(const cv::Mat &src, int i, int j)
{
    cv::Vec3s sum = {0, 0, 0};
    int c = 0;
    int filter_rows = 3, filter_cols = 3;

    for (c = 0; c < 3; c++)
    {
        sum[c] = 1 * src.at<cv::Vec3s>(i - 2, j)[c] + (-4) * src.at<cv::Vec3s>(i - 1, j)[c] + 6 * src.at<cv::Vec3s>(i, j)[c] + (-4) * src.at<cv::Vec3s>(i + 1, j)[c] + 1 * src.at<cv::Vec3s>(i + 2, j)[c];
    }

    return (sum);
}

int lawsFilter(const cv::Mat &src, cv::Mat &dst)
{

    int i = 0, j = 0;

    Mat temp;
    temp = Mat::zeros(src.rows, src.cols, CV_16SC3);

    for (i = 0; i < src.rows; i++)
    {
        for (j = 2; j < src.cols - 2; j++)
        {
            temp.at<cv::Vec3s>(i, j) = applyHorizontalLawsL5(src, i, j);
        }
    }

    for (i = 2; i < src.rows - 2; i++)
    {
        for (j = 0; j < src.cols; j++)
        {
            dst.at<cv::Vec3s>(i, j) = applyVerticalLawsR5(temp, i, j);
        }
    }
    
    cv::convertScaleAbs(dst, dst);

    return 0;
}

int main (int argc, char*  argv[]) {

// #######################################STARTS#################################################################################
    char dirname[256];
    char buffer[256];
    FILE *fp;
    DIR *dirp;
    struct dirent *dp;
    int i;

    // Check for Sufficient Arguments
    if( argc < 2) {
    printf("Usage: %s <directory path>\n", argv[0]);
    exit(-1);
    }

    // Get the Directory Path
    strcpy(dirname, argv[1]);
    printf("Processing Directory %s\n", dirname );

    // Open the Directory
    dirp = opendir( dirname );
    if( dirp == NULL) {
    printf("Cannot Open Directory %s\n", dirname);
    exit(-1);
    }
// ################################################STOPS########################################################################
    
    // string target_image = "olympus/pic.0065.jpg";
    string target_image = "olympus/pic.0013.jpg";

    Mat src;
    Mat temp;
    Mat result_images;
    src = imread(target_image, IMREAD_COLOR);

    Mat src_Laws;
    src_Laws = Mat::zeros(src.rows, src.cols, CV_16SC3);

    Mat temp_Laws;
    temp_Laws = Mat::zeros(src.rows, src.cols, CV_16SC3);

    int N = 10;
    
    vector<pair<string, float>> distance_metric;
    
    int Hsize = 16;
    float distance;
    float distance_color;
    float distance_laws;

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
    float *feature_src_3DHist_laws = new float [Hsize*Hsize*Hsize];
    // Initializing the data to all zeros:
    for(int i = 0; i < Hsize*Hsize*Hsize; i++) {
        feature_src_3DHist_laws[i] = 0;
    }

    // Initialize 3D array for computing 3D RGB Histogram for temporary image:
    float *feature_temp_3DHist_laws = new float [Hsize*Hsize*Hsize];
    // Initializing the data to all zeros:
    for(int i = 0; i < Hsize*Hsize*Hsize; i++) {
        feature_temp_3DHist_laws[i] = 0;
    }

    // imshow("Test", src);
    // waitKey();
    lawsFilter(src, src_Laws);
    // Get rgb histogram for source image:
    getRGBHistFeature3D(src, Hsize, feature_src_3DHist_color);
    getRGBHistFeature3D(src_Laws, Hsize, feature_src_3DHist_laws);

// ################################################STARTS########################################################################
    // Loop over all the files in the image file listing
    while( (dp = readdir(dirp)) != NULL ) {

        // Check if the file is an image
        if( strstr(dp->d_name, ".jpg") || strstr(dp->d_name, ".png") ||	strstr(dp->d_name, ".ppm") || strstr(dp->d_name, ".tif") ) {

            // printf("Processing Image File: %s\n", dp->d_name);

            // build the overall filename
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);
// ################################################STOPS########################################################################

            // MY CODE STARTS FROM HERE

            temp = imread(buffer, IMREAD_COLOR);
            temp_Laws = Mat::zeros(src.rows, src.cols, CV_16SC3);

            lawsFilter(temp, temp_Laws);

            // Get the rgb chromaticity histogram feature from the Image:
            getRGBHistFeature3D(temp, Hsize, feature_temp_3DHist_color);
            getRGBHistFeature3D(temp_Laws, Hsize, feature_temp_3DHist_laws);

            // Compute the Histogram Intersection between two feature vectors:
            distance_color = histogramIntersection3D(feature_src_3DHist_color, feature_temp_3DHist_color, Hsize);

            // Compute the Histogram Intersection between two feature vectors:
            distance_laws = histogramIntersection3D(feature_src_3DHist_laws, feature_temp_3DHist_laws, Hsize);
            distance = 0.3*distance_color + 0.7*distance_laws;
            // cout << "Distance Color: " << distance_color << endl;
            // cout << "Distance Laws: " << distance_laws << endl;
            // cout << distance << endl;
            distance_metric.push_back({buffer, distance});

            // MY CODE ENDS HERE

// ################################################STARTS########################################################################
            // printf("Full Path Name: %s\n", buffer);
            // break;

        }
    }

    sort(distance_metric.begin(), distance_metric.end(), sortPairedVector);

    for (int i = 0; i < N; i++) {
        cout << distance_metric[i].first << " " << distance_metric[i].second << endl;
        result_images = imread(distance_metric[i].first, IMREAD_COLOR);
        namedWindow("Image" + to_string(i), 1);
        imshow("Image" + to_string(i), result_images);
    }

    // splitImage(src, src_top, src_bottom, row);
    // imshow("1", src_Laws);
    // imshow("2", temp_Laws);
    // imshow("3", src_grad);
    // imshow("4", temp_grad);
    
    waitKey();

    printf("Terminating\n");

    destroyAllWindows();

    return 0;

}