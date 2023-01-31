#include "myfunctions.hpp"

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
    
    string target_image = "olympus/pic.0851.jpg";

    Mat src;
    Mat temp;
    Mat result_images;
    Mat filter_gabor;
    src = imread(target_image, IMREAD_COLOR);

    Mat src_gabor;
    src.copyTo(src_gabor);

    Mat temp_gabor;

    int N = 4;
    
    vector<pair<string, float>> distance_metric;
    
    int Hsize = 8;
    float distance;

    // Initialize 3D array for computing 3D RGB Histogram for target image:
    float *feature_src_3DHist_gabor = new float [Hsize*Hsize*Hsize];
    // Initializing the data to all zeros:
    for(int i = 0; i < Hsize*Hsize*Hsize; i++) {
        feature_src_3DHist_gabor[i] = 0;
    }

    // Initialize 3D array for computing 3D RGB Histogram for temporary image:
    float *feature_temp_3DHist_gabor = new float [Hsize*Hsize*Hsize];
    // Initializing the data to all zeros:
    for(int i = 0; i < Hsize*Hsize*Hsize; i++) {
        feature_temp_3DHist_gabor[i] = 0;
    }

    int ker_size = 9;
    double sig = 5, lmd = 4, gamma = 0.04, psi = CV_PI/4;
    double theta[] = {0,CV_PI/12,CV_PI/4,CV_PI/2,-CV_PI/4,-CV_PI/12};

    int cnt = sizeof(theta)/sizeof(theta[0]);

    for(int i=0; i<cnt; i++) {
        filter_gabor = getGaborKernel(Size(ker_size,ker_size), sig, theta[i],lmd,gamma,psi,CV_32F);
        filter2D(src, src_gabor, CV_32F, filter_gabor);
    }

    // imshow("gabor", src_gabor);
    // waitKey();

    // Get rgb histogram for source image:
    getRGBHistFeature3D(src_gabor, Hsize, feature_src_3DHist_gabor);

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
            temp.copyTo(temp_gabor);

            for(int i=0; i<cnt; i++) {
                filter_gabor = getGaborKernel(Size(ker_size,ker_size), sig, theta[i],lmd,gamma,psi,CV_32F);
                filter2D(temp, temp_gabor, CV_32F, filter_gabor);
            }
            
            // Get the rgb chromaticity histogram feature from the Image:
            getRGBHistFeature3D(temp_gabor, Hsize, feature_temp_3DHist_gabor);

            // Compute the Histogram Intersection between two feature vectors:
            distance = histogramIntersection3D(feature_src_3DHist_gabor, feature_temp_3DHist_gabor, Hsize);

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
    
    waitKey();

    printf("Terminating\n");

    destroyAllWindows();

    return 0;

}