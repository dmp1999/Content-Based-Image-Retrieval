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

    Mat src;
    Mat temp;
    Mat result_images;
    
    string target_image = "olympus/pic.0164.jpg";

    int N = 4;

    vector<pair<string, float>> distance_metric;
    
    int Hsize = 8;
    float distance;

    // Initialize 3D array for computing 3D RGB Histogram for target image:
    float *feature_src_3DHist = new float [Hsize*Hsize*Hsize];
    // Initializing the data to all zeros:
    for(int i = 0; i < Hsize*Hsize*Hsize; i++) {
        feature_src_3DHist[i] = 0;
    }

    // Initialize 3D array for computing 3D RGB Histogram for target image:
    float *feature_temp_3DHist = new float [Hsize*Hsize*Hsize];
    // Initializing the data to all zeros:
    for(int i = 0; i < Hsize*Hsize*Hsize; i++) {
        feature_temp_3DHist[i] = 0;
    }

    src = imread(target_image, IMREAD_COLOR);
    // imshow("Test", src);
    // waitKey();
    getRGBHistFeature3D(src, Hsize, feature_src_3DHist);

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

            // // Initalizing the temp feature 2D array data to all zeros:
            // for(int i = 0; i < Hsize; i++) {
            //     for(int j = 0; j < Hsize; j++) {
            //         feature_temp_2DHist[i][j] = 0;
            //     }
            // }

            // Get the rg chromaticity histogram feature from the Image:
            getRGBHistFeature3D(temp, Hsize, feature_temp_3DHist);

            // Compute the Histogram Intersection between two feature vectors:
            distance = histogramIntersection3D(feature_src_3DHist, feature_temp_3DHist, Hsize);
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