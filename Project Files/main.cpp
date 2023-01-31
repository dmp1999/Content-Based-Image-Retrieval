// Project 2 of CS 5330: Pattern Recognition and Computer Vision
// Created by Dhruvil Parikh

#include "myfunctions.hpp"

int main (int argc, char*  argv[]) {

    char dirname[256];
    char decision;

    // Check for Sufficient Arguments
    if( argc < 2) {
    printf("Usage: %s <directory path>\n", argv[0]);
    exit(-1);
    }

    // Get the Directory Path
    strcpy(dirname, argv[1]);
    printf("Processing Directory %s...\n", dirname );

    while(true) {

        system("clear");

        // Ask user for the task they want to perform:
        int flag_task;
        cout << "Tasks:" << endl;
        cout << "Press \"1\" for Task1: Baseline Matching" << endl;
        cout << "Press \"2\" for Task2: Histogram Matching" << endl;
        cout << "Press \"3\" for Task3: Multi-Histogram Matching" << endl;
        cout << "Press \"4\" for Task4: Texture and Color" << endl;
        cout << "Please enter a number: ";
        cin >> flag_task;

        // Ask user for the target image file name:
        string target_image;
        cout << "Please enter the name of the Target Image: ";
        cin >> target_image;
        target_image = "/" + target_image;
        target_image = dirname + target_image;

        // Ask user for the number of top N matches:
        int N;
        cout << "Please enter the number of top matches you wish to see: ";
        cin >> N;
        N++;;

        // Ask user for the number of bins:
        int nBins;
        cout << "Please enter the number of bins: ";
        cin >> nBins;

        readFiles(dirname, target_image, flag_task, N, nBins);

        cout << "Do you wish to perform another task?" << endl;
        cout << "Press \"Y\" for YES and \"N\" for NO: ";
        cin >> decision;

        if(decision == 'N') {
            break;
        }
        
    }

    printf("Terminating the Program...\n");

    return 0;

}