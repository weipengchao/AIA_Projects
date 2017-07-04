//============================================================================
// Name        : main.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : only calls processing and test routines
//============================================================================

#include <iostream>

#include "Aia2.h"

using namespace std;

// usage: path to image in argv[1]
// main function. loads and saves image
int main(int argc, char** argv) {

	// will contain path to input image (taken from argv[1])
	string img, tmpl1, tmpl2;

	// check if image path was defined
	// check if image paths were defined
	if (argc != 4){
	    cerr << "Usage: aia2 <input image>  <class 1 example>  <class 2 example>" << endl;
	    cerr << "Press enter to continue..." << endl;
	    cin.get();
	    return -1;
	}else{
	    // if yes, assign it to variable fname
		//Mat img1;
		//img1 = imread(argv[1], 0);
		img = argv[1];
	    tmpl1 = argv[2];
	    tmpl2 = argv[3];

/*
		Mat threshold_output,Erode_output;
		int thresh = 128,k=1;

		threshold(img1, threshold_output, 128, 255 ,THRESH_BINARY_INV);

		//showImage(dst,"Test2",0);


		//cout << dst <<endl;
		//waitKey(0);

		// Kernel ist für das eroden zuständig 3x3
		Mat kernel = Mat::ones(8,8,threshold_output.type()) * 255;
		erode(threshold_output, Erode_output, kernel, Point(-1,-1), k);

		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		imshow("test23", Erode_output);
				waitKey(0);

		 findContours(Erode_output, contours, hierarchy, CV_RETR_CCOMP , CV_CHAIN_APPROX_NONE, Point(0, 0));

		for (int i = 1; i < contours.size(); i++){
				cout<<"Hiers cont"<< contours[i] << endl;
		}


		imshow("test", Erode_output);
		waitKey(0);

*/

	}

	// construct processing object
	Aia2 aia2;

	// run some test routines
	//aia2.test();

	// start processing
	aia2.run(img, tmpl1, tmpl2);

	return 0;

}
