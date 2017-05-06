//============================================================================
// Name        : Aia1.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description :
//============================================================================

#include "Aia1.h"
#include <cv.h>
#include <highgui.h>
// function that performs some kind of (simple) image processing
/*
img	input image
return	output image
*/
Mat Aia1::doSomethingThatMyTutorIsGonnaLike(Mat& img){
	Mat g_img,r_img;

	cvtColor(img,g_img, CV_BGR2GRAY );
	cv::Mat noise_img = Mat(g_img.size(),CV_64F);

	normalize(g_img, r_img, 0.0, 1.0, CV_MINMAX, CV_64F);
	cv::randn(noise_img, 0, 0.1);

	r_img = r_img + noise_img;
	r_img.convertTo(r_img, CV_32F, 255, 0);

	normalize(r_img, r_img, 0.0, 1.0, CV_MINMAX, CV_64F);


	// TO DO !!!
	return r_img;

}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

// function loads input image, calls processing function, and saves result
/*
fname	path to input image
*/
void Aia1::run(string fname){

	// window names
	string win1 = string ("Original image");
	string win2 = string ("Result");

	// some images
	Mat inputImage, outputImage;

	// load image
	cout << "load image" << endl;
	inputImage = imread( fname );
	cout << "done" << endl;

	// check if image can be loaded
	if (!inputImage.data){
	    cerr << "ERROR: Cannot read file " << fname << endl;
	    cout << "Press Enter to continue..." << endl;
	    cin.get();
	    exit(-1);
	}

	// show input image
	namedWindow( win1.c_str() );
	imshow( win1.c_str(), inputImage );

	// do something (reasonable!)
	outputImage = doSomethingThatMyTutorIsGonnaLike( inputImage );

	// show result
	namedWindow( win2.c_str() );
	imshow( win2.c_str(), outputImage );

	// save result
	imwrite("result.jpg", outputImage);

	// wait a bit
	waitKey(0);

}

// function loads input image and calls processing function
// output is tested on "correctness"
/*
fname	path to input image
*/
void Aia1::test(string fname){

	// some image variables
	Mat inputImage, outputImage;

	// load image
	inputImage = imread( fname );

	// check if image can be loaded
	if (!inputImage.data){
	    cerr << "ERROR: Cannot read file " << fname << endl;
	    exit(-1);
	}

	// create output
	outputImage = doSomethingThatMyTutorIsGonnaLike( inputImage );
	// test output
	test_doSomethingThatMyTutorIsGonnaLike(inputImage, outputImage);

}

// function loads input image and calls processing function
// output is tested on "correctness"
/*
inputImage	input image as used by doSomethingThatMyTutorIsGonnaLike()
outputImage	output image as created by doSomethingThatMyTutorIsGonnaLike()
*/
void Aia1::test_doSomethingThatMyTutorIsGonnaLike(Mat& inputImage, Mat& outputImage){

	// ensure that input and output have equal number of channels
	if ( (inputImage.channels() == 3) and (outputImage.channels() == 1) )
		cvtColor(inputImage, inputImage, CV_BGR2GRAY);

	// split (multi-channel) image into planes
	vector<Mat> inputPlanes, outputPlanes;
	split( inputImage, inputPlanes );
	split( outputImage, outputPlanes );

	// number of planes (1=grayscale, 3=color)
	int numOfPlanes = inputPlanes.size();

	// calculate and compare image histograms for each plane
	Mat inputHist, outputHist;
	// number of bins
	int histSize = 100;
	float range[] = { 0, 256 } ;
	const float* histRange = { range };
	bool uniform = true; bool accumulate = false;
	double sim = 0;
	for(int p = 0; p < numOfPlanes; p++){
		// calculate histogram
		calcHist( &inputPlanes[p], 1, 0, Mat(), inputHist, 1, &histSize, &histRange, uniform, accumulate );
		calcHist( &outputPlanes[p], 1, 0, Mat(), outputHist, 1, &histSize, &histRange, uniform, accumulate );
		// normalize
		inputHist = inputHist / sum(inputHist).val[0];
		outputHist = outputHist / sum(outputHist).val[0];
		// similarity as histogram intersection
		sim += compareHist(inputHist, outputHist, CV_COMP_INTERSECT);
	}
	sim /= numOfPlanes;

	// check whether images are to similar after transformation
	if (sim >= 0.8)
		cout << "The input and output image seem to be quite similar (similarity = " << sim << " ). Are you sure your tutor is gonna like your work?" << endl;

}

