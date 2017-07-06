//============================================================================
// Name        : Aia5.cpp
// Author      : Ronny Haensch
// Version     : 1.0
// Copyright   : -
// Description : 
//============================================================================

#include "Aia4.h"

// Computes the log-likelihood of each feature vector in every component of the supplied mixture model.
/*
model:     	structure containing model parameters
features:  	matrix of feature vectors
return: 	the log-likelihood log(p(x_i|y_i=j))
*/
Mat Aia5::calcCompLogL(vector<struct comp*>& model, Mat& features){
	// To Do !!!
	cout<<"Features:" << features.cols << endl;
	cout<<"Model:" << model.size() << endl;
	Mat result = Mat::zeros(model.size(),features.cols,CV_32FC1);

	float z = - features.rows * log(2 * CV_PI)/2;

	for (int featureNo = 0 ; featureNo < features.cols; featureNo++){
		for (int modelNo = 0; modelNo < model.size(); modelNo++ ){
		// Berechne Covarianz Matrix
			float detCovMat = - 0.5 * log(determinant(model.at(modelNo)->covar ) );
			Mat featuremean;
			subtract(features.col(featureNo),model.at(modelNo)->mean,featuremean);
			Mat invCovMat = model.at(modelNo)->covar.inv();
			// Zugriff auf Mat XY (0,0)
			float calcComponent = ((Mat)(- 0.5 * featuremean.t()*invCovMat*featuremean)).at<float>(0,0);

			  result.at<float>(modelNo,featureNo) = detCovMat + calcComponent +  z;
		}
	}
	return result;
}

// Computes the log-likelihood of each feature by combining the likelihoods in all model components.
/*
model:     structure containing model parameters
features:  matrix of feature vectors
return:	   the log-likelihood of feature number i in the mixture model (the log of the summation of alpha_j p(x_i|y_i=j) over j)
*/
Mat Aia5::calcMixtureLogL(vector<struct comp*>& model, Mat& features){
	// To Do !!!

	Mat result = Mat::zeros(1, features.cols, CV_32FC1);

	//Rückgabewert ist Log (trick)
	Mat compLogRes = calcCompLogL(model, features);
	//Log -> entfernen

	exp(compLogRes,compLogRes);

	// -> Zeigerzugriff
	for(unsigned modelNr = 0; modelNr < model.size(); modelNr++){
		// Lade cLR mit model
			compLogRes.row(modelNr) *= model.at(modelNr)->weight;
			//cout << "compLogRes:"<< modelNr << compLogRes.row(modelNr) << endl;
	}

	//Log trick
	log(compLogRes, compLogRes);

	Mat logC = Mat::zeros(1, features.cols,CV_32FC1);
	double min,max;

	// finde max cLR vom Feature
	for (int featureNo = 0; featureNo < features.cols; featureNo++){
			minMaxLoc(compLogRes.col(featureNo), &min, &max);
			logC.at<float>(featureNo) = max;
		}


	// Gehe alle Features durch und nehme aus jedem die Reihen und deren Model-Parameter für den calcCompLogL-Fkt-Aufruf
		for(unsigned modelNr = 0; modelNr < model.size(); modelNr++){
			Mat expRes = Mat::zeros(1, features.cols,CV_32FC1);

			exp(compLogRes.row(modelNr) - logC.row(modelNr), expRes);

			cout << "resultmatrix" << result.size() << endl;
			cout << "resultmatrix" << expRes.size() << endl;

			add(result, expRes, result);


		}

		log(result, result);
		add(result, logC, result);


	return result;
}

// Computes the posterior over components (the degree of component membership) for each feature.
/*
model:     	structure containing model parameters
features:  	matrix of feature vectors
return:		the posterior p(y_i=j|x_i)
*/
Mat Aia5::gmmEStep(vector<struct comp*>& model, Mat& features){
	// To Do !!!
	cout << "gmmEStep" << endl;
	Mat result = Mat::zeros(1,model.size(),CV_32FC1);

	Mat MixLogRes = calcMixtureLogL(model, features);
	Mat comLogRes = calcCompLogL(model, features);

	for (unsigned modelNo = 0; modelNo < model.size(); modelNo++){
		result.row(modelNo) = log(model.at(modelNo)-> weight) + comLogRes.row(modelNo) - MixLogRes;
	}
exp(result, result);

	return Mat();
}

// Updates a given model on the basis of posteriors previously computed in the E-Step.
/*
model:     structure containing model parameters, will be updated in-place
           new model structure in which all parameters have been updated to reflect the current posterior distributions.
features:  matrix of feature vectors
posterior: the posterior p(y_i=j|x_i)
*/
void Aia5::gmmMStep(vector<struct comp*>& model, Mat& features, Mat& posterior){
   // To Do !!!

	Mat N = Mat::zeros(posterior.cols, 1, CV_32FC1);

		for(unsigned modelNr = 0; modelNr < model.size(); modelNr++){

			for(int posteriorNr = 0; posteriorNr < posterior.cols; posteriorNr++){
				N.at<float>(modelNr, 0) = posterior.at<float>(modelNr, posteriorNr) + N.at<float>(modelNr, 0);
			}

			// weight updating
			model.at(modelNr)->weight = N.at<float>(modelNr, 0) / features.cols;


			// mean updating
			model.at(modelNr)->mean.at<float>(modelNr, 0) = 0;
			for(int posteriorNr = 0; posteriorNr < posterior.cols; posteriorNr++){
				model.at(modelNr)->mean.at<float>(modelNr, 0) = (features.at<float>(modelNr, posteriorNr) * posterior.at<float>(modelNr, posteriorNr)) / N.at<float>(modelNr, 0) + model.at(modelNr)->mean.at<float>(modelNr, 0);
			}

			// covar updating
			model.at(modelNr)->covar = Mat::zeros(model.at(modelNr)->covar.rows, model.at(modelNr)->covar.cols, CV_32FC1);
			for(int posteriorNr = 0; posteriorNr < posterior.cols; posteriorNr++){
				float calcRes = features.at<float>(modelNr, posteriorNr) - model.at(modelNr)->mean.at<float>(modelNr, posteriorNr);
				calcRes = calcRes * calcRes * posterior.at<float>(modelNr, posteriorNr);
				model.at(modelNr)->covar.at<float>(modelNr, 0) = calcRes / N.at<float>(modelNr, 0) + model.at(modelNr)->covar.at<float>(modelNr, 0);
			}
		}



}

/* *****************************
  GIVEN FUNCTIONS
***************************** */

// sets parameters and generates data for EM clustering
void Aia5::run(void){

	// *********************************

	// To Do: Adjust number of princ. comp. and number of clusters and evaluate performance
	// dimensionality of the generated feature vectors (number of principal components)
    int vectLen = 20;
    // number of components (clusters) to be used by the model
    int numberOfComponents = 10;
    
    // *********************************

     // read training data
    cout << "Reading training data..." << endl;
    vector<Mat> trainImgDB;
    readImageDatabase("./img/train/in/", trainImgDB);
    cout << "Done\n" << endl;
      
    // generate PCA-basis
    cout << "Generate PCA-basis from data:" << endl;
    vector<PCA> featProjection;
    genFeatureProjection(trainImgDB, featProjection, vectLen);
    cout << "Done\n" << endl;

    // this is going to contain the individual models (one per category)
    vector< vector<struct comp*> > models;
    
    // start learning
    cout << "Start learning..." << endl;
    for(int c=0; c<10; c++){
	cout << "\nTrain GMM of category " << c << endl;
	cout << " > Project on principal components of category " << c << " :" << endl;
	Mat fea = Mat(trainImgDB.at(c).rows, vectLen, CV_32FC1);
	featProjection.at(c).project( trainImgDB.at(c), fea );
	fea = fea.t();
	cout << "> Done" << endl;
	
	cout << " > Estimate probability density of category " << c << " :" << endl;
	// train the corresponding mixture model using EM...
	vector<struct comp*> model;
	trainGMM(fea, numberOfComponents, model);
	models.push_back(model);
	cout << "> Done" << endl;
    }
    cout << "Done\n" << endl;

    // read testing data
    cout << "Reading test data...:\t";
    vector<Mat> testImgDB;
    readImageDatabase("./img/test/in/" , testImgDB);
    cout << "Done\n" << endl;
    
    cout << "Test GMM: Start" << endl;
    Mat confMatrix = Mat::zeros(10, 10, CV_32FC1);
    int dtr = 0, n=0;
    // for each category within the test data
    for(int c_true=0; c_true<10; c_true++){
	n += testImgDB.at(c_true).rows;
	// init likelihood
	Mat maxMixLogL = Mat(1, testImgDB.at(c_true).rows, CV_32FC1);
	// estimated class
	Mat est = Mat::zeros(1, testImgDB.at(c_true).rows, CV_8UC1);
	
	for(int c_est=0; c_est<10; c_est++){
	    cout << " > Project on principal components of category " << c_est << " :\t";
	    Mat fea = Mat(testImgDB.at(c_true).rows, vectLen, CV_32FC1);
	    featProjection.at(c_est).project( testImgDB.at(c_true), fea );
	    fea = fea.t();
	    cout << "Done" << endl;
	
	    cout << " > Estimate class likelihood of category " << c_est << " :" << endl;
	    
	    // get data log
	    Mat mixLogL = calcMixtureLogL(models.at(c_est), fea);

	    // compare to current max
	    for(int i=0; i<fea.cols; i++){
		if ( ( maxMixLogL.at<float>(0,i) < mixLogL.at<float>(0,i) ) or (c_est == 0) ){
		    maxMixLogL.at<float>(0,i) = mixLogL.at<float>(0,i);
		    est.at<uchar>(0,i) = c_est;
		}
	    }
	    cout << "Done\n" << endl;
	}
	// make corresponding entry in confusion matrix
	for(int i=0; i<testImgDB.at(c_true).rows; i++){
	    //cout << (int)est.at<uchar>(0,i) << "\t" << c_true << endl;
	    confMatrix.at<float>( c_true, (int)est.at<uchar>(0,i) )++;
	    dtr += (int)est.at<uchar>(0,i) == c_true;
	}
    }
    cout << "Test GMM: Done\n" << endl;
    
    cout << endl << "Confusion matrix:" << endl;
    cout << confMatrix << endl;
    cout << endl << "No of correctly classified:\t" << dtr << " of " << n << "\t( " << (dtr/(double)n*100) << "% )" << endl;
    
}

// sets parameters and generates data for EM clustering
void Aia5::test(void){

	// dimensionality of the generated feature vectors
    int vectLen = 2;

    // number of components (clusters) to generate
    int actualComponentNum = 10;

    // maximal standard deviation of vectors in each cluster
    double actualDevMax = 0.3;

    // number of vectors in each cluster
    int trainingSize = 150;

    // initialise random component parameters (mean and standard deviation)
    Mat actualMean(actualComponentNum, vectLen, CV_32FC1);
    Mat actualSDev(actualComponentNum, vectLen, CV_32FC1); 
    randu(actualMean, 0, 1);
    randu(actualSDev, 0, actualDevMax);

    // print distribution parameters to screen
    cout << "true mean" << endl;
    cout << actualMean << endl;
    cout << "true sdev" << endl;
    cout << actualSDev << endl;

    // initialise random cluster vectors
    Mat trainingData = Mat::zeros(vectLen, trainingSize*actualComponentNum, CV_32FC1);
    int n=0;
    RNG rng;
    for(int c=0; c<actualComponentNum; c++){
		for(int s=0; s<trainingSize; s++){
			for(int d=0; d<vectLen; d++){
				trainingData.at<float>(d,n) = rng.gaussian( actualSDev.at<float>(c, d) ) + actualMean.at<float>(c, d);
			}
			n++;
		}
    }

    // train the corresponding mixture model using EM...
 	 vector<struct comp*> model;
    trainGMM(trainingData, actualComponentNum, model);
    
}

// Trains a Gaussian mixture model with a specified number of components on the basis of a given set of feature vectors.
/*
data:     		feature vectors, one vector per column
numberOfComponents:	the desired number of components in the trained model
model:			the model, that will be created and trained inside of this function
*/
void Aia5::trainGMM(Mat& data, int numberOfComponents, vector<struct comp*>& model){

    // the number and dimensionality of feature vectors
    int featureNum = data.cols;
    int featureDim = data.rows;

    // initialize the model with one component and arbitrary parameters
    struct comp* fst = new struct comp();
    fst->weight = 1;
    fst->mean = Mat::zeros(featureDim, 1, CV_32FC1);
    fst->covar = Mat::eye(featureDim, featureDim, CV_32FC1);
    model.push_back(fst);

    // the data-log-likelihood
    double dataLogL[2] = {0,0};
        
    // iteratively add components to the mixture model
    for(int i=1; i<=numberOfComponents; i++){
      
		cout << "Current number of components: " << i << endl;
		
		// the current combined data log-likelihood p(X|Omega)
		Mat mixLogL = calcMixtureLogL(model, data);
		dataLogL[0] = sum(mixLogL).val[0];
		dataLogL[1] = 0.;
		
		// EM iteration while p(X|Omega) increases
		int it = 0;
		while( (dataLogL[0] > dataLogL[1]) or (it == 0) ){
		  
			printf("Iteration: %d\t:\t%f\r", it++, dataLogL[0]);

			// E-Step (computes posterior)
			Mat posterior = gmmEStep(model, data);

			// M-Step (updates model parameters)
			gmmMStep(model, data, posterior);
			
			// update the current p(X|Omega)
			dataLogL[1] = dataLogL[0];
			mixLogL = calcMixtureLogL(model, data);
			dataLogL[0] = sum(mixLogL).val[0];
		}
		cout << endl;

		// visualize the current model (with i components trained)
		if (featureDim >= 2){
		   plotGMM(model, data);
		}

		// add a new component if necessary
		if (i < numberOfComponents){
			initNewComponent(model, data);    
		}
    }
    
    cout << endl << "**********************************" << endl;
    cout << "Trained model: " << endl;
    for(int i=0; i<model.size(); i++){
		cout << "Component " << i << endl;
		cout << "\t>> weight: " << model.at(i)->weight << endl;
		cout << "\t>> mean: " << model.at(i)->mean << endl;
		cout << "\t>> std: [" << sqrt(model.at(i)->covar.at<float>(0,0)) << ", " << sqrt(model.at(i)->covar.at<float>(1,1)) << "]" << endl;
		cout << "\t>> covar: " << endl;
		cout << model.at(i)->covar << endl;
		cout << endl;
    }

}

// Adds a new component to the input mixture model by spliting one of the existing components in two parts.
/*
model:		Gaussian Mixture Model parameters, will be updated in-place
features:	feature vectors
*/
void Aia5::initNewComponent(vector<struct comp*>& model, Mat& features){

    // number of components in current model
    int compNum = model.size();

    // number of features
    int featureNum = features.cols;

    // dimensionality of feature vectors (equals 3 in this exercise)
    int featureDim = features.rows;

    // the largest component is split (this is not a good criterion...)
    int splitComp = 0;
    for(int i=0; i<compNum; i++){
		if (model.at(splitComp)->weight < model.at(i)->weight){
			splitComp = i;
		}
    }

    // split component 'splitComp' along its major axis
    Mat eVec, eVal;
    eigen(model.at(splitComp)->covar, eVal, eVec);

    Mat devVec = 0.5 * sqrt( eVal.at<float>(0) ) * eVec.row(0).t();

    // create new model structure and compute new mean values, covariances, new component weights...
    struct comp* newModel = new struct comp;
    newModel->weight = 0.5 * model.at(splitComp)->weight;
    newModel->mean = model.at(splitComp)->mean - devVec;
    newModel->covar = 0.25 * model.at(splitComp)->covar;

    // modify the split component
    model.at(splitComp)->weight = 0.5*model.at(splitComp)->weight;
    model.at(splitComp)->mean += devVec;
    model.at(splitComp)->covar *= 0.25;

    // add it to old model
    model.push_back(newModel);

}

// Visualises the contents of a feature space and the associated mixture model.
/*
model: 		parameters of a Gaussian mixture model
features: 	feature vectors

Feature vectors are plotted as black points
Estimated means of components are indicated by blue circles
Estimated covariances are indicated by blue ellipses
If the feature space has more than 2 dimensions, only the first two dimensions are visualized.
*/
void Aia5::plotGMM(vector<struct comp*>& model, Mat& features){

    // size of the plot
    int imSize = 500;
  
    // get scaling factor to scale coordinates
    double max_x=0, max_y=0, min_x=0, min_y=0;
    for(int n=0; n<features.cols; n++){
		if (max_x < features.at<float>(0, n) )
			max_x = features.at<float>(0, n);
		if (min_x > features.at<float>(0, n) )
			min_x = features.at<float>(0, n);
		if (max_y < features.at<float>(1, n) )
			max_y = features.at<float>(1, n);
		if (min_y > features.at<float>(1, n) )
			min_y = features.at<float>(1, n);
    }  
    double scale = (imSize-1)/max((max_x - min_x), (max_y - min_y));
    // create plot
    Mat plot = Mat(imSize, imSize, CV_8UC3, Scalar(255,255,255) );

    // set feature points
    for(int n=0; n<features.cols; n++){
		plot.at<Vec3b>( ( features.at<float>(0, n) - min_x ) * scale, max( (features.at<float>(1,n)-min_y)*scale, 5.) ) = Vec3b(0,0,0);
    }
    // get ellipse of components
    Mat EVec, EVal;
    for(int i=0; i<model.size(); i++){

		eigen(model.at(i)->covar, EVal, EVec);
		double rAng = atan2(EVec.at<float>(0, 1), EVec.at<float>(0, 0));

		// draw components
		circle(plot,  Point( (model.at(i)->mean.at<float>(1,0)-min_y)*scale, (model.at(i)->mean.at<float>(0,0)-min_x)*scale ), 3, Scalar(255,0,0), 2);
		ellipse(plot, Point( (model.at(i)->mean.at<float>(1,0)-min_y)*scale, (model.at(i)->mean.at<float>(0,0)-min_x)*scale ), Size(sqrt(EVal.at<float>(1))*scale*2, sqrt(EVal.at<float>(0))*scale*2), rAng*180/CV_PI, 0, 360, Scalar(255,0,0), 2);

    }
    
    // show plot an abd wait for key
    imshow("Current model", plot);
    waitKey(0);

}

// constructs PCA-space based on all samples of each class
/*
imgDB		image data base; each matrix corresponds to one class; each row to one image
featSpace	the PCA-bases for each class
vectLen		number of principal components to be used
*/
void Aia5::genFeatureProjection(vector<Mat>& imgDB, vector<PCA>& featSpace, int vectLen){

    int c = 0;
    for(vector<Mat>::iterator cat = imgDB.begin(); cat != imgDB.end(); cat++, c++){
	cout << " > Generate PC of category " << c << " :\t";
      	PCA compPCA = PCA(*cat, Mat(), CV_PCA_DATA_AS_ROW, vectLen);
	featSpace.push_back(compPCA);
	cout << "Done" << endl;
    }
}

// reads image data base from disc
/*
dataPath	path to directory
db		each matrix in this vector corresponds to one class, each row of the matrix corresponds to one image 
*/
void Aia5::readImageDatabase(string dataPath, vector<Mat>& db){

    // directory delimiter. you might wanna use '\' on windows systems
    char delim = '/';

    char curDir[100];
    db.reserve(10);
    
    int numberOfImages = 0;
    for(int c=0; c<10; c++){

	list<Mat> imgList;
	sprintf(curDir, "%s%c%i%c", dataPath.c_str(), delim, c, delim);
 
	// read directory
	DIR* pDIR;
	struct dirent *entry;
	struct stat s;
	
	stat(curDir,&s);

	// if path is a directory
	if ( (s.st_mode & S_IFMT ) == S_IFDIR ){
	    if( pDIR=opendir(curDir) ){
		// for all entries in directory
		while(entry = readdir(pDIR)){
		    // is current entry a file?
		    stat((curDir + string("/") + string(entry->d_name)).c_str(),&s);
		    if ( ( (s.st_mode & S_IFMT ) != S_IFDIR ) and ( (s.st_mode & S_IFMT ) == S_IFREG ) ){
			// if all conditions are fulfilled: load data
			Mat img = imread((curDir + string(entry->d_name)).c_str(), 0);
			img.convertTo(img, CV_32FC3);
			img /= 255.;
			imgList.push_back(img);
			numberOfImages++;
		    }
		}
		closedir(pDIR);
	    }else{
		cerr << "\nERROR: cant open data dir " << dataPath << endl;
		exit(-1);
	    }
	}else{
	    cerr << "\nERROR: provided path does not specify a directory: ( " << dataPath << " )" << endl;
	    exit(-1);
	}
	
	int numberOfImages = imgList.size();
	int numberOfPixPerImg = imgList.front().cols * imgList.front().rows;
	    
	Mat feature = Mat(numberOfImages, numberOfPixPerImg, CV_32FC1);
	
	int i = 0;
	for(list<Mat>::iterator img = imgList.begin(); img != imgList.end(); img++, i++){
	    for(int p = 0; p<numberOfPixPerImg; p++){
		feature.at<float>(i, p) = img->at<float>(p);
	    }
	}
	db.push_back(feature);
    }  
}
