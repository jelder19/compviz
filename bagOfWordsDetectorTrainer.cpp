

#include "opencv2/opencv_modules.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/ml/ml.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;


int main( int argc, const char** argv ) {


}


Mat featuresUnclustered;
/*
extract feature row vectors from a set of images from your problem domain
*/

//Construct BOWKMeansTrainer

int dictionarySize=1000;
TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
int retries=1;
int flags=KMEANS_PP_CENTERS;

BOWKMeansTrainer bowTrainer(dictionarySize,tc,retries,flags);

/*cluster the feature vectors */
Mat dictionary=bowTrainer.cluster(featuresUnclustered);
