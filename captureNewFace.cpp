#include "opencv2/opencv.hpp"
#include <iostream>
#include <fstream>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;
using namespace cv;

string window_name = "Capture - Setting up a new face for recognition.";
String face_cascade_name = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";


int image_exists(int n, string pgm_path){
    struct stat st = {0};
    if(stat(format("%s/%d.pgm",pgm_path.c_str(),n).c_str(),&st) == -1)
        return 0;
    else
        return 1;

}


static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}


void captureTheFace(int picNum, Mat frame, VideoCapture cap, Mat *theFacePtr){

    vector<Mat> images;
    vector<int> labels;
    // Read in the data (fails if no valid input filename is given, but you'll get an error message):
    try {
        read_csv("faceLearn.csv", images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }
    // Get the height from the first image. We'll need this
    // later in code to reshape the images to their original
    // size AND we need to reshape incoming faces to this size:
    int im_width = 512;
    int im_height = 512;
    /*if(!images[0].empty()){
        im_width = images[0].cols;
        im_height = images[0].rows;
    }else{
        im_width = 512;
        im_height = 512;
    }
*/
    CascadeClassifier haar_cascade;
    haar_cascade.load(face_cascade_name);

  
    std::cout << "Press SPACE to take picture number " << picNum << "..." << endl;

      Mat original, gray, norm, float_gray, blur, num, den;

      for(;;) {
        cap >> frame;
        // Clone the current frame
        original = frame.clone();
        // Convert the current frame to grayscale and perform illumination normalization
        cvtColor(original, gray, CV_BGR2GRAY);
       
        // Find the faces in the frame:
        vector< Rect_<int> > faces;

        haar_cascade.detectMultiScale(gray, faces);
        
        if(faces.size()){

            // Process face:
            Rect face_i = faces[0];
            // Crop the face from the image:
            Mat face = gray(face_i);

            
            cv::resize(face, face, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);

            // convert to floating-point image
            //face.convertTo(float_gray, CV_32F, 1.0/255.0);
            // numerator = img - gauss_blur(img)
            //cv::GaussianBlur(float_gray, blur, Size(0,0), 2, 2);
            //num = float_gray - blur;
            // denominator = sqrt(gauss_blur(img^2))
            //cv::GaussianBlur(num.mul(num), blur, Size(0,0), 20, 20);
            //cv::pow(blur, 0.5, den);
            // output = numerator / denominator
            //norm = num / den;
            // normalize output into [0,1]
            //cv::normalize(norm, norm, 0.0, 1.0, NORM_MINMAX, -1);
            *theFacePtr = face;
            // First of all draw a green rectangle around the detected face:
            rectangle(original, face_i, CV_RGB(0, 255,0), 1);

            imshow("Face", face);
        }
        // Show the result:
        imshow("NewFaceCapture", original);
        
        // And display it:
        char key = (char) waitKey(32);
        // Exit this loop on
        if(key == 32)
            break;
    }

 
 }

int main(int argc, const char *argv[])
{   // Check for valid command line arguments, print usage
    // if no arguments were given.
    if (argc != 5) {
        cout << "usage: " << argv[0] << " <subject name> <subject id>  <num pictures> </path/to/csv>" << endl;
        cout << "\t <subject name> -- The name of the subject." << endl;
        cout << "\t <subject id> -- The id of the subject (number)." << endl;
        cout << "\t <num pictures> -- The number of pictures to take." << endl;
        cout << "\t </path/to/csv> -- The path to the csv that will be used for recognition." << endl;
        exit(1);
    }else{

        string subject_name = string(argv[1]);
        string subject_id = string(argv[2]);
        int numPix = atoi(string(argv[3]).c_str());
        string csv_path = string(argv[4]);

        cout << "Subject Name: " << subject_name << endl;
        cout << "Subject ID: " << subject_id << endl;
        cout << "Number of Pictures: " << numPix << endl;
        cout << "Path to CSV File: " << csv_path << endl;

VideoCapture cap;
int capNum;
for(capNum=-1;capNum<11;capNum++){
    cap(capNum); // attempt to open the camera
    if(!cap.isOpened()){  // check if we succeeded
        cout << "Capture " << capNum << " failed to open. Check to make sure the camera is configured correctly." << endl;
       // return -1;
    }else{
	cout << "Capture  " << capNum << "opened successfully" << endl;
	break;
    }
}

    if(!cap.isOpened())
	return -1;

    cout << "\t Checking directory structure" << endl;

    string pgm_path = format("facerec/data/at/%s",subject_name.c_str());
    int n = 0;
    struct stat st = {0};
    if (stat(pgm_path.c_str(), &st) == -1) {
        mkdir(pgm_path.c_str(), 0777);
        cout << "\t\t The following directory does not exist:" << pgm_path << " ..." << endl;
        cout << "\t\t We will create it for you ..." << endl;

        //NEW SUBJECT!!

     }else{
        cout << "\t\t Directory structure confirmed." << endl;
        while(image_exists(n,pgm_path)){
           n++;
        }
     }

    for(int i=n;i<numPix+n-1;i++){
        Mat rawFrame;

        cap >> rawFrame; // get a new frame from camera

        cout << "Preparing image " << i << ":" << endl;

        pgm_path =format("facerec/data/at/%s/%d.pgm",subject_name.c_str(),i);

        cout << "Preparing to write image to " << pgm_path << endl;

        Mat theFace;
        captureTheFace(i+1, rawFrame, cap, &theFace);

        imwrite(pgm_path, theFace);

        ofstream myfile;
        myfile.open(csv_path.c_str(),ios::app);
        if(myfile.is_open()){
            myfile << endl << pgm_path << ";" << subject_id;
            myfile.close();
        }else{
            cout << "Unable to open filestream " << csv_path.c_str() << endl;
            exit(1);
        }

    }

    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}


