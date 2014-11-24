/*
 * Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
 * Released to public domain under terms of the BSD Simplified license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the organization nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *   See <http://www.opensource.org/licenses/bsd-license>
 */
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

using namespace cv;
using namespace std;

void detectFaces(Mat frame, std::vector<cv::Rect_<int> > *theFaces,CascadeClassifier theCascade);
void recognizeFaces(Mat *frame, vector< Rect_<int> > faces, string csv_path, int image_width, int image_height);
static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels);
string getSubjectName(int thePrediction, double theConfidence, string csv_path);

String face_cascade_name = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";

Ptr<FaceRecognizer> model;
//Ptr<FaceRecognizer> LBPHmodel;
double thresh = 0.0;

int main(int argc, const char *argv[]) {
    // Check for valid command line arguments, print usage
    // if no arguments were given.
    if (argc != 2) {
        cout << "usage: " << argv[0] << "</path/to/csv.ext>" << endl;
        cout << "\t </path/to/csv.ext> -- Path to the CSV file with the face database." << endl;
        exit(1);
    }
    // Get the path to your CSV:
    string fn_csv = string(argv[1]);
    int deviceId = -1;
    // These vectors hold the images and corresponding labels:
    vector<Mat> images;
    vector<int> labels;
    // Read in the data (fails if no valid input filename is given, but you'll get an error message):
    try {
        read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }
    // Get the height from the first image. We'll need this
    // later in code to reshape the images to their original
    // size AND we need to reshape incoming faces to this size:
    int im_width = images[0].cols;
    int im_height = images[0].rows;
    // Create a FaceRecognizer and train it on the given images:
    cout << "Creating FisherFace recognizer model... ";
    model = createFisherFaceRecognizer();
    cout << "Done!" << endl;

   //cout << "Creating LBPH recognizer model... ";
   // LBPHmodel = createLBPHFaceRecognizer(1,8,8,8,DBL_MAX);
   // cout << "Done!" << endl;


    cout << "Training the models... ";
    model->train(images, labels);
    //model->set("threshold",500);
    //LBPHmodel->train(images, labels);
    cout << "Done!" << endl;
    // That's it for learning the Face Recognition model. You now
    // need to create the classifier for the task of Face Detection.
    // We are going to use the haar cascade you have specified in the
    // command line arguments:
    //Load the cascade for face DETECTION
    cout << "Loading the HAAR Cascade... ";
    CascadeClassifier haar_cascade;
    haar_cascade.load(face_cascade_name);
    cout << "Done!" << endl;



    // Get a handle to the Video device:
    VideoCapture cap(deviceId);
    // Check if we can use this device at all:
    if(!cap.isOpened()) {
        cerr << "Capture Device ID " << deviceId << "cannot be opened." << endl;
        return -1;
    }
    // Holds the current frame from the Video device:
    Mat frame;
    vector< Rect_<int> > faces;
    //int loopNum = 0; //Use this to decide whether to detect, or track
    for(;;) {
        cap >> frame;
        // Clone the current frame:
        Mat original = frame.clone();

        detectFaces(original, &faces, haar_cascade);

        if(faces.size()){
        //Recognize the face(s)
            recognizeFaces(&original, faces, fn_csv, im_width, im_height);
        }else{
            putText(original,"No Faces in the Frame", Point(10, 10), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(255,0,0), 2.0);
        }
       
        // Show the result:
        imshow("face_recognizer", original);
        // And display it:
        char key = (char) waitKey(20);
        // Exit this loop on escape:
        if(key == 27)
            break;
    }
    return 0;
}

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels) {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }
    string line, path, classlabel;

    cout << "Reading in images for comparison... ";
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, ';');
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
    cout << "Done!" << endl;
}

/** @function detectFaces */
void detectFaces(Mat frame, std::vector<cv::Rect_<int> > *theFaces,CascadeClassifier theCascade){
    
    Mat frame_gray;

    cvtColor(frame, frame_gray, CV_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    
    theCascade.detectMultiScale(frame_gray, *theFaces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
}


void recognizeFaces(Mat *frame, vector< Rect_<int> > faces, string csv_path, int target_width, int target_height){
       Mat gray;
       cvtColor(*frame, gray, CV_BGR2GRAY);
       for(int i = 0; i < faces.size(); i++) {
            // Process face by face:
            Rect face_i = faces[i];
            // Crop the face from the image. So simple with OpenCV C++:
            Mat face = gray(face_i);
            // Resizing the face is necessary for Eigenfaces and Fisherfaces. You can easily
            // verify this, by reading through the face recognition tutorial coming with OpenCV.
            // Resizing IS NOT NEEDED for Local Binary Patterns Histograms, so preparing the
            // input data really depends on the algorithm used.
            //
            // I strongly encourage you to play around with the algorithms. See which work best
            // in your scenario, LBPH should always be a contender for robust face recognition.
            //
            // Since I am showing the Fisherfaces algorithm here, I also show how to resize the
            // face you have just found:

            Mat face_resized, norm_resized, gray, norm, float_gray, blur, num, den;

             // convert to floating-point image
             //face.convertTo(float_gray, CV_32F, 1.0/255.0);
             // numerator = img - gauss_blur(img)
             //cv::GaussianBlur(float_gray, blur, Size(0,0), 2, 2);
             // num = float_gray - blur;
             // denominator = sqrt(gauss_blur(img^2))
             //cv::GaussianBlur(num.mul(num), blur, Size(0,0), 20, 20);
             //cv::pow(blur, 0.5, den);
             // output = numerator / denominator
             //norm = num / den;
             // normalize output into [0,1]
             //cv::normalize(norm, norm, 0.0, 1.0, NORM_MINMAX, -1);

             cv::resize(face, face_resized, Size(target_width, target_height), 1.0, 1.0, INTER_CUBIC);

            
            //imshow("Face",face);
            // Now perform the prediction:
            int predictionID;
            double predictionConfidence;
            model->predict(face_resized,predictionID,predictionConfidence);
            // And finally write all we've found out to the original image!
            // First of all draw a green rectangle around the detected face:
            rectangle(*frame, face_i, CV_RGB(0, 255,0), 1);
            // Calculate the position for annotated text (make sure we don't
            // put illegal values in there):
            int pos_x = std::max(face_i.tl().x - 10, 0);
            int pos_y = std::max(face_i.tl().y - 10, 0);
            // And now put it into the image:


            string subj_name = getSubjectName(predictionID,predictionConfidence,csv_path);
            putText(*frame, format("%s %d",subj_name.c_str(),predictionConfidence), Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);

            // LBPHmodel->predict(face,predictionID,predictionConfidence);
            // rectangle(*frame, face_i, CV_RGB(0,0,255), 1);
            // // Calculate the position for annotated text (make sure we don't
            // // put illegal values in there):
            // pos_x = std::max(face_i.tl().x + 10, 0);
            // pos_y = std::max(face_i.tl().y + 10, 0);
            // // And now put it into the image:
            // subj_name = getSubjectName(predictionID,predictionConfidence,csv_path);
            // putText(*frame, format("%s %d",subj_name.c_str(),predictionConfidence), Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,0,255), 2.0);
        }
}

string getSubjectName(int thePrediction, double theConfidence, string csv_path){
    double confidenceThresh = 300;

     if(theConfidence >= confidenceThresh){
       //  return "unknown";
     }

    string predictionString = format("%d",thePrediction);
    ifstream file (csv_path.c_str());
    string line;
    string base_path = "/home/ryan/compviz/facerec/data/at/";
    string subj_name;
    bool done = 0;
    while (file.good()){
        getline (file, line, ';'); // read a string until ';'
        if(line.find(predictionString) != std::string::npos){
            //this line contains the info we are looking for (the name of the matched subject) 
            return line.substr(base_path.length()+predictionString.length()+1,line.length()-base_path.length()-predictionString.length()-7);
        }
    }

    return "something went wrong";

}