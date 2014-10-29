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
void trackFaces(Mat *frame, Rect trackWindow, unsigned f, std::vector<cv::Rect_<int> > *faces);
string getSubjectName(int thePrediction, string csv_path);

String face_cascade_name = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";
Ptr<FaceRecognizer> model;

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

    cout << "Training the model... ";
    model->train(images, labels);
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
        


        // if(loopNum == 0){
        //     //Detect the face(s)
             detectFaces(original, &faces, haar_cascade);
        // }else if(loopNum == 300){
        //     loopNum = 0;
        // }else{
        // //track the face(s)
        //     for(unsigned f = 0; f < faces.size(); f++){
        //         trackFaces(&original, faces[f], f, &faces);
        //     }

        // }

        //Recognize the face(s)
        recognizeFaces(&original, faces, fn_csv, im_width, im_height);

        //increment the loopNum
        //loopNum++;

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

/** @function trackFaces */
void trackFaces(Mat *frame_ptr, Rect trackWindow, unsigned f, std::vector<cv::Rect_<int> > *faces) {

    Mat frame = *frame_ptr;
    Mat hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;
    int vmin = 10, vmax = 256, smin = 30;
    int hsize = 16;
    float hranges[] = {0,180};
    const float* phranges = hranges;

    int _vmin = vmin; 
    int _vmax = vmax;
    int ch[] = {0, 0};

    cvtColor(frame, hsv, COLOR_BGR2HSV);
    
    inRange(hsv, 
        Scalar(0, smin, MIN(_vmin,_vmax)),
        Scalar(180, 256, MAX(_vmin,_vmax)), 
    mask
    );

  hue.create(hsv.size(), hsv.depth());
  mixChannels(&hsv, 1, &hue, 1, ch, 1);

  Mat roi(hue, trackWindow), maskroi(mask, trackWindow);
  calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
  normalize(hist, hist, 0, 255, NORM_MINMAX);

  histimg = Scalar::all(0);
  int binW = histimg.cols / hsize;
  Mat buf(1, hsize, CV_8UC3);
  
  for(int i = 0; i < hsize; i++ )
    buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);
  
  cvtColor(buf, buf, COLOR_HSV2BGR);

  for(int i = 0; i < hsize; i++ ){
    int val = saturate_cast<int>(hist.at<float>(i)*histimg.rows/255);
    
    rectangle(histimg, 
        Point(i*binW,histimg.rows),
        Point((i+1)*binW,histimg.rows - val),
      Scalar(buf.at<Vec3b>(i)), -1, 8
    );
  }

  calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
  backproj &= mask;
  RotatedRect trackBox = CamShift(backproj, trackWindow, TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));
  
  if(trackWindow.area() <= 1) {
    int cols = backproj.cols, rows = backproj.rows, r = (MIN(cols, rows) + 5)/6;
    trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
                        trackWindow.x + r, trackWindow.y + r) &
    Rect(0, 0, cols, rows);
  }

  (*faces)[f] = trackWindow; 
  ellipse(*frame_ptr, trackBox, Scalar(0,0,255), 3);
}

string getSubjectName(int thePrediction, string csv_path){
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
            Mat face_resized;
            cv::resize(face, face_resized, Size(target_width, target_height), 1.0, 1.0, INTER_CUBIC);
            // Now perform the prediction, see how easy that is:
            int prediction = model->predict(face_resized);
            // And finally write all we've found out to the original image!
            // First of all draw a green rectangle around the detected face:
            rectangle(*frame, face_i, CV_RGB(0, 255,0), 1);


            // Calculate the position for annotated text (make sure we don't
            // put illegal values in there):
            int pos_x = std::max(face_i.tl().x - 10, 0);
            int pos_y = std::max(face_i.tl().y - 10, 0);
            // And now put it into the image:
            string subj_name = getSubjectName(prediction,csv_path);
            putText(*frame, subj_name, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
        }
}