#include <opencv2/opencv.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectFaces(Mat);
void trackFaces(Mat, Rect, unsigned);

/** Global variables */
String face_cascade_name = "haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
String window_name = "Capture - Face detection";
RNG rng(12345);
vector<Rect> faces;
Mat frame, hsv, hue, mask, hist, histimg = Mat::zeros(200, 320, CV_8UC3), backproj;
int vmin = 10, vmax = 256, smin = 30;
int hsize = 16;
float hranges[] = {0,180};
const float* phranges = hranges;

/** @function main */
int main(int argc, const char** argv) {
	CvCapture* capture;

	// load the cascades
	if(!face_cascade.load(face_cascade_name)){ printf("--(!)Error loading\n"); return -1; };

	// read the video stream
	capture = cvCaptureFromCAM(-1);

	if(capture){
	  while(true){
			frame = cvQueryFrame(capture);

			// apply the classifier to the frame
			if(!frame.empty()){

				// if faces is empty, populate faces vector
				if(faces.empty()){
					detectFaces(frame);
					imshow(window_name, frame); 
				}else{
					for(unsigned f = 0; f < faces.size(); f++){
						cin.ignore();

						trackFaces(frame, faces[f], f);
					}

					imshow(window_name, frame);
				}
			}else{ 
			
				printf(" --(!) No captured frame -- Break!"); 
				break; 
			}

			int c = waitKey(10);
			if((char)c == 'c') { break; }
		}
	}
	
	return 0;
}

/** @function detectFaces */
void detectFaces(Mat frame) {
	Mat frame_gray;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
}

/** @function trackFaces */
void trackFaces(Mat frame, Rect trackWindow, unsigned f) {
	int _vmin = vmin, _vmax = vmax;
	int ch[] = {0, 0};

	cvtColor(frame, hsv, COLOR_BGR2HSV);
	
	inRange(hsv, 
		Scalar(0, smin, MIN(_vmin,_vmax)),
    	Scalar(180, 256, MAX(_vmin, _vmax)), 
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

  faces[f] = trackWindow; 
	ellipse(frame, trackBox, Scalar(0,0,255), 3);
}