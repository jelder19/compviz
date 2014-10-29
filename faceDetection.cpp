#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );

/** Global variables */
String face_cascade_name = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";

//String eyes_cascade_name = "/usr/local/share/OpenCV/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

//String smile_cascade_name = "/usr/local/share/OpenCV/haarcascades/haarcascade_smile.xml";

CascadeClassifier face_cascade;
//CascadeClassifier eyes_cascade;
//CascadeClassifier smile_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);
int isStable = 0;
std::vector<Rect> oldFaces;

/** @function main */
int main( int argc, const char** argv )
{
 CvCapture* capture;
 Mat frame;

 //-- 1. Load the cascades
 if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
 //if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
 //if( !smile_cascade.load( smile_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

 //-- 2. Read the video stream
 capture = cvCaptureFromCAM( -1 );
 if( capture ){
   while( true ){
 frame = cvQueryFrame( capture );

 //-- 3. Apply the classifier to the frame
     if( !frame.empty() ){ 
      detectAndDisplay( frame ); 
     }else{ 
      printf(" --(!) No captured frame -- Break!"); break; }

     int c = waitKey(10);
     if( (char)c == 'c' ) { break; }
    }
 }
 return 0;
}


/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
std::vector<Rect> faces;
std::vector<Rect> facesToDraw;
Mat frame_gray;

cvtColor( frame, frame_gray, CV_BGR2GRAY );
equalizeHist( frame_gray, frame_gray );

//-- Detect faces
face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

if(faces.size() != oldFaces.size()){
  if(faces.size() != 0){
    cout << "We've found " << faces.size() << " face(s) at the following location(s):\n";
  }else{
    cout << "No faces detected! (we've lost you!)\n";
  }
  isStable = 0;
  oldFaces = faces;
}else{
  isStable = 1;
}


  for( size_t i = 0; i < faces.size(); i++ ){

    //use pythagorean thm to calculate the distance the face has moved
    double face_move_dist = sqrt(pow(abs(faces[i].x - oldFaces[i].x),2) +  pow(abs(faces[i].y - oldFaces[i].y),2));

    if(face_move_dist == 0){
      //this is a newly detected face
      cout << "     Face number " << i+1 << " is located at (" << faces[i].x + faces[i].width*0.5  << "," << faces[i].y + faces[i].height*0.5 << ")\n";

    }else if(face_move_dist > 50){ 
      //this face has moved enough to alert the user
      cout << "     Face number " << i+1 << " has moved to (" << faces[i].x + faces[i].width*0.5 << "," << faces[i].y + faces[i].height*0.5 << ")\n";
      oldFaces[i] = faces[i];
    }


    if(isStable){
      facesToDraw = faces;
    }else{
      facesToDraw = oldFaces;
    }

    Point center( facesToDraw[i].x + facesToDraw[i].width*0.5, facesToDraw[i].y + facesToDraw[i].height*0.5 );
    ellipse( frame, center, Size( facesToDraw[i].width*0.5, facesToDraw[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );

    Mat faceROI = frame_gray( facesToDraw[i] );
    
    // std::vector<Rect> smiles;

    // //-- In each face, detect eyes
    // smile_cascade.detectMultiScale( faceROI, smiles, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

    // for( size_t j = 0; j < smiles.size(); j++ ){
    //   Point center( faces[i].x + smiles[j].x + smiles[j].width*0.5, faces[i].y + smiles[j].y + smiles[j].height*0.5 );
    //   int radius = cvRound( (smiles[j].width + smiles[j].height)*0.25 );
    //   circle( frame, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
    // }
  }
//-- Show what you got
imshow( window_name, frame );
}