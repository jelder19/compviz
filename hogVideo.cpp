#include <opencv2/opencv.hpp>
 
using namespace std;
using namespace cv;

// global variables
String window = "Human Tracking";

// function headers
vector<Rect> detectPeople(Mat);
void camshiftTrack(Mat, Rect);

// main loop
int main (int argc, const char *argv[]) {
  
  String fileName;

  // video file not specified
  if(!argv[1]){
    cerr << "Error: usage ./hogVideo <filepath/to/video>" << endl;
    return -1;
  }else{
    fileName = argv[1];
  }

  // initialize video capture
  VideoCapture cap(fileName);

  if(!cap.isOpened())
    return -1;

  Mat frame;
  vector<Rect> people;
  namedWindow(window, CV_WINDOW_AUTOSIZE);
  int counter = 0;

  while(true){
    
    //-----------------------------------------------
    // FRAME CAPTURE
    //-----------------------------------------------
    cap >> frame;
    
    if(frame.empty())
      continue;

    //-----------------------------------------------
    // PEOPLE DETECTION
    //
    // If people are detected in a frame, we
    // can begin facial detection and recognition.
    //-----------------------------------------------
    if(!counter){
      people = detectPeople(frame);      
      counter = 20;
    }

    //-----------------------------------------------
    // FACIAL DETECTION & RECOGNITION
    // 
    // Detect some faces and perform recognition.
    // If a person is recognized, save a video of
    // them walking up to the door.
    //-----------------------------------------------
    if(!people.empty()){
      
    }

    //-----------------------------------------------
    // TRACK & SHOW
    //-----------------------------------------------

    // todo:
    // - sepate thread for displaying
    // - track people if no faces
    if(!people.empty()){

      for(size_t i = 0; i < people.size(); i++){
        Rect person = people[i];

        if(person.x > 0 && person.y > 0 && 
           person.x + person.width <= frame.cols && 
           person.y + person.height <= frame.rows){

          // optional, print someone's coordinates
          cout << "(" << people[i].x << "," << people[i].y << ") " << endl;
          camshiftTrack(frame, person);
        }
      }
    }

    imshow(window, frame);
    counter--;
    
    //-----------------------------------------
    // EXIT
    //-----------------------------------------
    if(waitKey(10) >= 0)
      break;
  }

  return 0;
}

/**
 * People detection using HoG.
 */

vector<Rect> detectPeople(Mat frame) {
  HOGDescriptor hog;
  vector<Rect> found, filtered;
  size_t i, j;

  hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
  hog.detectMultiScale(
    frame, 
    found, 
    0, 
    Size(8,8), 
    Size(32,32), 
    1.05, 
    2
  );

  for(i = 0; i < found.size(); i++){
    Rect r = found[i];
    
    for(j = 0; j < found.size(); j++) 
      if(j != i && (r & found[j]) == r)
        break;
    
    if(j == found.size())
      filtered.push_back(r);
  }

  for(i = 0; i < filtered.size(); i++){
    Rect r = filtered[i];
    
    r.x += cvRound(r.width*0.1);
    r.width = cvRound(r.width*0.8);
    r.y += cvRound(r.height*0.07);
    r.height = cvRound(r.height*0.8);
    
    rectangle(frame, r.tl(), r.br(), Scalar(0,255,0), 1);        
  }

  return filtered;
}

/**
 * Camshift tracking function.
 */

void camshiftTrack(Mat frame, Rect trackWindow) {
  Mat hsv, 
      hue, 
      mask, 
      hist,
      backproj, 
      histimg = Mat::zeros(200, 320, CV_8UC3); 
  
  int vmin = 10, vmax = 256, smin = 30;
  int _vmin = vmin, _vmax = vmax;
  int ch[] = {0, 0};
  int hsize = 16;
  float hranges[] = {0,180};
  const float* phranges = hranges;

  cvtColor(frame, hsv, COLOR_BGR2HSV);
  
  inRange(
    hsv, 
    Scalar(0, smin, MIN(_vmin,_vmax)),
    Scalar(180, 256, MAX(_vmin, _vmax)), 
    mask
  );

  hue.create(hsv.size(), hsv.depth());
  mixChannels(&hsv, 1, &hue, 1, ch, 1);

  Mat roi(hue, trackWindow), 
      maskroi(mask, trackWindow);
  
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
    int cols = backproj.cols, 
        rows = backproj.rows, 
        r = (MIN(cols, rows) + 5) / 6;
    
    trackWindow = Rect(trackWindow.x - r, trackWindow.y - r,
                    trackWindow.x + r, trackWindow.y + r) &
                  Rect(0, 0, cols, rows);
  }

  ellipse(frame, trackBox, Scalar(0,255,0), 1);
}
