#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
 
using namespace std;
using namespace cv;

// global variables
string window = "Human Tracking";
string faceCascadeName = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml";
string fileName;
string csvPath;
vector<Rect> faces;
CascadeClassifier faceCascade;
Ptr<FaceRecognizer> model;

// function headers
vector<Rect> detectPeople(Mat);
void camshiftTrack(Mat, Rect);
void detectFaces(Mat);
void recognizeFaces(Mat *frame, vector< Rect_<int> > faces, string csv_path, int image_width, int image_height);
string recognizeFace(Mat *faceFrame, string csv_path, int image_width, int image_height);
static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels);
string getSubjectName(int thePrediction, double theConfidence, string csv_path);

// main loop
int main (int argc, char *argv[]) {

  if(!(argc % 2)){
    cerr << "Usage: main [OPTIONS] [-v <path/to/video>] [-f <path/to/csv>]" << endl;
    return -1;
  }

  for(size_t i = 1; i < argc; i += 2){
    if(string(argv[i]) == "-v"){
      fileName = string(argv[i + 1]);
      cout << "Analyzing " << fileName << endl;
    }

    if(string(argv[i]) == "-f"){
      csvPath = string(argv[i + 1]);
      cout << "Enabling facial recognition with path " << csvPath << endl;
    }
  }

  VideoCapture cap;

  if(fileName.empty()){
    cap.open(1);
  }else{
    cap.open(fileName);
  }

  if(!cap.isOpened()){
    cerr << "Video capture failed to open" << endl;
    return -1;
  }

  Mat frame;
  vector<Rect> people;
  namedWindow(window, CV_WINDOW_AUTOSIZE);
  
  int peopleCounter = 0;
  int faceCounter = 0;
  int imWidth, imHeight;
  bool enableFaceRecognition;
  vector<Mat> images;
  vector<int> labels;
  
  if(!csvPath.empty()){
    try {
      read_csv(csvPath, images, labels);
    } catch (cv::Exception& e) {
      cerr << "Error opening file \"" << csvPath << "\". Reason: " << e.msg << endl;
      return -1;
    }

    if(images.size()){
      imWidth = images[0].cols;
      imHeight = images[0].rows;

      model = createFisherFaceRecognizer();
      model->train(images, labels);
      enableFaceRecognition = true;
    }else{
      enableFaceRecognition = false;
    }
  }else{
    enableFaceRecognition = false;
  }

  faceCascade.load(faceCascadeName);
  string name_conf = "";

  while(true){
    
    //-----------------------------------------------
    // FRAME CAPTURE
    //-----------------------------------------------
    cap >> frame;
    
    if(frame.empty()){
      cout << "frame is empty" << endl;
      continue;
    }
    //-----------------------------------------------
    // PEOPLE DETECTION
    //
    // If people are detected in a frame, we
    // can begin facial detection and recognition.
    //-----------------------------------------------

    // todo:
    // - if people are detected, start looking for faces
    // - maybe forget about HoG for awhile and only do faces
    
    if(!peopleCounter){
      people = detectPeople(frame);      
      
      if(!people.empty()){
        peopleCounter = 400;        
      }else{
        continue;
      }
    }

    //-----------------------------------------------
    // FACIAL DETECTION & RECOGNITION
    // 
    // Detect some faces and perform recognition.
    // If a person is recognized, save a video of
    // them walking up to the door.
    //-----------------------------------------------

    // todo:
    // - if a face shows up for one frame, ignore it
    // - if a good face is found, start performing recognition
    // - start recording video of walking up to door
    

    detectFaces(frame);

    if(!faceCounter){
      

      if(enableFaceRecognition && !faces.empty()){

        for(size_t i = 0; i < faces.size(); i++){
          Rect face = faces[i];
          Mat faceFrame = frame(face);
          imshow("The Face", faceFrame);
          name_conf = recognizeFace(&faceFrame, csvPath, imWidth, imHeight);

        }
      }
      

      faceCounter = 30;   
    }
    
    
    if(!faces.empty()){

      for(size_t i = 0; i < faces.size(); i++){
        Rect face = faces[i];

        if(face.x > 0 && face.y > 0 && 
           face.x + face.width <= frame.cols && 
           face.y + face.height <= frame.rows){

          // optional, print someone's coordinates
          //cout << "(" << people[i].x << "," << people[i].y << ") " << endl;
          imshow("VJ Face", frame(face));

          Rect trackFace = Rect(face.x+face.width*.1,face.y+face.height*.1,face.width*.9, face.height*.9);
          imshow("Track Face", frame(trackFace));
          //camshiftTrack(frame, trackFace);
          putText(frame, name_conf, Point(face.x, face.y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);
        }
      } 
    }

    imshow(window, frame);
    peopleCounter = peopleCounter > 0 ? peopleCounter - 1 : 0;
    faceCounter = faceCounter > 0 ? faceCounter - 1 : 0;


    cout << "peopleCounter: " << peopleCounter << endl;
    cout << "faceCounter: " << faceCounter << endl;
    
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

  cout << "--------------------------------------------" << endl;
  cout << "function: detectPeople" << endl;
  cout << "--------------------------------------------" << endl;

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

  cout << "TRACKING RIGHT NOW" << endl;
}

/**
 * Detect faces.
 */

void detectFaces(Mat frame) {
  Mat frameGray;

  cvtColor(frame, frameGray, CV_BGR2GRAY);
  equalizeHist(frameGray, frameGray);

  faceCascade.detectMultiScale(frameGray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

  for(size_t i = 0; i < faces.size(); i++){
    cout << "detected face" << endl;
    Point center(faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height * 0.5);
    ellipse(frame, center, Size(faces[i].width * 0.5, faces[i].height * 0.5), 0, 0, 360, Scalar(0, 255, 0), 2, 8, 0);
  } 
}


/**
 * Read a csv file.
 */

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

/**
 * Recognize a single face.
 */
string recognizeFace(Mat *faceFrame, string csv_path, int image_width, int image_height) {
  Mat gray;
  cvtColor(*faceFrame, gray, CV_BGR2GRAY);

  // Crop the face from the image. So simple with OpenCV C++:

  Mat face_resized;

  cv::resize(gray, face_resized, Size(image_width, image_height), 1.0, 1.0, INTER_CUBIC);

  
  //imshow("Face",face);
  // Now perform the prediction:
  int predictionID;
  double predictionConfidence;
  model->predict(face_resized,predictionID,predictionConfidence);


  // And now put it into the image:

  string subj_name = getSubjectName(predictionID,predictionConfidence,csv_path);
  //putText(*mainFrame, format("%s %f",subj_name.c_str(),predictionConfidence), Point(x, y), FONT_HERSHEY_PLAIN, 1.0, CV_RGB(0,255,0), 2.0);

  //cout << "name: " << subj_name << ", confidence: " << (double)predictionConfidence << endl;


  return format("%s %.2f",subj_name.c_str(),predictionConfidence);
}


/**
 * Recognize faces.
 */

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

            Mat face_resized, norm_resized, norm, float_gray, blur, num, den;

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
            //rectangle(*frame, face_i, CV_RGB(0, 255,0), 1);
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

/**
 * Return the subject name for a given
 * prediction.
 */

string getSubjectName(int thePrediction, double theConfidence, string csvPath) {
  double confidenceThresh = 800;

  if(theConfidence >= confidenceThresh){
       return "unknown";
  }

  string predictionString = format("%d",thePrediction);
  ifstream file(csvPath.c_str());
  string line;
  string base_path = "facerec/data/at/";
  string subj_name;
  bool done = 0;
  
  while(file.good()){
    getline(file, line, ';'); // read a string until ';'
    if(line.find(predictionString) != std::string::npos){
      //this line contains the info we are looking for (the name of the matched subject) 
      return line.substr(csvPath.length()+predictionString.length()+1,line.length()-csvPath.length()-predictionString.length()-7);
    }
  }

  return "something went wrong";
}

