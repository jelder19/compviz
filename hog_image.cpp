#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;
 
int main(int argc, const char * argv[]) {
  
  /**
   * Load image.
   */
  if(argc != 2){
    cout << " Usage: display_image ImageToLoadAndDisplay" << endl;
    return -1;
  }

  Mat img;
  img = imread(argv[1], CV_LOAD_IMAGE_COLOR);   

  if(!img.data){
    cout << "Could not open or find the image" << endl;
    return -1;
  }

  namedWindow("Display window", WINDOW_AUTOSIZE);
  
  /**
   * Initialize HoG.
   */
  HOGDescriptor hog;
  hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

  /**
   * Use HoG to find candidates and mark them on the loaded image.
   */
  vector<Rect> found, found_filtered;
  hog.detectMultiScale(img, found, 0, Size(8,8), Size(32,32), 1.05, 2);
  size_t i, j;

  for(i = 0; i < found.size(); i++){
    Rect r = found[i];
   
    for(j = 0; j < found.size(); j++) 
      if(j != i && (r & found[j]) == r)
        break;
    
    if(j == found.size())
      found_filtered.push_back(r);
  }

  for(i = 0; i < found_filtered.size(); i++){
    Rect r = found_filtered[i];
    
    r.x += cvRound(r.width * 0.1);
    r.width = cvRound(r.width * 0.8);
    r.y += cvRound(r.height * 0.07);
    r.height = cvRound(r.height * 0.8);

    cout << i << ": (" << r.x << ", " << r.y << ")" << endl;
    
    rectangle(img, r.tl(), r.br(), Scalar(255,0,0), 3);        
  }

  imshow("opencv", img);
  waitKey(0);

  return 0;
}