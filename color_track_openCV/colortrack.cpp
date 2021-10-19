/*
Color-based image matching.  Detects two colored regions in the image,
and logs both their centers of mass.

Basic OpenCV info:
  http://opencv.willowgarage.com/documentation/cpp/basic_structures.html

OpenCV calls yanked from Utkarsh's page, and translated to OpenCV2:
  http://www.aishack.in/2010/07/tracking-colored-objects-in-opencv/
  https://github.com/liquidmetal/AI-Shack--Tracking-with-OpenCV

Pixel access discussed here:
  http://stackoverflow.com/questions/4742251/pixel-access-in-opencv-2-2

Dr. Orion Sky Lawlor, 2013-08-25 (Public Domain)
*/
#include <stdio.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>

int tolH=10; // hue tolerance
int tolS=100; // saturation tolerance
int tolV=100; // value tolerance (huge, due to lighting variations)
int min_pixels=10; // minimum detected size (box with this many pixels on a side)
int box_pixels=100; // ignore detected pixels farther than this from the initial center of mass

/* Return the current wall clock time, in seconds */
double time_in_seconds(void) {
  return cv::getTickCount()/cv::getTickFrequency();
}


int mouse_x,mouse_y;
int mouse_flags=0;
void myMouseCallback(int event,int x,int y,int flags,void *param)
{
  if (event==CV_EVENT_MOUSEMOVE) {mouse_x=x; mouse_y=y;}
  mouse_flags=flags;
}


/*
 Search an image for a particular color, and return the coordinates
 of that color in the image.
*/
class color_matcher {
public:
  cv::Mat threshed;  // thresholded image: black for no match; white for good match
  
  float posX,posY; // (x,y) pixel coordinates of matching pixels' center of mass
  float area; // pixels hit by center of mass
  
  // Look for this HSV color in this image.
  bool match(const cv::Mat &imgHSV,const cv::Vec3b &target);
public:
  /* Set our position from the center of mass of this image. */
  void calcMoments(const cv::Mat &img) {
    cv::Moments mu=cv::moments(img);
    area = mu.m00; // sum of zero'th moment is area
    posX = mu.m10/area; // center of mass = w*x/weight
    posY = mu.m01/area;
    area /= 255; // scale from bytes to pixels
  }
};
  
bool color_matcher::match(const cv::Mat &imgHSV,const cv::Vec3b &target)
{
  // Find matching pixels (color in range)
  // Order here is: Hue, Saturation, Value
  threshed=cv::Mat(imgHSV.size(),1);
  cv::inRange(imgHSV, 
    cv::Scalar(target[0]-tolH, target[1]-tolS, target[2]-tolV),
    cv::Scalar(target[0]+tolH, target[1]+tolS, target[2]+tolV),
    threshed);
  
  // Calculate the moments to estimate average position
  calcMoments(threshed);
  
  if (area>min_pixels*min_pixels) {
    // Trim the image around the initial average's center of mass
    cv::Rect roi(-box_pixels,-box_pixels,2*box_pixels,2*box_pixels);
    roi+=cv::Point(posX,posY); // shift to initial center
    roi&=cv::Rect(cv::Point(0,0),threshed.size()); // trim to thresh's rectangle
    calcMoments(threshed(roi)); // recompute moments
    posX+=roi.x; posY+=roi.y; // shift back to pixels
    return true;
  }
  else return false;
}

int main(int argc,char *argv[])
{
  cv::VideoCapture *cap=0;
  long framecount=0;
  long frameskip=1; // draw GUI every this often
  double time_start=-1.0;
  
  // bright pink (in HSV)
  int target_H=170, target_S=180, target_V=200;
  
  int argi=1; // command line argument index
  while (argc>=argi+2) { /* keyword-value pairs */
    char *key=argv[argi++];
    char *value=argv[argi++];
    if (0==strcmp(key,"--cam")) cap=new cv::VideoCapture(atoi(value));
    else if (0==strcmp(key,"--file")) cap=new cv::VideoCapture(value);
    else if (0==strcmp(key,"--skip")) frameskip=atol(value);
    else if (0==strcmp(key,"--hue")) target_H=atol(value);
    else if (0==strcmp(key,"--saturation")) target_S=atol(value);
    else if (0==strcmp(key,"--value")) target_V=atol(value);
    else printf("Unrecognized command line argument '%s'!\n",key);
  }
  // Initialize capturing live feed from the camera
  if (!cap) cap=new cv::VideoCapture(0);

  // Couldn't get a device? Throw an error and quit
  if(!cap->isOpened())
  {
    printf("Could not initialize capturing...\n");
    return -1;
  }

  // Make the windows we'll be using
  cv::namedWindow("video");
  cv::setMouseCallback("video",myMouseCallback,NULL);
  cv::createTrackbar("Hue tolerance","video",&tolH,180,NULL);
  cv::createTrackbar("Saturation tolerance","video",&tolS,255,NULL);
  cv::createTrackbar("Value tolerance","video",&tolV,255,NULL);
  cv::createTrackbar("Area minimum","video",&min_pixels,200,NULL);

  // This image holds the "scribble" data...
  // the tracked positions of the target
  cv::Mat scribble;
  
  // Will hold a frame captured from the camera
  cv::Mat frame, frameHSV;
  
  // Image color matching:
  enum {nMatchers=1};
  color_matcher matchers[nMatchers];
  
  cv::Vec3b targets[nMatchers];
  targets[0]=cv::Vec3b(target_H,target_S,target_V); 

  // An infinite loop
  while(true)
  {
    (*cap)>>frame; // grab next frame from camera

    // If we couldn't grab a frame... quit
    if(frame.empty())
      break;
    
    framecount++;
    if (time_start<0) time_start=time_in_seconds();
    
    if (scribble.empty())
      scribble = cv::Mat::zeros(frame.size(),CV_8U);

    // Convert to HSV (for more reliable color matching)
    cv::cvtColor(frame, frameHSV, cv::COLOR_BGR2HSV);
    
    // Dump the value at the mouse location
    if (mouse_x>=0 && mouse_x<frameHSV.cols &&
        mouse_y>=0 && mouse_y<frameHSV.rows &&
        0!=(mouse_flags&CV_EVENT_FLAG_LBUTTON))
    {
      printf("HSV: ");
      cv::Vec3b pix=frameHSV.at<cv::Vec3b>(mouse_y,mouse_x);
      for (int k=0;k<3;k++) printf("%d ",(int)pix[k]);
      printf("\n");
      fflush(stdout); // <- for "tee" or "tail"
      targets[0]=pix;
    }
    
    // Do color matching
    for (int m=0;m<nMatchers;m++)
      matchers[m].match(frameHSV,targets[m]);
    
    // We want to draw a line only if its a valid position
    if  (matchers[0].area>min_pixels*min_pixels)
    {
      // Print the current robot position
      printf("position  %.2f  %.2f  %.0f  %.3f  xya t\n", 
        matchers[0].posX, matchers[0].posY, matchers[0].area,
        time_in_seconds()-time_start);
      fflush(stdout); // <- for "tee" or "tail"
      
      // Draw crosshairs at the matched point
      int crosshair=20;
      for (int polarity=-1;polarity<=+1;polarity+=2)
        cv::line(scribble, 
          cv::Point(matchers[0].posX-crosshair, matchers[0].posY+polarity*crosshair), 
          cv::Point(matchers[0].posX+crosshair, matchers[0].posY-polarity*crosshair), 
          255, 3);
    }

    if ( ((framecount)%frameskip)==0 ) { // GUI update every other video frame
      // Mix with video
      std::vector<cv::Mat> channels;
      channels.push_back(scribble); // blue channel (same as green)
      channels.push_back(scribble); // green channel
      channels.push_back(matchers[0].threshed); // red channel 
      cv::Mat colorThresh;
      cv::merge(channels,colorThresh); // threshold images packed into colors

      frame=frame*0.5+colorThresh;
      cv::imshow("video", frame);
    
      // Slowly scale back old scribbles to zero
      scribble=scribble*0.9-5;

      // Wait for a keypress (for up to 1ms)
      int c = cv::waitKey(1);
      if(c=='q')
      {
        // If any key is pressed, break out of the loop
        break;
      }
    }
  }

  return 0;
}
