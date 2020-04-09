/* 
Simple OpenCV example

Dr. Orion Lawlor, lawlor@alaska.edu, 2020-04-07 (Public Domain)
*/
#include <stdlib.h> /* for NULL */
#include <iostream>  
#include <opencv2/opencv.hpp>


int main(int argc,char *argv[])
{
  cv::VideoCapture *cap=new cv::VideoCapture(0);

  // Couldn't get a device? Throw an error and quit
  if(!cap->isOpened())
  {
    printf("Could not initialize capturing...\n");
    return -1;
  }

  cv::Mat frame,template_img,match;
  template_img=cv::imread("template.jpg");
  
  // An infinite loop
  while(true)
  {
    (*cap)>>frame; // grab next frame from camera

    // If we couldn't grab a frame... quit
    if(frame.empty())
      break;
    cv::imwrite("capture.jpg",frame);
    
    cv::matchTemplate(frame,template_img,match,cv::TM_CCOEFF_NORMED);
    
    
    cv::imshow("raw video",frame);
    cv::imshow("match",match);
    
    std::cout<<"Gradient in middle:"<<match.at<float>(100,100)<<"\n";
    
    
    // Wait for a keypress (for up to 1ms)
    int c = cv::waitKey(1);
    if(c=='q') break;
  }

  return 0;
} 


 

