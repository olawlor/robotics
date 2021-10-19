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

  cv::Mat frame,gray,gradX,gradY;
  
  // An infinite loop
  while(true)
  {
    (*cap)>>frame; // grab next frame from camera

    // If we couldn't grab a frame... quit
    if(frame.empty())
      break;
    
    // Grayscale
    cv::cvtColor(frame,gray,cv::COLOR_BGR2GRAY);
    // Gradient estimate (with filtering)
    int ksize=3;
    cv::Sobel(gray,gradX,CV_32F, 1,0, ksize);
    cv::Sobel(gray,gradY,CV_32F, 0,1, ksize);
    
    // Show raw video and gradient onscreen
    cv::imshow("raw video",frame);
    cv::imshow("gradient in X",gradX*0.01+0.5);
    
    // Wait for a keypress (for up to 1ms)
    int c = cv::waitKey(1);
    if(c=='q') break;
  }

  return 0;
} 


 

