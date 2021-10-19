/*****************************************************************************************
Dr. Lawlor's modified ArUco marker detector:
  - Finds markers in the webcam image
  - Reconstructs the camera's location relative to the marker


ArUco example Copyright 2011 Rafael Muñoz Salinas. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are
permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice, this list of
      conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice, this list
      of conditions and the following disclaimer in the documentation and/or other materials
      provided with the distribution.

THIS SOFTWARE IS PROVIDED BY Rafael Muñoz Salinas ''AS IS'' AND ANY EXPRESS OR IMPLIED
WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Rafael Muñoz Salinas OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those of the
authors and should not be interpreted as representing official policies, either expressed
or implied, of Rafael Muñoz Salinas.
********************************************************************************************/
#include <iostream>

#include <opencv2/opencv.hpp>

#include "aruco/aruco.h"
#include "aruco/cvdrawingutils.h"

using namespace cv;
using namespace aruco;
using namespace std;


bool showGUI=true, useRefine=false;
Mat TheInputImage,TheInputImageCopy;


/* Extract location data from this valid, detected marker. 
   Does not modify the location for an invalid marker.
*/
void extract_location(const Marker &marker)
{
  // Extract 3x3 rotation matrix
  Mat Rot(3,3,CV_32FC1);
  Rodrigues(marker.Rvec, Rot); // euler angles to rotation matrix

  // Full 4x4 output matrix:
  Mat full(4,4,CV_32FC1);

  // Copy rotation 3x3
  for (int i=0; i<3; i++)
    for (int j=0; j<3; j++)
      full.at<float>(i,j)=Rot.at<float>(i,j);
  
  // Copy translation vector
  full.at<float>(0,3)=marker.Tvec.at<float>(0,0);
  full.at<float>(1,3)=marker.Tvec.at<float>(1,0);
  full.at<float>(2,3)=marker.Tvec.at<float>(2,0);

  // Final row is identity (nothing happening on W axis)
  full.at<float>(3,0)=0.0;
  full.at<float>(3,1)=0.0;
  full.at<float>(3,2)=0.0;
  full.at<float>(3,3)=1.0;

  // Invert, to convert marker-from-camera into camera-from-marker
  Mat back=full.inv();

  if (true) {
    // Dump transform matrix to screen, for debugging
    for (int i=0; i<4; i++) {
      for (int j=0; j<4; j++)
        printf("%5.2f  ",back.at<float>(i,j));
      printf("\n");
    }
  }
  
  double scale=14.5; // size of marker in physical world units (cm)
  double x_shift=0.0, y_shift=0.0, z_shift=0.0; // location of maker in world
  float x=back.at<float>(0,3)*scale+x_shift;
  float y=back.at<float>(1,3)*scale+y_shift;
  float z=back.at<float>(2,3)*scale+z_shift;
  float angle=180.0/M_PI*atan2(-back.at<float>(1,0),back.at<float>(0,0));

  // Print grep-friendly output
  printf("Marker %d: Camera %.1f %.1f %.1f cm, heading %.1f degrees\n",
         marker.id, x,y,z,angle
        );
}


int main(int argc,char **argv)
{
  try {
  string TheInputVideo;
  string TheIntrinsicFile;
  int camNo=0;
  float TheMarkerSize=-1;
  int ThePyrDownLevel=0;
  vector<Marker> TheMarkers;
  CameraParameters cam_param;
  float minSize=0.02; // fraction of frame, minimum size of rectangle
  int skipCount=1; // only process frames ==0 mod this
  int skipPhase=0;

  int wid=0, ht=0;
  const char *dictionary="TAG25h9";
  cv::VideoCapture *cap=0;
  for (int argi=1; argi<argc; argi++) {
    if (0==strcmp(argv[argi],"--nogui")) showGUI=false;
    else if (0==strcmp(argv[argi],"--cam")) cap=new cv::VideoCapture(atoi(argv[++argi]));
    else if (0==strcmp(argv[argi],"--file")) cap=new cv::VideoCapture(argv[++argi]);
    else if (0==strcmp(argv[argi],"--dict")) dictionary=argv[++argi];
    else if (0==strcmp(argv[argi],"--refine")) useRefine=true;
    else if (0==strcmp(argv[argi],"--sz")) sscanf(argv[++argi],"%dx%d",&wid,&ht);
    else if (0==strcmp(argv[argi],"--skip")) sscanf(argv[++argi],"%d",&skipCount);
    else if (0==strcmp(argv[argi],"--min")) sscanf(argv[++argi],"%f",&minSize);
    else printf("Unrecognized argument %s\n",argv[argi]);
  }
  // Initialize capturing live feed from the camera
  if (!cap) cap=new cv::VideoCapture(0);
  
  if (wid) cap->set(cv::CAP_PROP_FRAME_WIDTH, wid);
  if (ht)  cap->set(cv::CAP_PROP_FRAME_HEIGHT, ht);

  //check video is open
  if (!cap->isOpened()) {
    cerr<<"Could not open video"<<endl;
    return -1;
  }

  TheIntrinsicFile="camera.xml";

  //read first image to get the dimensions
  (*cap)>>TheInputImage;

  //read camera parameters if passed
  if (TheIntrinsicFile!="") {
    cam_param.readFromXMLFile(TheIntrinsicFile);
    cam_param.resize(TheInputImage.size());
  }
  //Configure other parameters
  aruco::MarkerDetector::Params params;
  
//  if (ThePyrDownLevel>0)
//    params.pyrDown(ThePyrDownLevel);
//  params.setCornerRefinementMethod(MarkerDetector::CORNER_SUBPIX); // more accurate
  params.setCornerRefinementMethod(aruco::CORNER_LINES); // more reliable?
  params.setDetectionMode(aruco::DM_FAST,0.1); // for distant/small markers (smaller values == smaller markers, but slower too)
  MarkerDetector MDetector(dictionary); // dictionary of tags recognized

  if (showGUI) {
    //Create gui
    cv::namedWindow("in",1);
  }

  unsigned int framecount=0;
  uint32_t vidcap_count=0;
  double tick2sec=1.0/getTickFrequency();
  double last_time=(double)getTickCount()*tick2sec;
  pair<double,double> AvrgTime(0,0) ;//determines the average time required for detection

  //capture until press ESC or until the end of the video
  while (cap->grab()) {
    if (!cap->retrieve( TheInputImage) || !cap->isOpened()) {
      std::cout<<"ERROR!  Camera "<<camNo<<" no longer connected!\n";
      std::cerr<<"ERROR!  Camera "<<camNo<<" no longer connected!\n";
      exit(1);
    }

    // Skip frames (do no processing, lets us stay live on fast cameras)
    skipPhase=(skipPhase+1)%skipCount;
    if (skipPhase!=0) continue;

    double start_time = (double)getTickCount()*tick2sec;//for checking the speed
    //Detection of markers in the image passed
    MDetector.detect(TheInputImage,TheMarkers,cam_param,1.0,true);
    
    //check the speed by calculating the mean speed of all iterations
    double end_time=(double)getTickCount()*tick2sec;
    AvrgTime.first+=end_time-start_time;
    AvrgTime.second++;
    if (end_time>2.0+last_time) {
      last_time=end_time;
      cout<<"Aruco time: "<<1000*AvrgTime.first/AvrgTime.second<<" milliseconds"<<endl;
    }
    
    for (unsigned int i=0; i<TheMarkers.size(); i++) {
      Marker &marker=TheMarkers[i];
      extract_location(marker);
    }
    

    bool vidcap=false;
    // if ((framecount++%32) == 0) vidcap=true;
    if (showGUI || vidcap) {
      //print marker info and draw the markers in image
      TheInputImage.copyTo(TheInputImageCopy);
      for (unsigned int i=0; i<TheMarkers.size(); i++) {
        Marker &marker=TheMarkers[i];
        // cout<<TheMarkers[i]<<endl;
        marker.draw(TheInputImageCopy,Scalar(0,0,255),1);

        //draw a 3d cube on each marker if there is 3d info
        if (  cam_param.isValid()) {
          CvDrawingUtils::draw3dCube(TheInputImageCopy,marker,cam_param,1,true);
          CvDrawingUtils::draw3dAxis(TheInputImageCopy,marker,cam_param);
        }

      }

      if (true) {
        //print other rectangles that contains invalid markers
        for (unsigned int i=0; i<MDetector.getCandidates().size(); i++) {
          aruco::Marker m( MDetector.getCandidates()[i],999);
          m.draw(TheInputImageCopy,cv::Scalar(255,0,0));
        }
      }
    }
    if (showGUI) {
      //show input with augmented information and  the thresholded image
      cv::imshow("in",TheInputImageCopy);
      // cv::imshow("thres",MDetector.getThresholdedImage());

      char key=cv::waitKey(1);//wait for key to be pressed
      if (key=='q' || key=='x' || key==0x13) exit(0);
    } /* end showGUI */
  } /* end frame loop */

  } catch (std::exception &ex) {
    cout<<"Vision/ArUco exception: "<<ex.what()<<endl;
  }

}

