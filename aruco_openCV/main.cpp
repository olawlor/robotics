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


// Store info about how this marker is scaled, positioned and oriented
struct marker_info_t {
	int id; // marker's ID, from 0-1023 
	float true_size; // side length, in meters, of black part of pattern
	
	float x_shift; // translation from origin, in meters, of center of pattern
	float y_shift; 
	float z_shift; 
	
	float rotate2D; // rotation around pattern's center, in degrees (0 for upright, 90 for rotated on side)
	float rotate3D; // rotation about pattern's X axis, in degrees
};

const static marker_info_t marker_info[]={
	{-1,0.5}, // fallback default case
	
/* Simple demo markers, scattered along X axis */
	{0, 0.5, 1.0,0.0,0.0,   0,90 },
	{1, 0.5, 7.0,0.0,0.0,   0,90 },
	{2, 0.5, 10.0,0.0,0.0,  0,90 },
};

// Look up the calibration parameters for this marker
const marker_info_t &get_marker_info(int id) {
	for (int i=1;i<sizeof(marker_info)/sizeof(marker_info_t);i++) {
		if (marker_info[i].id==id) 
			return marker_info[i];
	}
	return marker_info[0];
}



bool showGUI=true, useRefine=false;
Mat TheInputImage,TheInputImageCopy;


/**
  Convert 3D position to top-down 2D onscreen location
*/
cv::Point2f to_2D(const Marker &m,float x=0.0,int xAxis=2,int yAxis=0)
{
	// Extract 3x3 rotation matrix
	Mat Rot(3,3,CV_32FC1);
	Rodrigues(m.Rvec, Rot); // euler angles to rotation matrix

	cv::Point2f ret;
	const marker_info_t &mi=get_marker_info(m.id);
	float scale=mi.true_size*70; // world meters to screen pixels
	ret.x=scale*(m.Tvec.at<float>(xAxis,0)+x*Rot.at<float>(xAxis,xAxis));
	ret.y=scale*(m.Tvec.at<float>(yAxis,0)+x*Rot.at<float>(yAxis,xAxis));
	
	printf("Screen point: %.2f, %.2f cm\n",ret.x,ret.y);
	ret.y+=240; // approximately centered in Y
	return ret;
}

/**
 Draw top-down image of reconstructed location of marker.
*/
void draw_marker_gui_2D(Mat &img,Scalar color,const Marker &m)
{
	int lineWidth=2;
	
	cv::line(img,
		to_2D(m,-0.5),to_2D(m,0.0),
		color,lineWidth,CV_AA);
	cv::line(img,
		to_2D(m,0.0),to_2D(m,+0.5),
		Scalar(255,255,255)-color,lineWidth,CV_AA);
}



/* Extract location data from this valid, detected marker. 
   Does not modify the location for an invalid marker.
*/
void extract_location(const Marker &marker)
{
	const marker_info_t &mi=get_marker_info(marker.id);

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
	
	if (mi.rotate2D==90) {
		for (int i=0; i<3; i++) {
			std::swap(full.at<float>(i,0),full.at<float>(i,2)); // swap X and Z
			full.at<float>(i,0)*=-1; // invert (new) X
		}
	}
	if (mi.rotate3D==90) {
		for (int i=0; i<3; i++) {
			std::swap(full.at<float>(i,1),full.at<float>(i,2)); // swap Y and Z
			//full.at<float>(i,1)*=-1; // invert (new) Y
			full.at<float>(i,0)*=-1; // invert (new) X
		}
	}


	// Invert, to convert marker-from-camera into camera-from-marker
	Mat back=full.inv();

  if (false) {
	  // Dump transform matrix to screen, for debugging
	  for (int i=0; i<4; i++) {
		  for (int j=0; j<4; j++)
			  printf("%.2f	",back.at<float>(i,j));
		  printf("\n");
	  }
  }
	
	double scale=mi.true_size;
  float x=back.at<float>(0,3)*scale+mi.x_shift;
	float y=back.at<float>(1,3)*scale+mi.y_shift;
	float z=back.at<float>(2,3)*scale+mi.z_shift;
	float angle=180.0/M_PI*atan2(back.at<float>(1,0),-back.at<float>(0,0));

	// Print grep-friendly output
	printf("Marker %d: Camera %.3f %.3f %.3f meters, heading %.1f degrees\n",
	       marker.id, x,y,z,angle
	      );
}


int main(int argc,char **argv)
{
	try {
	string TheInputVideo;
	string TheIntrinsicFile;
	int camNo=1;
	float TheMarkerSize=-1;
	int ThePyrDownLevel=0;
	VideoCapture vidcap;
	vector<Marker> TheMarkers;
	CameraParameters cam_param;
	float minSize=0.02; // fraction of frame, minimum size of rectangle
	int skipCount=1; // only process frames ==0 mod this
	int skipPhase=0;

	int wid=640, ht=480;
	const char *dictionary="ARUCO_MIP_36h12";
	for (int argi=1; argi<argc; argi++) {
		if (0==strcmp(argv[argi],"--nogui")) showGUI=false;
		else if (0==strcmp(argv[argi],"--cam")) camNo=atoi(argv[++argi]);
		else if (0==strcmp(argv[argi],"--dict")) dictionary=argv[++argi];
		else if (0==strcmp(argv[argi],"--refine")) useRefine=true;
		else if (0==strcmp(argv[argi],"--sz")) sscanf(argv[++argi],"%dx%d",&wid,&ht);
		else if (0==strcmp(argv[argi],"--skip")) sscanf(argv[++argi],"%d",&skipCount);
		else if (0==strcmp(argv[argi],"--min")) sscanf(argv[++argi],"%f",&minSize);
		else printf("Unrecognized argument %s\n",argv[argi]);
	}

	//read from camera
	vidcap.open(camNo);
	
	vidcap.set(CV_CAP_PROP_FRAME_WIDTH, wid);
	vidcap.set(CV_CAP_PROP_FRAME_HEIGHT, ht);

	//check video is open
	if (!vidcap.isOpened()) {
		cerr<<"Could not open video"<<endl;
		return -1;
	}

	TheIntrinsicFile="camera.yml";

	//read first image to get the dimensions
	vidcap>>TheInputImage;

	//read camera parameters if passed
	if (TheIntrinsicFile!="") {
		cam_param.readFromXMLFile(TheIntrinsicFile);
		cam_param.resize(TheInputImage.size());
	}
	//Configure other parameters
	aruco::MarkerDetector::Params params;
	
//	if (ThePyrDownLevel>0)
//		params.pyrDown(ThePyrDownLevel);
//	params.setCornerRefinementMethod(MarkerDetector::CORNER_SUBPIX); // more accurate
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
	while (vidcap.grab()) {
		if (!vidcap.retrieve( TheInputImage) || !vidcap.isOpened()) {
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

