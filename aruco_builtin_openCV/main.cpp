/* 
 Simple OpenCV example of calling the built-in Aruco library.
 
 See https://docs.opencv.org/4.0.0/d5/dae/tutorial_aruco_detection.html
 https://docs.opencv.org/4.0.0/d9/d6a/group__aruco.html
 https://learnopencv.com/augmented-reality-using-aruco-detecteds-in-opencv-c-python/

 Dr. Orion Lawlor, lawlor@alaska.edu, 2023-11-01 (Public Domain)
*/
#include <stdlib.h> /* for NULL */
#include <iostream>  
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>

int main(int argc,char *argv[])
{
    auto style = cv::aruco::DICT_APRILTAG_25h9;
    auto dictionary = cv::aruco::getPredefinedDictionary(style);   
    float realSize = 0.305; // real-world units size of markers

    int camera_number=0;
    cv::VideoCapture cap(camera_number);
    if (!cap.isOpened()) {
        std::cerr<<"Cannot open video stream "<<camera_number<<"\n";
        return 1;
    } 

    int wid=1280;
    int ht=720;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, wid);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, ht);

    cv::Mat frame(cv::Size(wid, ht), CV_8UC3, cv::Scalar(0,0,255));
    
    auto parameters = cv::aruco::DetectorParameters::create();
    
    // Read camera calibration from camera.yml file
    cv::Mat cameraMatrix, distCoeff; 
    cv::FileStorage cameraFile;
    cameraFile.open("camera.yml",cv::FileStorage::READ);
    cameraFile["camera_matrix"] >> cameraMatrix;
    cameraFile["distortion_coefficients"] >> distCoeff;
    std::cout<<"Read cameraMatrix "<<cameraMatrix<<" and distCoeff "<<distCoeff<<std::endl;
                 
    // Stores the corners of the detecteds, and any rejected rectangles
    std::vector<std::vector<cv::Point2f> > detectedCorners, rejectedCandidates;
     
    // ID numbers for any detected markers
    std::vector<int> detectedIds;

    std::vector<cv::Vec3d> rvec;  // rotation vectors
    std::vector<cv::Vec3d> tvec;  // translation vectors

    while(1) {
        cap >> frame;
        if (frame.empty()) break;
        
        // Detect markers in the image
        cv::aruco::detectMarkers(frame, dictionary, 
            detectedCorners, detectedIds, parameters, rejectedCandidates, 
            cameraMatrix,distCoeff);
    
        // Show the detected markers
        if (detectedIds.size()>0) {
            std::cout<<" Detected "<<detectedIds.size()<<" markers:"<<"\n";
            
            cv::aruco::drawDetectedMarkers(frame, detectedCorners, detectedIds);
            
            cv::aruco::estimatePoseSingleMarkers(detectedCorners,realSize,
                cameraMatrix,distCoeff,
                rvec, tvec);
            
            for (size_t i=0;i<detectedIds.size();i++) {
                std::cout<<"    markerID "<<detectedIds[i]<<": "
                    <<"  origin "<<tvec[i]
                    <<"  rot "<<rvec[i]
                    <<std::endl;
                cv::drawFrameAxes(frame, cameraMatrix, distCoeff, rvec[i], tvec[i], 0.05);
            }
        }

        cv::imshow("Video",frame);

        int c = cv::waitKey(1);
        if (c==27 || c=='x') break; // escape / exit key
    }

    return 0;
} 


 

