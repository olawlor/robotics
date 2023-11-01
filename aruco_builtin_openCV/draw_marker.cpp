/*
 Create an aruco marker image.  This needs to be printed on a white background.
 
 Dr. Orion Lawlor, lawlor@alaska.edu, 2023-11-01 (Public Domain)
*/

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
 
int main() 
{
    int markerID = 17;
    int pixelSize = 800; 
    
    auto style = cv::aruco::DICT_APRILTAG_25h9;
    auto dictionary = cv::aruco::getPredefinedDictionary(style);
    
    cv::Mat img;
    // Generate the marker image
    cv::aruco::drawMarker(dictionary, markerID, pixelSize, img, 1);
    
    cv::imwrite("marker.png",img);
    
    return 0;
}


