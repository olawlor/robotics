/* 
Simple OpenCV example: does optical tracking of camera motion.

Dr. Orion Lawlor, lawlor@alaska.edu, 2025-11-01 (Public Domain)
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
    
    int vw=640, vh=480; // video width and height
    int stepX = 102, stepY = 85; // pixel chip step size for search
    cap->set(cv::CAP_PROP_FRAME_WIDTH, vw);
    cap->set(cv::CAP_PROP_FRAME_HEIGHT, vh);

    // Predeclaring all the images reduces reallocation costs
    cv::Mat frame,lastframe, refI, curI, match;

    int show_debug=1; // if >0, imshow the debug data
    // Display parts of the dataset here
    cv::Mat showdata;
    (*cap)>>showdata; //<- sets a background at least

    (*cap)>>lastframe; // need to boot up with a frame
  
  
    const int refWH = 128; // size of reference chip
    const int curWH = 64; // size of cur (search) chip
    const int shiftWH = (refWH-curWH)/2; // gap between images
    int mw=refWH-curWH+1, mh=refWH-curWH+1; // size of match output area
    cv::Rect matchR(shiftWH,shiftWH,mw,mh);
    cv::Mat matchRGB(mw, mh, CV_8UC3);
    
    // Number of match points on each axis
    int nx = (vw-refWH)/stepX + 1;
    int ny = (vh-refWH)/stepY + 1;
    int nm = nx*ny;
    std::vector<cv::Point> corr_shift(nm);
    std::vector<cv::Point> corr_center(nm);
    
    // An infinite loop
    while(true)
    {
        (*cap)>>frame; // grab next frame from camera

        // If we couldn't grab a frame... quit
        if(frame.empty()) break;
        cv::imwrite("capture.jpg",frame);
        
        if (!lastframe.empty())
        {
            if (show_debug>0) frame.copyTo(showdata);
        
            // Step across image, finding correlations
            for (int ix=0;ix<nx;ix++)
            for (int iy=0;iy<ny;iy++)
            {
                // Top left corner of search area is (dx,dy)
                int dx = ix*stepX; // for (int dx=0;dx<vw-refWH;dx+=stepX)
                int dy = iy*stepY; // for (int dy=0;dy<vh-refWH;dy+=stepY)

                // Extract chunk of reference image (this frame)
                cv::Rect refR(dx,dy,refWH,refWH);
                refI=frame(refR);
                
                // Extract smaller chunk of last frame
                cv::Rect curR(dx+shiftWH,dy+shiftWH,curWH,curWH);
                curI=lastframe(curR);
                
                // Match them
                cv::matchTemplate(refI,curI,match,cv::TM_CCOEFF_NORMED);
                
                // Find the peak
                cv::Point maxLoc; double maxVal=0;
                cv::minMaxLoc(match,0,&maxVal,0,&maxLoc);
                
                // Save that correlation peak
                cv::Point shift=maxLoc-cv::Point(shiftWH,shiftWH);
                cv::Point center=cv::Point(dx+refWH/2,dy+refWH/2);
                int ic=ix+iy*nx; // raster index
                corr_shift[ic]=shift;
                corr_center[ic]=center;
                
                // Save debug match image
                if (show_debug>0) {
                    cv::cvtColor(match*255.0,matchRGB,cv::COLOR_GRAY2RGB);
                    matchRGB.copyTo(showdata(matchR+cv::Point(dx,dy)));
                    cv::line(showdata,center,center+shift,
                        cv::Scalar(0,0,255),2);
                }
            }
            
            // FIXME: outlier removal fit to corr_shift and corr_center
            
            if (show_debug>0) cv::imshow("Correlations",showdata);
        }
        
        // Shift the lastframe into the current frame
        std::swap(lastframe,frame);
        
        // Wait for a keypress (for up to 1ms)
        int c = cv::waitKey(1);
        if(c=='q') break;
    }

    return 0;
} 


 

