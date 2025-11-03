/* 
OpenCV example: does optical tracking of camera motion.

Dr. Orion Lawlor, lawlor@alaska.edu, 2025-11-01 (Public Domain)
*/
#include <stdlib.h> /* for NULL */
#include <iostream>  
#include <opencv2/opencv.hpp>


// Represents an observed camera 2D translation and rotation
class camera_trot {
public:
    double dx; // shift in X (pixels)
    double dy; // shift in Y (pixels)
    double da; // shift in angle (radians CCW)
};
    
const cv::Point2d rcen = {320, 240}; // pixel center of rotation
const double rotscale = 1.0; // scale to make rotation in radians

// Return the tangent vector when a chip at this center point gets rotated
cv::Point2d tangent_rot(const cv::Point2d &center) {
    cv::Point2d rv = center-rcen; // radius vector
    cv::Point2d tv = {-rv.y,rv.x}; // tangent vector (CCW rotation)
    return tv;
}

// Return a prediction of the pixel offset at this point
cv::Point2d predict(const camera_trot &trot, const cv::Point2d &center)
{
    cv::Point2d tv = tangent_rot(center);
    cv::Point2d pred = { 
        trot.dx + trot.da*tv.x,
        trot.dy + trot.da*tv.y
    };
    return pred;
}

// Return the pixel distance between our fit and this center point and pixel shift
double fit_check(const camera_trot &trot, const cv::Point2d &center, const cv::Point2d &shift) 
{
    return cv::norm(shift - predict(trot,center));
}

// Do a least-squares fit of these camera shifts to a translation/rotation.
//  See also: https://math.stackexchange.com/questions/2136719/least-squares-fit-to-find-transform-between-points
camera_trot fit_trot(const std::vector<cv::Point2d> &centers,
    const std::vector<cv::Point2d> &shifts)
{
    int n = (int)centers.size(); // number of points
    int w = 3; // number of output parameters to fit: dx, dy, da
    cv::Mat A(cv::Size(w,2*n),CV_64F); // weights
    cv::Mat B(cv::Size(1,2*n),CV_64F); // targets

    for (int i=0;i<n;i++) {
        // Compute tangent vector for rotation at this chip center
        cv::Point2d tv = tangent_rot(centers[i]);
        
        // X axis matchup
        A.at<double>(2*i+0,0) = 1.0; // *dx -> x shift (+ pixels)
        A.at<double>(2*i+0,1) = 0.0; // *dy
        A.at<double>(2*i+0,2) = tv.x; // *da -> x rotate (*radians = pixels)
        
        B.at<double>(2*i+0,0) = shifts[i].x; // observed pixels

        // Same for Y        
        A.at<double>(2*i+1,0) = 0.0; // *dx 
        A.at<double>(2*i+1,1) = 1.0; // *dy -> y shift
        A.at<double>(2*i+1,2) = tv.y; // *da -> y rotate
        
        B.at<double>(2*i+1,0) = shifts[i].y;
    }
    
    cv::Mat out;
    cv::solve(A,B,out,cv::DECOMP_SVD);
    
    //std::cout<<"A "<<A<<" * out "<<out<<" = B "<<B<<"\n";
    //std::cout<<out<<"\n";
    camera_trot trot={out.at<double>(0,0), out.at<double>(0,1), out.at<double>(0,2)};
    return trot;
}




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
    std::vector<cv::Point2d> corr_shift(nm);
    std::vector<cv::Point2d> corr_center(nm);
    
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
            
            corr_shift.clear();
            corr_center.clear();
        
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
                cv::Point2d shift=cv::Point2d(maxLoc.x-shiftWH,maxLoc.y-shiftWH);
                cv::Point2d center=cv::Point2d(dx+refWH/2,dy+refWH/2);
                
                if (cv::norm(shift)<0.95*shiftWH) 
                { // plausible shift, save it
                    corr_shift.push_back(shift);
                    corr_center.push_back(center);
                }
                
                // Save debug match image
                if (show_debug>0) {
                    cv::cvtColor(match*255.0,matchRGB,cv::COLOR_GRAY2RGB);
                    matchRGB.copyTo(showdata(matchR+cv::Point(dx,dy)));
                    cv::line(showdata,center,center+shift,
                        cv::Scalar(0,0,255),2);
                }
            }
            camera_trot trot={0.0,0.0,0.0};
            
            printf("     Erasing: ");
            // Outlier removal loop, to fit to corr_shift and corr_center
            while (corr_shift.size()>12) {
                trot=fit_trot(corr_center,corr_shift);
                
                // remove the worst point and re-fit
                int worst_i=-1; double worst_err=2.0;
                for (int i=0;i<corr_center.size();i++) {
                    double err = fit_check(trot, corr_center[i], corr_shift[i]);
                    //printf(" err[%d]=%.1f ",i,err);
                    if (err>worst_err) {
                        worst_err=err;
                        worst_i=0;
                    }
                }
                printf("\nFit: dx %8.1f   dy %8.1f    da %8.3f    n %4d\n",
                        trot.dx, trot.dy, trot.da, (int)corr_center.size());
                if (worst_i<0) break;
                printf("  outlier[%d]=%.1f ",worst_i,worst_err);
                corr_center.erase(corr_center.begin()+worst_i);
                corr_shift.erase(corr_shift.begin()+worst_i);
            }
            printf("\nFinal Fit: dx %8.1f   dy %8.1f    da %8.3f    n %4d\n",
                    trot.dx, trot.dy, trot.da, (int)corr_center.size());
                
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


 

