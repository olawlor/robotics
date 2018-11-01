/* 
Extend image gradients into lines, which detects circles or ellipses,
and especially bullseyes.  The trick is that the gradient of a circle's
edges always points toward the center of the circle.

Also estimates the bullseye's radius and orientation, both using the 
center of mass of the black pixels relative to the gradient-accumulated
center point.  The orientation estimate is only good to +-5 degrees at best.
  
  Orion Sky Lawlor, lawlor@alaska.edu, 2013-10-23 (Public Domain)
*/
#include <stdlib.h> /* for NULL */
#include <iostream>  

#include <opencv2/opencv.hpp>


/* The image gradient needs to exceed this value to draw a line. 
   Low values allow low-contrast images to work, but are slower
   due to the many spurious lines generated.
*/
int minDiff=60;

/* Show features with this many cumulative gradients.
   Low values detect bullseyes easily, but may lead to false positives.
*/
int showScale=90;

int searchRadius=20;

/*
  Pixels less than this fraction of the mean brightness are considered black.
  Higher values are more forgiving of low-contrast imagery.
*/
float mean_threshold=0.9;

/* This is the fraction of the bulls-eye image that is black.
   It's used to estimate the detected radius.
 */
float coverage=0.25; 

/* Return the current wall clock time, in seconds */
double time_in_seconds(void) {
  static double first=-1;
  double cur=cv::getTickCount()/cv::getTickFrequency();
  if (first<0) first=cur;
        return cur-first;
}

typedef unsigned short accum_t;
accum_t fetchAccum(const cv::Mat &accum,int x,int y) {
  return ((const accum_t *)accum.data)[y*accum.cols+x];
}

/* Increment pixels along this line. */
void accumulateLine(cv::Mat &accum,
  cv::Point S,cv::Point E)
{
  accum_t *accumDat=(accum_t *)accum.data;
  cv::Rect r(2,2,accum.cols-4,accum.rows-4);
  if (!cv::clipLine(r,S,E)) return;
  
  float rounding=0.49999; // compensates for rounding down
  
  int dx=E.x-S.x;
  int dy=E.y-S.y;
  if (abs(dx)>abs(dy)) 
  { /* X-major line */
    if (E.x<S.x) std::swap(S,E);
    float m=(E.y-S.y)/float(E.x-S.x);
    float b=S.y-m*S.x+rounding;
    for (int x=S.x;x<=E.x;x++)
    {
      float y=m*x+b;
      //if (y<0 || y>=accum.rows) abort();
      accumDat[((int)y)*accum.cols+x]+=1.0;
    }
  }
  else  /* dx<=dy */
  { /* Y-major line */
    if (E.y==S.y) return; // start and end are equal
    
    if (E.y<S.y) std::swap(S,E);
    float m=(E.x-S.x)/float(E.y-S.y);
    float b=S.x-m*S.y+rounding;
    for (int y=S.y;y<=E.y;y++)
    {
      float x=m*y+b;
      //if (x<0 || x>=accum.cols) abort();
      accumDat[y*accum.cols+(int)x]+=1.0;
    }
  }
}

/* Estimate properties of this bulls-eye, based on a rough center location */
class bull_props {
public:
  float cx,cy; // center pixel
  float radius; // apparent radius, in pixels
  float orientx,orienty; // screen-coordinates orientation vector
  float angle; // screen-coordinates orientation, in radians
  cv::Scalar color; // color

  bull_props(cv::Mat &accum,cv::Mat &annot,cv::Mat &rgb,cv::Mat &gray,
    int x_,int y_,int searchRadius);
};
bull_props::bull_props(cv::Mat &accum,cv::Mat &annot,cv::Mat &rgb,cv::Mat &gray,
    int x_,int y_,int searchRadius)
{
  cx=x_; cy=y_;
  
  float C=fetchAccum(accum,x_,y_); // 5 point stencil here
  float L=fetchAccum(accum,x_-1,y_), R=fetchAccum(accum,x_+1,y_);
  float B=fetchAccum(accum,x_,y_-1), T=fetchAccum(accum,x_,y_+1);

  cx+=-(R-L)/(2.0*(R+L-2.0*C)); // parabolic peak-polishing
  cy+=-(T-B)/(2.0*(T+B-2.0*C)); 
  
/*
  Two passes: first pass is with initial default search radius.
  Second pass is with the refined radius estimated during the first pass.
*/
  for (int pass=0;pass<=0;pass++) 
  {
    /*
      Prepare a subset image marking the black pixels.
    */
    cv::Rect search(x_-searchRadius,y_-searchRadius,2*searchRadius,2*searchRadius);
    search &= cv::Rect(0,0,rgb.cols,rgb.rows); // intersect with image pixels
  
    cv::Mat thresh=gray(search).clone();
    cv::Scalar mean=cv::mean(thresh);
  
    for (int y=0;y<thresh.rows;y++)
    for (int x=0;x<thresh.cols;x++)
    {
      /* Binary threshold this image */
      unsigned char *data=(unsigned char *)thresh.data;
      int i=y*thresh.cols+x;
      cv::Point center_relative=cv::Point(x,y)+cv::Point(search.x,search.y)-cv::Point(x_,y_);
      bool in_circle=cv::norm(center_relative)<searchRadius;
      if (in_circle &&  data[i]<mean_threshold*mean[0]) 
      { // mark known-black pixels
        data[i]=1;
        ((accum_t *)annot.data)[(y+search.y)*annot.cols+(x+search.x)]=0x7777; // mark in annotated image too
      }
      else 
      {
        data[i]=0;
      }
    }
  
    cv::Moments mo=cv::moments(thresh);
  
    // estimate onscreen size from number of matching pixels
    radius=sqrt(mo.m00/coverage)/2.0; // <- side length squared proportional to area
  
    // estimate orientation from bias in center of mass
    //   (FIXME: this is quite inaccurate in the presence of noise)
    orientx=(mo.m10/mo.m00-(cx-search.x)); 
    orienty=(mo.m01/mo.m00-(cy-search.y));
  
    if (radius<4.0) { // we didn't find it.  Look wider.
      searchRadius=2*searchRadius;
    } else { // we found it OK--refine
      searchRadius=(int)1.2*radius; // update search radius for second pass
    }
  }
  
  float inscale=radius/sqrt(orientx*orientx+orienty*orienty);
  cv::line(annot,
    cv::Point(cx+ inscale*orientx, cy+ inscale*orienty),
    cv::Point(cx+2*radius*orientx, cy+2*radius*orienty),
    cv::Scalar(50000),
    3);
  
  angle=atan2(orienty,orientx);
}


int main(int argc,char *argv[])
{  

  cv::VideoCapture *cap=0;
  long framecount=0;
  double time_start=-1.0;
  
  int argi=1; // command line argument index
  while (argc>=argi+2) { /* keyword-value pairs */
    char *key=argv[argi++];
    char *value=argv[argi++];
    if (0==strcmp(key,"--cam")) cap=new cv::VideoCapture(atoi(value));
    else if (0==strcmp(key,"--file")) cap=new cv::VideoCapture(value);
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
  cv::createTrackbar("minimum gradient","video",&minDiff,255,NULL);
  cv::createTrackbar("output scale","video",&showScale,2000,NULL);
  cv::createTrackbar("search radius","video",&searchRadius,200,NULL);

  cv::Mat frame,gray,gradX,gradY;
  cv::Mat accum;
  
  double t_last=0.0; // last frame time
  double t_last_print=1.0; // last framerate printout time
  
  // An infinite loop
  while(true)
  {
    double t_cur=time_in_seconds();
    double dt=t_cur-t_last;
    t_last=t_cur;
    if (t_cur>t_last_print+1.0) {
      std::cout<<"Framerate: "<<1.0/dt<<" fps\n";
      t_last_print=t_cur;
    }
  
    (*cap)>>frame; // grab next frame from camera

    // If we couldn't grab a frame... quit
    if(frame.empty())
      break;
    
    framecount++;
    
  /* Convert steep gradients to lines */
    // Accumulator for gradient power
    accum=cv::Mat::zeros(frame.rows,frame.cols,CV_16U);
    
    // Grayscale
    cv::cvtColor(frame,gray,CV_BGR2GRAY);
    // Gradient estimate (with filtering)
    int ksize=3;
    cv::Sobel(gray,gradX,CV_32F, 1,0, ksize);
    cv::Sobel(gray,gradY,CV_32F, 0,1, ksize);
    
    float *gradXF=(float *)gradX.data;
    float *gradYF=(float *)gradY.data;
    float minDiffSq=minDiff*minDiff;
    for (int y=0;y<frame.rows;y++)
    for (int x=0;x<frame.cols;x++)
    {
      int i=y*frame.cols+x;
      float dx=gradXF[i], dy=gradYF[i];
      float mag=dx*dx+dy*dy;
      if (mag>minDiffSq) {
        mag=sqrt(mag); // now a length
        float s=20.1/mag; // scale factor from gradient to line length
        accumulateLine(accum,
          cv::Point(x+dx*s,y+dy*s),
          cv::Point(x-dx*s,y-dy*s));
        
        /* // This doesn't support alpha blending (WHY NOT?!)
        cv::line(annot,
          cv::Point(x+dx*s,y+dy*s),
          cv::Point(x-dx*s,y-dy*s),
          cv::Scalar(255,0,0,10),0.1,CV_AA);
        */
      }
    }
    
  // Circle areas where there's a high gradient *and* a local maximum.
    if (showScale<=0) showScale=1;
    cv::Mat annot=(65535/showScale)*accum.clone(); // annotated version (for display)
    int bullcount=0;
    
    const accum_t *readAccum=(const accum_t *)accum.data;
    int de=10; // must be maximum among neighborhood of this size
    for (int y=de;y<accum.rows-de;y++)
    for (int x=de;x<accum.cols-de;x++)
    {
      int i=y*accum.cols+x;
      int cur=readAccum[i];
      if (cur>=showScale) 
      { /* it's big--but is there a bigger one nearby? */
        bool biggest=true;
        for (int dy=-de;dy<de && biggest;dy++)
        for (int dx=-de;dx<de;dx++)
        {
          float her=readAccum[(y+dy)*accum.cols+(x+dx)];
          /* To break ties, I'm putting a slight tilt along both axes. */
          her+=dx*(1.0/1057)+dy*(1.0/8197);
          
          if (cur<her) {
            biggest=false;
            break;
          }
        }
        
        if (biggest) 
        { /* This is it!  Draw a circle around it. */
          bull_props props(accum,annot,frame,gray,
            x,y,searchRadius);
          
          if (bullcount==0) std::cout<<"\n";
          printf("bullseye  %d  %.2f  %.2f   %.3f  %.1f   %.3f  # XY rA t\n",
            bullcount++, props.cx, props.cy,  props.radius, props.angle*180.0/M_PI,
            t_cur);
          
          // Draw onscreen
          if (1) cv::circle(annot,
            cv::Point(x,y),
            1.2*props.radius,
            cv::Scalar(65535),
            1,
            CV_AA);
        }
      }
    }
    
  // Dump accumulator onscreen (debugging)
    cv::imshow("video",annot);
    // Wait for a keypress (for up to 1ms)
    int c = cv::waitKey(1);
    if(c=='q')
    {
      break;
    }
  }

  return 0;
} 


 

