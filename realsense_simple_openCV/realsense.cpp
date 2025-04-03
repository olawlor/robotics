/**
  Grab basic depth data from Intel RealSense camera.
  
  This needs librealsense2, from https://github.com/IntelRealSense/librealsense
  
  Dr. Orion Lawlor, lawlor@alaska.edu (public domain)
  Modified from tiny example by BjarneG at https://communities.intel.com/thread/121826
*/
#include <librealsense2/rs.hpp>  
#include <opencv2/opencv.hpp> 

  
using namespace std;  
using namespace cv;  

/// Make it easy to swap between float (fast for big arrays) and double
typedef float real_t;

/// Simple 3D vector
class vec3 {
public:
  real_t x;
  real_t y;
  real_t z;
  
  vec3(real_t X=0.0, real_t Y=0.0, real_t Z=0.0) {
    x=X; y=Y; z=Z;
  }
};

int main()  
{  
    rs2::pipeline pipe;  
    rs2::config cfg;  
  
    bool bigmode=true;

    int fps=6;
    fps=30; // USB 3.0 only
    int depth_w=1280, depth_h=720; // high res mode: definitely more detail visible
    int color_w=1280, color_h=720; 
    if (!bigmode) { // low res
      fps=15;
      depth_w=480; depth_h=270;
      color_w=424; color_h=240;
    }

    cfg.enable_stream(RS2_STREAM_DEPTH, depth_w,depth_h, RS2_FORMAT_Z16, fps);  
    cfg.enable_stream(RS2_STREAM_COLOR, color_w,color_h, RS2_FORMAT_BGR8, fps);  
  
    rs2::pipeline_profile selection = pipe.start(cfg);  

    auto sensor = selection.get_device().first<rs2::depth_sensor>();
    float scale =  sensor.get_depth_scale();
    printf("Depth scale: %.3f\n",scale);
    double depth2cm = scale * 100.0; 
    double depth2screen=255.0*scale/4.5;
    
    int framecount=0;
    int nextwrite=1;
    
    rs2::frameset frames;  
    while (true)  
    {  
        // Wait for a coherent pair of frames: depth and color  
        frames = pipe.wait_for_frames();  
        rs2::depth_frame depth_frame = frames.get_depth_frame();  
        rs2::video_frame color_frame = frames.get_color_frame();  
        framecount++;
  
        if ((depth_w != depth_frame.get_width()) ||
          (depth_h != depth_frame.get_height()) || 
          (color_w != color_frame.get_width()) ||
          (color_h != color_frame.get_height()))
        {
          std::cerr<<"Realsense capture size mismatch!\n";
          exit(1);
        }
  
        typedef unsigned short depth_t;
        depth_t *depth_data = (depth_t*)depth_frame.get_data();  
        void *color_data = (void*)color_frame.get_data();  
        
        // Make OpenCV versions of raw pixels:
        //Mat depth_raw(Size(depth_w, depth_h), CV_16U, depth_data, Mat::AUTO_STEP);  
        //Mat color(Size(color_w, color_h), CV_8UC3, color_data, Mat::AUTO_STEP);  
        
        // Display raw data onscreen
        //imshow("Depth", depth_raw);
        //imshow("RGB", color);
        
        Mat debug_image(Size(depth_w, depth_h), CV_8UC3, cv::Scalar(0));
        
        for (int y = 0; y < depth_h; y++)
        for (int x = 0; x < depth_w; x++)
        {
          int i=y*depth_w + x;
          float depth=depth_data[i]*depth2cm; // depth, in cm
          
          // Calculate pixel look direction
          float direction_from_pixel = 1.1 / depth_w; // radians of field of view per pixel
          float dirX = (x-depth_w/2)*direction_from_pixel;
          float dirY = (y-depth_h/2)*direction_from_pixel;
          
          // Camera coordinates 3D location
          float camX = dirX * depth;
          float camY = dirY * depth;
          float camZ = depth;
          
          // Round to grid
          float gridSize = 30.0; // 1 foot grid
          float gridX = fmod(camX,gridSize);
          float gridY = fmod(camY,gridSize);
          float gridZ = fmod(camZ,gridSize);
          
          // Convert to an output color
          const float colorScale = 255.0/gridSize;
          cv::Vec3b debug_color = {  // <- OpenCV BGR color order
            (unsigned char)(colorScale*gridZ), 
            (unsigned char)(colorScale*gridY), 
            (unsigned char)(colorScale*gridX)
          };
          
          debug_image.at<cv::Vec3b>(y,x)=debug_color;
        }   
        imshow("Depth image",debug_image);
        
        
        int k = waitKey(10);  
        if (k == 27 || k=='q')  
            break;  
    }  
  
  
    return 0;  
}  
