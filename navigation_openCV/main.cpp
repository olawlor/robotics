/* 
2D Robot sensor data and navigation display

Dr. Orion Lawlor, lawlor@alaska.edu, 2020-04-07 (Public Domain)
*/
#include <stdlib.h> /* for NULL */
#include <iostream>  
#include <opencv2/opencv.hpp>

#include "osl/robot_world.h"

/// Shoot a laser at this robot-relative angle in degrees (counterclockwise).
/// Return the distance to the first obstacle the laser hits, out to a maximum of 3.0 meters.
double shoot_lidar(const mobile_robot &robot,double angle_degrees=0);

sinwave_obstacles obs(vec3(1.1,0.5,1));

const float maxlidar = 3.0;
double shoot_lidar(const mobile_robot &robot,double angle_degrees)
{
  float pixelstep = 0.01; // meter step size along beam
  for (float lidar=0.0; lidar<=maxlidar; lidar+=pixelstep)
  {
    vec3 p=robot.get_forward(lidar,0.0,angle_degrees);
    if (obs.is_obstacle(p)) // hit an obstacle at this location
    {
      return lidar;
    }
  }
  return maxlidar;
}


// Meters to pixels
cv::Point makePoint(vec3 world)
{
    float scale = 1024/5.0;
    return cv::Point(
        10 + scale * world.x,
        1000 + scale * (-world.y) // flip Y axis from OpenCV's down to Y+ up
    );
}

int main(int argc,char *argv[])
{
  // Stores our navigation data
  cv::Mat img(1024,1024, CV_8UC3, cv::Scalar(0));

  mobile_robot robot;
  
  // An infinite loop
  while(true)
  {
    // Draw lidar fan
    for (float angle=-90;angle<=+90;angle+=5)
    {
        double dist = shoot_lidar(robot,angle);
        
        // Draw lidar results
        cv::Point end = makePoint(robot.get_forward(dist,0,angle));
        cv::line(img, 
            makePoint(robot.get_forward(0)),
            end,
            cv::Scalar(0,0,255), // red (BGR)
            1);
        
        // Draw white dot at end (if it hit something)
        if (dist<maxlidar)
            cv::line(img, 
                end,
                end,
                cv::Scalar(255,255,255), // white
                4);
        
    }
  
    // Draw robot's position
    cv::line(img, 
        makePoint(robot.get_forward(0)),
        makePoint(robot.get_forward(0.2)),
        cv::Scalar(0,255,0), // green (BGR)
        2);
    
    // Show our current image
    cv::imshow("img",img);
    
    // Wait for a keypress (for up to 10ms)
    int c = cv::waitKey(10);
    if(c=='q') break;
    
    // Keyboard navigation
    if (c=='w') robot.forward(0.1);
    if (c=='s') robot.forward(-0.1);
    if (c=='a') robot.left(5);
    if (c=='d') robot.left(-5);
    
    printf("Robot at %f, %f   angle %f\n", robot.pos.x, robot.pos.y, robot.angle);
    
  }

  return 0;
} 


 

