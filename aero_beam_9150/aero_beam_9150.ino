/**
 Aero Beam motion controller using an MPU-9150 accelerometer:
   - Arduino pin 2 connected to electronic speed controller (ESC) for fan motor.
   - Arduino SDA/SCL connected to MPU-9150 I2C lines.
   - MPU-9150 power and ground connected.
 
 Dr. Orion Lawlor, lawlor@alaska.edu, 2015-11-08 (Public Domain)
*/
#include <Wire.h>
#include <Servo.h>
#include "mpu9150.h"

Servo esc; // hooked to 3-phase brushless fan motor

void setup()
{      
  // Initialize the Serial Bus for printing data.
  Serial.begin(57600);
  Serial.println("Connecting to MPU-9150...");
  esc.attach(2);
  esc.writeMicroseconds(1000);

  // Initialize the 'Wire' class for the I2C-bus.
  Wire.begin();

  MPU9150_readings::setup();

  Serial.println("Connected!  Enter commands now.");
}

float p_gain=1.0; // proportional to angle error
float d_gain=0.13; // derivative of angle (gyro rate)
float i_gain=0.5; // integral of error
float center_term=1230; // enough microseconds to cancel gravity

int run=0; // run on physical ('r' key) or virtual only (spacebar)
long verbose=0, XYZdata=0;
long tot_err=0, tot_samp=0;
long history=0;
long experiment_start=0;
int manual=0; // manual motor control


void printgains() {
  Serial.print("p_gain = "); Serial.print(p_gain);
  Serial.print("\td_gain = "); Serial.print(d_gain);
  Serial.print("\ti_gain = "); Serial.print(i_gain);
  Serial.print("\tcenter = "); Serial.print(center_term);
  Serial.println();
}

void user_interface(char c) {
    if (c=='p') { // adjust p gain
      p_gain=Serial.parseFloat();
      printgains();
    }
    else if (c=='d') { // adjust d gain
      d_gain=Serial.parseFloat();
      printgains();
    }
    else if (c=='i') { // adjust i gain
      i_gain=Serial.parseFloat();
      printgains();
    }
    else if (c=='c') { // center term
      center_term=Serial.parseFloat();
      printgains();
    }
    else if (c=='m') { // manual control
      manual=Serial.parseInt();
      Serial.print("New manual power: "); Serial.println(manual);
    }
    
    else if (c=='x') { // show raw xyz data for a short time
      XYZdata=100;
    }
    else if (c=='X') { // show raw xyz data for a long time
      XYZdata=1000;
    }
    
    else if (c=='v') { // verbose--run short experiment
      experiment_start=millis(); tot_err=0; tot_samp=0;
      verbose=100;
    }
    else if (c=='V') { // verbose--run long experiment
      experiment_start=millis(); tot_err=0; tot_samp=0;
      verbose=800;
    }
    else if (c=='r') { // run experiment
      Serial.println("RUNNING REAL MOTOR");
      run=1;
      manual=0;
      printgains();
    }
    else if (c==' ') { // stop experiment
      Serial.println("STOPPING REAL MOTOR");
      run=0;
      verbose=0;
      manual=0;
      printgains();
    }
    else if (c=='\n' || c=='\r') { // ignore it
      
    }
    else {
      Serial.print("Unknown command ");
      Serial.println(c);
    }
}


// Main overall control loop:
void loop()
{
  // Check user interface
  if (Serial.available()) user_interface(Serial.read());

  // Pull sensor data:
  MPU9150_readings imu=MPU9150_readings::read();

  // Compute PID terms:
  int err=imu.A[1]; // accelerometer Y axis == error
  int rate=imu.G[0]+49; // gyro X axis == rate (plus drift fix)
  
  history+=err;
  long wind=2000; // maximum "wind-up": limit history to avoid oscilations
  if (history>wind) history=wind;
  if (history<-wind) history=-wind;

  // Sum PID terms and gains
  int cmd=center_term-(0.01*p_gain)*err+d_gain*rate-(0.01*i_gain)*history;

  // Limit resulting command
  int cmd_min=1000, cmd_max=1500;
  if (cmd<cmd_min) cmd=cmd_min;
  if (cmd>cmd_max) cmd=cmd_max;

  // Send value to motor:
  if (manual) esc.writeMicroseconds(manual);
  else if (verbose && run) esc.writeMicroseconds(cmd);
  else  esc.writeMicroseconds(1000);

  // Show data onscreen:
  if (verbose) {
    verbose--;
    Serial.print("TCERH:\t");
    Serial.print(millis()-experiment_start);
    Serial.print("\t");
    Serial.print(cmd);
    Serial.print("\t");
    Serial.print(err);
    Serial.print("\t");
    Serial.print(rate);
    Serial.print("\t");
    Serial.print(history);
    Serial.println();

    if (err>0) tot_err+=err;
    else tot_err-=err;
    tot_samp++;
    if (verbose==0) {
      Serial.print("Average error for run: ");
      Serial.println(tot_err*1.0/tot_samp);
      printgains();
    }
  }

  // Show XYZ data onscreen
  if (XYZdata) {
    XYZdata--;
    Serial.print("A ");
    for (int i=0;i<3;i++) { Serial.print(imu.A[i]); Serial.print(" "); }
    
    Serial.print("  G  ");
    for (int i=0;i<3;i++) { Serial.print(imu.G[i]); Serial.print(" "); }
    
    Serial.print("  C  ");
    for (int i=0;i<3;i++) { Serial.print(imu.C[i]); Serial.print(" "); }
    
    Serial.print("  T  ");
    Serial.print(imu.T);
    Serial.println();
  }

  // Limit control rate to 100Hz (max)
  delay(10);
}


