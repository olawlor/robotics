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

const int gyro_vertigo = +49; // <- TUNE THIS for your IMU's rotation rate

Servo esc; // hooked to 3-phase brushless fan motor

const int esc_off=1000; // minimum pulse (arms ESC)
const int esc_center=1200; // default center term (cancels gravity)
const int esc_cap=1350; // maximum pulse to send

void setup()
{      
  // Initialize the Serial Bus for printing data.
  Serial.begin(115200);
  Serial.println("Connecting to MPU-9150...");
  esc.attach(2);
  esc.writeMicroseconds(esc_off);

  // Initialize the 'Wire' class for the I2C-bus.
  Wire.begin();

  MPU9150_readings::setup();

  Serial.println("Connected!  Enter commands now (h for help)");
}

float p_gain=1.0; // proportional to angle error
float d_gain=0.15; // derivative of angle (gyro rate)
float i_gain=0.1; // integral of error
float center_term=esc_center; // enough microseconds to cancel gravity

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

int thermostat=0;
void user_interface(char c) {
    if (c=='h' || c=='?') { // Help mode
      Serial.println(F("Aero beam serial commands:\n"
        "  p <float gain>: proportional gain (default 1.0)\n"
        "  i <float gain>: integral gain (default 0.1)\n"
        "  d <float gain>: derivative gain (default 0.15)\n"
        "\n"
        "  c <float pulse>: center term (default 1200 us servo pulse)\n"
        "  m <float pulse>: manual motor term (default 0, disabled)\n"
        "\n"
        "  v: run short experiment (V for longer experiment)\n"
        "  r: enable real motor (space to stop)\n"
        "\n"
        "  x: show XYZ accelerometer data briefly (X shows for longer)\n"
        "  P: enable PID control (default)\n"
        "  T: enable thermostat (on/off) control instead of PID\n"
      ));
    }
    else if (c=='T') { // thermostat algorithm!
      Serial.println("Thermostat mode enabled (PID disabled)");
      thermostat=1;
    }
    else if (c=='P') { // PID algorithm on
      Serial.println("PID enabled (thermostat mode disabled)");
      thermostat=0;
    }
    else if (c=='p') { // adjust p gain
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
      XYZdata=200;
    }
    else if (c=='X') { // show raw xyz data for a long time
      XYZdata=1000;
    }
    
    else if (c=='v') { // verbose--run short experiment
      experiment_start=millis(); tot_err=0; tot_samp=0;
      verbose=200;
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
    else if (c=='\n' || c=='\r') { // ignore newlines
      
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
  int rate=imu.G[0]+gyro_vertigo; // gyro X axis == rate (plus drift fix)
  
  history+=err;
  long wind=4000; // maximum "wind-up": limit history to avoid oscilations
  if (history>wind) history=wind;
  if (history<-wind) history=-wind;

  // Sum PID terms and gains
  int cmd=-(0.01*p_gain)*err+d_gain*rate-(0.01*i_gain)*history;
  
  if (thermostat) { // overwrite PID with simple thermostat
    if (err<0) cmd=+100; // higher
    else cmd=-100; // lower
  }
  
  cmd += center_term;

  // Limit resulting command
  int cmd_min=esc_off, cmd_max=esc_cap;
  if (cmd<cmd_min) cmd=cmd_min;
  if (cmd>cmd_max) cmd=cmd_max;

  // Send value to motor:
  if (manual) esc.writeMicroseconds(manual);
  else if (verbose && run) esc.writeMicroseconds(cmd);
  else  esc.writeMicroseconds(esc_off);

  // Show data onscreen:
  if (verbose) {
    verbose--;
    Serial.print("TCERHA:\t");
    Serial.print(millis()-experiment_start);
    Serial.print("\t");
    Serial.print(cmd);
    Serial.print("\t");
    Serial.print(err);
    Serial.print("\t");
    Serial.print(rate);
    Serial.print("\t");
    Serial.print(history);
    Serial.print("\t");
    Serial.print(thermostat?"thermostat":"PID");
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
