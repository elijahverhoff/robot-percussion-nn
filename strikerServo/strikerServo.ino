#include <Servo.h>

Servo strikerServo;

const int SERVO_PIN = 9;
const int LED_PIN   = 13;

const int REST_ANGLE   = 110;
const int STRIKE_ANGLE = 95;     // tune for your strike delay
const int STRIKE_HOLD_MS = 20;
const int RETURN_MS    = 120;

void setup() {
  Serial.begin(115200);
  strikerServo.attach(SERVO_PIN);
  strikerServo.write(REST_ANGLE);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  Serial.println("ready");
}

void strike() {
  digitalWrite(LED_PIN, HIGH);
  strikerServo.write(STRIKE_ANGLE);
  delay(STRIKE_HOLD_MS);
  digitalWrite(LED_PIN, LOW);
  strikerServo.write(REST_ANGLE);
  delay(RETURN_MS);
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();
    if (c == 's') strike();
  }
}
