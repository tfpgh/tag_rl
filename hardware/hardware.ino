#include <WiFi.h>
#include <WiFiUdp.h>
#include <esp_wifi.h>

const char *WIFI_SSID = "TagRL";
const char *WIFI_PASS = "TagRLPassword";
const int UDP_PORT = 8888;
const int TIMEOUT_MS = 200; // Kill motors after no packets for this long

// DRV8833 pins
const int PIN_LEFT1 = D6;
const int PIN_LEFT2 = D5;
const int PIN_RIGHT1 = D4;
const int PIN_RIGHT2 = D3;

WiFiUDP udp;
unsigned long lastPacketTime = 0;

void motorsSet(int16_t left, int16_t right) {
  int pwmLeft = map(abs(left), 0, 32767, 0, 255);
  int pwmRight = map(abs(right), 0, 32767, 0, 255);

  if (left > 0) {
    analogWrite(PIN_LEFT1, 255);
    analogWrite(PIN_LEFT2, 255 - pwmLeft);
  } else if (left < 0) {
    analogWrite(PIN_LEFT1, 255 - pwmLeft);
    analogWrite(PIN_LEFT2, 255);
  } else {
    analogWrite(PIN_LEFT1, 255);
    analogWrite(PIN_LEFT2, 255);
  }

  if (right > 0) {
    analogWrite(PIN_RIGHT1, 255);
    analogWrite(PIN_RIGHT2, 255 - pwmRight);
  } else if (right < 0) {
    analogWrite(PIN_RIGHT1, 255 - pwmRight);
    analogWrite(PIN_RIGHT2, 255);
  } else {
    analogWrite(PIN_RIGHT1, 255);
    analogWrite(PIN_RIGHT2, 255);
  }
}

void setup() {
  Serial.begin(115200);

  analogWriteFrequency(PIN_LEFT1, 20000);
  analogWriteFrequency(PIN_LEFT2, 20000);
  analogWriteFrequency(PIN_RIGHT1, 20000);
  analogWriteFrequency(PIN_RIGHT2, 20000);

  pinMode(PIN_LEFT1, OUTPUT);
  pinMode(PIN_LEFT2, OUTPUT);
  pinMode(PIN_RIGHT1, OUTPUT);
  pinMode(PIN_RIGHT2, OUTPUT);

  motorsSet(0, 0);

  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.println(WiFi.macAddress());
  while (WiFi.status() != WL_CONNECTED) {
    delay(100);
  }
  Serial.println(WiFi.localIP());

  esp_wifi_set_ps(WIFI_PS_NONE);

  udp.begin(UDP_PORT);
}

void loop() {
  int len = udp.parsePacket();
  if (len == 4) {
    uint8_t buf[4];
    udp.read(buf, 4);
    int16_t a = (int16_t)(buf[0] | (buf[1] << 8));
    int16_t b = (int16_t)(buf[2] | (buf[3] << 8));
    motorsSet(a, b);
    lastPacketTime = millis();
  }

  if (lastPacketTime != 0 && millis() - lastPacketTime > TIMEOUT_MS) {
    motorsSet(0, 0);
    lastPacketTime = 0;
    Serial.println("Timeout");
  }
}
