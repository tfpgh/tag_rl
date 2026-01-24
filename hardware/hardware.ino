#include <WiFi.h>
#include <WiFiUdp.h>

const char *WIFI_SSID = "Middlebury-IoT";
const char *WIFI_PASS = "INeedToGetAPassword";
const int UDP_PORT = 8888;
const int TIMEOUT_MS = 200; // Kill motors after no packets for this long

// DRV8833 pins
const int PIN_LEFT1 = 0;
const int PIN_LEFT2 = 1;
const int PIN_RIGHT1 = 2;
const int PIN_RIGHT2 = 3;

WiFiUDP udp;
unsigned long lastPacketTime = 0;

void motorsSet(int16_t left, int16_t right) {
  int pwmLeft = map(abs(left), 0, 32767, 0, 255);
  int pwmRight = map(abs(right), 0, 32767, 0, 255);

  analogWrite(PIN_LEFT1, left > 0 ? pwmLeft : 0);
  analogWrite(PIN_LEFT2, left < 0 ? pwmLeft : 0);
  analogWrite(PIN_RIGHT1, right > 0 ? pwmRight : 0);
  analogWrite(PIN_RIGHT2, right < 0 ? pwmRight : 0);
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

  Serial.println(WiFi.macAddress());

  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(100);
  }
  Serial.println(WiFi.localIP());

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
