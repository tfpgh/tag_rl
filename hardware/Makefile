FQBN := esp32:esp32:XIAO_ESP32C3
PORT := /dev/cu.usbmodem101

build:
	arduino-cli compile --fqbn $(FQBN) --build-path ./build .

upload: build
	arduino-cli upload --fqbn $(FQBN) --port $(PORT) --input-dir ./build .
	$(MAKE) monitor

monitor:
	arduino-cli monitor --port $(PORT)

clean:
	rm -rf build/

.PHONY: build upload monitor clean
