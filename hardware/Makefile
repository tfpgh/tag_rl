FQBN := esp32:esp32:esp32c3
PORT := /dev/ttyUSB0

build:
	arduino-cli compile --fqbn $(FQBN) --build-path ./build .
	cp build/compile_commands.json .

upload: build
	arduino-cli upload --fqbn $(FQBN) --port $(PORT) .

monitor:
	arduino-cli monitor --port $(PORT)

clean:
	rm -rf build/

.PHONY: build upload monitor clean
