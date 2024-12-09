sudo chmod 777 /dev/ttyUSB0
sudo chmod 777 /dev/ttyUSB2
sudo chmod 777 /dev/ttyAMA4
sudo chmod 777 /sys/bus/usb-serial/devices/ttyUSB2/latency_timer
sudo echo 1 > /sys/bus/usb-serial/devices/ttyUSB2/latency_timer
sudo cat  /sys/bus/usb-serial/devices/ttyUSB2/latency_timer