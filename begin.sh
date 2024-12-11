sudo chmod 777 /dev/ttyUSB1
sudo chmod 777 /dev/ttyUSB0
sudo chmod 777 /dev/ttyAMA4
sudo chmod 777 /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
sudo echo 1 > /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
sudo cat  /sys/bus/usb-serial/devices/ttyUSB0/latency_timer