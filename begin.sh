#!/bin/bash
# Using stable /dev/serial/by-id paths instead of ttyUSB* which can change
sudo chmod 777 /dev/ttyUSB0
sudo chmod 777 /dev/ttyUSB1
sudo chmod 777 /dev/ttyAMA4
sudo chmod 777 /sys/bus/usb-serial/devices/ttyUSB1/latency_timer
sudo echo 1 > /sys/bus/usb-serial/devices/ttyUSB1/latency_timer
sudo cat  /sys/bus/usb-serial/devices/ttyUSB1/latency_timer
#echo "=========================================="
#echo "Setting up USB serial ports..."
#echo "=========================================="

# Get actual device paths
#SERVO_BY_ID="/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT6Z5NIL-if00-port0"
#IMU_BY_ID="/dev/serial/by-id/usb-1a86_USB2.0-Ser_-if00-port0"

# Resolve to actual ttyUSB* devices
#if [ -L "$SERVO_BY_ID" ]; then
#    SERVO_DEV=$(readlink -f "$SERVO_BY_ID")
#    echo "Servo device (FTDI): $SERVO_BY_ID -> $SERVO_DEV"
#    sudo chmod 777 "$SERVO_DEV"
#    sudo chmod 777 "$SERVO_BY_ID"
#    echo "  ✓ Permissions set for servo port"
#else
#    echo "  ⚠️  Servo device not found!"
#    exit 1
#fi

#if [ -L "$IMU_BY_ID" ]; then
#    IMU_DEV=$(readlink -f "$IMU_BY_ID")
#    echo "IMU device (CH340): $IMU_BY_ID -> $IMU_DEV"
#    sudo chmod 777 "$IMU_DEV"
#    sudo chmod 777 "$IMU_BY_ID"
#    echo "  ✓ Permissions set for IMU port"
#else
#    echo "  ⚠️  IMU device not found!"
#    exit 1
#fi

# Set permissions for ttyAMA4
#if [ -e "/dev/ttyAMA4" ]; then
#sudo chmod 777 /dev/ttyAMA4
#    echo "  ✓ Permissions set for ttyAMA4"
#fi

# Set latency timer for FTDI device (servos)
#SERVO_TTY=$(basename "$SERVO_DEV")
#LATENCY_TIMER_PATH="/sys/bus/usb-serial/devices/$SERVO_TTY/latency_timer"

#if [ -f "$LATENCY_TIMER_PATH" ]; then
#    sudo chmod 777 "$LATENCY_TIMER_PATH"
#    echo 1 | sudo tee "$LATENCY_TIMER_PATH" > /dev/null
#    LATENCY_VALUE=$(cat "$LATENCY_TIMER_PATH")
#    echo "  ✓ Latency timer set for $SERVO_TTY: $LATENCY_VALUE"
#else
#    echo "  ⚠️  Latency timer file not found: $LATENCY_TIMER_PATH"
#fi

#echo "=========================================="
#echo "Setup completed!"
#echo "=========================================="