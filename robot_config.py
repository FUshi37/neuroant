# Robot configuration file
# Modify these parameters according to your hexapod robot setup

import numpy as np

# Robot physical parameters
ROBOT_CONFIG = {
    # Link lengths for forward kinematics (meters)
    'link_lengths': {
        'l1': 0.126,  # Hip to knee
        'l2': 0.266,  # Knee to ankle
        'l3': 0.100,  # Ankle to foot
    },

    # Servo ID mapping for each leg
    # Assuming 3 servos per leg, 6 legs total
    'servo_mapping': {
        'FL': [0, 1, 2],  # Front Left: hip, knee, ankle
        'FR': [3, 4, 5],  # Front Right
        'ML': [6, 7, 8],  # Middle Left
        'MR': [9, 10, 11], # Middle Right
        'RL': [12, 13, 14], # Rear Left
        'RR': [15, 16, 17], # Rear Right
    },

    # Neutral pose angles (degrees)
    'neutral_angles': np.array([
        180, 180, 60,   # FL
        180, 180, 60,   # FR
        180, 180, 60,   # ML
        180, 180, 60,   # MR
        180, 180, 60,   # RL
        180, 180, 60,   # RR
    ]),

    # Angle limits (degrees) - based on real robot measurements
    'angle_limits': {
        'min': np.array([
            120.0,  # Joint 0
            130.0,  # Joint 1
            60.0,   # Joint 2
            130.0,  # Joint 3
            130.0,  # Joint 4
            60.0,   # Joint 5
            130.0,  # Joint 6
            130.0,  # Joint 7
            60.0,   # Joint 8
            130.0,  # Joint 9
            130.0,  # Joint 10
            60.0,   # Joint 11
            120.0,  # Joint 12
            130.0,  # Joint 13
            60.0,   # Joint 14
            100.0,  # Joint 15
            130.0,  # Joint 16
            60.0    # Joint 17
        ]),
        'max': np.array([
            225.0,  # Joint 0
            315.0,  # Joint 1
            270.0,  # Joint 2
            235.0,  # Joint 3
            315.0,  # Joint 4
            270.0,  # Joint 5
            225.0,  # Joint 6
            315.0,  # Joint 7
            270.0,  # Joint 8
            230.0,  # Joint 9
            315.0,  # Joint 10
            270.0,  # Joint 11
            260.0,  # Joint 12
            315.0,  # Joint 13
            270.0,  # Joint 14
            240.0,  # Joint 15
            315.0,  # Joint 16
            270.0   # Joint 17
        ]),
    },

    # IMU mounting offsets (degrees)
    'imu_offsets': {
        'roll': 0.0,
        'pitch': 0.0,
        'yaw': 0.0,
    },

    # Control parameters
    'control': {
        'frequency': 50,  # Hz
        'interpolation_steps': 50,  # Smooth motion interpolation
        'voltage_threshold': 11.1,  # Minimum voltage (V)
    },

    # Serial ports - Using by-id paths for stable device identification
    'serial_ports': {
        'servos': '/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT6Z5NIL-if00-port0',
        'imu': '/dev/serial/by-id/usb-1a86_USB2.0-Ser_-if00-port0',
    },
}

# Conversion functions
def real_to_sim_angles(real_angles):
    """Convert real robot angles to simulation angles (radians)"""
    # Add your conversion logic here
    # This might include axis direction flips, offset adjustments, etc.
    sim_angles = np.zeros(18)
    for i in range(18):
        sim_angles[i] = real_angles[i] * np.pi / 180.0

        # Example: flip certain joint directions
        if i % 3 == 1:  # Knee joints
            sim_angles[i] *= -1

    return sim_angles

def sim_to_real_angles(sim_angles):
    """Convert simulation angles to real robot angles (degrees)"""
    real_angles = np.zeros(18)
    for i in range(18):
        real_angles[i] = sim_angles[i] * 180.0 / np.pi

        # Apply inverse transformations
        if i % 3 == 1:  # Knee joints
            real_angles[i] *= -1

        # Apply angle limits
        real_angles[i] = np.clip(real_angles[i],
                                ROBOT_CONFIG['angle_limits']['min'][i],
                                ROBOT_CONFIG['angle_limits']['max'][i])

    return real_angles

def angles_to_ticks(angles):
    """Convert angles (degrees) to Dynamixel ticks"""
    ticks = np.zeros(18, dtype=int)
    for i in range(18):
        # Dynamixel XM430: 0-4095 ticks = 0-360 degrees
        ticks[i] = int((angles[i] / 360.0) * 4096)
        ticks[i] = np.clip(ticks[i], 0, 4095)
    return ticks

def ticks_to_angles(ticks):
    """Convert Dynamixel ticks to angles (degrees)"""
    angles = np.zeros(18)
    for i in range(18):
        angles[i] = (ticks[i] / 4096.0) * 360.0
    return angles
