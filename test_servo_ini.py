import numpy as np
import getch
from Servos import *
servos=Servos()
#servos.light_LED()
#goal_position=2048*np.ones(1,18)
DXLn_ID=range(18)
print(DXLn_ID)
servos.read_voltage(2)
position_Read=servos.read_all_positions()
print("read position:",position_Read) ,  

print("Press any key to enable legs")
if getch.getch() != chr(0x1b):
    servos.enable_torque(DXLn_ID)


goal_position=np.array([180,205,103,180,205,103,180,205,103,180,205,103,180,205,103,180,205,103])
servos.Robot_initialize(goal_position)
position_Read=servos.read_all_positions()
print("read position:",position_Read)
print("Press any key to disable legs")
if getch.getch() != chr(0x1b):
    servos.disable_torque(DXLn_ID)
