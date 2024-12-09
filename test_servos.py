import numpy as np
import getch
from Servos import *

DXLn_ID=[6,7,8,9,10,11]
servos=Servos()
#servos.light_LED()
#goal_position=2048*np.ones(1,18)
servos.read_voltage(1)
goal_position=np.array([180,204,85,180,204,85])
position_Read=servos.read_position_loop()
print("read position:",position_Read)
servos.enable_torque(DXLn_ID)
print("Press any key to continue! (or press ESC to move leg2!)")
if getch.getch()  == chr(0x1b):
    servos.write_some_positions(goal_position,DXLn_ID)
time.sleep(1)
position_Read=servos.read_all_positions()
print("read position:",position_Read)

DXLn_ID=[0,1,2,3,4,5]
servos.enable_torque(DXLn_ID)
print("Press any key to continue! (or press ESC to move leg1!)")
goal_position=np.array([180,204,85,180,204,85])
if getch.getch()  == chr(0x1b):
    servos.write_some_positions(goal_position,DXLn_ID)
time.sleep(1)
position_Read=servos.read_all_positions()
print("read position:",position_Read)

DXLn_ID=[12,13,14,15,16,17]
servos.enable_torque(DXLn_ID)
print("Press any key to continue! (or press ESC to move leg3!)")
goal_position=np.array([180,204,85,180,204,85])
if getch.getch()  == chr(0x1b):
    servos.write_some_positions(goal_position,DXLn_ID)
time.sleep(1)
position_Read=servos.read_all_positions()
print("read position:",position_Read)

print("Press any key to continue! (or press ESC to disable leg3!)")
if getch.getch()  == chr(0x1b):
    servos.disable_torque(DXLn_ID)
position_Read=servos.read_position_loop()
DXLn_ID=[6,7,8,9,10,11]
print("Press any key to continue! (or press ESC to disable leg2!)")
if getch.getch()  == chr(0x1b):
    servos.disable_torque(DXLn_ID)

DXLn_ID=[0,1,2,3,4,5]
position_Read=servos.read_position_loop()
print("Press any key to continue! (or press ESC to disable leg1!)")
if getch.getch()  == chr(0x1b):
    servos.disable_torque(DXLn_ID)
position_Read=servos.read_position_loop()

#servos.write_all_positions(goal_position)

