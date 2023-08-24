import threading
import time
from queue import Queue
import datetime

from multiprocessing import Process,Queue
import numpy as np
from CPGs import *
import time
import matplotlib.pyplot as plt

from Servos import *

servos=Servos()
DXLn_ID=range(18)
print(DXLn_ID)
ID=0
servos.read_voltage(1)
print("voltage",)
#servos.enable_torque([10])
servos.set_position_control()
#servos.read_position_loop()
for i in range(100):

	#servos.read_voltage(ID)
	start_time=time.time()
	#position_Read=servos.read_all_positions()
	#print("read position:",position_Read)
	#servos.set_current_based_control()
	position_Read=servos.read_all_positions()
	#print("read position:",position_Read)
	#servos.read_goal_current()
	#current_Read=servos.read_all_current()

	#print("goal current:",servos.read_goal_current())
	#while(time.time()-start_time<0.010):
	#    1
	end_time=time.time()
	print("last_time",(end_time-start_time)*1000)
	
