import os
import numpy as np
import time
import getch
'''
if os.name == 'nt':
    import msvcrt
    def getch.getch() :
        return msvcrt.getch().decode()
else:
    import sys, tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    def getch.getch() :
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch
'''

from dynamixel_sdk import *                    # Uses Dynamixel SDK library
class Servos:
    def __init__(self):
        self.ADDR_PRO_TORQUE_ENABLE = 64  # Control table address is different in Dynamixel model
        self.ADDR_PRO_LED_RED = 65
        self.ADDR_PRO_GOAL_POSITION = 116
        self.ADDR_PRO_PRESENT_POSITION = 132
        self.ADDR_PRO_PRESENT_INPUT_VOLTAGE=144
        self.ADDR_PRO_PRESENT_CURRENT = 126
        self.ADDR_PRO_PRESENT_VELOCITY = 128
        self.ADDR_PRO_OPERATION_MODE= 11
        self.ADDR_PRO_GOAL_CURRENT=102


        self.POSITION_CONTROL=3
        self.CURRENT_BASED_POSITION_CONTROL=5


        # Data Byte Length
        self.LEN_PRO_LED_RED = 1
        self.LEN_PRO_GOAL_POSITION = 4
        self.LEN_PRO_PRESENT_POSITION = 4
        self.LEN_PRO_PRESENT_INPUT_VOLTAGE=2
        self.LEN_PRO_PRESENT_CURRENT = 2
        self.LEN_PRO_PRESENT_VELOCITY = 4
        self.LEN_PRO_OPERATION_MODE = 1
        self.LEN_PRO_GOAL_CURRENT = 2


        # Protocol version
        self.PROTOCOL_VERSION = 2.0  # See which protocol version is used in the Dynamixel

        # Default setting
        self.BAUDRATE = 4000000  # Dynamixel default baudrate : 57600
        self.DEVICENAME = '/dev/ttyUSB2'  # Check which port is being used on your controller
        # ex) Windows: "COM1"   Linux: "/dev/ttyUSB0" Mac: "/dev/tty.usbserial-*"

        self.TORQUE_ENABLE = 1  # Value for enabling the torque
        self.TORQUE_DISABLE = 0  # Value for disabling the torque
        self.Enable=1
        self.DXL_MINIMUM_POSITION_VALUE = 100  # Dynamixel will rotate between this value
        self.DXL_MAXIMUM_POSITION_VALUE = 4000  # and this value (note that the Dynamixel would not move when the position value is out of movable range. Check e-manual about the range of the Dynamixel you use.)
        self.DXL_MOVING_STATUS_THRESHOLD = 20  # Dynamixel moving status threshold
        self.portHandler = PortHandler(self.DEVICENAME)

        # Initialize PacketHandler instance
        # Set the protocol version
        # Get methods and members of Protocol1PacketHandler or Protocol2PacketHandler
        self.packetHandler = PacketHandler(self.PROTOCOL_VERSION)

        self.DXLn_ID = range(18)

        # Open port
        if self.portHandler.openPort():
            print("Succeeded to open the port")
        else:
            print("Failed to open the port")
            print("Press any key to terminate...")
            getch.getch() 
            quit()

        # Set port baudrate
        if self.portHandler.setBaudRate(self.BAUDRATE):
            print("Succeeded to change the baudrate")
        else:
            print("Failed to change the baudrate")
            print("Press any key to terminate...")
            getch.getch() 
            quit()


    def write_all_positions(self, dxl_goal_position):
        # Initialize GroupSyncWrite instance
        groupSyncWrite = GroupSyncWrite(self.portHandler, self.packetHandler, self.ADDR_PRO_GOAL_POSITION, self.LEN_PRO_GOAL_POSITION)
        goal_position=np.zeros(18)
        goal_position=dxl_goal_position.astype(int)

        for index_ID in self.DXLn_ID:
            #goal_position[index_ID]=int(dxl_goal_position[index_ID]) # tick 整数类型
            #print("ID: ",index_ID,"position:",goal_position[index_ID])
            param_goal_position = [DXL_LOBYTE(DXL_LOWORD(goal_position[index_ID])),
                                   DXL_HIBYTE(DXL_LOWORD(goal_position[index_ID])),
                                   DXL_LOBYTE(DXL_HIWORD(int(goal_position[index_ID]))),
                                   DXL_HIBYTE(DXL_HIWORD(int(goal_position[index_ID])))]
            dxl_addparam_result = groupSyncWrite.addParam(index_ID, param_goal_position)
            if dxl_addparam_result != True:
                print("[ID:%03d] groupSyncWrite addparam failed" % index_ID)
                quit()

        dxl_comm_result = groupSyncWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))

        # Clear syncwrite parameter storage
        groupSyncWrite.clearParam()
        
    
    def write_all_positions_angles(self, dxl_goal_position):
        # Initialize GroupSyncWrite instance
        groupSyncWrite = GroupSyncWrite(self.portHandler, self.packetHandler, self.ADDR_PRO_GOAL_POSITION, self.LEN_PRO_GOAL_POSITION)
        goal_position=np.zeros(18).astype(int)
        #goal_position=dxl_goal_position.astype(int)

        for index_ID in self.DXLn_ID:
            #dxl_goal_position[count]=int(dxl_goal_position_input[count]/360*4096) # 将度数转化为tick
            #print("ID: ",index_ID,"position:",dxl_goal_position[count])
            goal_position[index_ID]=int(dxl_goal_position[index_ID]/360*4096) #  将度数转化为tick
            #print("ID: ",index_ID,"position:",goal_position[index_ID])
            param_goal_position = [DXL_LOBYTE(DXL_LOWORD(goal_position[index_ID])),
                                   DXL_HIBYTE(DXL_LOWORD(goal_position[index_ID])),
                                   DXL_LOBYTE(DXL_HIWORD(int(goal_position[index_ID]))),
                                   DXL_HIBYTE(DXL_HIWORD(int(goal_position[index_ID])))]
            dxl_addparam_result = groupSyncWrite.addParam(index_ID, param_goal_position)
            if dxl_addparam_result != True:
                print("[ID:%03d] groupSyncWrite addparam failed" % index_ID)
                quit()

        dxl_comm_result = groupSyncWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))

        # Clear syncwrite parameter storage
        groupSyncWrite.clearParam()
        print("write angles")

    def read_all_positions(self):
        groupSyncRead = GroupSyncRead(self.portHandler, self.packetHandler, self.ADDR_PRO_PRESENT_POSITION,
                                      self.LEN_PRO_PRESENT_POSITION)
        dxl_present_position = np.zeros( 18)
        flag_read_success = np.ones(18)
        for index_ID in self.DXLn_ID:
            dxl_addparam_result = groupSyncRead.addParam(index_ID)
            if dxl_addparam_result != True:
                print("[ID:%03d] groupSyncRead addparam failed" % index_ID)
                quit()

        while 1:
             # Syncread present position
            dxl_comm_result = groupSyncRead.txRxPacket()
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
                flag_read_success[0] = 1
                continue

            for index_ID in self.DXLn_ID:
                #print(index_ID)
                dxl_getdata_result = groupSyncRead.isAvailable(index_ID, self.ADDR_PRO_PRESENT_POSITION,
                                                                   self.LEN_PRO_PRESENT_POSITION)
                if dxl_getdata_result != True:
                    print("[ID:%03d] groupSyncRead getdata failed" % index_ID)
                    flag_read_success[index_ID] = 1
                    
                    #quit()
                dxl_present_position[index_ID] = groupSyncRead.getData(index_ID, self.ADDR_PRO_PRESENT_POSITION,
                                                                           self.LEN_PRO_PRESENT_POSITION)*360/4096
                flag_read_success[index_ID] = 0
            

            if np.sum(flag_read_success) == 0:
                flag=1
                break
        return  dxl_present_position


    def enable_torque(self,dxln_ID_selected):
        for index_ID in dxln_ID_selected:
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, index_ID, self.ADDR_PRO_TORQUE_ENABLE,
                                                      self.TORQUE_ENABLE)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            else:
                print("Dynamixel#%d has been successfully connected" % index_ID)

    def disable_torque(self,dxln_ID_selected):
        for index_ID in dxln_ID_selected:
            dxl_comm_result, dxl_error = self.packetHandler.write1ByteTxRx(self.portHandler, index_ID, self.ADDR_PRO_TORQUE_ENABLE,
                                                      self.TORQUE_DISABLE)
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
            elif dxl_error != 0:
                print("%s" % self.packetHandler.getRxPacketError(dxl_error))
            else:
                print("Dynamixel#%d has been successfully connected" % index_ID)


    def light_LED(self):
        groupSyncWrite = GroupSyncWrite(self.portHandler, self.packetHandler, self.ADDR_PRO_LED_RED,
                                        self.LEN_PRO_LED_RED)

        for index_ID in self.DXLn_ID:

            dxl_addparam_result = groupSyncWrite.addParam(index_ID, [self.Enable])
            if dxl_addparam_result != True:
                print("[ID:%03d] groupSyncWrite addparam failed" % index_ID)
                quit()

        dxl_comm_result = groupSyncWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))

        # Clear syncwrite parameter storage
        groupSyncWrite.clearParam()

    def read_position_loop(self):
        groupSyncRead = GroupSyncRead(self.portHandler, self.packetHandler, self.ADDR_PRO_PRESENT_POSITION,
                                      self.LEN_PRO_PRESENT_POSITION)
        dxl_present_position = np.zeros(18)
        flag_read_success = np.ones(18)
        for index_ID in self.DXLn_ID:
            dxl_addparam_result = groupSyncRead.addParam(index_ID)
            if dxl_addparam_result != True:
                print("[ID:%03d] groupSyncRead addparam failed" % index_ID)
                quit()

        while 1:
            # Syncread present position
            print("Press any key to continue! (or press ESC to quit!)")
            if getch.getch()  == chr(0x1b):
                break
            dxl_comm_result = groupSyncRead.txRxPacket()
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
                flag_read_success[0] = 1
                continue

            for index_ID in self.DXLn_ID:
                #print(index_ID)
                dxl_getdata_result = groupSyncRead.isAvailable(index_ID, self.ADDR_PRO_PRESENT_POSITION,
                                                               self.LEN_PRO_PRESENT_POSITION)
                if dxl_getdata_result != True:
                    print("[ID:%03d] groupSyncRead getdata failed" % index_ID)
                    flag_read_success[index_ID] = 1
                    quit()
                dxl_present_position[index_ID] = groupSyncRead.getData(index_ID, self.ADDR_PRO_PRESENT_POSITION,
                                                                       self.LEN_PRO_PRESENT_POSITION)/4096*360
                flag_read_success[index_ID] = 0
            print(dxl_present_position)


        return dxl_present_position
   
    def read_all_positions_tick(self):
        groupSyncRead = GroupSyncRead(self.portHandler, self.packetHandler, self.ADDR_PRO_PRESENT_POSITION,
                                      self.LEN_PRO_PRESENT_POSITION)
        dxl_present_position = np.zeros( 18)
        flag_read_success = np.ones(18)
        for index_ID in self.DXLn_ID:
            dxl_addparam_result = groupSyncRead.addParam(index_ID)
            if dxl_addparam_result != True:
                print("[ID:%03d] groupSyncRead addparam failed" % index_ID)
                quit()

        while 1:
             # Syncread present position
            dxl_comm_result = groupSyncRead.txRxPacket()
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))

            for index_ID in self.DXLn_ID:
                #print(index_ID)
                dxl_getdata_result = groupSyncRead.isAvailable(index_ID, self.ADDR_PRO_PRESENT_POSITION,
                                                                   self.LEN_PRO_PRESENT_POSITION)
                if dxl_getdata_result != True:
                    print("[ID:%03d] groupSyncRead getdata failed" % index_ID)
                    flag_read_success[index_ID] = 1
                    #quit()
                dxl_present_position[index_ID] = groupSyncRead.getData(index_ID, self.ADDR_PRO_PRESENT_POSITION,
                                                                           self.LEN_PRO_PRESENT_POSITION)
                flag_read_success[index_ID] = 0

            if np.sum(flag_read_success) == 0:
                break
        return  dxl_present_position   
    

    def write_some_positions(self, dxl_goal_position_input,DXLn_ID):
        # Initialize GroupSyncWrite instance
        groupSyncWrite = GroupSyncWrite(self.portHandler, self.packetHandler, self.ADDR_PRO_GOAL_POSITION, self.LEN_PRO_GOAL_POSITION)
        count=0
        dxl_goal_position=DXLn_ID.copy()

        for index_ID in DXLn_ID:
            dxl_goal_position[count]=int(dxl_goal_position_input[count]/360*4096) # 将度数转化为tick
            print("ID: ",index_ID,"position:",dxl_goal_position[count])
            param_goal_position = [DXL_LOBYTE(DXL_LOWORD(dxl_goal_position[count])),
                                   DXL_HIBYTE(DXL_LOWORD(dxl_goal_position[count])),
                                   DXL_LOBYTE(DXL_HIWORD(dxl_goal_position[count])),
                                   DXL_HIBYTE(DXL_HIWORD(dxl_goal_position[count]))]
                                   
            dxl_addparam_result = groupSyncWrite.addParam(index_ID, param_goal_position)
            count=count+1
            if dxl_addparam_result != True:
                print("[ID:%03d] groupSyncWrite addparam failed" % index_ID)
                quit()

        dxl_comm_result = groupSyncWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))

        # Clear syncwrite parameter storage
        groupSyncWrite.clearParam()

    def Robot_initialize(self,dxl_goal_position):
        # 输入角度
        position_Read=self.read_all_positions_tick()
        print("position_Read",position_Read)
        #print("dxl_goal_position",dxl_goal_position)

        Interpolation_n=50
        for j in range(Interpolation_n):
            position_goal=np.trunc(position_Read+(dxl_goal_position/360*4096-position_Read)/Interpolation_n*(j+1)) #tick
            #print("position goal",position_goal)
            self.write_all_positions(position_goal)
            time.sleep(0.001)
            if j%10==0:
                print(self.read_all_current())
            #position_temp=self.read_all_positions()
            #print("posoyion temp:",position_temp)
            
    def read_voltage(self,dxl_ID):
        dxl_present_voltage, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(self.portHandler, dxl_ID,
                                                                                       self.ADDR_PRO_PRESENT_INPUT_VOLTAGE)
        print("Vlotage:",dxl_present_voltage)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))

        if dxl_present_voltage<11.1:
            print("!!!!voltage is not enough")
        return dxl_present_voltage

    def read_current_single(self,dxl_ID):
        dxl_present_current, dxl_comm_result, dxl_error = self.packetHandler.read2ByteTxRx(self.portHandler, dxl_ID,
                                                                                    self.ADDR_PRO_PRESENT_CURRENT)
        print("CURRENT:",dxl_present_current*2.69)
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            print("%s" % self.packetHandler.getRxPacketError(dxl_error))
        return dxl_present_current*2.69


    def read_current_2legs(self,leg_couple_index):
        sum_current=0
        for index in range(6):
            ID=int(int(leg_couple_index)*int(6)+int(index))
            #print("ID:",ID,"leg_couple_index",leg_couple_index,"index:",index)
            present_current=self.read_current_single(ID)
            sum_current=sum_current+present_current
        print("sum_current",sum_current)

    def read_all_current(self):
        groupSyncRead = GroupSyncRead(self.portHandler, self.packetHandler, self.ADDR_PRO_PRESENT_CURRENT,
                                      self.LEN_PRO_PRESENT_CURRENT)
        dxl_present_current = np.zeros(18)
        flag_read_success = np.ones(18)
        for index_ID in self.DXLn_ID:
            dxl_addparam_result = groupSyncRead.addParam(index_ID)
            if dxl_addparam_result != True:
                print("[ID:%03d] groupSyncRead addparam failed" % index_ID)
                quit()

        while 1:
             # Syncread present position
            dxl_comm_result = groupSyncRead.txRxPacket()
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
                flag_read_success[index_ID] = 1

            for index_ID in self.DXLn_ID:
                #print(index_ID)
                dxl_getdata_result = groupSyncRead.isAvailable(index_ID, self.ADDR_PRO_PRESENT_CURRENT,
                                                                   self.LEN_PRO_PRESENT_CURRENT)
                if dxl_getdata_result != True:
                    print("[ID:%03d] groupSyncRead getdata failed" % index_ID)
                    flag_read_success[0] = 1
                    continue
                    #quit()
                dxl_present_current[index_ID] = groupSyncRead.getData(index_ID, self.ADDR_PRO_PRESENT_CURRENT,
                                                                           self.LEN_PRO_PRESENT_CURRENT)
                if dxl_present_current[index_ID]>61400:
                    dxl_present_current[index_ID]=dxl_present_current[index_ID]-65536
                dxl_present_current[index_ID]=dxl_present_current[index_ID]
                flag_read_success[index_ID] = 0
            
            #if np.sum(flag_read_success) != 0:
                #flag=0
                #break

            if np.sum(flag_read_success) == 0:
                flag=1
                break
        return  dxl_present_current
    
    
    def read_all_torque(self):
        groupSyncRead = GroupSyncRead(self.portHandler, self.packetHandler, self.ADDR_PRO_PRESENT_CURRENT,
                                      self.LEN_PRO_PRESENT_CURRENT)
        dxl_present_current = np.zeros(18)
        flag_read_success = np.ones(18)
        for index_ID in self.DXLn_ID:
            dxl_addparam_result = groupSyncRead.addParam(index_ID)
            if dxl_addparam_result != True:
                print("[ID:%03d] groupSyncRead addparam failed" % index_ID)
                quit()

        while 1:
             # Syncread present position
            dxl_comm_result = groupSyncRead.txRxPacket()
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
                flag_read_success[index_ID] = 1

            for index_ID in self.DXLn_ID:
                #print(index_ID)
                dxl_getdata_result = groupSyncRead.isAvailable(index_ID, self.ADDR_PRO_PRESENT_CURRENT,
                                                                   self.LEN_PRO_PRESENT_CURRENT)
                if dxl_getdata_result != True:
                    print("[ID:%03d] groupSyncRead getdata failed" % index_ID)
                    flag_read_success[0] = 1
                    continue
                    #quit()
                dxl_present_current[index_ID] = groupSyncRead.getData(index_ID, self.ADDR_PRO_PRESENT_CURRENT,
                                                                           self.LEN_PRO_PRESENT_CURRENT)
                if dxl_present_current[index_ID]>61400:
                    dxl_present_current[index_ID]=dxl_present_current[index_ID]-65536
                    flag=-1
                else:
                    flag=1
                if abs(dxl_present_current[index_ID])>6:
                    dxl_present_current[index_ID]=(abs(dxl_present_current[index_ID])*2.69/1000*1.82-0.2576)*flag
                else:
                    dxl_present_current[index_ID]=(dxl_present_current[index_ID])*2.69/1000*1.82
                #dxl_present_current[index_ID]=dxl_present_current[index_ID]
                flag_read_success[index_ID] = 0
            
            #if np.sum(flag_read_success) != 0:
                #flag=0
                #break

            if np.sum(flag_read_success) == 0:
                flag=1
                break
        return  dxl_present_current
    
    def read_all_velocity(self):
        groupSyncRead = GroupSyncRead(self.portHandler, self.packetHandler, self.ADDR_PRO_PRESENT_VELOCITY,
                                      self.LEN_PRO_PRESENT_VELOCITY)
        dxl_present_current = np.zeros(18)
        flag_read_success = np.ones(18)
        for index_ID in self.DXLn_ID:
            dxl_addparam_result = groupSyncRead.addParam(index_ID)
            if dxl_addparam_result != True:
                print("[ID:%03d] groupSyncRead addparam failed" % index_ID)
                quit()

        while 1:
             # Syncread present position
            dxl_comm_result = groupSyncRead.txRxPacket()
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))

            for index_ID in self.DXLn_ID:
                #print(index_ID)
                dxl_getdata_result = groupSyncRead.isAvailable(index_ID, self.ADDR_PRO_PRESENT_VELOCITY,
                                                                   self.LEN_PRO_PRESENT_VELOCITY)
                if dxl_getdata_result != True:
                    print("[ID:%03d] groupSyncRead getdata failed" % index_ID)
                    flag_read_success[index_ID] = 1
                    #quit()
                dxl_present_current[index_ID] = groupSyncRead.getData(index_ID, self.ADDR_PRO_PRESENT_VELOCITY,
                                                                           self.LEN_PRO_PRESENT_VELOCITY)
                if dxl_present_current[index_ID]>61400:
                    dxl_present_current[index_ID]=65536-dxl_present_current[index_ID]
                dxl_present_current[index_ID]=dxl_present_current[index_ID]*2.69
                flag_read_success[index_ID] = 0

            if np.sum(flag_read_success) == 0:
                break
        return  dxl_present_current

    def set_current_based_control(self):

        # change mode
        groupSyncWrite = GroupSyncWrite(self.portHandler, self.packetHandler, self.ADDR_PRO_OPERATION_MODE,
                                        self.LEN_PRO_OPERATION_MODE)


        for index_ID in self.DXLn_ID:
            dxl_addparam_result = groupSyncWrite.addParam(index_ID, [self.CURRENT_BASED_POSITION_CONTROL])
            if dxl_addparam_result != True:
                print("[ID:%03d] groupSyncWrite addparam failed" % index_ID)
                quit()

        dxl_comm_result = groupSyncWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        print("changing to current based control is successful")

        # Clear syncwrite parameter storage
        groupSyncWrite.clearParam()

        # set goal current=750
        groupSyncWrite = GroupSyncWrite(self.portHandler, self.packetHandler, self.ADDR_PRO_GOAL_CURRENT,
                                        self.LEN_PRO_GOAL_CURRENT)
        goal_current = 460*np.ones(18).astype(int)


        for index_ID in self.DXLn_ID:
            # goal_position[index_ID]=int(dxl_goal_position[index_ID]) # tick 整数类型
            # print("ID: ",index_ID,"position:",goal_position[index_ID])
            param_goal_position = [DXL_LOBYTE(goal_current[index_ID]),
                                   DXL_HIBYTE(goal_current[index_ID])]
            dxl_addparam_result = groupSyncWrite.addParam(index_ID, param_goal_position)
            if dxl_addparam_result != True:
                print("[ID:%03d] groupSyncWrite addparam failed" % index_ID)
                quit()

        dxl_comm_result = groupSyncWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))

        # Clear syncwrite parameter storage
        groupSyncWrite.clearParam()
    
    def set_position_control(self):

        # change mode
        groupSyncWrite = GroupSyncWrite(self.portHandler, self.packetHandler, self.ADDR_PRO_OPERATION_MODE,
                                        self.LEN_PRO_OPERATION_MODE)


        for index_ID in self.DXLn_ID:
            dxl_addparam_result = groupSyncWrite.addParam(index_ID, [self.POSITION_CONTROL])
            if dxl_addparam_result != True:
                print("[ID:%03d] groupSyncWrite addparam failed" % index_ID)
                quit()

        dxl_comm_result = groupSyncWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))
        print("changing to position control is successful")

        # Clear syncwrite parameter storage
        groupSyncWrite.clearParam()

        # set goal current=750
        groupSyncWrite = GroupSyncWrite(self.portHandler, self.packetHandler, self.ADDR_PRO_GOAL_CURRENT,
                                        self.LEN_PRO_GOAL_CURRENT)
        goal_current = 460*np.ones(18).astype(int)


        for index_ID in self.DXLn_ID:
            # goal_position[index_ID]=int(dxl_goal_position[index_ID]) # tick 整数类型
            # print("ID: ",index_ID,"position:",goal_position[index_ID])
            param_goal_position = [DXL_LOBYTE(goal_current[index_ID]),
                                   DXL_HIBYTE(goal_current[index_ID])]
            dxl_addparam_result = groupSyncWrite.addParam(index_ID, param_goal_position)
            if dxl_addparam_result != True:
                print("[ID:%03d] groupSyncWrite addparam failed" % index_ID)
                quit()

        dxl_comm_result = groupSyncWrite.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))

        # Clear syncwrite parameter storage
        groupSyncWrite.clearParam()



    def read_goal_current(self):
        groupSyncRead = GroupSyncRead(self.portHandler, self.packetHandler, self.ADDR_PRO_GOAL_CURRENT,
                                      self.LEN_PRO_GOAL_CURRENT)
        dxl_goal_current = np.zeros( 18)
        flag_read_success = np.ones(18)
        for index_ID in self.DXLn_ID:
            dxl_addparam_result = groupSyncRead.addParam(index_ID)
            if dxl_addparam_result != True:
                print("[ID:%03d] groupSyncRead addparam failed" % index_ID)
                quit()

        while 1:
             # Syncread present position
            dxl_comm_result = groupSyncRead.txRxPacket()
            if dxl_comm_result != COMM_SUCCESS:
                print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))

            for index_ID in self.DXLn_ID:
                #print(index_ID)
                dxl_getdata_result = groupSyncRead.isAvailable(index_ID, self.ADDR_PRO_GOAL_CURRENT,
                                                                   self.LEN_PRO_GOAL_CURRENT)
                if dxl_getdata_result != True:
                    print("[ID:%03d] groupSyncRead getdata failed" % index_ID)
                    flag_read_success[index_ID] = 1
                    #quit()
                dxl_goal_current[index_ID] = groupSyncRead.getData(index_ID, self.ADDR_PRO_GOAL_CURRENT,
                                                                           self.LEN_PRO_GOAL_CURRENT)
                flag_read_success[index_ID] = 0

            if np.sum(flag_read_success) == 0:
                break
        return  dxl_goal_current*2.69

    def read_all_positions_velocity_current(self):
        groupBulkRead = GroupBulkRead(self.portHandler, self.packetHandler)
        groupSyncRead = GroupSyncRead(self.portHandler, self.packetHandler, self.ADDR_PRO_PRESENT_POSITION,
                                      self.LEN_PRO_PRESENT_POSITION)
        dxl_present_position = np.zeros( 18)
        dxl_present_velocity = np.zeros(18)
        dxl_present_current = np.zeros(18)
        flag_read_success = np.ones(18)
        for index_ID in self.DXLn_ID:
            # read  position
            dxl_addparam_result = groupBulkRead.addParam(index_ID, self.ADDR_PRO_PRESENT_POSITION, self.LEN_PRO_PRESENT_POSITION)
            if dxl_addparam_result != True:
                print("[ID:%03d] groupBulkRead  position addparam failed" % index_ID)
        for index_ID in self.DXLn_ID:
            # read velocity
            dxl_addparam_result = groupBulkRead.addParam(index_ID, self.ADDR_PRO_PRESENT_VELOCITY,
                                                         self.LEN_PRO_PRESENT_VELOCITY)
            if dxl_addparam_result != True:
                print("[ID:%03d] groupBulkRead addparam  velocity failed" % index_ID)
        
        for index_ID in self.DXLn_ID:
            #read current
            dxl_addparam_result = groupBulkRead.addParam(index_ID, self.ADDR_PRO_PRESENT_CURRENT,
                                                         self.LEN_PRO_PRESENT_CURRENT)
            if dxl_addparam_result != True:
                print("[ID:%03d] groupBulkRead addparam  current failed" % index_ID)





        while 1:
             # Syncread present position
             dxl_comm_result = groupBulkRead.txRxPacket()
             if dxl_comm_result != COMM_SUCCESS:
                 print("%s" % self.packetHandler.getTxRxResult(dxl_comm_result))

             for index_ID in self.DXLn_ID:
                 # print(index_ID)
                 dxl_getdata_result = groupBulkRead.isAvailable(index_ID, self.ADDR_PRO_PRESENT_POSITION,
                                                                self.LEN_PRO_PRESENT_POSITION)
                 if dxl_getdata_result != True:
                     print("[ID:%03d] groupBulkRead getdata failed" % index_ID)
                     flag_read_success[index_ID] = 1
                     # quit()
                 dxl_present_position[index_ID] = groupBulkRead.getData(index_ID, self.ADDR_PRO_PRESENT_POSITION,
                                                                        self.LEN_PRO_PRESENT_POSITION)



                 dxl_getdata_result = groupBulkRead.isAvailable(index_ID, self.ADDR_PRO_PRESENT_VELOCITY,
                                                                self.LEN_PRO_PRESENT_VELOCITY)
                 if dxl_getdata_result != True:
                     print("[ID:%03d] groupBulkRead getdata failed" % index_ID)
                     flag_read_success[index_ID] = 1
                     # quit()
                 dxl_present_velocity[index_ID] = groupBulkRead.getData(index_ID, self.ADDR_PRO_PRESENT_VELOCITY,
                                                                        self.LEN_PRO_PRESENT_VELOCITY)

                 dxl_getdata_result = groupBulkRead.isAvailable(index_ID, self.ADDR_PRO_PRESENT_CURRENT,
                                                                self.LEN_PRO_PRESENT_CURRENT)
                 if dxl_getdata_result != True:
                     print("[ID:%03d] groupBulkRead getdata failed" % index_ID)
                     flag_read_success[index_ID] = 1
                     # quit()
                 dxl_present_current[index_ID] = groupBulkRead.getData(index_ID, self.ADDR_PRO_PRESENT_CURRENT,
                                                                        self.LEN_PRO_PRESENT_CURRENT)

                 flag_read_success[index_ID] = 0

             if np.sum(flag_read_success) == 0:
                break

        return  dxl_present_position,dxl_present_velocity,dxl_present_current


















