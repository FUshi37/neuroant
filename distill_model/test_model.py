from action_net import *
import torch
import numpy as np
import time
from multiprocessing import Process
from multiprocessing import Process, Queue



obs_dim_t = 11

act_dim = 1
action_type = 'Continuous'
num_agents=6
n_embd=64
device = torch.device("cpu")
obs_dim=obs_dim_t*num_agents
    
    
action_net2=ActionNet(obs_dim, n_embd, act_dim,device)
action_net3=ActionNet(obs_dim, n_embd, act_dim,device)
action_net4=ActionNet(obs_dim, n_embd, act_dim,device)
action_net5=ActionNet(obs_dim, n_embd, act_dim,device)
action_net6=ActionNet(obs_dim, n_embd, act_dim,device)
    
action_net1=ActionNet(obs_dim, n_embd*1, act_dim*num_agents,device)

model1_dir="/home/fast3/Desktop/DynamixelSDK-3.7.31/python/tests/protocol2_0/distill_model/action_net_one_BC_mlp_new2_1_70.pt"
model2_dir="distill_model/action_net2_BC_mlp_1_20.pt"
model3_dir="distill_model/action_net3_BC_mlp_1_20.pt"
model4_dir="distill_model/action_net4_BC_mlp_1_20.pt"
model5_dir="distill_model/action_net5_BC_mlp_1_20.pt"
model6_dir="distill_model/action_net6_BC_mlp_1_20.pt"

saved_model_1 = torch.load(model1_dir,map_location=torch.device('cpu'))
saved_model_2 = torch.load(model2_dir,map_location=torch.device('cpu'))
saved_model_3 = torch.load(model3_dir,map_location=torch.device('cpu'))
saved_model_4 = torch.load(model4_dir,map_location=torch.device('cpu'))
saved_model_5 = torch.load(model5_dir,map_location=torch.device('cpu'))
saved_model_6 = torch.load(model6_dir,map_location=torch.device('cpu'))
action_net1.load_state_dict(saved_model_1)
action_net2.load_state_dict(saved_model_2)
action_net3.load_state_dict(saved_model_3)
action_net4.load_state_dict(saved_model_4)
action_net5.load_state_dict(saved_model_5)
action_net6.load_state_dict(saved_model_6)

obs=torch.rand((1,66))
for i in range(100):
    start_t=time.time()
    ac1=action_net1(obs)
    #ac2=action_net2(obs)
    #ac3=action_net3(obs)
    #ac4=action_net4(obs)
    #ac5=action_net5(obs)
    #ac6=action_net6(obs)


    end_t=time.time()
    print("last time",(end_t- start_t)*1000)

print("end")
