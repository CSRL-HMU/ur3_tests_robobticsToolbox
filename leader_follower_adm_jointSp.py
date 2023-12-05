import roboticstoolbox as rt
import numpy as np
import scipy as sp
import spatialmath as sm
import rtde_receive
import rtde_control
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import time
import keyboard


# Define follower (UR3)
rtde_c_follower = rtde_control.RTDEControlInterface("192.168.1.66")
rtde_r_follower = rtde_receive.RTDEReceiveInterface("192.168.1.66")

# Define leader (UR3e)
rtde_c_leader = rtde_control.RTDEControlInterface("192.168.1.60")
rtde_r_leader = rtde_receive.RTDEReceiveInterface("192.168.1.60")

# Get initial configuration
q0_leader = np.array(rtde_r_leader.getActualQ())
q0_follower = np.array(rtde_r_follower.getActualQ())

# Declare math pi
pi = math.pi

# Create the robot (both for Leader and Follower)
ur = rt.models.UR3()

# Control cycle
dt = 0.002

# Init time
t = 0.0

# Start logging
qlog_leader = q0_leader
tlog = t

# get time now
t_now = time.time()

# Tool mass
tool_mass = 0.145
# gravity acceleration
gAcc = 9.81

# initialize qdot for leader (admittance simulation)
qddot_leader = np.zeros(6)
qdot_leader = np.zeros(6)

# Define the admittance inertia
M_leader = np.identity(6)
M_leader[np.ix_([0, 1], [0, 1])] = 0.3 * M_leader[np.ix_([0, 1], [0, 1])]
M_leader[np.ix_([2, 3], [2, 3])] = 0.1 * M_leader[np.ix_([2, 3], [2, 3])]
M_leader[np.ix_([4, 5], [4, 5])] = 0.05 * M_leader[np.ix_([4, 5], [4, 5])]

# Compute inverse (constant)
Minv_leader = np.linalg.inv(M_leader);

# Define the admittance damping
D_leader = np.identity(6)
D_leader[np.ix_([0, 1], [0, 1])] = 0.3 * D_leader[np.ix_([0, 1], [0, 1])]
D_leader[np.ix_([2, 3], [2, 3])] = 0.3 * D_leader[np.ix_([2, 3], [2, 3])]
D_leader[np.ix_([4, 5], [4, 5])] = 0.3 * D_leader[np.ix_([4, 5], [4, 5])]

# initialize leader Jacobian
J_leader = np.zeros([6,6])

# this is the rotation between ur-rtde and Corke's robotics toolbox
Rrtde = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

# get force measurement bias
f_leader_offset_tool = np.array(rtde_r_leader.getActualTCPForce()) #- tool_mass * gAcc

# Move leader to the initial follower's pose
rtde_c_leader.moveJ(q0_follower, 0.5, 0.5)

# Initialize
k_follower = 0;




for i in range(50000):

    if keyboard.is_pressed('a'):
        print('Stopping robot')
        break

    # print(time.time() - t_now)
    t_now =  time.time()
    t_start = rtde_c_leader.initPeriod()


    # print(actual_q)


    # Integrate time
    t = t + dt

    # Get joint values

    q_leader = np.array(rtde_r_leader.getActualQ())
    q_follower = np.array(rtde_r_follower.getActualQ())

    g_leader = ur.fkine(q_leader)
    R_leader = np.array(g_leader.R)
    p_leader = np.array(g_leader.t)
    Je_leader = np.array(ur.jacobe(q_leader))

    J_leader[:3] = R_leader @ Je_leader[:3]
    J_leader[-3:] = R_leader @ Je_leader[-3:]



    f_leader = np.array(rtde_r_leader.getActualTCPForce()) - f_leader_offset_tool
    f_leader[:3] = Rrtde @ f_leader[:3]
    f_leader[-3:] = Rrtde @ f_leader[-3:]
    #f_leader[3] = f_leader[3] - tool_mass * gAcc
    print(f_leader)

    fnorm = np.linalg.norm(f_leader[:3])
    nF = f_leader[:3] / fnorm

    if fnorm<4.0:
        f_leader[:3] = np.zeros(3)
    else:
        f_leader[:3] = f_leader[:3] - 4.0 * nF

    taunorm = np.linalg.norm(f_leader[-3:])
    nTau = f_leader[-3:] / taunorm

    if taunorm < 0.5:
        f_leader[-3:] = np.zeros(3)
    else:
        f_leader[-3:] = f_leader[-3:] - 0.5 * nTau




    # v = np.concatenate(( - k * (p - pT) , np.zeros(3) ))
    qdot_leader = qdot_leader + qddot_leader * dt


    # v = 0.001 * f
    # v = [  0, 0.0 , 0.01 , 0, 0 ,0]
    # Admittance
    tau_leader =  np.transpose( J_leader) @ f_leader
    qddot_leader = Minv_leader @ (- D_leader @ qdot_leader + tau_leader)
    # print(v)





    rtde_c_leader.speedJ(qdot_leader, 3.0, dt)

    tlog = np.vstack((tlog,t))
    qlog_leader = np.vstack((qlog_leader, q_leader))

    # print(t)


    rtde_c_leader.waitPeriod(t_start)

    k_follower = k_follower + 1
    if k_follower == 4:

        rtde_c_follower.speedJ(qdot_leader - 1.0*(q_follower - q_leader), 3.0, 4.0*dt)
        k_follower = 0




    # print(f[:3])
    # print(R @ f[-3:])

rtde_c_leader.speedStop()
rtde_c_leader.stopScript()

rtde_c_follower.speedStop()
rtde_c_follower.stopScript()


# plt.plot(tlog,elog[:,2])
# plt.show()
# ur.plot( q=qlog, backend='pyplot',dt=dt)
