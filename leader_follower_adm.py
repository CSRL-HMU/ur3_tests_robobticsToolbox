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

rtde_c_follower = rtde_control.RTDEControlInterface("192.168.1.66")
rtde_r_follower = rtde_receive.RTDEReceiveInterface("192.168.1.66")

rtde_c_leader = rtde_control.RTDEControlInterface("192.168.1.60")
rtde_r_leader = rtde_receive.RTDEReceiveInterface("192.168.1.60")

q0_leader = np.array(rtde_r_leader.getActualQ())
q0_follower = np.array(rtde_r_follower.getActualQ())

pi = math.pi

# create the robot
ur = rt.models.UR3()


dt = 0.002

t = 0.0


qlog_leader = q0_leader
tlog = t


t_now = time.time()

v_leader = np.zeros(6)
vdot_leader = np.zeros(6)

M_leader = np.identity(6)
M_leader[np.ix_([0,1,2],[0,1,2])] = 10.0 * M_leader[np.ix_([0,1,2],[0,1,2])]
M_leader[np.ix_([3,4,5],[3,4,5])] = 0.01 * M_leader[np.ix_([3,4,5],[3,4,5])]

Minv_leader = np.linalg.inv(M_leader);

D_leader = np.identity(6)
D_leader[np.ix_([0,1,2],[0,1,2])] = 2.0 * D_leader[np.ix_([0,1,2],[0,1,2])]
D_leader[np.ix_([3,4,5],[3,4,5])] = 1.0 * D_leader[np.ix_([3,4,5],[3,4,5])]

J_leader = np.zeros([6,6])

Rrtde = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

f_leader_offset_tool = np.array(rtde_r_leader.getActualTCPForce())


rtde_c_leader.moveJ(q0_follower, 0.5, 0.5)

kk = 0;


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
    # f_leader = f_leader - f_leader_offset
    # print(f_leader)





    # v = np.concatenate(( - k * (p - pT) , np.zeros(3) ))
    v_leader = v_leader + vdot_leader * dt


    # v = 0.001 * f
    # v = [  0, 0.0 , 0.01 , 0, 0 ,0]
    # Admittance
    vdot_leader = Minv_leader @ (- D_leader @ v_leader + f_leader)
    # print(v)

    # v_leader[-3:] = np.zeros(3)
    qdot_leader = np.linalg.inv(J_leader) @ v_leader


    rtde_c_leader.speedJ(qdot_leader, 3.0, dt)

    tlog = np.vstack((tlog,t))
    qlog_leader = np.vstack((qlog_leader, q_leader))

    # print(t)


    rtde_c_leader.waitPeriod(t_start)

    kk = kk + 1
    if kk == 4:

        rtde_c_follower.speedJ(qdot_leader - 1.0*(q_follower - q_leader), 3.0, 4.0*dt)
        kk = 0




    # print(f[:3])
    # print(R @ f[-3:])

rtde_c_leader.speedStop()
rtde_c_leader.stopScript()

rtde_c_follower.speedStop()
rtde_c_follower.stopScript()


# plt.plot(tlog,elog[:,2])
# plt.show()
# ur.plot( q=qlog, backend='pyplot',dt=dt)
