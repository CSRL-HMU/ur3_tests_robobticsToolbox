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

rtde_c = rtde_control.RTDEControlInterface("192.168.1.60")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.1.60")

init_q = rtde_r.getActualQ()


# rtde_r = rtde_receive.RTDEReceiveInterface("127.0.0.1")
# rtde_frequency = 500.0
# rtde_c = RTDEControl("127.0.0.1", rtde_frequency, RTDEControl.FLAG_USE_EXT_UR_CAP)


pi = math.pi

simEnable = False
visEnable = True
framerate = 30.0

rng = np.random.default_rng()

# create the robot
ur = rt.models.UR3()

# Initialization

if simEnable:
    q0 = np.array([0.0 , 0.5 , pi/4 , 0.6 , -pi/3 ,  1])
    q0 = np.array(rtde_r.getActualQ())
else:
    q0 = np.array(rtde_r.getActualQ())



q = q0

print(q)

dt_sim = 0.001
dt_real = 0.002
dt_vis = 1.0/framerate


qdot = np.array([-0.5 ,0 ,0.5 ,0 ,0 ,0])

t = 0.0
t_vis = 0.0
qlog = q

g0 = ur.fkine(q)
R0 = g0.R
p0 = g0.t

pT = p0 + np.array([-0.1 , 0.0 , 0.1])

k = 4.0

tlog = t
elog = p0 -pT

t_now = time.time()

for i in tqdm(range(5000)):

    print(time.time() - t_now)
    t_now =  time.time()
    t_start = rtde_c.initPeriod()

    actual_q = rtde_r.getActualQ()
    # print(actual_q)


    if simEnable:
        # Integrate time
        t = t + dt_sim
        t_vis = t_vis + dt_sim
        
        # Integrate joint values
        q = q + qdot * dt_sim
    else:
        # Integrate time
        t = t + dt_real
        t_vis = t_vis + dt_real

        # Get joint values
        q = np.array(rtde_r.getActualQ())

    g = ur.fkine(q)
    R = g.R
    p = g.t
    J = ur.jacob0(q)


    v = np.concatenate(( - k * (p - pT) , np.zeros(3) ))

    qdot = 0.0*np.linalg.inv(J) @ v

    if simEnable==0:
        rtde_c.speedJ(qdot, 1.0, dt_real)

    tlog = np.vstack((tlog,t))
    elog = np.vstack((elog,p-pT))

    # print(t)

    if visEnable:
        if t_vis>dt_vis:
            qlog = np.vstack((qlog,q))
            t_vis = 0.0

    rtde_c.waitPeriod(t_start)

    f = rtde_r.getActualTCPForce()
    # print(f)

rtde_c.speedStop()
rtde_c.stopScript()
    


plt.plot(tlog,elog[:,2])
plt.show()
ur.plot( q=qlog, backend='pyplot',dt=dt_vis)
