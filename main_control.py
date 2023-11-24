import roboticstoolbox as rt
import numpy as np
import scipy as sp
import spatialmath as sm
import rtde_control
import rtde_receive
from tqdm import tqdm
import matplotlib.pyplot as plt




simEnable = True
visEnable = True
framerate = 30.0

rng = np.random.default_rng()

# create the robot
ur = rt.models.UR3()

# Initialization
q0 = np.array(rng.standard_normal(6))
q = q0

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


for i in tqdm(range(5000)):

    


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
        #q = rtde ... 

    g = ur.fkine(q)
    R = g.R
    p = g.t
    J = ur.jacob0(q)


    v = np.concatenate(( - k * (p - pT) , np.zeros(3) ))

    qdot = np.linalg.inv(J) @ v 



    tlog = np.vstack((tlog,t))
    elog = np.vstack((elog,p-pT))

    # print(t)

    if visEnable:
        if t_vis>dt_vis:
            qlog = np.vstack((qlog,q))
            t_vis = 0.0

    


plt.plot(tlog,elog[:,2])
plt.show()
ur.plot( q=qlog, backend='pyplot',dt=dt_vis)
