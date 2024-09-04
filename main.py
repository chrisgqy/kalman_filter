import numpy as np
from matplotlib import pyplot as plt
from KalmanFilter import KalmanFilter

# real_position = 0.0
# real_v = 0.9

x = 0.0
v = 0.5
acc_var = 0.1

real_position = 0
real_velocity = 0.9
meas_var = 0.1 ** 2

kalman_filter = KalmanFilter(init_x = x, init_v = v, init_acc_var=acc_var)

dt = 0.1
steps = 1000
measure_steps = 20



mus = []
covs = []
real_xs = []
real_vs = []

for step in range(steps):
    # introduce change in real value to see if KF can keep up
    if step > 500:
        real_velocity *= 0.9
    covs.append(kalman_filter.cov)
    mus.append(kalman_filter.mean)

    real_position = real_position + dt * real_velocity
    kalman_filter.prediction(dt = dt)
    
    if step != 0 and step % measure_steps == 0:
        kalman_filter.update(meas_val = real_position + np.random.randn() * np.sqrt(meas_var), 
                             meas_var = meas_var) 

    real_xs.append(real_position)
    real_vs.append(real_velocity)


plt.ion()
plt.figure()

plt.subplot(2, 1, 1)
plt.title('Predicted Position')
plt.plot([mean[0] for mean in mus], 'r')
plt.plot(real_xs, 'b')
plt.plot([mean[0] - 2*np.sqrt(cov[0,0]) for mean, cov in zip(mus, covs)], 'r--')
plt.plot([mean[0] + 2*np.sqrt(cov[0,0]) for mean, cov in zip(mus, covs)], 'r--')



plt.subplot(2, 1, 2)
plt.title('Predicted Velocity')
plt.plot([mean[1] for mean in mus], 'r')
plt.plot(real_vs, 'b') 
plt.plot([mean[1] - 2*np.sqrt(cov[1,1]) for mean, cov in zip(mus, covs)], 'r--')
plt.plot([mean[1] + 2*np.sqrt(cov[1,1]) for mean, cov in zip(mus, covs)], 'r--')



plt.show()
plt.ginput()

