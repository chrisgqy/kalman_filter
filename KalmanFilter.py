import numpy as np

class KalmanFilter():
    def __init__(self, init_x, init_v, init_acc_var):
        # initialize variables
        self._x = np.array([init_x, init_v])
        
        # initialize covariance matrix 
        self._P = np.eye(2)
        
        # initialize acceleration 
        self._acc_var = init_acc_var


    def prediction(self, dt):
        F = np.array([[1, dt], [0, 1]])
        G = np.array([0.5*dt**2, dt])
        
        # Prediction of x: position and velocity
        # X = F X + G U
        pred_x = F.dot(self._x)  # + G.dot(acceleration), which is zero when intialized 

        # Prediction of covariance matrix
        # P = F P Ft + G Gt a
        pred_P = F.dot(self._P).dot(F.T) + G.dot(G.T)*self._acc_var

        self._x = pred_x
        self._P = pred_P

    def update(self, meas_val, meas_var):

        H = np.array([[1, 0]])
        z = np.array([meas_val])
        r = np.array([meas_var])

        # Y = z - H X
        # S = H P Ht + R
        y = z - H.dot(self._x)
        s = H.dot(self._P).dot(H.T) + r

        # K = P Ht S^-1 -- Kalman Gain
        # X = X + K Y -- update X
        # P = (I - K H) P -- update covariance matrix
        k = self._P.dot(H.T).dot(np.linalg.inv(s))
        x_update = self._x + k.dot(y)
        p_update = (np.eye(2) - k.dot(H)).dot(self._P)

        self._x = x_update
        self._P = p_update


    @property 
    def cov(self) -> np.array:
        return self._P
    
    @property 
    def mean(self) -> np.array:
        return self._x

    @property
    def pos(self) -> float:
        return self._x[0]
    
    @property 
    def velo(self) -> float:
        return self._x[1]