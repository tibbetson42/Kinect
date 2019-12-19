from scipy.linalg   import block_diag
import NN_functions as nnf
import numpy        as np


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class Weight_Adapter():
    def __init__(self,model,sampleX):

        weights = model.get_weights();
        # Initialize per cheng paper on adaptive weights
        num_joints = sampleX[0].shape[-2]
        layers = weights[0].shape[1]
        # Offline trained model - everything but last layer
        self.U = [weights[i] for i in range(0,len(weights)-2)]
        self.g = nnf.create_model(layer1 =layers, layer2 = layers,num_joints = num_joints,for_adapter = True)
        # also pass one sample to it to initialize
        #pdb.set_trace()
        g_out = self.g.predict(sampleX)
        self.g.set_weights(self.U)

        #Format adaptive weights. W is in NN format, Theta is column stack of W
        self.W = weights[-2::]
        self.Theta = self.W2Theta(self.W)

        # Initialize state variables. X = past window, true value
        self.Xest = None
        self.X = None
        self.Xerr = None
        self.Xshape = sampleX.shape
        self.glength = g_out.size + 1

        # Adaption Gain
        self.F = 100 * np.eye(len(self.Theta))

        # Learning parameters
        self.Lambda1 = 0.998
        self.Lambda2 = 1.0

        return

    def start(self,X):
        print('starting...')
        self.X = X
        self.Phi  = self.getPhi( self.g.predict(self.X) )
        self.Xest = np.matmul( self.Phi, self.Theta ).reshape(self.Xshape)
        # new data flag
        self.new = False
        #pdb.set_trace()
        return

    def feedData(self,X):
        print('feeding data')
        self.X = X
        self.new = True
        return

    def mainLoop(self):
        if not self.new:
            return
        #err between the our new observed X, and predicted
        self.Xerr  = (self.X - self.Xest)
        # update parameters by Cheng equation 6
        self.Theta = self.updateTheta( self.Theta, self.F, self.Phi, self.Xerr.reshape((-1,1)))
        # update F by equation 7
        self.F     = self.updateF( self.F, self.Phi )

        # Predict with the model g's n-1 layers, then apply theta to guess next X
        self.Phi   = self.getPhi( self.g.predict(self.X) )
        self.Xest  = np.matmul( self.Phi, self.Theta ).reshape(self.Xshape)

        #wait for new X observation
        self.new = False

        return

    def getPhi(self,g_out):
        g_out = np.append(g_out.reshape((1,-1)),1).reshape((1,-1))
        Phi = g_out
        for i in range(0,self.W[0].shape[1]-1):
            Phi = block_diag(Phi,g_out)
        #pdb.set_trace()
        return Phi

    def updateF(self,F,Phi):
        #equation 7 in cheng's paper
        L1 = self.Lambda1
        L2 = self.Lambda2

        # Simplfication of writing
        PhiT = Phi.transpose()

        # Denominator as a matrix
        scaling = np.linalg.inv(L1 + L2 * np.linalg.multi_dot( [ Phi, F, PhiT ]  ))

        # Scale denominator (9x9 if X is 9x1) to match F (369x369 if 40 hidden layer nodes)
        dg = np.diag(scaling)
        Sdiag = np.zeros(Phi.shape[1])
        for si, s in enumerate(dg):
            Sdiag[si*self.glength:(si+1)*(self.glength)] = s
        S = np.diag(Sdiag)

        # Evaluate
        F_plus = 1/L1 * (F  - L2 * np.linalg.multi_dot([F,PhiT,Phi,F,S]))
        #pdb.set_trace()
        return F_plus


    def updateTheta(self,Theta,F,Phi,error):
        Theta_plus = self.Theta + np.linalg.multi_dot([F,Phi.transpose(),error])
        return Theta_plus

    def W2Theta(self, W):
        Theta = None
        for i in range(0,len(W[1])):
            ti = np.vstack([W[0][:,i].reshape((-1,1)),W[1][i]])
            if Theta is None:
                Theta = ti
            else:
                Theta = np.vstack([Theta,ti])
        return Theta
