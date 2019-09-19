#author: Kevin Shen
import numpy as np
#Todo: 
#1) build in the Emulate/fitting functionality
#2) built in first two derivatives
#3) vectorize

class Spline:
    def __init__(self,rcut,knots):
        self.rcut = float(rcut)
        self.knots = knots
        self.nknot = len(knots)
        self.BFunc = np.array([[ 1.,  4.,  1.,  0.],
                  [-3.,  0.,  3.,  0.],
                  [ 3., -6.,  3.,  0.],
                  [-1.,  3., -3.,  1.]], float) / 6.

        self.Update(knots)

    def Update(self,knots=None):
        """Updates SPCoeff to use values of knots"""
        self.SPDist = [ self.rcut/self.nknot, self.nknot/self.rcut ]
        if knots is not None:
            self.knots = knots
        self.SPCoeff = np.zeros([4,self.nknot])
        self.SPCoeff = self.getSpCoeff(self.knots)

    def Val(self,r):
        if r > self.rcut:
            return 0
        x = r*self.SPDist[1]
        k = np.min( [np.floor(x).astype(int), self.nknot-1] )
        t = x-k
        v = self.SPCoeff[0,k] + t*( self.SPCoeff[1,k] + t*( self.SPCoeff[2,k] + t*self.SPCoeff[3,k] ) )
        return v
 
    def DVal(self,r):
        if r > self.rcut:
            return 0
        x = r*self.SPDist[1]
        k = np.min( [np.floor(x).astype(int), self.nknot-1] )
        t = x-k
        dv = ( self.SPCoeff[1,k] + t*( 2*self.SPCoeff[2,k] + t * 3 * self.SPCoeff[3,k] ) ) * self.SPDist[1]
        return dv
    
    def __call__(self,arg):
        return self.Val(arg)

    def getSpCoeff(self,knots):
        nk = len(knots)
        Y = np.zeros(nk+3, float)
        Y[1:-2] = knots
        #make Y(0) for zero zecond deriv
        Y[0] = 2.*Y[1] - Y[2]
        #make the last two knots for zero val and first deriv
        Y[-2] = -0.5 * Y[-3]
        Y[-1] = Y[-3]
        #calculate coefficients
        C = np.zeros((4, nk), float)
        for i in range(1, nk + 1):
            C[:,i-1] = np.dot(self.BFunc, Y[i-1:i+3])
        return C

    def dUdKnot(self,x):
        """Returns value of derivative @ `x` with respect to knot values
        Parameters
        ----------
        x: float
            Point at which to evaluate derivative. In natural units, not spline interval units.
        
        Returns
        -------
        ndarray
            vector/array of derivative @ `x` w.r.t. each knot

        Notes
        -----
        Spline is linear in knot values, :math:`sum_i k_i f_i(x)`
        """
        if x > self.rcut:
            return 0
        x = x * self.SPDist[1]
        i = min(int(x), self.nknot - 1)
        t = x - float(i)
        d = np.zeros(self.nknot, float)
        SPdm1 = 0.166666666666667 + t * (-0.5 + t * (0.5 - t * 0.166666666666667))
        SPd0 = 0.666666666666667 + t * t * (-1.0 + t * 0.5)
        SPdp1 = 0.166666666666667 + t * (0.5 + t * (0.5 - t * 0.5))
        SPdp2 = t * t * t * 0.166666666666667

        if i == 0:
            #note: y(-1) = 2 y(0) - y(1) for linearity condition
            d[i    ] = SPd0 + 2.0 * SPdm1
            d[i + 1] = SPdp1 - SPdm1
            d[i + 2] = SPdp2
        elif i == self.nknot - 1:
            #note: y(N) = -0.5 y(N-1) and y(N+1) = y(N-1) for zero slope and value condition
            d[i - 1] = SPdm1
            d[i    ] = SPd0 - 0.5 * SPdp1 + SPdp2
        elif i == self.nknot - 2:
            #note: y(N) = -0.5 y(N-1) and y(N+1) = y(N-1) for zero slope and value condition
            d[i - 1] = SPdm1
            d[i    ] = SPd0
            d[i + 1] = SPdp1 - 0.5 * SPdp2
        else:
            d[i - 1] = SPdm1
            d[i    ] = SPd0
            d[i + 1] = SPdp1
            d[i + 2] = SPdp2
        return d

    def fitDists(self,npt=4.0):
        """Returns default/recommended locations @ which to fit a function
        Parameters
        ----------
        npt: int
            Number of internal points per sub-interval to fit with. Default is 4.

        Returns
        -------
        ndarray
            array of points at which to fit
        """

        step = 1.0/float(npt)
        tVals = np.arange(step/2.0, self.nknot-1e-10, step)
        DistVals = tVals * self.SPDist[0]
        return DistVals

    def fitCoeff(self,DistVals, Vals, Weights=None):
        """Finds knot values that best fit a function.
        Parameters
        ----------
        DistVals: ndarray
            Nx1 array of distances @ which to fit
        Vals: ndarray
            Nx1 array of values @ pts in the DistVals array
        Weights: None, 1-D array
            If included, weighs the points to be used in the fitting

        Returns
        -------
        ndarray
            nknot x1 array of fitted knot values

        Notes
        -----
        Recommended to fit at multiple *internal* points per sub-interval. One default is to use 4 internal points.
        """
        #set up spline interval values
        tVals = DistVals/self.SPDist[0]
        iVals = tVals.astype(int)
        tVals = tVals-iVals
        rs = DistVals+1.0e-300

        #determine derivatives, which are the XCoefs
        XCoefs = np.zeros((len(DistVals), self.nknot), float)
        for (i,x) in enumerate(DistVals):
            XCoefs[i] = self.dUdKnot(x)

        #determine values by least squares. no weighing right now
        if Weights == None:
            knots = np.linalg.lstsq(XCoefs, Vals)[0]            
        else:
            knots = sim.mathutil.LeastSquaresFit(XCoefs, Vals)

        print("LSQ: {}".format(knots))

        #regularize right knot to go to zero smoothly
        self.Update(knots)

    def convert(self, new_rcut, new_nknot):
        """Convert current spline to a new rcut interval, new nknot"""
        sp = Spline(new_rcut, np.zeros(new_nknot))
        xfit = sp.fitDists()
        vals = np.zeros(xfit.shape)
        for ix,xv in enumerate(xfit):
                vals[ix] = self.Val(xv)
        sp.fitCoeff( xfit, vals )
        
        self.rcut = new_rcut
        self.knots = sp.knots
        self.nknot = len(sp.knots)

        self.Update()

