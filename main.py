from scipy import signal
from scipy import linalg
from scipy import integrate
import math
import numpy as np

def generateCellEigComplexReal(eig_complex, eig_real):
    eigenCell = []

    for i in range(0,len(eig_complex), 2):
        array = np.zeros(shape=(2,2), dtype=object)
        array[0,0] = eig_complex[i].imag
        array[0,1] = eig_complex[i].real
        array[1,0] = eig_complex[i+1].imag
        array[1,1] = eig_complex[i+1].real
        eigenCell.append(array)
    
    for i in eig_real:
        array = np.zeros(shape=(1,1), dtype=object)
        array[0,0] = i.real
        eigenCell.append(array)

    return eigenCell

def splitEigsComplexReal(eigen):
    return [x for x in eigen if x.imag!=0], [x for x in eigen if x.imag==0]

def generateLuenbergerD(eigen):
    eig_complex, eig_real = splitEigsComplexReal(eigen)

    if(any(~np.isnan(eig_complex))):
        eig_complex = sorted(eig_complex)
    
    eigenCell = generateCellEigComplexReal(eig_complex, eig_real)

    D = linalg.block_diag(*eigenCell[:])

    return D

    
# function dwdt = nonlin(t,w,f,g,h,u,D,F)
        # x = w(1:dimx);
        # dwdt = [f(x); 
                # zeros(size(D,2),1);] + [zeros(dimx) zeros(dimx, size(D,2));
                                        # zeros(size(D,2), dimx) D;]*w + [zeros(dimx,1);
                                                                                # F;]*h(x) + [g(x);
                                                                                            # zeros(size(D,2),1);]*u(t);                                                                     
    # end

def performMultipleLuenbergerSimulations(f,g,h,dimx,u,D,F,w0_array,nsims,tsim,dt):
    def dxdt(t,x):
        return f(x) + g(x) * u(t)

    def dzdt(t,z):
        return D*z+F*h(x)
    #     x = w[0:dimx]
    #     A = np.concatenate((f(x),np.zeros((D.shape[1],1))))
    #     B = np.concatenate((np.zeros((dimx,dimx)),np.zeros((D.shape[1],dimx)))) 
    #     C = np.concatenate((np.zeros((dimx,D.shape[1])),D))
    #     Z = np.concatenate((B,C), axis=-1)
    #     M = np.concatenate((np.zeros((dimx,1)), F))
    #     N = np.concatenate((g(x), np.zeros((D.shape[1],1))))
    #     h_x = h(x)
    #     u_t = u(t)
    #     res =  A + Z * w + M * h_x + N * u_t 

    tspan = (0, tsim)
    tq = np.arange(0,tsim,dt)
    output_data = np.zeros(shape=(len(tq), dimx + D.shape[1], nsims))

    for i in range(nsims):
        x0 = w0_array[0:2, i]
        z0 = w0_array[2:5, i]
        x_hat = integrate.solve_ivp(dxdt, tspan, w0)
        z = integrate.solve_ivp(dzdt, tspan, z0)

        # wq = interp1(t,w,tq)



if __name__ == "__main__":

    f = lambda x: np.array([x[1]**3, -x[0]])
    h = lambda x: x[0]
    g = lambda x: np.array([0,0]);
    u = lambda x: 0;

    dimx = 2

    b, a = signal.bessel(3,2*math.pi, 'low', analog=True, norm='phase')

    eigen = np.roots(a)

    D = generateLuenbergerD(eigen)
    F = np.array([[1],[1],[1]])

    nsims = 10;

    tsim = 40;

    dt = 1e-2;

    w0_array = np.zeros(shape=(dimx + D.shape[1], nsims));

    for i in range(nsims):
        w0_array[0,i] = 0.1*(i+1);

    performMultipleLuenbergerSimulations(f,g,h,dimx,u,D,F,w0_array,nsims,tsim,dt)

