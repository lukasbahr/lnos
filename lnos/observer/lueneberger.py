import numpy as np
from scipy import linalg, integrate, interpolate
from torchdiffeq import odeint
import torch

class LuenebergerObserver():

    def __init__(self, dim_x: int, dim_y: int, f: callable, g: callable, h: callable, u: callable):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = self.dim_y * (self.dim_x + 1)

        self.f = f
        self.g = g
        self.h = h
        self.u = u

        self.F = torch.zeros((self.dim_z, 1))
        self.eigenD = torch.zeros((self.dim_z, 1))
        self.D = torch.zeros((self.dim_z, self.dim_z))

    def tensorDFromEigen(self, eigen: torch.tensor) -> torch.tensor:
        """
        Generates the state matrix D to use in a Luenberger
        observer given the desired eigenvalues.
        """
        self.eigenD = eigen
        eig_complex, eig_real = [x for x in eigen if x.imag != 0], [
            x for x in eigen if x.imag == 0]

        if(any(~np.isnan(eig_complex))):
            eig_complex = sorted(eig_complex)
            eigenCell = self.eigenCellFromEigen(eig_complex, eig_real)
            D = linalg.block_diag(*eigenCell[:])

            return torch.tensor(D,dtype=torch.float32)

    @staticmethod
    def eigenCellFromEigen(eig_complex: torch.tensor, eig_real: torch.tensor) -> []:
        """
        Generates a cell array containing 2X2 real
        matrices for each pair of complex conjugate eigenvalues passed in the
        arguments, and real scalar for each real eigenvalue.
        """
        eigenCell = []

        for i in range(0, len(eig_complex), 2):
            array = np.zeros(shape=(2, 2))
            array[0, 0] = eig_complex[i].real
            array[0, 1] = eig_complex[i].imag
            array[1, 0] = eig_complex[i+1].imag
            array[1, 1] = eig_complex[i+1].real
            eigenCell.append(array)

        for i in eig_real:
            array = np.zeros(shape=(1, 1))
            array[0, 0] = i.real
            eigenCell.append(array)

        return eigenCell

    def simulateLueneberger(self, y_0: torch.tensor, tsim: tuple, dt) -> [torch.tensor, torch.tensor]:
        """
        Runs and outputs the results from 
        multiple simulations of an input-affine nonlinear system driving a 
        Luenberger observer target system.
        """
        def dydt(t, y):
            y = y.float()
            x = y[0:self.dim_x]
            z = y[self.dim_x:len(y)]
            x_dot = self.f(x) + self.g(x) * self.u(t)
            z_dot = torch.matmul(self.D, z)+self.F*self.h(x)
            return torch.cat((torch.tensor(x_dot), z_dot))

        # Output timestemps of solver
        tq = torch.arange(tsim[0], tsim[1], dt)

        # Solve
        sol = odeint(dydt, y_0, tq)

        return tq, sol
