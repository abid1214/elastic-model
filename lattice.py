from math import cos, sin, pi
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class WSe:

    ################ INIT ##################################
    def __init__(self, L, k, d=1, x0=0, periodic=False):
        self.L  = L
        self.k  = k
        self.x0 = x0
        self.N = L*L
        self.d = d
        self.periodic = periodic
        self.remove_list = []

        a1 = np.array([0,d])
        a2 = np.array([-d*cos(pi/6), d*sin(pi/6)])

        b1 = np.array([-d*cos(pi/6), d*(1 + sin(pi/6))])
        b2 = np.array([2*d*cos(pi/6), 0])

        self.W_list  = np.array([     i*b1 + j*b2 for j in range(L) for i in range(L)])
        self.Se_list = np.array([a1 + i*b1 + j*b2 for j in range(L) for i in range(L)])

        self.W_orig_list  = np.copy(self.W_list)
        self.Se_orig_list = np.copy(self.Se_list)
    ########################################################


    ################ PLOTTING ##############################
    def plot_lattice(self, title='', fig=None, all=False):
        if not fig:
            fig = plt.figure()
        ax = plt.subplot(111)
        ax.scatter(self.W_list[:,0], self.W_list[:,1], color='k')
        ax.scatter(self.Se_list[:,0], self.Se_list[:,1], color='r')
        if len(self.remove_list)>0:
            self.plot_points(self.remove_list, 'w')
        if all:
            ax.axis([-self.d*cos(pi/6)*self.L, 2*self.d*cos(pi/6)*self.L, \
                    0, self.d*(1+sin(pi/6))*self.L])
        else:
            l = int(self.L/2)
            idx = self.ij_to_idx(l,l)
            cen_pos = self.get_position(idx+self.N)
            ax.axis('equal')
            r= l*self.d/2
            ax.axis([cen_pos[0] - r, cen_pos[0] + r, cen_pos[1] - r, cen_pos[1] + r])

        plt.title(title)
        return fig

    def plot_points(self, idx_list, c='k'):
        idx_list = np.array(idx_list)
        W_idx  = idx_list[idx_list <  self.N]
        Se_idx = idx_list[idx_list >= self.N] - self.N

        W_list  = np.array([self.W_list[idx]  for idx in W_idx])
        Se_list = np.array([self.Se_list[idx] for idx in Se_idx])

        ax = plt.subplot(111)
        if len(W_list) > 0:
            ax.scatter(W_list[:,0], W_list[:,1], color=c)
        if len(Se_list) > 0:
            ax.scatter(Se_list[:,0], Se_list[:,1], color=c)

    ########################################################

    def _nb_W_edge_cases(self, idx, nb):
        (i,j) = self.idx_to_ij(idx)
        if i == 0:
            return [nb[0]]
        elif j == 0:
            return [nb[0], nb[1]]
        else:
            return nb

    def _nb_Se_edge_cases(self, idx, nb):
        (i,j) = self.idx_to_ij(idx)
        if i == self.L - 1:
            return [nb[0]]
        elif j == self.L - 1:
            return [nb[0], nb[1]]
        else:
            return nb

    def neighbors(self, idx):
        '''
        gives the indices of the neighbors of a particle labled by idx
        if idx < N, then it denotes a W atom, else it denotes an Se atom.
        '''
        if idx in self.remove_list:
            return []
        if idx < self.N:
            #atom is W, so neighbors are Se
            nb_list = [idx, idx - 1, idx - (self.L + 1)]
            nb_list = np.array(self._nb_W_edge_cases(idx, nb_list))
            nb_list =  nb_list + self.N
        else:
            #atom is Se, so neighbors are W
            i = idx%self.N
            nb_list = [i, i + 1, i + (self.L + 1)]
            nb_list = np.array(self._nb_Se_edge_cases(idx, nb_list))
        return list(set(nb_list) - set(self.remove_list))

    def idx_to_ij(self, idx):
        i = idx%self.N
        return (i%self.L, int(i/self.L))

    def ij_to_idx(self, i, j):
        return self.L*j + i

    def get_position(self, idx):
        atom_list = self.W_list if idx < self.N else self.Se_list
        return atom_list[idx%self.N]


    def U_ij(self, i, j):
        '''
        gives the energy between an Se site and W site. One of i or j
        must be less than N and the other must be greater than N
        '''
        assert((i in range(self.N) and j in range(self.N, 2*self.N)) or \
               (j in range(self.N) and i in range(self.N, 2*self.N)))

        (W_idx, Se_idx) = (i, j - self.N) if i < self.N else (j,i - self.N)
        x = np.linalg.norm(self.W_list[W_idx] - self.Se_list[Se_idx])
        return .5*self.k*(x - self.x0)*(x - self.x0)

    def energy(self):
        W_list = [self.ij_to_idx(i, j) for i in range(1, self.L) for j in range(1, self.L)] \
                if self.periodic else range(self.N)
        return sum([self.U_ij(i,j) for i in W_list for j in self.neighbors(i)])

    def E_diff(self, idx, x):
        atom_list = self.W_list if idx < self.N else self.Se_list
        old_x = atom_list[idx%self.N]
        nb = self.neighbors(idx)

        E_old = sum([self.U_ij(idx, n) for n in nb])
        self.move_atom(idx, x)

        E_new = sum([self.U_ij(n, idx) for n in nb])
        self.move_atom(idx, old_x)
        return E_new - E_old

    def get_min_pos(self, idx):
        x0 = self.get_position(idx)
        def U(x):
            nb      = self.neighbors(idx)
            nb_pos  = [ self.get_position(n) for n in nb ]
            x_list  = [np.linalg.norm(n - x) for n in nb_pos]
            delta_x = np.array([x - self.x0 for x in x_list])
            return .5*self.k*np.dot(delta_x, delta_x)

        def gradU(x):
            nb = self.neighbors(idx)
            nb_pos = [ self.get_position(n) for n in nb ]
            x_list = [ np.linalg.norm(x-n) for n in nb_pos ]
            return self.k*sum([(x - nb_pos[i])*(x_list[i] - self.x0)/x_list[i] for i in range(len(nb))])

        return minimize(U, x0, jac=gradU).x


    def move_atom(self, idx, x):
        atom_list = self.W_list if idx < self.N else self.Se_list
        delta_x = x - atom_list[idx%self.N]
        atom_list[idx%self.N] = x
        if self.periodic:
            (i, j) = self.idx_to_ij(idx)
            if i == 0:
                per_idx = self.ij_to_idx(self.L - 1, j)
            elif i == self.L - 1:
                per_idx = self.ij_to_idx(0, j)
            elif j == 0:
                per_idx = self.ij_to_idx(i, self.L - 1)
            elif j == self.L - 1:
                per_idx = self.ij_to_idx(i, 0)
            else:
                return
            atom_list[per_idx] += delta_x

    def remove_atom(self, idx):
        self.remove_list.append(idx)

    def get_XY_list(self):
        W_X, W_Y   = list(zip(*self.W_orig_list))
        Se_X, Se_Y = list(zip(*self.Se_orig_list))

        return W_X + Se_X,  W_Y + Se_Y

    def get_displacement_field(self, mean_sub = False):
        W_diff_list  = self.W_list  - self.W_orig_list
        Se_diff_list = self.Se_list - self.Se_orig_list

        Dx = np.concatenate((W_diff_list[:,0], Se_diff_list[:,0]))
        Dy = np.concatenate((W_diff_list[:,1], Se_diff_list[:,1]))

        if mean_sub:
            Dx -= np.mean(Dx)
            Dy -= np.mean(Dy)

        return Dx, Dy

    def get_strain_fields(self, mean_sub=False):
        d, L = self.d, self.L
        dx = d/32.
        xi = np.arange(-d*cos(pi/6)*L, 2*d*cos(pi/6)*L, dx)
        yi = np.arange(0, d*(1+sin(pi/6))*L, dx)
        xi,yi = np.meshgrid(xi,yi)

        x, y = get_XY_list()
        Dx, Dy = self.get_displacement_field(mean_sub)

        Dxi = griddata((x,y),Dx,(xi,yi),method='cubic')
        Dyi = griddata((x,y),Dy,(xi,yi),method='cubic')

        Dxi[np.isnan(Dxi)] = 0
        Dyi[np.isnan(Dyi)] = 0

        Dxi = gaussian_filter(Dxi, .5*L.d/dx)
        Dyi = gaussian_filter(Dyi, .5*L.d/dx)

        exxi = np.gradient(Dxi, dx)[1]
        eyyi = np.gradient(Dyi, dx)[0]

        return xi, yi, exxi, eyyi
