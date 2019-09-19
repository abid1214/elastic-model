from lattice import WSe
import numpy as np
import matplotlib.pyplot as plt
import pickle

def mc_minimize(lattice, epochs=1, all=True):
    moves = 0
    per = 10000

    while moves < 2*lattice.N*epochs:

        i, j = np.random.randint(1, lattice.L-1, size=2)
        idx = lattice.ij_to_idx(i,j) + int(np.random.random() < 0.5)*lattice.N

        x_old = lattice.get_position(idx)
        x = lattice.get_min_pos(idx)

        if np.linalg.norm(x - x_old)/lattice.d > 1e-5:
            lattice.move_atom(idx, x)
            if moves%per == 0:
                print(moves/(2*lattice.N*epochs))
                print("\tenergy = {}\n".format(lattice.energy()))
                f = lattice.plot_lattice(title=str(int(moves/per)), all=all)
                f.savefig("plots/{}.png".format(str(int(moves/per)).zfill(4)))
                plt.close()
            moves += 1

if __name__ == '__main__':
    l, x0, k = 50, 1.01, 1
    L = WSe(l, k, x0=x0, periodic=True)
    idx = L.ij_to_idx(int(l/2),int(l/2))
    L.remove_atom(idx+L.N)
    mc_minimize(L, epochs=500, all=False)
    pickle.dump(L, open("lattice_50.p", 'wb'))


