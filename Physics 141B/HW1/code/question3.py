import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt

# dealing with tight-binding ring

# HOPPING VALUES FOR PARTS 1 AND 2
# to = 1 # hopping from odd to even 
# te = 4 # hopping from even to odd

# HOPPING VALUES FOR PART 3
to = 4 # hopping from odd to even 
te = 1 # hopping from even to odd

N = 10 # number of sites

# Construct the Hamiltonian Matrix for the ring
H = np.zeros((N, N))

for i in range(N):
    if (i % 2 == 1):
        H[i][((i+1)%N)] = to
        H[i][((i-1)%N)] = to
    else:
        H[i][((i+1)%N)] = te
        H[i][((i-1)%N)] = te

# Break the ring -- COMMENT THIS OUT AND RUN FOR PART 1
H[N-2][N-1] = 0
H[N-1][N-2] = 0


# print(H)

vals, vecs = eig(H)
smallest_five_indices = np.argsort(vals)[:5]


amps = np.zeros((5,N))
for i in range(5):
    for j in range(N):
        amps[i][j] = np.abs(vecs[i][j])**2

# print(amps)
    
x_axis_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

print(amps)
for i in range(5):
    plt.plot(x_axis_vals, amps[i] + i, label=f'Site {i}')  # Offset each row by i

plt.xlabel('Index')
plt.ylabel('Value + Offset')
plt.title('Broken 10 site ring, with switched hopping parameters')
plt.legend()

# Save the plot
# plt.savefig("Q3plot.png", dpi=300)  # Save with high resolution

# Show that plot
plt.show()
