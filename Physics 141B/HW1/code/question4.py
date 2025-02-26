import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt

# dealing with tight-binding ring

# HOPPINGS FOR PART 1
# to = 1 # hopping from odd to even
# te = 4 # hopping from even to odd

# HOPPINGS FOR PART 2
to = 4 # hopping from odd to even
te = 1 # hopping from even to odd

N = 11 # number of sites

# Construct the Hamiltonian Matrix
H = np.zeros((N, N))

for i in range(N):
    if (i % 2 == 1):
        H[i][((i+1)%N)] = to
        H[i][((i-1)%N)] = to
    else:
        H[i][((i+1)%N)] = te
        H[i][((i-1)%N)] = te


# print(H)

vals, vecs = eig(H)

# for i in range(len(vals)):
#     print("Eigenvalue: ", vals[i])
#     print("")
#     print("Corresponding Eigenvector: ", vecs[i])
#     print("----------------------------------------------------")

amps = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        amps[i][j] = np.abs(vecs[i][j])**2
    
x_axis_vals = range(N)

print(amps)
for i in range(N):
    plt.plot(x_axis_vals, amps[i] + i, label=f'Site {i+1}')  # Offset each row by i

plt.xlabel('Index')
plt.ylabel('Value + Offset')
plt.title('Q4(ii)')
plt.legend()

# Save the plot
# plt.savefig("Q3plot.png", dpi=300)  # Save with high resolution

# Show that plot
plt.show()
