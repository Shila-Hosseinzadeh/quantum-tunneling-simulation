import numpy as np

# Get system parameters from user
N = int(input("Number of molecules (N): "))
n = int(input("Atoms per molecule (n): "))

# Get energy and hopping parameters
print(f"\nEnter {n} on-site energies (space separated):")
epsilons = list(map(float, input().split()))

print(f"\nEnter {n} upper hopping values (space separated):")
upper_hops = list(map(float, input().split()))

print(f"\nEnter {n} lower hopping values (space separated):")
lower_hops = list(map(float, input().split()))

# Validate input lengths
if len(epsilons) != n or len(upper_hops) != n or len(lower_hops) != n:
    print("Error: All input lists must match length n!")
    exit()

# Initialize all matrices
total_atoms = N * n
H_full = np.zeros((total_atoms, total_atoms))      # Full Hamiltonian
H_diagonal = np.zeros((total_atoms, total_atoms))  # Diagonal-only part
H_upper_full = np.zeros((total_atoms, total_atoms)) # Upper triangular part
H_lower_full = np.zeros((total_atoms, total_atoms)) # Lower triangular part
H_ground = np.zeros((n, n))                       # Single molecule Hamiltonian
upper_base = np.zeros((n, n))                     # Base upper hopping matrix
lower_base = np.zeros((n, n))                     # Base lower hopping matrix

# Build base hopping matrices (periodic boundary for single molecule)
upper_base[-1, 0] = upper_hops[-1]  # Only the last hopping term
lower_base[0, -1] = lower_hops[-1]  # Only the first hopping term

# Construct full Hamiltonian
for mol in range(N):
    start = mol * n

    # On-site energies (diagonal)
    np.fill_diagonal(H_full[start:start+n, start:start+n], epsilons)
    np.fill_diagonal(H_diagonal[start:start+n, start:start+n], epsilons)

    # Intra-molecular hoppings
    for i in range(n-1):
        # Upper diagonal terms
        H_full[start+i, start+i+1] = upper_hops[i]
        H_upper_full[start+i, start+i+1] = upper_hops[i]

        # Lower diagonal terms
        H_full[start+i+1, start+i] = lower_hops[i]
        H_lower_full[start+i+1, start+i] = lower_hops[i]

    # Inter-molecular hoppings (periodic boundary)
    if mol < N-1:
        H_full[start+n-1, start+n] = upper_hops[-1]
        H_upper_full[start+n-1, start+n] = upper_hops[-1]

        H_full[start+n, start+n-1] = lower_hops[-1]
        H_lower_full[start+n, start+n-1] = lower_hops[-1]

# Build single molecule Hamiltonian
np.fill_diagonal(H_ground, epsilons)
for i in range(n-1):
    H_ground[i, i+1] = upper_hops[i]
    H_ground[i+1, i] = lower_hops[i]
# Add periodic boundary for single molecule
#H_ground[-1, 0] = upper_hops[-1]
#-H_ground[0, -1] = lower_hops[-1]


# Calculate all eigenvalues
full_eigenvalues = np.linalg.eigvalsh(H_full)
diag_eigenvalues = np.linalg.eigvalsh(H_diagonal)
ground_eigenvalues = np.linalg.eigvalsh(H_ground)



# Display comprehensive results
print("\nCOMPREHENSIVE RESULTS:")
print("=====================")
print(f"\n1. Base Matrices ({n}×{n}):")
print("Upper base (periodic):\n", upper_base)
print("Lower base (periodic):\n", lower_base)

print(f"\n2. Single Molecule Hamiltonian ({n}×{n}):")
print(H_ground)

print(f"\n3. Full System Hamiltonian ({total_atoms}×{total_atoms}):")
print("Diagonal part:\n", H_diagonal)
print("Upper triangular part:\n", H_upper_full)
print("Lower triangular part:\n", H_lower_full)
print("Combined Hamiltonian:\n", H_full)


print("\n4. Eigenvalue Analysis:")
print("Single molecule energies:", np.round(ground_eigenvalues, 4))
print("Full system energies:", np.round(full_eigenvalues, 4))
