import math
import torch
import matplotlib.pyplot as plt

# ============================================================
# Discrete soap film / minimal surface with Adam
#
# We do NOT solve the minimal-surface PDE directly.
# We only minimize the total area of a triangulated surface.
# ============================================================

torch.set_default_dtype(torch.float64)

# Grid resolution: N x N vertices
N = 41

# Square domain: (x, y) in [-1, 1] x [-1, 1]
x = torch.linspace(-1.0, 1.0, N)
y = torch.linspace(-1.0, 1.0, N)
Y, X = torch.meshgrid(y, x, indexing="ij")

# ------------------------------------------------------------
# 1. Fixed wire-frame boundary
#
# top edge:    z = +A cos(pi x / 2)
# bottom edge: z = -A cos(pi x / 2)
# left/right:  z = 0
#
# At x = +/-1, cos(pi/2) = 0, so the four corners match.
# ------------------------------------------------------------
A = 0.8

boundary_z = torch.zeros_like(X)
profile = A * torch.cos(math.pi * x / 2.0)

boundary_z[-1, :] = +profile   # y = +1
boundary_z[0, :]  = -profile   # y = -1
boundary_z[:, 0]  = 0.0        # x = -1
boundary_z[:, -1] = 0.0        # x = +1

boundary_mask = torch.zeros((N, N), dtype=torch.bool)
boundary_mask[0, :] = True
boundary_mask[-1, :] = True
boundary_mask[:, 0] = True
boundary_mask[:, -1] = True

# ------------------------------------------------------------
# 2. Initial surface
#
# This interpolates between upper/lower boundaries, but is
# generally NOT the minimal-area surface.
# ------------------------------------------------------------
Z0 = A * Y * torch.cos(math.pi * X / 2.0)

# Add a small interior perturbation so Adam visibly relaxes it.
torch.manual_seed(1)
noise = 0.08 * torch.randn_like(Z0)
Z0 = Z0 + torch.where(boundary_mask, torch.zeros_like(noise), noise)

# Force boundary to be exact.
Z0 = torch.where(boundary_mask, boundary_z, Z0)

# Adam will optimize this full array; torch.where below freezes boundary.
Z_param = torch.nn.Parameter(Z0.clone())


# ------------------------------------------------------------
# 3. Construct current surface while keeping boundary fixed
# ------------------------------------------------------------
def current_Z():
    return torch.where(boundary_mask, boundary_z, Z_param)


# ------------------------------------------------------------
# 4. Exact area of the triangulated surface
#
# Every rectangular grid cell is split into two triangles:
#
# p00 ---- p10
#  |      / |
#  |    /   |
#  |  /     |
# p01 ---- p11
#
# triangle area = 1/2 |(b-a) x (c-a)|
# ------------------------------------------------------------
def surface_area(Z):
    P = torch.stack((X, Y, Z), dim=-1)

    p00 = P[:-1, :-1, :]
    p10 = P[:-1, 1:, :]
    p01 = P[1:, :-1, :]
    p11 = P[1:, 1:, :]

    cross1 = torch.cross(p10 - p00, p11 - p00, dim=-1)
    cross2 = torch.cross(p11 - p00, p01 - p00, dim=-1)

    area1 = 0.5 * torch.linalg.norm(cross1, dim=-1)
    area2 = 0.5 * torch.linalg.norm(cross2, dim=-1)

    return area1.sum() + area2.sum()


# ------------------------------------------------------------
# 5. Adam optimization
# ------------------------------------------------------------
optimizer = torch.optim.Adam([Z_param], lr=0.01)

epochs = 5000
history = []

initial_area = surface_area(current_Z()).item()

for epoch in range(epochs + 1):
    optimizer.zero_grad()

    Z = current_Z()
    area = surface_area(Z)

    area.backward()
    optimizer.step()

    history.append(area.item())

    if epoch % 250 == 0:
        print(
            f"epoch={epoch:5d}   "
            f"area={area.item():.10f}"
        )

Z_final = current_Z().detach()
final_area = surface_area(Z_final).item()

print("\n==============================")
print(f"initial area = {initial_area:.10f}")
print(f"final area   = {final_area:.10f}")
print(f"decrease     = {initial_area - final_area:.10f}")
print(f"relative     = {(initial_area-final_area)/initial_area*100:.4f}%")
print("==============================")

# Confirm boundary stayed fixed
boundary_error = torch.max(
    torch.abs(
        Z_final[boundary_mask] - boundary_z[boundary_mask]
    )
).item()

print(f"max boundary error = {boundary_error:.3e}")


# ------------------------------------------------------------
# 6. Save figures
# ------------------------------------------------------------
Xn = X.numpy()
Yn = Y.numpy()
Z0n = Z0.detach().numpy()
Zfn = Z_final.numpy()

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(Xn, Yn, Z0n, rstride=1, cstride=1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Initial surface")
fig.tight_layout()
fig.savefig("initial_surface.png", dpi=180)
plt.close(fig)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
ax.plot_surface(Xn, Yn, Zfn, rstride=1, cstride=1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Adam-optimized soap film")
fig.tight_layout()
fig.savefig("optimized_surface.png", dpi=180)
plt.close(fig)

fig = plt.figure(figsize=(8, 5))
plt.plot(history)
plt.xlabel("Adam step")
plt.ylabel("surface area")
plt.title("Area minimization")
plt.grid(True)
fig.tight_layout()
fig.savefig("area_history.png", dpi=180)
plt.close(fig)

print("\nSaved:")
print("  initial_surface.png")
print("  optimized_surface.png")
print("  area_history.png")
