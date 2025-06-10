import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap

# parameters
size = 200
du, dv = 0.16, 0.08
f, k = 0.025, 0.055
dt = 1.0

# initialize
U = np.ones((size, size))
V = np.zeros((size, size))

# seed small square
# r = 20
# U[size//2 - r:size//2 + r, size//2 - r:size//2 + r] = 0.50
# V[size//2 - r:size//2 + r, size//2 - r:size//2 + r] = 0.25

# add random noise
U += 0.05 * np.random.randn(size, size)
V += 0.05 * np.random.randn(size, size)

# laplacian operator
def laplacian(X):
    return (
        -X +
        0.2 * (np.roll(X, 1, 0) + np.roll(X, -1, 0) +
               np.roll(X, 1, 1) + np.roll(X, -1, 1)) +
        0.05 * (np.roll(np.roll(X, 1, 0), 1, 1) + np.roll(np.roll(X, 1, 0), -1, 1) +
                np.roll(np.roll(X, -1, 0), 1, 1) + np.roll(np.roll(X, -1, 0), -1, 1))
    )

# update
def update():
    global U, V
    Lu = laplacian(U)
    Lv = laplacian(V)
    reaction = U * V * V
    U += (du * Lu - reaction + f * (1 - U)) * dt
    V += (dv * Lv + reaction - (f + k) * V) * dt
    # keep values bounded
    np.clip(U, 0, 1, out=U)
    np.clip(V, 0, 1, out=V)

# plot
fig, ax = plt.subplots()
pink_cmap = LinearSegmentedColormap.from_list('bright_pink', ['white', "#ff01ee"])
img = ax.imshow(V, cmap=pink_cmap, vmin=0, vmax=1)
ax.axis('off')

# animation
def animate(frame):
    for _ in range(50):
        update()
    img.set_data(V)
    return [img]

ani = FuncAnimation(fig, animate, frames=1000, interval=50, blit=True)
plt.tight_layout()
plt.show()
