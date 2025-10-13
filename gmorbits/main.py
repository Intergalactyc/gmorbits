from gmorbits.methods import (
    ExplicitEulerMethod,
    SymplecticEulerMethod,
)
from gmorbits.potentials import KeplerPotential
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from gmorbits.constants import G


def plot_result(times, result_x, result_v):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    ax.set_xlim(np.min(result_x[:, 0]), np.max(result_x[:, 0]))
    ax.set_ylim(np.min(result_x[:, 1]), np.max(result_x[:, 1]))
    ax.set_zlim(np.min(result_x[:, 2]), np.max(result_x[:, 2]))

    (trail,) = ax.plot([], [], [], "-", lw=2, color="tab:blue")
    (point,) = ax.plot([], [], [], "o", markersize=8, color="tab:red")

    def init():
        trail.set_data([], [])
        trail.set_3d_properties([])
        point.set_data([], [])
        point.set_3d_properties([])
        return trail, point

    def update(frame):
        # Full trail up to current time step
        trail.set_data(result_x[:frame, 0], result_x[:frame, 1])
        trail.set_3d_properties(result_x[:frame, 2])
        # Single point at current position
        point.set_data([result_x[frame, 0]], [result_x[frame, 1]])
        point.set_3d_properties([result_x[frame, 2]])
        return trail, point

    ani = FuncAnimation(
        fig,
        update,
        frames=result_x.shape[0],
        init_func=init,
        interval=1,
        repeat=False,
    )

    plt.show()
    # ani.save("animation.mp4", fps=30, dpi=150)


if __name__ == "__main__":
    M = 15

    pot = KeplerPotential(M)
    m = SymplecticEulerMethod(pot, 0.25)  # ExplicitEulerMethod(pot, 0.25)

    x0 = [10, 0, 0]
    r = np.linalg.norm(x0)
    vc = np.sqrt(G * M / r)
    T = 2 * np.pi * r / vc
    t, x, v = m.iterate(x0, [0, vc, 0], T * 2, "out.csv")

    plot_result(t, x, v)
