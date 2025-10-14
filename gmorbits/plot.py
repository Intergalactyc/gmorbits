import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os


def plot_result(result, dim: int = 2, saveto: os.PathLike = None):
    assert dim in {2, 3}, "Dimension must be 2 or 3"

    t, x, v, T, U, L = result

    Ln = np.apply_along_axis(np.linalg.norm, 1, L)

    fig = plt.figure(figsize=(12, 10))
    if dim == 3:
        ax = fig.add_subplot(2, 2, (1, 2), projection="3d")
    else:
        ax = fig.add_subplot(2, 2, (1, 2))
    eax = fig.add_subplot(2, 2, 3)
    lax = fig.add_subplot(2, 2, 4)

    xlow, ylow, zlow = np.min(x[:, :], axis=0)
    xhigh, yhigh, zhigh = np.max(x[:, :], axis=0)

    padx = max(0.1 * (xhigh - xlow), 1e-1)
    ax.set_xlim(xlow - padx, xhigh + padx)
    pady = max(0.1 * (yhigh - ylow), 1e-1)
    ax.set_ylim(ylow - pady, yhigh + pady)

    tlow = np.min(t)
    thigh = np.max(t)

    Elow = np.min(U)
    Ehigh = np.max(T)
    padE = max(0.1 * (Ehigh - Elow), 1e-2)
    eax.set_xlim(tlow, thigh)
    eax.set_ylim(Elow - padE, Ehigh + padE)

    Llow = np.min(Ln)
    Lhigh = np.max(Ln)
    padL = max(0.1 * (Lhigh - Llow), 1e-5)
    lax.set_xlim(tlow, thigh)
    lax.set_ylim(Llow - padL, Lhigh + padL)

    if dim == 3:
        padz = max(0.1 * (zhigh - zlow), 0.1)
        ax.set_zlim(zlow - padz, zhigh + padz)
        ax.set_zlabel("Z")
        (trail,) = ax.plot([], [], [], "--", lw=1, color="tab:blue")
        (point,) = ax.plot([], [], [], "o", markersize=8, color="tab:red")
    else:
        (trail,) = ax.plot([], [], "--", lw=1, color="tab:blue")
        (point,) = ax.plot([], [], "o", markersize=8, color="tab:red")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    (Tplot,) = eax.plot([], [], "--", lw=1, color="tab:blue", label="Kinetic Energy")
    (Uplot,) = eax.plot([], [], "--", lw=1, color="tab:green", label="Potential Energy")
    (Eplot,) = eax.plot([], [], "-", lw=2, color="tab:red", label="Total Energy")
    eax.legend()
    eax.set_xlabel("Time")
    eax.set_ylabel("Specific Energy")

    (Lplot,) = lax.plot(
        [], [], "-", lw=3, color="tab:red", label="||Angular Momentum||"
    )
    lax.legend()
    lax.set_xlabel("Time")
    lax.set_ylabel("Specific Angular Momentum Norm")

    def init():
        for p in [trail, point, Tplot, Uplot, Eplot, Lplot]:
            p.set_data([], [])

        if dim == 3:
            trail.set_3d_properties([])
            point.set_3d_properties([])

        return trail, point, Tplot

    def update(frame):
        trail.set_data(x[:frame, 0], x[:frame, 1])
        point.set_data([x[frame, 0]], [x[frame, 1]])
        Tplot.set_data(t[:frame], T[:frame])
        Uplot.set_data(t[:frame], U[:frame])
        Eplot.set_data(t[:frame], T[:frame] + U[:frame])
        Lplot.set_data(t[:frame], Ln[:frame])

        if dim == 3:
            point.set_3d_properties([x[frame, 2]])
            trail.set_3d_properties(x[:frame, 2])

        return trail, point, Tplot

    fig.tight_layout()

    ani = FuncAnimation(
        fig,
        update,
        frames=x.shape[0],
        init_func=init,
        interval=10,
        repeat=False,
    )

    plt.show()
    if saveto:
        ani.save("animation.mp4", fps=30, dpi=150)
