import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import numpy as np
import os
from gmorbits.constants import _EPSILON


_n = "\n"


def animate_orbit(
    t,
    x,
    title: str = "",
    saveto: os.PathLike = None,
    *,
    stride: int = 3,
    trails: bool = True,
    track: bool = False,
    track_index: int = 0,
    particle_color=(1, 0, 0),
    trail_length: int = None,
    fade_length: int = 100,
    min_alpha: float = 0.25,
    trail_color=(0, 0, 1),
    min_frame_delay: int = 25,
):
    # apply frame skipping
    t = t[::stride]
    x = x[::stride]

    m, n, _ = x.shape

    fig, ax = plt.subplots()

    if trail_length is None:
        trail_length = m + 1

    if track:
        relative = x[:, track_index, :][:, np.newaxis, :]
        positions = x - relative
    else:
        positions = x

    scat = ax.scatter(
        positions[0, :, 0],
        positions[0, :, 1],
        s=50,
        zorder=3,
        c=[particle_color for _ in range(n)],
    )

    if trails:
        _trails = []
        for _ in range(n):
            lc = LineCollection([], lw=2)
            ax.add_collection(lc)
            _trails.append(lc)

    ax.set_xlim(positions[:, :, 0].min() - 0.1, positions[:, :, 0].max() + 0.1)
    ax.set_ylim(positions[:, :, 1].min() - 0.1, positions[:, :, 1].max() + 0.1)

    def update(frame):
        scat.set_offsets(positions[frame])

        if trails:
            start = max(0, frame - trail_length + 1)
            for i, lc in enumerate(_trails):
                pts = positions[start : frame + 1, i, :]
                if len(pts) < 2:
                    lc.set_segments([])
                    continue

                segments = np.stack([pts[:-1], pts[1:]], axis=1)
                n_seg = len(segments)

                alphas = np.full(n_seg, min_alpha)
                fade_len = min(fade_length, n_seg)
                if fade_len > 0:
                    alphas[-fade_len:] = np.linspace(min_alpha, 1.0, fade_len)

                colors = np.zeros((n_seg, 4))
                colors[:, :3] = trail_color
                colors[:, 3] = alphas

                lc.set_segments(segments)
                lc.set_colors(colors)

        ax.set_title(f"{title}{_n if title else ''}Time = {t[frame]:.2f}")
        return (scat, *_trails) if trails else (scat,)

    anim = FuncAnimation(fig, update, frames=m, interval=min_frame_delay, blit=False)

    if saveto:
        anim.save(
            filename=saveto,
            writer="pillow",
            fps=len(t) / (t[-1] - t[0]),
            # progress_callback=lambda i, n: print(f"Saving frame {i}/{n}"),
        )
        plt.close()
    else:
        plt.show()


def plot_energies(t, T, U, title: str = "", saveto: os.PathLike = None):
    E = T + U

    fig, ax = plt.subplots()
    ax.plot(t, T, label="Kinetic energy")
    ax.plot(t, U, label="Potential energy")
    ax.plot(t, E, label="Total energy")
    ax.legend()

    ax.set_xlabel("Time")
    ax.set_ylabel("Energy")
    ax.set_title(f"{title}{_n if title else ''}Total energy over time")

    fig.tight_layout()

    if saveto:
        plt.savefig(saveto, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_angular_momentum(t, L, title: str = "", saveto: os.PathLike = None):
    fig, ax = plt.subplots()
    if (Ldim := L.shape[1]) == 1:
        ax.plot(t, L[:, 0])
    elif Ldim == 3:
        ax.plot(t, L[:, 0], label="Lx")
        ax.plot(t, L[:, 1], label="Ly")
        ax.plot(t, L[:, 2], label="Lz")
        ax.legend()
    else:
        raise Exception(f"Unrecognized shape of L (Ldim {Ldim})")

    ax.set_xlabel("Time")
    ax.set_ylabel("Angular Momentum")
    ax.set_title(f"{title}{_n if title else ''}Total angular momentum over time")

    if saveto:
        plt.savefig(saveto, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_twoform(t, w, title: str = "", saveto: os.PathLike = None):
    fig, ax = plt.subplots()

    _w = w / _EPSILON**2

    # werr = np.abs(_w - _w[0])
    ax.plot(t, _w, label=r"$\omega(\delta_1,\delta_2)$")
    # ax.plot(t, werr, label="absolute error")

    ax.set_xlabel("Time")
    ax.set_ylabel("Two-form, normalized to initial value of 1")
    ax.set_title(
        f"{title}{_n if title else ''}Symplectic two-form of propagated pair of vectors over time"
    )

    ax.legend()

    if saveto:
        plt.savefig(saveto, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_result(
    result,
    cname: str = None,
    mname: str = None,
    dirpath: os.PathLike = None,
    dim: int = 2,
    *,
    trails: str | bool = "auto",
    **kwargs,
):
    assert dim in {2, 3}, "Dimension must be 2 or 3"

    t, x, v, T, U, L, w = result

    if dirpath is not None:
        subdir = os.path.join(dirpath, f"{cname.lower().replace(' ', '_')}")
        os.makedirs(subdir, exist_ok=True)

    def fpath(name):
        if dirpath is None:
            return None
        return os.path.join(subdir, f"{mname.lower().replace(' ', '_')}-{name}")

    if isinstance(trails, str) and trails.lower() == "auto":
        trails = x.shape[1] < 4

    title = f"{cname}: {mname}"
    animate_orbit(t, x, saveto=fpath("orbit.gif"), trails=trails, **kwargs)
    plot_energies(t, T, U, title, fpath("energies.png"))
    plot_angular_momentum(t, L, title, fpath("angularMomentum.png"))
    if w is not None:
        plot_twoform(t, w, title, fpath("2form.png"))
