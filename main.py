# also see https://github.com/xnx/collision
from collections.abc import Sequence, MutableSequence, Iterable
from dataclasses import dataclass

import matplotlib.axes
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
from matplotlib.patches import Circle
from matplotlib import animation
from itertools import combinations


@dataclass
class Particle:
    """A class representing a two-dimensional particle."""

    r: npt.NDArray[np.single]
    v: npt.NDArray[np.single]
    radius: float
    mass: float
    style: dict | None  # keyword arguments to the Circle constructor


def is_overlapping(
    r1: npt.NDArray[np.single], rad1: float, r2: npt.NDArray[np.single], rad2: float
) -> bool:
    return np.linalg.norm(r1 - r2) < rad1 + rad2


def draw(ax: matplotlib.axes.Axes, p: Particle):
    circle = Circle(xy=tuple(p.r.tolist()), radius=p.radius, **p.style)
    ax.add_patch(circle)
    return circle


def place_particle(
    particles: Sequence[Particle], radius: float, mass: float, style: dict | None
) -> Particle:
    """Place single particle."""
    while True:
        r: npt.NDArray[np.single] = radius + (1 - 2 * radius) * np.random.random(2)
        v_r = 0.1 * np.sqrt(np.random.random()) + 1.0
        v_phi = 2 * np.pi * np.random.random()
        v: npt.NDArray[np.single] = np.array([v_r * np.cos(v_phi), v_r * np.sin(v_phi)])
        if not any(is_overlapping(r, radius, p2.r, p2.radius) for p2 in particles):
            return Particle(r=r, v=v, radius=radius, mass=mass, style=style)


def place_particles(
    particles: MutableSequence[Particle], n: int, radius: float, mass: float, style=None
) -> None:
    """Place multiple particles (all with same radius, mass, and style)."""
    for _ in range(n):
        particles.append(place_particle(particles, radius, mass, style))


def advance(p: Particle, dt: float) -> None:
    """Advance the Particle's position forward in time by dt."""
    p.r += dt * p.v


def collide(p1: Particle, p2: Particle) -> None:
    """Update speed under elastic collision."""
    m1, m2 = p1.mass, p2.mass
    total_m = m1 + m2
    r1, r2 = p1.r, p2.r
    d = np.linalg.norm(r1 - r2) ** 2
    v1, v2 = p1.v, p2.v
    u1 = v1 - 2 * m2 / total_m * np.dot(v1 - v2, r1 - r2) / d * (r1 - r2)
    u2 = v2 - 2 * m1 / total_m * np.dot(v2 - v1, r2 - r1) / d * (r2 - r1)
    p1.v = u1
    p2.v = u2


def handle_collisions(particles: Sequence[Particle]) -> None:
    """Detect and handle any collisions between the Particles."""
    pairs = combinations(range(len(particles)), 2)
    for i, j in pairs:
        p1, p2 = particles[i], particles[j]
        if is_overlapping(p1.r, p1.radius, p2.r, p2.radius):
            collide(p1, p2)


def handle_boundary_collisions(p):
    """Bounce the particles off the walls elastically."""
    for i in [0, 1]:
        if p.r[i] - p.radius < 0:
            p.r[i] = p.radius
            p.v[i] = -p.v[i]
        if p.r[i] + p.radius > 1:
            p.r[i] = 1 - p.radius
            p.v[i] = -p.v[i]


def advance_all(particles: Sequence[Particle], dt: float) -> None:
    """Advance the animation by dt, returning the updated Circles list."""
    for i, p in enumerate(particles):
        advance(p, dt)
        handle_boundary_collisions(p)
    handle_collisions(particles)


def animate(
    particles: Sequence[Particle], dt: float, frames: int, steps: int = 1
) -> None:
    fig, ax = plt.subplots()
    for s in ["top", "bottom", "left", "right"]:
        ax.spines[s].set_linewidth(2)
    ax.set_aspect("equal", "box")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    circles: Sequence[Circle] = [draw(ax, p) for p in particles]

    def init() -> Iterable[Artist]:
        return circles

    def update(frame) -> Iterable[Artist]:
        for _ in range(steps):
            advance_all(particles, dt / steps)
        for c, p in zip(circles, particles):
            c.center = p.r
        return circles

    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=frames,
        interval=1,
        blit=True,
    )
    anim.save("particles.mp4", writer="ffmpeg", fps=60)


def main() -> None:
    particles = [
        Particle(np.array([0.5, 0.5]), np.array([0, 0]), 0.1, 2.0, {"color": "red"})
    ]
    place_particles(particles, 200, 0.01, 1.0, {"color": "blue"})
    animate(particles, 0.005, 600, 8)


if __name__ == "__main__":
    main()
