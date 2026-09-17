from typing import List
from matplotlib.animation import FuncAnimation
import numpy as np
from src.utils.types import transform
from src.wavelet import Wavelet
from matplotlib import pyplot as plt


class StandingPulse:
    def __init__(self, coefficients: np.ndarray, wave_vectors: np.ndarray | float, phases: np.ndarray | float):
        n_wavelets = len(coefficients)
        self.shape = (n_wavelets, 2)
        self.coefficients = transform(coefficients, self.shape)
        self.kss = transform(wave_vectors, self.shape)
        self.phases = transform(phases, self.shape)
        self.wavelets = [
            Wavelet(cs[0], cs[1], kcs[0], kcs[1], phi[0], phi[1]) for cs, kcs, phi in
            zip(self.coefficients, self.kss, self.phases)
        ]

    def __call__(self, x, bias=0):
        y = bias
        for wavelet in self.wavelets:
            y += wavelet(x)
        return y

    def __add__(self, other):
        return StandingPulse.from_wavelets([*self.wavelets, *other.wavelets])

    def draw(self, xrange, bias=0, grid=False):
        plt.plot(xrange, self(xrange, bias))
        plt.grid(grid)
        plt.show()

    @staticmethod
    def generate_random_standing_pulse(n_cos: int = 1, n_sin: int = 1):
        wavelets = []
        for i in range(n_cos):
            wavelets.append(Wavelet.generate_random_cosine_wavelet())
        for j in range(n_sin):
            wavelets.append(Wavelet.generate_random_sine_wavelet())

        return StandingPulse.from_wavelets(wavelets)

    @staticmethod
    def from_wavelets(wavelets: List[Wavelet]):
        coefficients = []
        kss = []
        phases = []
        for wavelet in wavelets:
            coefficients.append(wavelet.sextuplet[0:2])
            kss.append(wavelet.sextuplet[2:4])
            phases.append(wavelet.sextuplet[4:6])

        return StandingPulse(
            coefficients=np.array(coefficients),
            wave_vectors=np.array(kss),
            phases=np.array(phases)
        )


class Pulse(StandingPulse):
    def __init__(self, coefficients: np.ndarray, wave_vectors: np.ndarray | float, phases: np.ndarray | float, ws: np.ndarray | float):
        if ws is None:
            raise TypeError("""
                The 'ws' argument cannot be None. It is the time coefficients for the wavelets in the pulse which make them a pulse.
                If you want a standing pulse, use the StandingPulse class instead.
                """)
        super().__init__(coefficients, wave_vectors, phases)
        self.time_weights = transform(ws, self.shape)
        self.wavelets = lambda t: [
            Wavelet(cs[0], cs[1], kcs[0], kcs[1], phi[0] + tw[0] * t, phi[1] + tw[1] * t) for cs, kcs, phi, tw in
            zip(self.coefficients, self.kss, self.phases, self.time_weights)
        ]

    @staticmethod
    def from_wavelets(wavelets: List[Wavelet], ws: np.ndarray | float):
        coefficients = []
        kss = []
        phases = []
        for wavelet in wavelets:
            coefficients.append(wavelet.sextuplet[0:2])
            kss.append(wavelet.sextuplet[2:4])
            phases.append(wavelet.sextuplet[4:6])

        return Pulse(
            coefficients=np.array(coefficients),
            wave_vectors=np.array(kss),
            phases=np.array(phases),
            ws=ws
        )

    @staticmethod
    def generate_random_pulse(n_cos: int = 1, n_sin: int = 1, ws: np.ndarray | float | None = None):
        wavelets = []
        for _ in range(n_cos):
            wavelets.append(Wavelet.generate_random_cosine_wavelet())
        for _ in range(n_sin):
            wavelets.append(Wavelet.generate_random_sine_wavelet())

        if ws is None:
            ws = np.random.normal(-1, 1, size=(1, 2))

        return Pulse.from_wavelets(wavelets, ws)

    def __call__(self, x, t=0, bias=0):
        y = bias
        for wavelet in self.wavelets(t):
            y += wavelet(x)
        return y

    def animate(self, xrange, t_range, bias=0, interval=50, grid=False, save_path=None):
        fig, ax = plt.subplots()
        line, = ax.plot(xrange, self(xrange, t_range[0], bias))
        ax.set_xlim(xrange.min(), xrange.max())

        y_all = np.array([self(xrange, t, bias) for t in t_range])
        ax.set_ylim(y_all.min(), y_all.max())
        ax.grid(grid)
        title = ax.set_title(f"t = {t_range[0]:.3f}")
        print("Pulse parameters: ")
        print("Coefficients: ", self.coefficients)
        print("Phases: ", self.phases)
        print("Time weights: ", self.time_weights)

        def update(frame):
            t = t_range[frame]
            line.set_ydata(self(xrange, t, bias))
            title.set_text(f"t = {t:.3f}")
            return line, title

        anim = FuncAnimation(fig, update, frames=len(t_range), interval=interval, blit=False)

        if save_path:
            anim.save(save_path)
        else:
            plt.show()

        return anim
