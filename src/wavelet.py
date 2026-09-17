from numpy import cos, sin, round, random
class Wavelet:
    def __init__(self, c: int | float = 1, s: int | float = 1, kc: int | float = 1, ks: int | float = 1,
                 phi_c: int | float = 0, phi_s: int | float = 0, truncate=6):
        self.coefficients = {
            "cos": c,
            "sin": s,
            "kc": kc,
            "ks": ks,
            "phase_c": phi_c,
            "phase_s": phi_s
        }
        self.sextuplet = (c, s, kc, ks, phi_c, phi_s)
        self.truncate = truncate

    def __call__(self, x):
        cos_term = sin_term = 0
        if self.coefficients["cos"] != 0:
            cos_term = self.coefficients["cos"] * cos((self.coefficients["kc"] * x) + self.coefficients["phase_c"])
        if self.coefficients["sin"] != 0:
            sin_term = self.coefficients["sin"] * sin((self.coefficients["ks"] * x) + self.coefficients["phase_s"])
        return round(cos_term + sin_term, self.truncate)

    def __add__(self, other):
        from src.pulse import StandingPulse
        return StandingPulse.from_wavelets([self, other])

    @staticmethod
    def generate_random_cosine_wavelet():
        return Wavelet(
            c=random.random(),
            s=0,
            kc=random.random(),
            ks=0,
            phi_c=random.random(),
            phi_s=0
        )

    @staticmethod
    def generate_random_sine_wavelet():
        return Wavelet(
            s=random.random(),
            c=0,
            ks=random.random(),
            kc=0,
            phi_s=random.random(),
            phi_c=0
        )
