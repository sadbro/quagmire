
from src.pulse import *

def test_standing_pulse():
    pulse = StandingPulse.generate_random_standing_pulse()
    x = np.arange(-36, 36, 0.05)
    pulse.draw(x)

def test_pulse():
    pulse = Pulse.generate_random_pulse(ws=[[-1, 1]])
    x = np.arange(-40, 40, 0.05)
    t = np.arange(0, 100, 0.1)
    pulse.animate(x, t, interval=100, save_path=None)
