# constants.py
import numpy as np

# Screen dimensions
WIDTH, HEIGHT = 640, 360
SCREEN_SIZE = 4 * 2.54  # cm
VIEWER_DISTANCE = 24 * 2.54  # cm
PIXELS_PER_CM = 400 / 27.6  # pixels/cm measured

# Colors
gamma = 1.37
dBstep_size = 1
background_color = 20  # 31.5 apostlib = 10 candelas is standard. ideally we can calibrate our setup
background_level = 255 * (background_color / 255) ** (gamma)
min_level = background_level
max_level = 255  # ideal brightness stimuli
dBlevelsAvailable = np.log10(max_level / min_level) * 10
dBlevelsCount = round(np.trunc(dBlevelsAvailable / dBstep_size))
dBlevels = np.linspace(0, 0 + (dBlevelsCount-1)*dBstep_size, dBlevelsCount)
dot_levels = max_level * (10 ** (-dBlevels/10))
dot_colors = 255 * (dot_levels / 255) ** (1 / gamma)
BACKGROUND = (background_color, background_color, background_color)  # background
WHITE = (max_level, max_level, max_level)
ORANGE = (255, 165, 0)

# Cross mark dimensions
CROSS_SIZE = 6
CROSS_WIDTH = 2

# Game settings
GAME_DURATION = 30  # 5 minutes in seconds
response_window = 1  # 1 second to respond after the dot disappears
time_pause_limit = [1, 3]
