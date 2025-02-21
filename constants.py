# constants.py
import numpy as np

# Screen dimensions
WIDTH, HEIGHT = 1280, 720
SCREEN_SIZE = 40 * 2.54  # cm
VIEWER_DISTANCE = 28 * 2.54  # cm
PIXELS_PER_CM = 400 / 27.6  # pixels/cm measured

# Colors
# TV used: Samsung NU7100 40" https://www.rtings.com/tv/reviews/samsung/nu7100
# Maximum of 261 cd/m^2 for small area = 261 nits = 820 apostilb (asb)
# Maximum SDR 100% screen = 291 cd/m^2. But I use 35/50 backlight.
# 0 isn't 0 cd/m^2, but 35 feels like about 30% less, bright, so we will just linearly scale it
# Max brightness with my setup is 204 cd/m^2
# https://www.ncbi.nlm.nih.gov/books/NBK585112/#:~:text=Zero%20decibels%20(dB)%20represent%20the,lowest%20sensitivity%20and%20vice%20versa.
# Humpfrey visual field test has 0 dB (maximum brightness) stimuli of 10,000 apostilbs
max_brightness = 204
humpfrey_max_stimulus = 3183
# Maximum contrast of 5397:1 according to rtings.com
# Minimum brightness thus is about 0.05 cd/m^2. This is ok, since much lower than desired background luminance of 10 cd/m^2

gamma = 1.37 # gamma for screen was found using 50% grey vs black/white checkered grid method
dBstep_size = 1
# 31.5 apostlib = 10 candelas/m^2 is standard. ideally we can calibrate our setup. https://webeye.ophth.uiowa.edu/ips/articles/Conventional-Perimetry-Part-I.pdf
# background color = standard background / max * 255 = 10 / 204 * 255 = 12.5
background_level = 12.5
background_color = int(255 * (background_level / 255) ** (1 / gamma)) #28 for my setup
min_color = background_color
max_level = 255
max_color = 255
max_stimulus_dB = -np.log10((204-10) / 3183) * 10 # 12.15 dB for my setup (more positive is less bright, less positive is more bright).
min_stimulus_dB = -np.log10(1 / 3183) * 10 # 35 dB for my setup
stimuli_colors = np.arange(background_color,max_color+1)
stimuli_levels = 255 * (stimuli_colors / 255) ** (gamma)
stimuli_dBlevels = - np.log(stimuli_levels/3183) * 10
#dBlevelsAvailable = np.log10(max_level / min_level) * 10
#dBlevelsCount = round(np.trunc(dBlevelsAvailable / dBstep_size))
#dBlevels = np.linspace(0, 0 + (dBlevelsCount-1)*dBstep_size, dBlevelsCount)
#dot_levels = max_level * (10 ** (-dBlevels/10))
#dot_colors = 255 * (dot_levels / 255) ** (1 / gamma)
BACKGROUND = (background_color, background_color, background_color)  # background
WHITE = (max_level, max_level, max_level)
ORANGE = (255, 165, 0)
stimuli_target_size_degrees = 0.43 # https://www.ncbi.nlm.nih.gov/books/NBK585112/ # Target 	Size ( in square mm)	Degrees:III	4	0.43 degrees

# Cross mark dimensions
CROSS_SIZE = 6
CROSS_WIDTH = 2

# Game settings
GAME_DURATION = 120  # 5 minutes in seconds
response_window = 1  # 1 second to respond after the dot disappears
time_pause_limit = [1, 3]
stimulus_duration = 0.2 #https://www.ncbi.nlm.nih.gov/books/NBK585112/ #

#scotoma_points = [[3, 3], [9, 3]]  # 24-2 test's scotoma points
scotoma_points = [[13, -1.5]]  # 24-2 test's scotoma points
scotoma_margin = 7.5  # Maximum distance for points to remain
