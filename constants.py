# constants.py
import numpy as np
from bimodal_distribution import generate_bimodal_prior

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
humpfrey_background = 10
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
max_stimulus_dB = -np.log10((max_brightness-humpfrey_background) / humpfrey_max_stimulus) * 10 # 12.15 dB for my setup (more positive is less bright, less positive is more bright).
min_stimulus_level = max_brightness * (1 / 255) ** (gamma)
#min_stimulus_dB = -np.log10(1 / humpfrey_max_stimulus) * 10 # 35 dB for my setup
stimuli_colors = np.arange(background_color+1,max_color+1,1) #every 1 or 2 integers, ie, 28, 30, 32 ...252, 254
stimuli_cdm2 = max_brightness * (stimuli_colors / 255) ** (gamma)
stimuli_dBlevels = - np.log10((stimuli_cdm2-humpfrey_background)/3183) * 10
min_stimulus_dB = max(stimuli_dBlevels)
dBlevels_count = len(stimuli_dBlevels)
#dBlevelsAvailable = np.log10(max_level / min_level) * 10
#dBlevelsCount = round(np.trunc(dBlevelsAvailable / dBstep_size))
#dBlevels = np.linspace(0, 0 + (dBlevelsCount-1)*dBstep_size, dBlevelsCount)
#dot_levels = max_level * (10 ** (-dBlevels/10))
#dot_colors = 255 * (dot_levels / 255) ** (1 / gamma)
BACKGROUND = (background_color, background_color, background_color)  # background
WHITE = (max_level, max_level, max_level)
ORANGE = (255, 165, 0)
stimuli_target_size_degrees = 0.43 # https://www.ncbi.nlm.nih.gov/books/NBK585112/ # Target 	Size ( in square mm)	Degrees:III	4	0.43 degrees

#b_values = np.linspace(max_stimulus_dB+2, min_stimulus_dB-2, 66)
b_values = np.linspace(max_stimulus_dB-2, min_stimulus_dB+2, 66)
# Define the parameters for the bimodal distribution
mean_b1 = 35  # Mean of the first Gaussian
std_b1 = 10  # Standard deviation of the first Gaussian
mean_b2 = 0  # Mean of the second Gaussian
std_b2 = 10  # Standard deviation of the second Gaussian

# Set the weight for each Gaussian
weight_b1 = 2  # First Gaussian has twice the weight of the second
weight_b2 = 1  # Second Gaussian has normal weight

# Generate the bimodal prior distribution using the function
print(f'b_values{b_values}')
prior = generate_bimodal_prior(b_values, mean_b1, std_b1, mean_b2, std_b2, weight_b1, weight_b2)
k_guess = 2
max_prob_guess = 0.95
min_prob_guess = 0.05

lookup_file = 'optimal_choices.pkl'

# Cross mark dimensions
CROSS_SIZE = 6
CROSS_WIDTH = 2

# Game settings
GAME_DURATION = 2  # 5 minutes in seconds
response_window = 1  # 1 second to respond after the dot disappears
time_pause_limit = [1, 3]
stimulus_duration = 0.2 #https://www.ncbi.nlm.nih.gov/books/NBK585112/ #

#scotoma_points = [[3, 3], [9, 3]]  # 24-2 test's scotoma points  for kay
scotoma_points = [[13, -1.5]]  # 24-2 test's scotoma points for harrison
scotoma_margin = 7.5  # Maximum distance for points to remain, 7.5 for harrison? 6 for kay
total_point_radius = 24
point_degree_spacing = 2
