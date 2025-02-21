# main.py

import pygame
from game import main as game_main
from constants import WIDTH, HEIGHT

def initialize_pygame():
    # Initialize the Pygame library and set up the screen dimensions
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Visual Field Test")
    return screen

def run_game():
    screen = initialize_pygame()
    # Start the game logic from the game.py
    game_main(screen)

if __name__ == "__main__":
    run_game()
    pygame.quit()
