# core/config.py

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Expanded action space including movement and jump decisions
MOVEMENT_ACTIONS = {
    0: {'move': 'right', 'jump': False},  # Move right
    1: {'move': 'right', 'jump': True},   # Move right and jump
    2: {'move': 'nothing', 'jump': False}, # Stand still
    3: {'move': 'nothing', 'jump': True},  # Jump in place
}

ACTION_DIM = len(MOVEMENT_ACTIONS)  # 4 actions

# Colors
BG_COLOR = (255, 255, 255)  # White color
