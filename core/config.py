# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Simplified action space (movement only)
MOVEMENT_ACTIONS = {
    0: 'left',
    1: 'right',
    2: 'nothing'
}

ACTION_DIM = len(MOVEMENT_ACTIONS)  # 3 actions

# Colors
BG_COLOR = (255, 255, 255)  # Gray color
