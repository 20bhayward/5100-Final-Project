import random
from components.obstacle import Obstacle

SCREEN_HEIGHT = 400
AGENT_SIZE = 30
GROUND_LEVEL = SCREEN_HEIGHT - AGENT_SIZE

# Function to spawn obstacles randomly
def spawn_obstacle(obstacles):
    if random.randint(1, 100) < 4:  # Random chance of spawning
        obstacles.append(Obstacle())

# Function to update obstacles
def update_obstacles(obstacles):
    for obstacle in obstacles[:]:
        obstacle.update()
        if obstacle.rect.right < 0:  # Remove obstacle if off-screen
            obstacles.remove(obstacle)

def check_collisions(agent, obstacles):
    for obstacle in obstacles:
        if agent.rect.colliderect(obstacle.rect):
            # Get overlap area in both horizontal and vertical directions
            overlap_x = min(agent.rect.right, obstacle.rect.right) - max(agent.rect.left, obstacle.rect.left)
            overlap_y = min(agent.rect.bottom, obstacle.rect.bottom) - max(agent.rect.top, obstacle.rect.top)

            # Vertical collision (agent is landing on top of the obstacle)
            if agent.rect.bottom <= obstacle.rect.top + overlap_y and agent.velocity_y >= 0:
                # Allow landing on top of the obstacle
                agent.rect.bottom = obstacle.rect.top  # Adjust position to stand on obstacle
                agent.velocity_y = 0  # Stop vertical velocity (no more falling)
                agent.is_jumping = False  # Stop jumping
                return False  # No termination (safe landing)

            # Side collision (agent hits obstacle from the side)
            if overlap_x < overlap_y:
                return True  # Terminate the game due to side collision

    return False  # No collision detected
