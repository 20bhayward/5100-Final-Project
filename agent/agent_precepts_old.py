# agent_precepts.py
import pygame
import numpy as np
from pygame import Vector2

class AgentPrecepts:
    def __init__(self, agent, level, pygame_manager):
        self.agent = agent
        self.level = level
        self.pygame_manager = pygame_manager
        
        # Vision system parameters
        self.vision_rays = [
            (-60, 150),  # Far left ray
            (-30, 150),  # Mid left ray
            (-15, 200),  # Near left ray
            (0, 250),    # Center ray (longest)
            (15, 200),   # Near right ray
            (30, 150),   # Mid right ray
            (60, 150),   # Far right ray
            (90, 100),   # Straight down ray
        ]
        
        # Platform detection parameters
        self.platform_check_distance = 100
        self.gap_check_depth = 50
        
        # Movement tracking
        self.last_positions = []
        self.position_history_size = 10

    def get_environment_state(self):
        """
        Get comprehensive state information about the agent's environment.
        
        Returns:
            dict: A dictionary containing all relevant state information
        """
        ray_info = self.cast_rays()
        platform_info = self.get_platform_info()
        movement_info = self.get_movement_info()
        hazard_info = self.get_hazard_info()
        goal_info = self.get_goal_info()
        
        # Combine all state information
        state = {
            # Ray information
            'ray_distances': ray_info['distances'],
            'ray_types': ray_info['types'],
            
            # Platform information
            'on_platform': platform_info['on_platform'],
            'platform_ahead': platform_info['platform_ahead'],
            'gap_ahead': platform_info['gap_ahead'],
            'gap_below': platform_info['gap_below'],  # New
            'gap_distance': platform_info['gap_distance'],
            'moving_platform': platform_info['on_moving_platform'],
            'safe_direction': platform_info['safe_direction'],  # New
            
            # Movement information
            'velocity': movement_info['velocity'],
            'on_ground': movement_info['on_ground'],
            'is_jumping': movement_info['is_jumping'],
            
            # Hazard information
            'hazard_ahead': hazard_info['hazard_ahead'],
            'hazard_distance': hazard_info['hazard_distance'],
            'hazard_nearby': hazard_info['hazard_nearby'],  # New
            'safe_to_jump': hazard_info['safe_to_jump'],    # New
            
            # Goal information
            'goal_distance': goal_info['distance'],
            'goal_direction': goal_info['direction']
        }
        
        return state
    
    def cast_rays(self):
        """
        Cast rays to detect objects in the environment.
        
        Returns:
            dict: Contains normalized distances and object types for each ray
        """
        distances = []
        types = []
        
        for angle, length in self.vision_rays:
            # Calculate ray endpoint
            rad_angle = np.radians(angle)
            end_x = self.agent.rect.centerx + np.cos(rad_angle) * length
            end_y = self.agent.rect.centery + np.sin(rad_angle) * length
            
            ray_start = Vector2(self.agent.rect.centerx, self.agent.rect.centery)
            ray_end = Vector2(end_x, end_y)
            
            # Check intersections
            min_dist = length
            obj_type = 0  # 0:nothing, 1:platform, 2:hazard, 3:goal, 4:moving platform
            
            # Check platforms
            for platform in self.pygame_manager.block_list:
                if hasattr(platform, 'is_moving') and platform.is_moving:
                    intersection = self._line_rect_intersection(ray_start, ray_end, platform.rect)
                    if intersection:
                        dist = ray_start.distance_to(intersection)
                        if dist < min_dist:
                            min_dist = dist
                            obj_type = 4  # Moving platform
                elif not any(goal for goal in self.level.goal_list if platform == goal):
                    intersection = self._line_rect_intersection(ray_start, ray_end, platform.rect)
                    if intersection:
                        dist = ray_start.distance_to(intersection)
                        if dist < min_dist:
                            min_dist = dist
                            obj_type = 1  # Static platform
            
            # Check hazards (traps)
            for hazard in self.level.trap_list:
                intersection = self._line_rect_intersection(ray_start, ray_end, hazard.rect)
                if intersection:
                    dist = ray_start.distance_to(intersection)
                    if dist < min_dist:
                        min_dist = dist
                        obj_type = 2  # Hazard
            
            # Check goal
            for goal in self.level.goal_list:
                intersection = self._line_rect_intersection(ray_start, ray_end, goal.rect)
                if intersection:
                    dist = ray_start.distance_to(intersection)
                    if dist < min_dist:
                        min_dist = dist
                        obj_type = 3  # Goal
            
            distances.append(min_dist / length)  # Normalize to [0,1]
            types.append(obj_type)
        
        return {'distances': distances, 'types': types}
    
    def get_platform_info(self):
        """
        Get detailed information about platforms around the agent.
        
        Returns:
            dict: Information about platform presence and gaps
        """
        # Check current platform
        on_platform = False
        on_moving_platform = False
        platform_ahead = False
        gap_ahead = False
        gap_below = False  # New
        gap_distance = float('inf')
        safe_direction = 1  # 1 for right, -1 for left
        
        # Ground check
        ground_check = pygame.Rect(
            self.agent.rect.centerx - 5,
            self.agent.rect.bottom,
            10,
            2
        )
        
        # Down check for immediate gap
        down_check = pygame.Rect(
            self.agent.rect.centerx - 5,
            self.agent.rect.bottom + 1,
            10,
            100  # Check quite far down
        )
        
        # Check if we're over a gap
        blocks_below = [block for block in self.pygame_manager.block_list 
                        if down_check.colliderect(block.rect)]
        gap_below = len(blocks_below) == 0
        
        # Determine safe direction when over gap
        if gap_below:
            # Check for platforms on both sides
            left_check = pygame.Rect(
                self.agent.rect.left - 100,  # Check 100 pixels to the left
                self.agent.rect.bottom - 10,
                100,
                20
            )
            right_check = pygame.Rect(
                self.agent.rect.right,
                self.agent.rect.bottom - 10,
                100,
                20
            )
            
            platforms_left = any(block.rect.colliderect(left_check) 
                                for block in self.pygame_manager.block_list)
            platforms_right = any(block.rect.colliderect(right_check) 
                                 for block in self.pygame_manager.block_list)
            
            if platforms_left and not platforms_right:
                safe_direction = -1
            elif platforms_right and not platforms_left:
                safe_direction = 1
            else:
                # If platforms on both sides or no platforms, prefer right
                safe_direction = 1
        
        # Check current platform
        for platform in self.pygame_manager.block_list:
            if ground_check.colliderect(platform.rect):
                on_platform = True
                if hasattr(platform, 'is_moving') and platform.is_moving:
                    on_moving_platform = True
                break
        
        # Forward platform/gap check
        check_x = self.agent.rect.right
        check_width = self.platform_check_distance
        if self.agent.change_x < 0:  # If moving left
            check_x = self.agent.rect.left - self.platform_check_distance
        
        platform_check = pygame.Rect(
            check_x,
            self.agent.rect.bottom - 10,
            check_width,
            20
        )
        
        gap_check = pygame.Rect(
            check_x,
            self.agent.rect.bottom,
            check_width,
            self.gap_check_depth
        )
        
        # Check for platforms ahead
        for platform in self.pygame_manager.block_list:
            if platform_check.colliderect(platform.rect):
                platform_ahead = True
                break
        
        # Check for gaps ahead
        if not any(platform.rect.colliderect(gap_check) for platform in self.pygame_manager.block_list):
            gap_ahead = True
            if on_platform:
                current_platform = None
                for platform in self.pygame_manager.block_list:
                    if ground_check.colliderect(platform.rect):
                        current_platform = platform
                        break
                if current_platform:
                    if self.agent.change_x >= 0:
                        gap_distance = current_platform.rect.right - self.agent.rect.x
                    else:
                        gap_distance = self.agent.rect.x - current_platform.rect.left
        
        return {
            'on_platform': on_platform,
            'platform_ahead': platform_ahead,
            'gap_ahead': gap_ahead,
            'gap_below': gap_below,  # New
            'gap_distance': gap_distance / self.platform_check_distance,  # Normalize
            'on_moving_platform': on_moving_platform,
            'safe_direction': safe_direction  # New
        }
    
    def get_movement_info(self):
        """
        Get information about the agent's movement state.
        
        Returns:
            dict: Movement-related state information
        """
        # Update position history
        self.last_positions.append((self.agent.rect.x, self.agent.rect.y))
        if len(self.last_positions) > self.position_history_size:
            self.last_positions.pop(0)
        
        # Calculate velocity
        velocity = Vector2(0, 0)
        if len(self.last_positions) >= 2:
            prev_pos = Vector2(*self.last_positions[-2])
            curr_pos = Vector2(*self.last_positions[-1])
            velocity = curr_pos - prev_pos
        
        return {
            'velocity': (velocity.x / self.agent.max_speed_x, velocity.y / self.agent.terminal_velocity),
            'on_ground': self.agent.on_ground,
            'is_jumping': self.agent.is_jumping
        }
    
    def get_hazard_info(self):
        """
        Get detailed information about nearby hazards and safe directions.
        
        Returns:
            dict: Information about hazard presence, distance, and safe direction
        """
        hazard_ahead = False
        hazard_distance = float('inf')
        safe_direction = 1  # Default to right
        hazard_nearby = False
        
        # Create detection areas for different directions
        look_ahead = 150
        detection_height = self.agent.rect.height + 40
        
        # Right detection area
        right_detect = pygame.Rect(
            self.agent.rect.right,
            self.agent.rect.top - 20,
            look_ahead,
            detection_height
        )
        
        # Left detection area
        left_detect = pygame.Rect(
            self.agent.rect.left - look_ahead,
            self.agent.rect.top - 20,
            look_ahead,
            detection_height
        )
        
        # Immediate area (for nearby hazards)
        nearby_detect = pygame.Rect(
            self.agent.rect.left - 50,  # Check 50 pixels on each side
            self.agent.rect.top - 20,
            self.agent.rect.width + 100,
            detection_height
        )
        
        # Check for hazards in each direction
        hazards_right = []
        hazards_left = []
        
        for hazard in self.level.trap_list:
            # Check right
            if right_detect.colliderect(hazard.rect):
                hazards_right.append(abs(hazard.rect.centerx - self.agent.rect.centerx))
            # Check left
            if left_detect.colliderect(hazard.rect):
                hazards_left.append(abs(hazard.rect.centerx - self.agent.rect.centerx))
            # Check nearby
            if nearby_detect.colliderect(hazard.rect):
                hazard_nearby = True
        
        # Determine if there are hazards ahead based on agent's direction
        if self.agent.change_x >= 0:  # Moving/facing right
            if hazards_right:
                hazard_ahead = True
                hazard_distance = min(hazards_right)
        else:  # Moving/facing left
            if hazards_left:
                hazard_ahead = True
                hazard_distance = min(hazards_left)
        
        # Determine safe direction
        if hazards_right and hazards_left:
            # Hazards on both sides - prefer the side with farther hazard
            min_right = min(hazards_right) if hazards_right else float('inf')
            min_left = min(hazards_left) if hazards_left else float('inf')
            safe_direction = -1 if min_right < min_left else 1
        elif hazards_right:
            safe_direction = -1  # Hazard only on right, go left
        elif hazards_left:
            safe_direction = 1   # Hazard only on left, go right
        
        # Also consider the goal direction if no immediate hazards
        if not (hazards_right or hazards_left):
            goal = next(iter(self.level.goal_list))
            safe_direction = 1 if goal.rect.centerx > self.agent.rect.centerx else -1
        
        # Check if jumping is safe
        jump_clearance = pygame.Rect(
            self.agent.rect.x - 20,
            self.agent.rect.top - 60,  # Check above
            self.agent.rect.width + 40,
            60
        )
        safe_to_jump = not any(jump_clearance.colliderect(hazard.rect) 
                            for hazard in self.level.trap_list)
        
        return {
            'hazard_ahead': hazard_ahead,
            'hazard_distance': hazard_distance / look_ahead,  # Normalize
            'safe_direction': safe_direction,
            'hazard_nearby': hazard_nearby,
            'safe_to_jump': safe_to_jump
        }
        
    def get_goal_info(self):
        """
        Get information about the goal's position relative to the agent.
        
        Returns:
            dict: Goal distance and direction information
        """
        goal = next(iter(self.level.goal_list))
        dx = goal.rect.centerx - self.agent.rect.centerx
        dy = goal.rect.centery - self.agent.rect.centery
        
        distance = np.sqrt(dx*dx + dy*dy)
        max_distance = np.sqrt(self.level.width**2 + self.level.height**2)
        
        direction = np.arctan2(dy, dx) / np.pi  # Normalize to [-1, 1]
        
        return {
            'distance': distance / max_distance,  # Normalize
            'direction': direction
        }
    
    def get_goal_distance(self):
        """
        Calculate the Euclidean distance from the agent to the nearest goal.
        
        Returns:
            float: Normalized distance to the goal
        """
        goal = next(iter(self.level.goal_list))
        distance = ((goal.rect.centerx - self.agent.rect.centerx) ** 2 +
                    (goal.rect.centery - self.agent.rect.centery) ** 2) ** 0.5
        max_distance = np.sqrt(self.level.width**2 + self.level.height**2)
        return distance / max_distance
    
    def get_goal_position(self):
        """
        Get the x-coordinate of the goal.
        
        Returns:
            float: Normalized x-coordinate relative to the level width
        """
        goal = next(iter(self.level.goal_list))
        return (goal.rect.centerx) / self.level.width
    
    def _line_rect_intersection(self, start, end, rect):
        """
        Helper method to detect line intersection with rectangle.
        
        Args:
            start (Vector2): Start point of the line
            end (Vector2): End point of the line
            rect (pygame.Rect): Rectangle to check intersection with
        
        Returns:
            Vector2 or None: Point of intersection if it exists
        """
        def line_line_intersection(line1_start, line1_end, line2_start, line2_end):
            x1, y1 = line1_start
            x2, y2 = line1_end
            x3, y3 = line2_start
            x4, y4 = line2_end
            
            denominator = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))
            if denominator == 0:
                return None
                
            t = (((x1 - x3) * (y3 - y4)) - ((y1 - y3) * (x3 - x4))) / denominator
            if 0 <= t <= 1:
                intersection_x = x1 + t * (x2 - x1)
                intersection_y = y1 + t * (y2 - y1)
                return Vector2(intersection_x, intersection_y)
            return None
        
        # Check all rectangle edges
        rect_lines = [
            (Vector2(rect.left, rect.top), Vector2(rect.right, rect.top)),
            (Vector2(rect.right, rect.top), Vector2(rect.right, rect.bottom)),
            (Vector2(rect.right, rect.bottom), Vector2(rect.left, rect.bottom)),
            (Vector2(rect.left, rect.bottom), Vector2(rect.left, rect.top))
        ]
        
        closest_intersection = None
        min_distance = float('inf')
        
        for line_start, line_end in rect_lines:
            intersection = line_line_intersection(start, end, line_start, line_end)
            if intersection:
                distance = start.distance_to(intersection)
                if distance < min_distance:
                    min_distance = distance
                    closest_intersection = intersection
        
        return closest_intersection
