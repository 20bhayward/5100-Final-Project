# camera_precepts.py

import pygame
import numpy as np
from pygame import Vector2

class CameraBasedPrecepts:
    def __init__(self, agent, level, pygame_manager):
        self.agent = agent
        self.level = level
        self.pygame_manager = pygame_manager

        # Camera view parameters
        self.screen_width = 800  # From config.SCREEN_WIDTH
        self.screen_height = 600  # From config.SCREEN_HEIGHT

        # Platform analysis parameters
        self.min_gap_width = 20  # Minimum gap width to consider for jumping
        self.max_gap_width = 200  # Maximum gap width the agent can jump
        self.safe_landing_width = 10  # Minimum width needed for safe landing

        # Jump trajectory parameters
        self.gravity = abs(self.agent.gravity_acc)
        self.jump_velocity = self.agent.jump_speed  # Should match agent's jump_speed

    def get_visible_state(self):
        """Get the state of objects visible in the camera view."""
        camera_rect = pygame.Rect(
            -self.pygame_manager.camera_x,
            -self.pygame_manager.camera_y,
            self.screen_width,
            self.screen_height
        )

        visible_platforms = []
        visible_hazards = []
        visible_goals = []

        for sprite in self.pygame_manager.get_all_sprites():
            sprite_rect = sprite.rect
            if camera_rect.colliderect(sprite_rect):
                # Use world coordinates instead of camera-relative
                if sprite in self.pygame_manager.get_blocks():
                    platform_info = {
                        'rect': sprite_rect.copy(),  # Use actual world coordinates
                        'is_moving': hasattr(sprite, 'is_moving') and sprite.is_moving,
                        'speed': sprite.speed if hasattr(sprite, 'speed') else 0,
                        'direction': sprite.direction if hasattr(sprite, 'direction') else None,
                        'sprite': sprite
                    }
                    visible_platforms.append(platform_info)
                elif sprite in self.level.trap_list:
                    visible_hazards.append(sprite_rect.copy())
                elif sprite in self.level.goal_list:
                    visible_goals.append(sprite_rect.copy())

        platform_analysis = self._analyze_platforms(visible_platforms)
        hazard_analysis = self._analyze_hazards(visible_hazards)
        jump_opportunities = self._analyze_jump_opportunities(visible_platforms, visible_hazards)
        movement_state = self._get_movement_state()

        return {
            'platforms': platform_analysis,
            'hazards': hazard_analysis,
            'jumps': jump_opportunities,
            'movement': movement_state,
            'goal_visible': len(visible_goals) > 0,
            'goal_direction': self._get_goal_direction(visible_goals)
        }

    def _analyze_platforms(self, visible_platforms):
        """Analyze platforms by combining adjacent blocks into continuous platforms."""
        # Sort platforms by x position
        sorted_platforms = sorted(visible_platforms, key=lambda p_info: p_info['rect'].x)
        
        # Combine adjacent blocks into continuous platforms
        continuous_platforms = []
        current_platform = None
        
        for platform_info in sorted_platforms:
            if current_platform is None:
                current_platform = {
                    'rect': platform_info['rect'].copy(),
                    'is_moving': platform_info['is_moving'],
                    'speed': platform_info['speed'],
                    'direction': platform_info['direction'],
                    'sprite': platform_info['sprite']
                }
            else:
                # Check if this block is adjacent to current platform (within 1 block width = 40 pixels)
                if platform_info['rect'].left - current_platform['rect'].right <= 5:
                    # Extend current platform
                    current_platform['rect'].width = (
                        platform_info['rect'].right - current_platform['rect'].left
                    )
                else:
                    # Gap found, store current platform and start new one
                    continuous_platforms.append(current_platform)
                    current_platform = {
                        'rect': platform_info['rect'].copy(),
                        'is_moving': platform_info['is_moving'],
                        'speed': platform_info['speed'],
                        'direction': platform_info['direction'],
                        'sprite': platform_info['sprite']
                    }
        
        if current_platform:
            continuous_platforms.append(current_platform)

        # Now analyze gaps between continuous platforms
        gaps = []
        for i in range(len(continuous_platforms) - 1):
            current = continuous_platforms[i]
            next_platform = continuous_platforms[i + 1]
            
            # Calculate actual gap width between continuous platforms
            gap_width = next_platform['rect'].left - current['rect'].right
            
            if self.min_gap_width <= gap_width <= self.max_gap_width:
                jumpable = self._is_gap_jumpable(
                    gap_width,
                    current['rect'].y - next_platform['rect'].y,
                    current,
                    next_platform
                )
                gaps.append({
                    'start_x': current['rect'].right,
                    'width': gap_width,
                    'start_y': current['rect'].y,
                    'end_y': next_platform['rect'].y,
                    'jumpable': jumpable
                })
        
        # Get current and next platform relative to agent
        current_platform = self._get_current_platform(continuous_platforms)
        next_platform = self._get_next_platform(continuous_platforms) if current_platform else None

        return {
            'gaps': gaps,
            'current_platform': current_platform,
            'next_platform': next_platform,
            'platform_count': len(continuous_platforms)
        }

    def _analyze_hazards(self, visible_hazards):
        """Analyze visible hazards and their threat level."""
        immediate_threats = []
        for hazard in visible_hazards:
            # Calculate distance and direction to hazard
            dx = hazard.centerx - self.agent.rect.centerx
            dy = hazard.centery - self.agent.rect.centery
            distance = np.sqrt(dx*dx + dy*dy)

            if distance < 200:  # Adjusted threshold for visibility
                immediate_threats.append({
                    'rect': hazard,  # Include the rect for visualization
                    'distance': distance,
                    'direction': np.sign(dx),
                    'above': dy < 0,
                    'requires_jump': self._requires_jump_to_avoid(hazard)
                })

        # Sort threats by distance (closest first)
        immediate_threats.sort(key=lambda x: x['distance'])

        return {
            'immediate_threats': immediate_threats,
            'threat_count': len(immediate_threats),
            'visible_hazards': visible_hazards  # Include visible hazards for visualization
        }

    def _is_gap_jumpable(self, width, height_diff, current_platform_info, target_platform_info):
        """Determine if a gap is jumpable considering platform movement."""
        # Predict time of flight based on horizontal distance
        v0_x = self.agent.max_speed_x
        if v0_x == 0:
            return False  # Cannot jump if horizontal speed is zero
        t_flight = width / v0_x

        # Predict future position of the target platform
        target_platform_rect = target_platform_info['rect'].copy()
        if target_platform_info['is_moving'] and target_platform_info['direction'] == 'horizontal':
            movement = target_platform_info['speed'] * t_flight
            target_platform_rect.x += movement

        # Recalculate width and height_diff based on predicted position
        adjusted_width = target_platform_rect.left - current_platform_info['rect'].right
        adjusted_height_diff = current_platform_info['rect'].y - target_platform_rect.y

        # Use adjusted values to determine jumpability
        v0_y = self.jump_velocity  # Should be negative (e.g., -7)
        gravity = self.gravity     # Should be positive (e.g., 0.4)

        # Time to reach the peak of the jump
        t_up = -v0_y / gravity
        # Maximum height reached relative to jump point
        h_max = v0_y * t_up + 0.5 * gravity * t_up ** 2

        # Check if agent can reach the required height difference
        can_reach_target_height = h_max >= adjusted_height_diff

        # Total time in air (up and down)
        t_total = 2 * t_up

        # Check if agent can cover the adjusted horizontal distance in total time
        can_cover_distance = v0_x * t_total >= adjusted_width

        # Return True or False based on calculations
        return can_reach_target_height and can_cover_distance

    def _analyze_jump_opportunities(self, platforms, hazards):
        """Analyze potential jump opportunities and their success probability."""
        jumps = []
        agent_pos = Vector2(self.agent.rect.x, self.agent.rect.y)

        # Get current platform
        current_platform_info = self._get_current_platform(platforms)
        if not current_platform_info:
            return []

        current_platform_rect = current_platform_info['rect']

        for platform_info in platforms:
            if platform_info == current_platform_info:
                continue

            platform_rect = platform_info['rect']

            # Predict future position of the target platform
            dx = platform_rect.left - current_platform_rect.right

            v0_x = self.agent.max_speed_x
            if v0_x == 0:
                continue  # Avoid division by zero
            t_flight = dx / v0_x

            if platform_info['is_moving'] and platform_info['direction'] == 'horizontal':
                movement = platform_info['speed'] * t_flight
                platform_rect.x += movement

            # Recalculate dx and dy with predicted position
            adjusted_dx = platform_rect.left - current_platform_rect.right
            dy = current_platform_rect.y - platform_rect.y

            if self._is_gap_jumpable(adjusted_dx, dy, current_platform_info, platform_info):
                # Calculate success probability
                success_prob = self._calculate_jump_success_prob(adjusted_dx, dy, current_platform_info, platform_info)

                # Append jump opportunity
                jumps.append({
                    'start_x': current_platform_rect.right - 20,
                    'end_x': platform_rect.left + 20,
                    'height_diff': dy,
                    'distance': adjusted_dx,
                    'safe_landing': True,  # Simplified; should check hazards
                    'success_prob': success_prob
                })

        return sorted(jumps, key=lambda j: j['success_prob'], reverse=True)

    def _calculate_jump_success_prob(self, distance, height_diff, current_platform_info, target_platform_info):
        """Calculate probability of successful jump based on distance and height."""
        if not self._is_gap_jumpable(distance, height_diff, current_platform_info, target_platform_info):
            return 0.0

        # Base probability inversely proportional to distance (normalized)
        base_prob = max(0.0, 1.0 - (distance / self.max_gap_width))

        # Adjust for height difference
        height_factor = max(0.0, 1.0 - (abs(height_diff) / 100.0))  # Penalize large height differences

        # Combine factors
        success_prob = base_prob * height_factor

        return success_prob

    def _get_current_platform(self, platforms):
        """Get the platform the agent is currently on."""
        agent_feet = pygame.Rect(
            self.agent.rect.left,
            self.agent.rect.bottom,
            self.agent.rect.width,
            2
        )

        for platform_info in platforms:
            platform_rect = platform_info['rect']
            if agent_feet.colliderect(platform_rect):
                return platform_info
        return None

    def _get_next_platform(self, platforms):
        """Get the next platform in the agent's path."""
        current = self._get_current_platform(platforms)
        if not current:
            return None

        current_rect = current['rect']
        next_platforms = [p for p in platforms if p['rect'].left > current_rect.right]
        if next_platforms:
            next_platform = min(next_platforms, key=lambda p: p['rect'].left)
            return next_platform
        else:
            return None

    def _requires_jump_to_avoid(self, hazard):
        """Determine if a hazard requires jumping to avoid."""
        return (hazard.top < self.agent.rect.bottom and
                abs(hazard.centerx - self.agent.rect.centerx) < 50)

    def _get_movement_state(self):
        """Get the current movement state of the agent."""
        return {
            'velocity_x': self.agent.change_x / self.agent.max_speed_x if self.agent.max_speed_x != 0 else 0,
            'velocity_y': self.agent.change_y / self.agent.terminal_velocity if self.agent.terminal_velocity != 0 else 0,
            'on_ground': self.agent.on_ground,
            'is_jumping': self.agent.is_jumping,
            'can_jump': self.agent.on_ground
        }

    def _get_goal_direction(self, visible_goals):
        """Get normalized direction to visible goal."""
        if not visible_goals:
            goal = next(iter(self.level.goal_list))  # Get the level's goal
            return np.sign(goal.rect.centerx - self.agent.rect.centerx)

        goal = visible_goals[0]
        dx = goal.centerx - self.agent.rect.centerx
        return np.sign(dx)

    def calculate_max_jump_distance(self):
        """Calculate the maximum horizontal distance the agent can jump."""
        v0_x = self.agent.max_speed_x
        v0_y = self.agent.jump_speed  # Should be negative
        gravity = self.gravity

        # Time to reach the peak
        t_up = -v0_y / gravity
        # Total time of flight
        t_total = 2 * t_up
        # Maximum horizontal distance
        max_distance = v0_x * t_total
        return max_distance
