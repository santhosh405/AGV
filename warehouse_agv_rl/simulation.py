"""Pygame animation for a trained policy."""

from __future__ import annotations

import math

import pygame


def run_simulation(env, policy, fps: int = 5, title: str = "Warehouse AGV"):
    """Animate the AGV following a policy until it reaches the goal."""
    pygame.init()
    pygame.display.set_caption(title)
    cell_size = 54
    sidebar_width = 180
    screen_width = env.cols * cell_size + sidebar_width
    screen_height = env.rows * cell_size
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()

    state = env.reset()
    total_reward = 0.0
    steps = 0
    action = 0
    running = True
    paused = False
    done = False
    max_steps = max(50, env.rows * env.cols * 5)
    speed = fps

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    state = env.reset()
                    total_reward = 0.0
                    steps = 0
                    done = False
                    action = 0
                elif event.key in (pygame.K_UP, pygame.K_w):
                    speed = max(1, speed + 1)
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    speed = max(1, speed - 1)
                elif event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False

        if not paused and not done and steps < max_steps:
            action = int(policy[state])
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            steps += 1
            if state == env.goal_state:
                done = True

        screen.fill((245, 245, 245))
        board_x = 0
        board_y = 0
        for row in range(env.rows):
            for col in range(env.cols):
                rect = pygame.Rect(board_x + col * cell_size, board_y + row * cell_size, cell_size, cell_size)
                cell_type = env.grid[row][col]
                if cell_type == env.SHELF:
                    color = (40, 40, 40)
                elif cell_type == env.HAZARD:
                    color = (255, 140, 0)
                elif (row, col) == env.start_pos:
                    color = (70, 130, 255)
                elif (row, col) == env.goal_pos:
                    color = (40, 200, 90)
                else:
                    color = (220, 220, 220)
                pygame.draw.rect(screen, color, rect, border_radius=4)
                pygame.draw.rect(screen, (180, 180, 180), rect, 1)

        row, col = env.state_to_pos(state)
        agv_center = (board_x + col * cell_size + cell_size // 2, board_y + row * cell_size + cell_size // 2)
        pygame.draw.circle(screen, (255, 0, 0), agv_center, cell_size // 3)

        hud_x = env.cols * cell_size + 20
        hud_y = 20
        pygame.draw.rect(screen, (230, 230, 230), (hud_x - 10, hud_y - 10, sidebar_width - 10, 220), border_radius=8)
        label_font = pygame.font.SysFont(None, 28)
        value_font = pygame.font.SysFont(None, 24)
        labels = [
            ("Step", str(steps)),
            ("Reward", f"{total_reward:.1f}"),
            ("Action", {0: "Up", 1: "Down", 2: "Left", 3: "Right"}.get(action, "?")),
            ("Speed", f"{speed}x"),
        ]
        for idx, (label, value) in enumerate(labels):
            label_surf = label_font.render(f"{label}:", True, (25, 25, 25))
            value_surf = value_font.render(str(value), True, (25, 25, 25))
            screen.blit(label_surf, (hud_x, hud_y + idx * 40))
            screen.blit(value_surf, (hud_x + 90, hud_y + idx * 40))

        status = "Paused" if paused else ("Goal reached" if done else "Running")
        status_surf = value_font.render(status, True, (40, 40, 40))
        screen.blit(status_surf, (hud_x, hud_y + 170))

        pygame.display.flip()
        clock.tick(speed)

        if done or steps >= max_steps:
            if done:
                status_surf = value_font.render("Goal reached!", True, (20, 120, 20))
                screen.blit(status_surf, (hud_x, hud_y + 150))
                pygame.display.flip()
                pygame.time.wait(500)
            running = False

    pygame.quit()
    return steps, total_reward
