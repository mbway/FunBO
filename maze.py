#!/usr/bin/env python3

import sys
import time
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pygame as pg
import pygame.locals as pgl

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

class Ball:
    def __init__(self, x, y):
        self.pos = pg.math.Vector2(x, y)
        self.vel = pg.math.Vector2(0, 0)

    def physics_update(self, delta):
        self.pos += self.vel*delta

    def draw(self, surface):
        pg.draw.rect(surface, (200, 0, 0), self.get_rect())

    def get_rect(self):
        x, y = (round(self.pos.x), round(self.pos.y))
        w = 20
        return pg.Rect(x-w/2, y-w/2, w, w)

    def is_colliding(self, world):
        rects = [wall.rect for wall in world]
        i = self.get_rect().collidelist(rects)
        return i != -1  # returns index so -1 => no collision

class Wall:
    def __init__(self, x, y, width, height):
        self.rect = pg.Rect(x, y, width, height)

    def draw(self, surface):
        pg.draw.rect(surface, (0, 200, 0), self.rect)


class Simulation:
    def __init__(self):
        pg.init()
        # flags are for OpenGL, fullscreen, resizable etc
        self.w, self.h = 500, 800
        self.screen = pg.display.set_mode((self.w, self.h), 0, 32) # resolution, flags, depth
        pg.display.set_caption('Simulation')
        self.surface = pg.Surface(self.screen.get_size()).convert()
        self.clock = pg.time.Clock()

    def get_world(self, world_num):
        if world_num == 1:
            return [
                Wall(0, 500, 300, 50),
                Wall(self.w-250, 100, 250, 200),
                Wall(0, 100, 100, 200)
            ]
        elif world_num == 2:
            return [
                Wall(self.w/2-100, 600, 200, 50),
                Wall(self.w-150, 100, 150, 300),
                Wall(0, 100, 150, 300),
            ]
        else:
            raise ValueError()

    def render_trail(self, ball_control, world_num=1):
        self.run(fps=5000, ball_control=ball_control, quiet=True, leave_trail=True, world_num=world_num)
        data = pg.image.tostring(self.surface, 'RGBA')
        img = Image.frombytes('RGBA', (self.w, self.h), data)
        return img

    def show_trail(self, ball_control, world_num=1):
        img = self.render_trail(ball_control, world_num)
        scale = 100
        fig, ax = plt.subplots(figsize=(self.w/scale, self.h/scale))
        ax.grid(None)
        ax.imshow(np.array(img), interpolation=None)
        plt.show()

    def run(self, fps=20, ball_control=None, quiet=False, leave_trail=True, world_num=1):
        """
        Args:
            ball_control: either a function of the y position of the ball which
                returns the current velocity, or None to have the ball
                controlled interactively.
        """
        # simulation data
        physics_delta = 0.1
        ball = Ball(self.w/2, self.h)
        ball_a = -50  # ball upwards acceleration

        world = self.get_world(world_num)

        trail = [] # record the (x,y) locations of the ball
        physics_timer = 0
        wallclock_timer = time.time()
        self.surface.fill(BLACK)
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.display.quit()
                    pg.quit()
                    sys.exit()
                elif ball_control is None and event.type == pg.KEYDOWN: # no ball control => interactive
                    if event.key == pg.K_LEFT:
                        ball.vel.x = -100
                    elif event.key == pg.K_RIGHT:
                        ball.vel.x = 100

            if not leave_trail:
                self.surface.fill(BLACK)

            ball.vel.y += ball_a * physics_delta
            if ball_control is not None:
                state = self.h - ball.pos.y # from 0 to self.h
                ball.pos.x = ball_control(state)
                #ball.vel.x = ball_control(state)
                #ball.vel.x += ball_control(state) * physics_delta

            ball.physics_update(physics_delta)
            trail.append((ball.pos.x, ball.pos.y))
            ball.draw(self.surface)

            for wall in world:
                wall.draw(self.surface)

            # ball reached the top
            if ball.pos.y <= 0:
                ball.pos.y = 0
                break
            # ball went off the screen
            if ball.pos.x <= 0 or ball.pos.x >= self.w:
                break
            # ball collided with a wall
            if ball.is_colliding(world):
                break

            physics_timer += physics_delta
            self.screen.blit(self.surface, (0, 0))
            pg.display.flip()
            self.clock.tick(fps) # limit the FPS

        score = self.h - ball.pos.y
        if not quiet:
            real_time = time.time() - wallclock_timer
            print('simulation finished with score {} in {:.1f}s ({:.1f} realtime)'.format(score, physics_timer, real_time))
        return score, trail

def main():
    sim = Simulation()
    try:
        world_num = int(sys.argv[1])
    except ValueError:
        print('usage: maze.py <world_num>')
        sys.exit(1)
    sim.run(fps=20, ball_control=None, quiet=False, world_num=world_num)

if __name__ == '__main__':
    main()
