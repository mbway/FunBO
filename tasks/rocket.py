#!/usr/bin/env python3

import sys
import time
import numpy as np
import pygame as pg
import pygame.locals as pgl
from collections import namedtuple

class Rocket:
    def __init__(self, x, y):
        self.pos = pg.math.Vector2(x, y)
        self.vel = pg.math.Vector2(0, 0)
        self.fuel = 1000
        self.max_power = 50

    def physics_update(self, delta):
        self.pos += self.vel*delta

    def draw(self, surface, power):
        p = 1 - power/self.max_power
        pg.draw.rect(surface, (255, 255*p, 255*p), self.get_rect())

    def get_rect(self):
        x, y = (round(self.pos.x), round(self.pos.y))
        w, h = 20, 60
        return pg.Rect(x-w/2, y-h/2, w, h)

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
        self.w, self.h = 500, 700
        self.screen = pg.display.set_mode((self.w, self.h), 0, 32) # resolution, flags, depth
        pg.display.set_caption('Simulation')
        self.surface = pg.Surface(self.screen.get_size()).convert()
        self.clock = pg.time.Clock()

    State = namedtuple('State', ['pos', 'vel', 'acc', 'fuel'])

    def run(self, fps=20, rocket_control=None, quiet=False):
        """
        Args:
        """
        # simulation data
        physics_delta = 0.1
        rocket = Rocket(self.w/2, 100)
        gravity = 30
        rocket_power = 0

        ground = Wall(0, self.h-100, 500, 100)

        states = []
        physics_timer = 0
        wallclock_timer = time.time()
        self.surface.fill((0, 0, 0))
        while True:
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.display.quit()
                    pg.quit()
                    sys.exit()
                elif rocket_control is None and event.type == pg.KEYDOWN: # no rocket control => interactive
                    if event.key == pg.K_UP:
                        rocket_power = rocket.max_power
                elif rocket_control is None and event.type == pg.KEYUP: # no rocket control => interactive
                    if event.key == pg.K_UP:
                        rocket_power = 0

            self.surface.fill((0, 0, 0))

            a = gravity

            if rocket_control is not None:
                rocket_power = rocket_control(physics_timer)

            if rocket.fuel > 0:
                a += -rocket_power
                rocket.fuel -= rocket_power * physics_delta
            else:
                rocket_power = 0

            rocket.vel.y += a * physics_delta

            rocket.physics_update(physics_delta)
            states.append(Simulation.State(rocket.pos.y, rocket.vel.y, a, rocket.fuel))
            rocket.draw(self.surface, rocket_power)

            ground.draw(self.surface)

            if rocket.is_colliding([ground]):
                break

            physics_timer += physics_delta
            self.screen.blit(self.surface, (0, 0))
            pg.display.flip()
            self.clock.tick(fps) # limit the FPS

        global_score = rocket.vel.y
        if not quiet:
            real_time = time.time() - wallclock_timer
            print('simulation finished with score {} in {:.1f}s ({:.1f} realtime)'.format(global_score, physics_timer, real_time))
        return global_score, states

def main():
    sim = Simulation()
    sim.run(fps=20, rocket_control=None, quiet=False)

if __name__ == '__main__':
    main()
