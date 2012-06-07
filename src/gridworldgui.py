#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: GRIDWORLD.PY
Date: Monday, January 11 2010
Description: A class for creating and exploring gridworlds.
"""

import os, sys, getopt, pdb, string
import time

import pygame
import pygame.locals as pgl

import numpy as np
import random as pr
from gridworld8 import Gridworld8 as Gridworld

class GridworldGui( Gridworld ):

    def __init__(self, *args, **kwargs):
        # initialize the base gridworld class (no gui)
        Gridworld.__init__(self, *args, **kwargs)

        nrows = 5
        ncols = 5
        size = 32
        # compute the appropriate height and width (with room for cell borders)
        self.height = nrows * size + nrows + 1
        self.width = ncols * size + ncols + 1
        self.size = size

        # initialize pygame ( SDL extensions )
        pygame.init()
        pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Gridworld')
        self.screen = pygame.display.get_surface()
        self.surface = pygame.Surface(self.screen.get_size())

        self.background()
        self.screen.blit(self.surface, (0,0))
        pygame.display.flip()

        self.updategui = True # switch to stop updating gui if you want to collect a trace quickly

    def state2coord(self, state, center = False):
        # state is a number between 0 and nstates

        i,j = self.states[state]

        return self.indx2coord(i, j, center)

    def state2circle(self, state, bg = True, blit=True):
        if bg:
            self.background()

        x,y = self.state2coord(state, center=True)
        pygame.draw.circle(self.surface, (255,0,0), (y,x), self.size / 2)

        if blit:
            self.screen.blit(self.surface, (0,0))
            pygame.display.flip()


    def drawvalues(self, state, w):

        # write the value for each state for debugging purposes

        a = self.linear_policy(w,state)
        val = np.dot(w, self.phi(state,a))

        #font = pygame.font.SysFont("Times New Roman", 10)
        font = pygame.font.SysFont("FreeSans", 10)
        txt = font.render("%.1f" % val, True, (0,0,0))
        x,y = self.state2coord(state, center=False)

        self.surface.blit(txt, (y,x))

    def drawarrow(self, state, action):
        # observe the current best policy?

        # Note: template alread in "graphics" coordinates
        template = np.array([(-8,0),(0,0),(8,0),(0,8),(8,0),(0,-8)])
        x,y = self.state2coord(state, center = True)

        #         0 : left
        #         1: up left
        #         2 : up
        #         3 : up right
        #         4 : right
        #         5 : right down
        #         6 : down
        #         7 : down left

        v = 1.0 / np.sqrt(2)
        rot45 = np.array([(v,v),(-v,v)])


        rot90 = np.array([(0,1),(-1,0)])

        if action == 0:
            template = np.dot(template,rot90)
            template = np.dot(template,rot90)

        if action == 1:
            template = np.dot(template, rot90)
            template = np.dot(template, rot90)
            template = np.dot(template, rot45)

        if action == 2:
            template = np.dot(template,rot90)
            template = np.dot(template,rot90)
            template = np.dot(template,rot90)

        if action == 3:
            template = np.dot(template,rot90)
            template = np.dot(template,rot90)
            template = np.dot(template,rot90)
            template = np.dot(template,rot45)

        if action == 4:
            template = template

        if action == 5:
            template = np.dot(template, rot45)

        if action == 6:
            template = np.dot(template,rot90)

        if action == 7:
            template = np.dot(template, rot90)
            template = np.dot(template, rot45)

        arrowpoints = [(y + z[0],x + z[1]) for z in template]
        pygame.draw.lines(self.surface,(0,255,0),0, arrowpoints, 1)

    def save(self, filename):
        pygame.image.save(self.surface, filename)


    def callback(self, *args, **kwargs):
        iter, current = args[0], args[1]
        self.background()


        for s in range(self.nstates):
            a = self.linear_policy(current,s)
            self.drawarrow(s,a)
            self.drawvalues(s,current)


        self.screen.blit(self.surface, (0,0))
        pygame.display.flip()

    def indx2coord(self,i,j, center = False):
        # the +1 indexing business is to ensure that the grid cells
        # have borders of width 1px

        if center:
            return i * (self.size + 1) + 1 + self.size / 2, \
                j * (self.size + 1) + 1 + self.size /2
        else:
            return i * (self.size + 1) + 1, j * (self.size + 1) + 1

    def coord2indx(self,x,y):
        return x / (self.size + 1), y / (self.size + 1)

    def background(self):

        self.surface.fill((0,0,0))
        for s in range(self.nstates):
            i,j = self.states[s]
            x,y = self.indx2coord(i,j)
            coords = pygame.Rect(y,x,self.size,self.size)
            pygame.draw.rect(self.surface, (255,255,255), coords)

            # some code to draw "lake" cells blue
            if hasattr(self, 'lake'):
                if s in self.lake:
                    pygame.draw.rect(self.surface,(0,0,255), coords)
            if hasattr(self, 'goal'):
                if s == self.goal:
                    pygame.draw.rect(self.surface,(0,255,0), coords)

    def ml_circle(self,x,y):
        # draw a circle in the grid cell corresponding to the cursor
        # location.  this was used to test the gui code and may be
        # useful for future gui interaction (placing obstacles etc.).

        # find the enclosing grid square
        i,j = self.coord2indx(y,x)

        # compute the grid center
        x,y = self.indx2coord(i,j,center = True)

        s = i * self.ncols + j

        # need to examine what states are aliased under this clustering scheme (i.e. what are the clusters?)
        if s in range(self.nstates):
            print s, self.phi(s,0), self.phi(s,1), self.phi(s,2), self.phi(s,3)

        pygame.draw.circle(self.surface, (255,0,0), (y,x), self.size / 2)


    def move(self, a, obs = False):

        (previous, a, reward, current) = Gridworld.move(self, a)

        if self.updategui:
            self.state2circle(current)


        return (previous, a, reward, current)

    def mainloop(self, w):

        # TODO: interative method to explore feature space

        self.callback(0,w)
        self.screen.blit(self.surface,(0,0))
        pygame.display.flip()

        while True:
            for event in pygame.event.get():
                if event.type == pgl.QUIT:
                    sys.exit()
                elif event.type == pgl.KEYDOWN and event.key == pgl.K_ESCAPE:
                    sys.exit()
                elif event.type == pgl.MOUSEMOTION:

                    # self.background()
                    #print type
                    self.callback(0,w)
                    self.ml_circle(*event.pos)

                    # do this again to make the circle visible
                    self.screen.blit(self.surface, (0,0))
                    pygame.display.flip()


                else:
                    print event

            self.screen.blit(self.surface,(0,0))
            pygame.display.flip()

    def run_agent(self,w):
        """
        This method runs the agent's policy from every state in the GUI for visual examination.
        """


        for i in range(self.nstates):

            print "Testing state %d" % i

            self.state2circle(i)
            self.current = i

            time.sleep(1)

            steps = 0
            while not (self.current in self.endstates) or steps > 5:

                action = self.linear_policy(w,self.current)
                print action
                print self.move(action)
                time.sleep(1)
                steps += 1



if __name__ == '__main__':

    gw = GridworldGui()
    #gw.save("gridworld.png")
    t = gw.trace(1000)
