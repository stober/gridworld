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
#from gridworld8 import Gridworld8 as Gridworld
from gridworld8 import SparseGridworld8 as Gridworld

class GridworldGui( Gridworld ):

    def __init__(self, *args, **kwargs):
        # initialize the base gridworld class (no gui)
        Gridworld.__init__(self, *args, **kwargs)

        nrows = self.nrows
        ncols = self.ncols
        size = 16
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
        self.bg = pygame.Surface(self.screen.get_size())
        self.bg_rendered = False # optimize background render

        pdb.set_trace()
        self.background()
        self.screen.blit(self.surface, (0,0))
        pygame.display.flip()

        self.build_templates()
        self.updategui = True # switch to stop updating gui if you want to collect a trace quickly

    def draw_state_labels(self):
        
        font = pygame.font.SysFont("FreeSans", 10)
        for k,v in self.states.items():
            x,y = self.indx2coord(v[0],v[1],False)
            txt = font.render("%d" % k, True, (0,0,0))
            self.surface.blit(txt, (y,x))

        self.screen.blit(self.surface, (0,0))
        pygame.display.flip()

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


    def draw_values(self, vals):
        """
        vals: a dict with state labels as the key
        """
        font = pygame.font.SysFont("FreeSans", 10)

        for k,v in self.states.items():
            x,y = self.indx2coord(v[0],v[1],False)
            v = vals[k]
            txt = font.render("%.1f" % v, True, (0,0,0))
            self.surface.blit(txt, (y,x))

        self.screen.blit(self.surface, (0,0))
        pygame.display.flip()


    def draw_linear_values(self, state, w):

        # write the value for each state for debugging purposes

        a = self.linear_policy(w,state)
        val = np.dot(w, self.phi(state,a))

        #font = pygame.font.SysFont("Times New Roman", 10)
        font = pygame.font.SysFont("FreeSans", 10)
        txt = font.render("%.1f" % val, True, (0,0,0))
        x,y = self.state2coord(state, center=False)

        self.surface.blit(txt, (y,x))

    def set_arrows(self, pi):
        self.bg_rendered = False # rerender background
        self.arrows = pi

    def draw_arrows(self,surface):
        """
        self.arrows needs to be set.
        """

        for s in self.sindices:
            a = self.arrows[s]
            x,y = self.state2coord(s, center = True)
            arrowpoints = [(y + z[0],x + z[1]) for z in self.t[a]]
            pygame.draw.lines(surface,(55,55,55),0, arrowpoints, 1)



    def build_templates(self):

        # Note: template already in "graphics" coordinates
        template = np.array([(-1,0),(0,0),(1,0),(0,1),(1,0),(0,-1)])
        template = self.size / 3 * template # scale template

        v = 1.0 / np.sqrt(2)
        rot90 = np.array([(0,1),(-1,0)])
        rot45 = np.array([(v,-v),(v,v)]) # neg


        
        # align the template with the first action.
        t0 = np.dot(template, rot90) 
        t0 = np.dot(t0, rot90)
        t0 = np.dot(t0, rot90)
        
        t1 = np.dot(t0, rot45)
        t2 = np.dot(t1, rot45)
        t3 = np.dot(t2, rot45)
        t4 = np.dot(t3, rot45)
        t5 = np.dot(t4, rot45)
        t6 = np.dot(t5, rot45)
        t7 = np.dot(t6, rot45)

        self.t = [t0,t1,t2,t3,t4,t5,t6,t7]
        

    def save(self, filename):
        pygame.image.save(self.surface, filename)

    def test_drawactions(self):

        while True:
            for event in pygame.event.get():
                if event.type == pgl.QUIT:
                    sys.exit()
                elif event.type == pgl.KEYDOWN and event.key == pgl.K_ESCAPE:
                    sys.exit()


            for s in range(self.nstates):
                a = 1
                self.draw_arrow(s,a)

            self.screen.blit(self.surface, (0,0))
            pygame.display.flip()

    def redraw(self):
        self.screen.blit(self.surface, (0,0))
        pygame.display.flip()
    

    def callback(self, *args, **kwargs):
        iter, current = args[0], args[1]
        self.background()


        for s in range(self.nstates):
            a = self.linear_policy(current,s)
            self.draw_arrow(s,a)
            #self.drawvalues(s,current)


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

        if self.bg_rendered:
            self.surface.blit(self.bg,(0,0))
        else:
            self.bg.fill((0,0,0))
            for s in range(self.nstates):
                i,j = self.states[s]
                x,y = self.indx2coord(i,j)
                coords = pygame.Rect(y,x,self.size,self.size)
                pygame.draw.rect(self.bg, (255,255,255), coords)

            # some code to draw "lake" cells blue
                if hasattr(self, 'lake'):
                    if s in self.lake:
                        pygame.draw.rect(self.bg,(0,0,255), coords)
                if hasattr(self, 'endstates'):
                    if s in self.endstates:
                        pygame.draw.rect(self.bg,(0,255,0), coords)
            if hasattr(self, 'arrows'):
                self.draw_arrows(self.bg)
            self.bg_rendered = True # don't render again unless flag is set
            self.surface.blit(self.bg,(0,0))
                              
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


    def follow(self, s, policy):
        self.current = s
        while not self.current  in self.endstates:
            print policy(self.current)
            self.move(policy(self.current))
            time.sleep(0.5)
            

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


    gw = GridworldGui(ncols = 32, nrows = 64, walls = [], endstates = [])
    #gw.save("gridworld.png")
    #t = gw.trace(1000)
    gw.test_drawactions()
