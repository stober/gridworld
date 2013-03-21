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
from gridworld8 import SparseRBFGridworld8,SparseGridworld8,ObserverGridworld,SparseAliasGridworld8,RBFObserverGridworld
from gridworld8 import wall_pattern

def gridworld_gui_factory(baseclass):     
    """     
    In order to make it easy to add a gui to any gridworld, this factory method modifies the baseclass (Gridworld) so that the subclass is a Gui+Gridworld or the desired type.     
    """

    class GridworldGui(baseclass):

        def __init__(self, *args, **kwargs):
            size=16
            # initialize the base gridworld class (no gui)
            super(GridworldGui, self).__init__(*args, **kwargs)
            self.initialize_pygame(size=size)

        def initialize_pygame(self,size=16):
            nrows = self.nrows
            ncols = self.ncols
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

        def coord2state(self, coord):
            i,j = self.coord2indx(coord[1],coord[0])
            return self.rstates[(i,j)]

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

        def normalize(self, minval, maxval, v):
            if float(maxval - minval) != 0:
                n = float(v - minval) / float(maxval - minval)
            else:
                n = 0
            #n = float(v + np.abs(minval)) / float(np.abs(minval) + float(np.abs(maxval)))
            return 255 - int(n * 255.0)

        def set_heatmap(self, m):
            self.bg_rendered = False
            maxval = np.max(m)
            minval = np.min(m)
            self.heatmap = []
            for v in m:
                color = (255,255,self.normalize(minval,maxval,v)) # uses blue for now
                self.heatmap.append(color)

        def set_colormap(self, sc):
            """
            Recieves a dict of the form state -> color and colors the grid accordingly. 
            """
            self.bg_rendered = False
            self.colormap = {}
            for s,c in sc.items():
                self.colormap[s] = c

        def draw_colormap(self, surface):
            for s in self.sindices:
                x,y = self.state2coord(s)
                color = self.colormap.get(s,(255,255,255))
                coords = pygame.Rect(y,x,self.size,self.size)
                pygame.draw.rect(surface, color, coords)

        def draw_heatmap(self, surface):
            """
            self.heatmap needs to be set
            """

            for s in self.sindices:
                x,y = self.state2coord(s)
                color = self.heatmap[s]
                coords = pygame.Rect(y,x,self.size,self.size)
                pygame.draw.rect(surface, color, coords)

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
                arrowpoints = [(y + z[0],x + z[1]) for z in self.t[self.allowed_actions[a]]]
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

            pi = np.zeros(self.nstates, dtype=int)
            for s in range(self.nstates):
                pi[s] = self.linear_policy(current,s)

            self.set_arrows(pi)
            self.background()
            self.redraw()

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
                for s,(i,j) in self.states.items():
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

                if hasattr(self, 'heatmap'):
                    self.draw_heatmap(self.bg)

                if hasattr(self, 'colormap'):
                    self.draw_colormap(self.bg)

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

            (previous, a, reward, current) = super(GridworldGui,self).move( a)

            if self.updategui:
                self.state2circle(current)

            return (previous, a, reward, current)

        def trace(self, tlen = 1000, policy = None, obs = False, show = True, additional_endstates = None, reset_on_cycle = False, reset_on_endstate = False, stop_on_cycle = False):
            
            t = None
            kwargs = {'tlen' : tlen, 
                      'policy' : policy, 
                      'obs' : obs,
                      'additional_endstates' : additional_endstates,
                      'reset_on_cycle' : reset_on_cycle,
                      'reset_on_endstate' : reset_on_endstate,
                      'stop_on_cycle' : stop_on_cycle}
            if not show:
                pval = self.updategui
                self.updategui = False
                t = super(GridworldGui,self).trace(**kwargs)
                self.updategui = pval
            else:
                t = super(GridworldGui,self).trace(**kwargs)
            return t

        def mainloop(self):

            self.screen.blit(self.surface,(0,0))
            pygame.display.flip()
            self.state2circle(self.current)

            while True:
                for event in pygame.event.get():
                    if event.type == pgl.QUIT:
                        sys.exit()
                    elif event.type == pgl.KEYDOWN and event.key == pgl.K_ESCAPE:
                        sys.exit()
                    elif event.type == pgl.MOUSEMOTION:
                        pass
                    elif event.type == pgl.KEYDOWN and event.key == pgl.K_DOWN:
                        self.move(4)
                    elif event.type == pgl.KEYDOWN and event.key == pgl.K_UP:
                        self.move(0)
                    elif event.type == pgl.KEYDOWN and event.key == pgl.K_LEFT:
                        #self.move(2)
                        self.move(1)
                    elif event.type == pgl.KEYDOWN and event.key == pgl.K_RIGHT:
                        self.move(6)

                    elif event.type == pgl.MOUSEBUTTONDOWN:
                        state = self.coord2state(event.pos)
                        obs = self.observe(state)
                        print state,obs
                        self.test_rbf(state)

                    else:
                        pass

                self.screen.blit(self.surface,(0,0))
                pygame.display.flip()

        def test_rbf(self, s):
            if not hasattr(self, 'rbf_loc'):
                return
            else:
                #obs = self.observe(s)
                r = self.rfunc(s)
                print len(r)
                # t = np.zeros(self.nstates)
                # t[s] = 1.0
                print r
                self.set_heatmap(r)
                self.background()

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

    return GridworldGui

# Here we use the factory to construct our desired subclasses using different Gridworld base classes.
GridworldGui = gridworld_gui_factory(SparseGridworld8)
RBFGridworldGui = gridworld_gui_factory(SparseRBFGridworld8)
ObserverGridworldGui = gridworld_gui_factory(ObserverGridworld)
AliasGridworldGui = gridworld_gui_factory(SparseAliasGridworld8)
RBFObserverGridworldGui = gridworld_gui_factory(RBFObserverGridworld)

if __name__ == '__main__':

    walls = wall_pattern(64,64)
    gw = GridworldGui(nrows = 64, ncols = 64, walls = walls, endstates = [0], size=8)
    t = gw.trace(1000)
    gw.mainloop()
    
    #gw.save("gridworld.png")
    #t = gw.trace(1000)
    #gw.test_drawactions()
