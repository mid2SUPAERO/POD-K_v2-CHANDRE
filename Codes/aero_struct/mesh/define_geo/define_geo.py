# -*- coding: utf-8 -*-
"""
Script to create an aircraft wing type Airbus.

@author: gberthel
"""
import numpy as np
class airbus_wing():
    def __init__(self,b = 64.75,S = 443.,phi = 30.*np.pi/180.,diedre = 1.5*np.pi/180.,BF = 5.96,Mach = 0.85):
        self.BF = BF
        self.b = b
        self.S = S
        self.phi = phi
        self.diedre = diedre
        self.eta_cassure = 0.4
        self.Y_4 = self.b/2.
        self.Y_3 = self.eta_cassure * self.Y_4
        self.Y_2 = self.BF / 2.
        self.eps_1 = 0.38
        self.Mach = Mach
        self.ep_aero = 0.89 - (self.Mach + 0.02) * np.sqrt(np.cos(self.phi))
        self.twist = 0.
    
    def compute_diedre(self):
        return self.diedre
    
    def compute_span(self):
        self.span = self.Y_4 - self.Y_2
        span_1 = self.Y_3 - self.Y_2
        span_2 = self.span - span_1
        return span_1, span_2
        
    def compute_cord(self):
        self.L_1 = (self.S-(self.Y_3-self.Y_2)*(self.Y_3+self.Y_2)*np.tan(self.phi)) / \
        ((1+self.eps_1)*self.span+self.BF-(3.*(1.-self.eps_1)*(self.Y_3-self.Y_2)*(self.Y_3+self.Y_2)/(2.*(self.b+self.BF))))
        self.L_2 = self.L_1+(self.Y_3-self.Y_2)*(np.tan(self.phi)-(3.*(1-self.eps_1)*self.L_1)/(4.*(self.span)))
        self.L_4 = self.eps_1*self.L_1
        self.L_3 = self.L_4+(self.L_1-self.L_4)*(self.Y_4-self.Y_3)/(self.Y_4-self.Y_2)
        return self.L_1,self.L_2,self.L_3,self.L_4
    
    def compute_ep(self):
        e_emplanture = 1.24 * self.ep_aero * self.L_2
        e_cassure = 0.94 * self.ep_aero * self.L_3
        e_extremite = 0.86 * self.ep_aero * self.L_4
        return e_emplanture, e_cassure, e_extremite
    
    def compute_phi(self):
        return self.phi, self.phi
    
    def compute_twist(self):
        return self.twist, self.twist, self.twist