# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np


class GM:
    """
    Generative Model class.
    
    Attributes
    ----------
    
    """

    def __init__(self, dx, dy, T):
        """Initialization method."""
        self.dx = dx
        self.dy = dy
        self.T = T
               
class SSM:
    """
    State Space Model class.
    
    Attributes
    ----------
    """
    
    def __init__(self):
        """Initialization method."""
        self.f = None
        self.g = None
        
  
class RSSSM:
    """
    Regime Switching State Space Model class.

    Attributes
    ----------
    """