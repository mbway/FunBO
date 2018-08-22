#!/usr/bin/env python3
""" FunBO: continuous function Bayesian optimisation """

from . import utils
from .coordinator import *
from .optimiser import Optimiser, Trial
from .acquisition import UCB, RBF_weighted
from .aux_optimisers import *
from .surrogates import Surrogate, GPySurrogate, DistributedGPSurrogate
from .grid import *

