import numpy as np
import tensorflow as tf
from .models


def generate_greedy(transformer, start_sequence, steps):
    
    for step in range(steps):
        
        output = evaluate(transformer, )
