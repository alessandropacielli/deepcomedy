"""
This file contains a composable implementation of a Predicate :: a -> bool, and a decorator that wraps any function into a predicate

Example usage:
    TODO


Reference:
https://stackoverflow.com/questions/9184632/pointfree-function-combination-in-python

For FP enthusiasts :)
"""

from functools import update_wrapper

class Predicate(object):
    def __init__(self, predicate):
        self.predicate = predicate

    def __call__(self, obj):
        return self.predicate(obj)

    def __copy_pred(self):
        return copy.copy(self.predicate)

    def __and__(self, predicate):
        def func(obj):
            return self.predicate(obj) and predicate(obj)
        return Predicate(func)

    def __or__(self, predicate):
        def func(obj):
            return self.predicate(obj) or predicate(obj)
        return Predicate(func)
    

def predicate(func):
    """Decorator that constructs a predicate (``Predicate``) instance from
    the given function."""
    result = Predicate(func)
    
    # Retain func metadata
    update_wrapper(result, func)
    
    return result
