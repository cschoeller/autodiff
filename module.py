from typing import Sequence, Mapping

from variable import Variable


class Module:

    def __init__(self):
        self.parameters = []
        
    def zero_grad(self):
        for param in self.parameters:
            param.grad = 0
    
    def _add_params_recursive(self, value):
        if isinstance(value, Variable):
            self.parameters.append(value)
            return
        elif isinstance(value, Sequence):
            for v in value:
                self._add_params_recursive(v)
        elif isinstance(value, Mapping):
            for v in value.values():
                self._add_params_recursive(v)
        return
    
    def __setattr__(self, name, value):
        self._add_params_recursive(value)
        super().__setattr__(name, value)