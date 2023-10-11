class Variable():

    def __init__(self, val, prev = None, requires_grad = True):
        self.val = val
        self._backward = lambda: None
        self.grad = 0
        self._prev = [] if prev is None else prev
        self._requires_grad = requires_grad
        
    def __repr__(self):
        return self.__str__()
        
    def __str__(self):
        return f"Variable({self.val})"

    def _to_var(self, val):
        return val if isinstance(val, Variable) else Variable(val, requires_grad=False)
    
    def __add__(self, other):
        other = self._to_var(other)
        out = Variable(self.val + other.val, prev=[self, other])
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        
        return out
        
    def __radd__(self, other):
        return self.__add__(other)
        
    def __mul__(self, other):
        other = self._to_var(other)
        out = Variable(self.val * other.val, prev=[self, other])
        
        def _backward():
            self.grad += other.val * out.grad
            other.grad += self.val * out.grad
        out._backward = _backward
        
        return out
        
    def __rmul__(self, other):
        return self.__mul__(other)
   
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "__pow__ only supports int and float."
        other = self._to_var(other)
        out = Variable(self.val**other.val, prev=[self, other])
        
        def _backward():
           self.grad += other.val * (self.val ** (other.val - 1)) * out.grad
           # NOTE: Derivative of the exponent would involve complex logarithms.
        out._backward = _backward
        
        return out       
        
    def __rpow__(self, other):
        other = self._to_var(other)
        return other.__pow__(self)
        
    def __neg__(self):
        return -1 * self
        
    def __sub__(self, other):
        other = self._to_var(other)
        return self.__add__(-other)
        
    def __rsub__(self, other):
        other = self._to_var(other)
        return other.__add__(-self)
        
    def __truediv__(self, other):
        return self * other**(-1)
        
    def __rtruediv__(self, other):
        return other * self**(-1)
        
    def _bfs_traversal(self):
        node_queue = [self]
        while node_queue:
            node = node_queue.pop()
            node_queue[0:0] = node._prev # prepend
            yield node
    
    def backward(self):
        """Compute gradients by traversing the DAG with BFS."""
        if self._backward is None:
            raise AttributeError("No computation executed.")
          
        self.grad = 1.
        for node in self._bfs_traversal():
            if node._requires_grad:
                node._backward()
