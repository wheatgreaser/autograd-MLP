import numpy as np
import matplotlib.pyplot as plt
import math
import random

class Value:
    def __init__(self, data, prev={}, op = ''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self.prev = prev
        self.op = op

    def __repr__(self):
        return f"Value = {self.data}"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, {self, other}, '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, {self, other}, '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other
    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self + (-1 * other)
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out
    def tanh(self):
        n = self.data
        t = (math.exp(2 * n) - 1)/(math.exp(2 * n) + 1)
        out = Value(t, {self}, 'tanh')
        def _backward():
            self.grad += out.grad * (1 - t**2)
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
    
    def __call__(self, x):
        out = sum((wi * xi for wi, xi in zip(self.w,x)), self.b)
        return out.tanh()
    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    def parameters(self):
        params = []
        for neuron in self.neurons:
            ps = neuron.parameters()
            params.extend(ps)
        return params

    
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    

n = MLP(3, [4, 4, 1])

xs = [[2.0, 3.0, -1.0],
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 2.0, -0.5]]
ys = [1.0, -1.0, -1.0, 1.0]
vs = []
ms = []
vms = []
mms = []
for _ in range(len(n.parameters())):
    vs.append(0)
    ms.append(0)
    vms.append(0)
    mms.append(0)
for k in range(50):
    ypred = [n(x) for x in xs]
    loss = sum((yp - ytrue)**2 for yp, ytrue in zip(ypred, ys))
    for p in n.parameters():
        p.grad = 0
    loss.backward()
    
    for i,p in enumerate(n.parameters()):
        ms[i] = (0.9 * ms[i]) + ((0.1)* p.grad)
        vs[i] = (0.999 * vs[i]) + ((0.001)* (p.grad ** 2))
        mms[i] = ms[i] / (1 - 0.9**(k+1))
        vms[i] = vs[i] / (1 - 0.999**(k+1))
        p.data += -0.05 * (mms[i] / (math.sqrt(vms[i]) + 1e-8))
    print(k, loss.data)



print(ypred)


    

