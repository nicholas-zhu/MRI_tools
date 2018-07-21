import numpy as np


'''
copy from sigpy
'''

class Linop(object):
    '''
    linear operator class
    forward operator
    adjoint operator
    '''

    def __init__(self, oshape, ishape, name=None):
        self.oshape = list(oshape)
        self.ishape = list(ishape)

        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name

    def _check_domain(self, input):

        for i1, i2 in zip(input.shape, self.ishape):
            if i2 != -1 and i1 != i2:
                raise ValueError('input shape mismatch for {s}, got {input_shape}'.format(
                    s=self, input_shape=input.shape))

    def _check_codomain(self, output):

        for o1, o2 in zip(output.shape, self.oshape):
            if o2 != -1 and o1 != o2:
                raise ValueError('output shape mismatch for {s}, got {output_shape}'.format(
                    s=self, output_shape=output.shape))

    def _apply(self, input):
        raise NotImplementedError

    def apply(self, input):
        self._check_domain(input)
        with util.get_device(input):
            output = self._apply(input)
        self._check_codomain(output)

        return output
    
    def _adjoint_linop(self):
        raise NotImplementedError
    
    @property
    def H(self):
        return self._adjoint_linop()

    def __call__(self, input):
        return self.__mul__(input)

    def __mul__(self, input):
        if isinstance(input, Linop):
            return Compose([self, input])
        elif np.isscalar(input):
            M = Multiply(self.ishape, input)
            return Compose([self, M])
        elif isinstance(input, util.get_xp(input).ndarray):
            return self.apply(input)

        return NotImplemented

    def __rmul__(self, input):

        if np.isscalar(input):
            M = Multiply(self.oshape, input)
            return Compose([M, self])
        
        return NotImplemented

    def __add__(self, input):

        if isinstance(input, Linop):
            return Add([self, input])
        else:
            raise NotImplementedError

    def __neg__(self):

        return -1 * self

    def __sub__(self, input):

        return self.__add__(-input)

    def __repr__(self):

        return '<{oshape}x{ishape}> {repr_str} Linop>'.format(
            oshape=self.oshape, ishape=self.ishape, repr_str=self.repr_str)
    
class Identity(Linop):
    '''Identity linear operator.
    Args:
        shape : input shape
    '''

    def __init__(self, shape):

        super().__init__(shape, shape)

    def _apply(self, input):
        return input

    def _adjoint_linop(self):
        return self


class Move(Linop):
    '''Move input between devices.
    Args:
        shape: input/output shape.
        odevice - output device
        idevice - input device
    '''

    def __init__(self, shape, odevice, idevice):

        self.odevice = odevice
        self.idevice = idevice
        
        super().__init__(shape, shape)

    def _apply(self, input):
        return util.move(input, self.odevice)
    
    def _adjoint_linop(self):
        return Move(self.ishape, self.idevice, self.odevice)


class Conj(Linop):
    '''
    Complex conjugate of linear operator.
    '''
    def __init__(self, A):

        self.A = A

        super().__init__(A.oshape, A.ishape, repr_str=A.repr_str)

    def _apply(self, input):

        device = util.get_device(input)
        with device:
            input = device.xp.conj(input)

        output = self.A._apply(input)

        device = util.get_device(output)
        with device:
            return device.xp.conj(output)
        
    def _adjoint_linop(self):
        return Conj(self.A.H)


class Add(Linop):
    '''Addition of linear operators.
    ishape, and oshape must match.
    Parameters
    ----------
    linops - list of linear operators
    Returns: linops[0] + linops[1] + ... + linops[n - 1]
    '''

    def __init__(self, linops):
        
        _check_linops_same_ishape(linops)
        _check_linops_same_oshape(linops)

        self.linops = linops
        oshape = linops[0].oshape
        ishape = linops[0].ishape

        super().__init__(oshape, ishape,
                         repr_str=' + '.join([linop.repr_str for linop in linops]))

    def _apply(self, input):

        output = 0
        for linop in self.linops:
            outputi = linop._apply(input)
            with util.get_device(outputi):
                output += outputi

        return output
        
    def _adjoint_linop(self):
        return Add([linop.H for linop in self.linops])


def _check_compose_linops(linops):
    
    for linop1, linop2 in zip(linops[:-1], linops[1:]):
        if (linop1.ishape != linop2.oshape):
            raise ValueError('cannot compose {linop1} and {linop2}.'.format(
                linop1=linop1, linop2=linop2))


def _combine_compose_linops(linops):

    combined_linops = []
    for linop in linops:
        if isinstance(linop, Compose):
            combined_linops += linop.linops
        else:
            combined_linops.append(linop)

    return combined_linops

        
class Compose(Linop):
    '''
    Composition of linear operators.
    Parameters
    ----------
    linops
    Returns: linops[0] * linops[1] * ... * linops[n - 1]
    '''

    def __init__(self, linops):

        _check_compose_linops(linops)
        self.linops = _combine_compose_linops(linops)

        super().__init__(self.linops[0].oshape, self.linops[-1].ishape,
                         repr_str=' * '.join([linop.repr_str for linop in linops]))

    def _apply(self, input):

        output = input
        for linop in self.linops[::-1]:
            output = linop._apply(output)
            linop._check_codomain(output)
        
        return output

    def _adjoint_linop(self):
        return Compose([linop.H for linop in self.linops[::-1]])