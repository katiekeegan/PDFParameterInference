
"""
Module to provide implementations of some common probability distributions
mainly developed to test the idea of distributional differentiability as
needed by the QuantOm project developments.

Author: A. Attia (attia@anl.gov)
"""

import numbers
import math
import numpy as np
from scipy.optimize import fsolve  # function solve (x at which f=0)
from scipy.interpolate import CubicSpline
from scipy.special import erfinv  # Inverse of the error function (for evaluating 1D ICDF of a Gaussian)

class Distribution():
    """
    Base class for probability distributions developed here
    """


class OneDimensionalDistribution():
    """
    Base class for One-Dimensional probability distributions developed here
    """

    def numeric_icdf(self, p, x_grid=None, cdf_vals=None, interpolate_first=True):
        """
        Evaluate (numerically), the inverse of the cumulative density function for a given probability `p`

        :param float p: cumulative probability; must belong to [0, 1]
        :param iterable|None x_grid: if passed, coordinate points at which `cdf_vals` are (or to be) evaluated
        :param iterable|None cdf_vals: if passed, values of the cumulative density at each of the `x_grid`
        :param bool interpolate_first: If true, a cubic spline `cdf := g(x)` is fitted for the CDF,
            and is then used to find `x` by solving function ::math:`g(x)-p=0` to find the point `x` at which
            the cubic spline gives CDF value of `p`.

        ..note::
            Inverse CDF is valid only for one-dimensional distribution as the inverse of CDF is not a
            valid function in multidimensions.

        ..note::
            This implementation will be valid for any one dimensional distribution, and is thus made available
            in this base class.
        """
        if x_grid is cdf_vals is None:
            pass

        elif cdf_vals is None:
            print("Can't pass `cdf_vals` without `x_grid`; the passed `cdf_vals` are discarded")
            cdf_vals = None

        elif x_grid is not None and cdf_vals is not None:
            assert len(x_grid) == len(cdf_vals), "Both `x_grid` and `cdf_vals` must be iterables of same length!"

        if x_grid is None:
            width = 4 * math.sqrt(self.variance)
            x_grid = np.linspace(self.mean - width, self.mean + width, 100)

        if cdf_vals is None:
            pdf_vals = [self.pdf(x) for x in x_grid]
            cdf_vals = [sum(pdf_vals[: i]) for i in range(1, len(pdf_vals)+1)]

        # One of the two approaches for interpolation
        if interpolate_first:
            # Interpolate the CDF
            cdf_func = CubicSpline(x_grid, cdf_vals)

            # Solve for the inverse at probability p
            x = fsolve(lambda r: cdf_func(r)-p, p)

        else:
            # Numerically evaluate inverse CDF from CDF values
            # Numerically interpolate p from given CDF values considering x are the (ICDF) function values
            x = np.interp(p, xp=cdf_vals, fp=x_grid)

        return x


class OneDimensionalGaussian(OneDimensionalDistribution):
    """
    A one dimensional Normal/Gaussian distribution.
    """

    def __init__(self, mean=0.0, variance=1.0):
        """
        Initialize a one-dimensional normal (1D Gaussian) distribution
        with passed mean and variance
        """
        self.mean = mean
        self.variance = variance

    def pdf(self, x, normalize=True):
        """
        Evaluate the probability density function value at the passed coordinate point `x`.

        :param float x: coordinate point (realization of the random variable)
        :param bool normalize: if `True`, multiply the density value by the normalization constant

        :returns: the value of the density function (normalized or unnormalized based on `normalize`)
        """
        assert isinstance(x, numbers.Number), f"The coordinate point must be a number; received {type(x)}"

        # Unnormalized density
        density = math.exp(-0.5 * (x-self.mean)**2 / (self.variance))

        # Normalization constant
        if normalize:
            norm = 1.0 / math.sqrt(2 * self.variance * math.pi)
            density *= normalize

        return density

    def cdf(self, x):
        """
        Evaluate (exactly) and return the value of the cumulative density function at a given point `x`

        :param float x: coordinate point (realization of the random variable)

        ..note::
            The CDF of a 1D normal distribution is
            ..math:
                \\frac{1}{2} \\left[ 1 + erf\\left( \\frac{x-\\mu}{\\sigma\\sqrt{2}} \\right) \\right]

            where ::math:`erf` is the error function

        ..note::
            The CDF maps the real space ( of ::math:`x`) to the probability space, that is
            CDF: ::math:`x \\rightarrow [0, 1]`
        """
        assert isinstance(x, numbers.Number), f"The coordinate point must be a number; received {type(x)}"
        return 0.5 * (1 + math.erf((x-self.mean) / (math.sqrt(2 * self.variance)) ) )

    def icdf(self, p):
        """
        Evaluate the inverse of the cumulative density function (ICDF) at a given probability;
        thus return the point ::math:`x` that achieves the passed cumulative probability/density.

        :param float p: cumulative probability (must be within ::math:`[0, 1]`)
        """
        assert isinstance(p, numbers.Number), f"The cumulative probability must be a number; received {type(p)}"
        assert 0 <=p <=1, f"Cumulative probability/density must be in [0, 1]; received {x}"

        return erfinv (p * 2 - 1) * math.sqrt(2*self.variance) + self.mean


    @property
    def mean(self):
        return self._mean
    @mean.setter
    def mean(self, val):
        assert isinstance(val, numbers.Number), f"The mean must be a number; received {type(val)}"
        self._mean = val

    @property
    def variance(self):
        return self._variance
    @variance.setter
    def variance(self, val):
        assert isinstance(val, numbers.Number), f"The variance must be a number; received {type(val)}"
        assert val > 0, f"The variance must be positive; received {val}"
        self._variance = val


