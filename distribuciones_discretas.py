import numpy as np
from scipy.stats import binom

class Binomial:
    def __init__(self, n, p):
        self.n = n
        self.p = p

    def media(self):
        return self.n * self.p

    def varianza(self):
        return self.n * self.p * (1 - self.p)

    def desviacion_estandar(self):
        return np.sqrt(self.varianza())

    def pmf(self, k):
        return binom.pmf(k, self.n, self.p)

    def cdf(self, k):
        return binom.cdf(k, self.n, self.p)