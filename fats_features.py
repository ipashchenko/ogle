import math
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from statsmodels.tsa import stattools
from scipy.interpolate import interp1d
import lomb


def get_all_subclasses(cls):
    """
    https://stackoverflow.com/a/17246726
    """
    all_subclasses = []

    for subclass in cls.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))

    return all_subclasses


class Feature(object):

    Data = ['magnitude']
    depends = set()

    @classmethod
    def get_subclass_name(cls):
        return cls.__name__

    def __init__(self, **kwargs):
        self._value = None

    def fit(self, data):
        raise NotImplementedError

    @property
    def value(self):
        return self._value

    def __str__(self):
        return self.__class__.__name__


class Amplitude(Feature):
    """Half the difference between the maximum and the minimum magnitude"""

    def fit(self, data):
        magnitude = data[0]
        N = len(magnitude)
        sorted_mag = np.sort(magnitude)

        self._value = (np.median(sorted_mag[-int(math.ceil(0.05 * N)):]) -
                       np.median(sorted_mag[0:int(math.ceil(0.05 * N))])) / 2.0


class Rcs(Feature):
    """Range of cumulative sum"""

    def fit(self, data):
        magnitude = data[0]
        sigma = np.std(magnitude)
        N = len(magnitude)
        m = np.mean(magnitude)
        s = np.cumsum(magnitude - m) * 1.0 / (N * sigma)
        R = np.max(s) - np.min(s)
        self._value = R


class StetsonK(Feature):

    Data = ['magnitude', 'error']

    def fit(self, data):
        magnitude = data[0]
        error = data[2]

        mean_mag = (np.sum(magnitude/(error*error)) /
                    np.sum(1.0 / (error * error)))

        N = len(magnitude)
        sigmap = (np.sqrt(N * 1.0 / (N - 1)) *
                  (magnitude - mean_mag) / error)

        K = (1 / np.sqrt(N * 1.0) *
             np.sum(np.abs(sigmap)) / np.sqrt(np.sum(sigmap ** 2)))

        self._value = K


class Meanvariance(Feature):
    """variability index"""

    def fit(self, data):
        magnitude = data[0]
        self._value = np.std(magnitude) / np.mean(magnitude)


class Autocor_length(Feature):

    def __init__(self, lags=100, **kwargs):
        super(Autocor_length, self).__init__(**kwargs)
        self.nlags = lags

    def fit(self, data):

        magnitude = data[0]
        AC = stattools.acf(magnitude, nlags=self.nlags)
        k = next((index for index, value in
                 enumerate(AC) if value < np.exp(-1)), None)

        while k is None:
            self.nlags = self.nlags + 100
            AC = stattools.acf(magnitude, nlags=self.nlags)
            k = next((index for index, value in
                      enumerate(AC) if value < np.exp(-1)), None)

        self._value = k


class SlottedA_length(Feature):

    Data = ['magnitude', 'time']

    def __init__(self, T=-99, **kwargs):
        """
        lc: MACHO lightcurve in a pandas DataFrame
        k: lag (default: 1)
        T: tau (slot size in days. default: 4)
        """
        super(SlottedA_length, self).__init__(**kwargs)
        SlottedA_length.SAC = []

        self.T = T

    def slotted_autocorrelation(self, data, time, T, K,
                                second_round=False, K1=100):

        slots = np.zeros((K, 1))
        i = 1

        # make time start from 0
        time = time - np.min(time)

        # subtract mean from mag values
        m = np.mean(data)
        data = data - m

        prod = np.zeros((K, 1))
        pairs = np.subtract.outer(time, time)
        pairs[np.tril_indices_from(pairs)] = 10000000

        ks = np.int64(np.floor(np.abs(pairs) / T + 0.5))

        # We calculate the slotted autocorrelation for k=0 separately
        idx = np.where(ks == 0)
        prod[0] = ((sum(data ** 2) + sum(data[idx[0]] *
                   data[idx[1]])) / (len(idx[0]) + len(data)))
        slots[0] = 0

        # We calculate it for the rest of the ks
        if second_round is False:
            for k in np.arange(1, K):
                idx = np.where(ks == k)
                if len(idx[0]) != 0:
                    prod[k] = sum(data[idx[0]] * data[idx[1]]) / (len(idx[0]))
                    slots[i] = k
                    i = i + 1
                else:
                    prod[k] = np.infty
        else:
            for k in np.arange(K1, K):
                idx = np.where(ks == k)
                if len(idx[0]) != 0:
                    prod[k] = sum(data[idx[0]] * data[idx[1]]) / (len(idx[0]))
                    slots[i - 1] = k
                    i = i + 1
                else:
                    prod[k] = np.infty
            np.trim_zeros(prod, trim='b')

        slots = np.trim_zeros(slots, trim='b')
        return prod / prod[0], np.int64(slots).flatten()

    def fit(self, data):
        magnitude = data[0]
        time = data[1]
        N = len(time)

        if self.T == -99:
            deltaT = time[1:] - time[:-1]
            sorted_deltaT = np.sort(deltaT)
            self.T = sorted_deltaT[int(N * 0.05)+1]

        K = 100

        [SAC, slots] = self.slotted_autocorrelation(magnitude, time, self.T, K)
        # SlottedA_length.SAC = SAC
        # SlottedA_length.slots = slots

        SAC2 = SAC[slots]
        SlottedA_length.autocor_vector = SAC2

        k = next((index for index, value in
                 enumerate(SAC2) if value < np.exp(-1)), None)

        while k is None:
            K = K+K

            if K > (np.max(time) - np.min(time)) / self.T:
                break
            else:
                [SAC, slots] = self.slotted_autocorrelation(magnitude,
                                                            time, self.T, K,
                                                            second_round=True,
                                                            K1=K/2)
                SAC2 = SAC[slots]
                k = next((index for index, value in
                         enumerate(SAC2) if value < np.exp(-1)), None)

        self._value = slots[k] * self.T


class StetsonK_AC(Feature):

    Data = ['magnitude', 'time', 'error']
    depends = {"SlottedA_length"}

    def __init__(self, **kwargs):
        super(StetsonK_AC, self).__init__(**kwargs)
        self.sl_al = kwargs["SlottedA_length"]

    def fit(self, data):

        autocor_vector = self.sl_al.autocor_vector

        # autocor_vector = autocor_vector[slots]
        N_autocor = len(autocor_vector)
        sigmap = (np.sqrt(N_autocor * 1.0 / (N_autocor - 1)) *
                  (autocor_vector - np.mean(autocor_vector)) /
                  np.std(autocor_vector))

        K = (1 / np.sqrt(N_autocor * 1.0) *
             np.sum(np.abs(sigmap)) / np.sqrt(np.sum(sigmap ** 2)))

        self._value = K


class StetsonL(Feature):
    Data = ['magnitude', 'time', 'error', 'magnitude2', 'error2']

    def fit(self, data):

        aligned_magnitude = data[4]
        aligned_magnitude2 = data[5]
        aligned_error = data[7]
        aligned_error2 = data[8]

        N = len(aligned_magnitude)

        mean_mag = (np.sum(aligned_magnitude/(aligned_error*aligned_error)) /
                    np.sum(1.0 / (aligned_error * aligned_error)))
        mean_mag2 = (np.sum(aligned_magnitude2/(aligned_error2*aligned_error2)) /
                     np.sum(1.0 / (aligned_error2 * aligned_error2)))

        sigmap = (np.sqrt(N * 1.0 / (N - 1)) *
                  (aligned_magnitude[:N] - mean_mag) /
                  aligned_error)

        sigmaq = (np.sqrt(N * 1.0 / (N - 1)) *
                  (aligned_magnitude2[:N] - mean_mag2) /
                  aligned_error2)
        sigma_i = sigmap * sigmaq

        J = (1.0 / len(sigma_i) *
             np.sum(np.sign(sigma_i) * np.sqrt(np.abs(sigma_i))))

        K = (1 / np.sqrt(N * 1.0) *
             np.sum(np.abs(sigma_i)) / np.sqrt(np.sum(sigma_i ** 2)))

        self._value = J * K / 0.798


class Con(Feature):
    """Index introduced for selection of variable starts from OGLE database.


    To calculate Con, we counted the number of three consecutive measurements
    that are out of 2sigma range, and normalized by N-2
    Pavlos not happy
    """

    def __init__(self, consecutiveStar=3, **kwargs):
        super(Con, self).__init__(**kwargs)
        self.consecutiveStar = consecutiveStar

    def fit(self, data):

        magnitude = data[0]
        N = len(magnitude)
        if N < self.consecutiveStar:
            return 0
        sigma = np.std(magnitude)
        m = np.mean(magnitude)
        count = 0

        for i in xrange(N - self.consecutiveStar + 1):
            flag = 0
            for j in xrange(self.consecutiveStar):
                if(magnitude[i + j] > m + 2 * sigma or magnitude[i + j] < m - 2 * sigma):
                    flag = 1
                else:
                    flag = 0
                    break
            if flag:
                count = count + 1
        self._value = count * 1.0 / (N - self.consecutiveStar + 1)


# class VariabilityIndex(Base):

#     # Eta. Removed, it is not invariant to time sampling
#     '''
#     The index is the ratio of mean of the square of successive difference to
#     the variance of data points
#     '''
#     def __init__(self):
#         self.category='timeSeries'


#     def fit(self, data):

#         N = len(data)
#         sigma2 = np.var(data)

#         return 1.0/((N-1)*sigma2) * np.sum(np.power(data[1:] - data[:-1] , 2)
#      )


class Color(Feature):
    """Average color for each MACHO lightcurve
    mean(B1) - mean(B2)
    """
    Data = ['magnitude', 'time', 'magnitude2']

    def fit(self, data):
        magnitude = data[0]
        magnitude2 = data[3]
        self._value = np.mean(magnitude) - np.mean(magnitude2)

# The categories of the following featurs should be revised


class Beyond1Std(Feature):
    """Percentage of points beyond one st. dev. from the weighted
    (by photometric errors) mean
    """

    Data = ['magnitude', 'error']

    def fit(self, data):

        magnitude = data[0]
        error = data[2]
        n = len(magnitude)

        weighted_mean = np.average(magnitude, weights=1 / error ** 2)

        # Standard deviation with respect to the weighted mean

        var = sum((magnitude - weighted_mean) ** 2)
        std = np.sqrt((1.0 / (n - 1)) * var)

        count = np.sum(np.logical_or(magnitude > weighted_mean + std,
                                     magnitude < weighted_mean - std))

        self._value = float(count) / n


class SmallKurtosis(Feature):
    """Small sample kurtosis of the magnitudes.

    See http://www.xycoon.com/peakedness_small_sample_test_1.htm
    """

    def fit(self, data):
        magnitude = data[0]
        n = len(magnitude)
        mean = np.mean(magnitude)
        std = np.std(magnitude)

        S = sum(((magnitude - mean) / std) ** 4)

        c1 = float(n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
        c2 = float(3 * (n - 1) ** 2) / ((n - 2) * (n - 3))

        self._value = c1 * S - c2


class Std(Feature):
    """Standard deviation of the magnitudes"""

    def fit(self, data):
        magnitude = data[0]
        self._value = np.std(magnitude)


class Skew(Feature):
    """Skewness of the magnitudes"""

    def fit(self, data):
        magnitude = data[0]
        self._value = stats.skew(magnitude)


class StetsonJ(Feature):
    """Stetson (1996) variability index, a robust standard deviation"""
    Data = ['magnitude', 'time', 'error', 'magnitude2', 'error2']

    def fit(self, data):
        aligned_magnitude = data[4]
        aligned_magnitude2 = data[5]
        aligned_error = data[7]
        aligned_error2 = data[8]
        N = len(aligned_magnitude)

        mean_mag = (np.sum(aligned_magnitude/(aligned_error*aligned_error)) /
                    np.sum(1.0 / (aligned_error * aligned_error)))

        mean_mag2 = (np.sum(aligned_magnitude2 / (aligned_error2*aligned_error2)) /
                     np.sum(1.0 / (aligned_error2 * aligned_error2)))

        sigmap = (np.sqrt(N * 1.0 / (N - 1)) *
                  (aligned_magnitude[:N] - mean_mag) /
                  aligned_error)
        sigmaq = (np.sqrt(N * 1.0 / (N - 1)) *
                  (aligned_magnitude2[:N] - mean_mag2) /
                  aligned_error2)
        sigma_i = sigmap * sigmaq

        J = (1.0 / len(sigma_i) * np.sum(np.sign(sigma_i) *
             np.sqrt(np.abs(sigma_i))))

        self._value = J


class MaxSlope(Feature):
    """
    Examining successive (time-sorted) magnitudes, the maximal first difference
    (value of delta magnitude over delta time)
    """
    Data = ['magnitude', 'time']

    def fit(self, data):

        magnitude = data[0]
        time = data[1]
        slope = np.abs(magnitude[1:] - magnitude[:-1]) / (time[1:] - time[:-1])
        np.max(slope)

        self._value = np.max(slope)


class MedianAbsDev(Feature):

    def fit(self, data):
        magnitude = data[0]
        median = np.median(magnitude)

        devs = (abs(magnitude - median))

        self._value = np.median(devs)


class MedianBRP(Feature):
    """Median buffer range percentage

    Fraction (<= 1) of photometric points within amplitude/10
    of the median magnitude
    """

    def fit(self, data):
        magnitude = data[0]
        median = np.median(magnitude)
        amplitude = (np.max(magnitude) - np.min(magnitude)) / 10
        n = len(magnitude)

        count = np.sum(np.logical_and(magnitude < median + amplitude,
                                      magnitude > median - amplitude))

        self._value = float(count) / n


class PairSlopeTrend(Feature):
    """
    Considering the last 30 (time-sorted) measurements of source magnitude,
    the fraction of increasing first differences minus the fraction of
    decreasing first differences.
    """

    def fit(self, data):
        magnitude = data[0]
        data_last = magnitude[-30:]

        self._value = (float(len(np.where(np.diff(data_last) > 0)[0]) -
                       len(np.where(np.diff(data_last) <= 0)[0])) / 30)


class FluxPercentileRatioMid20(Feature):

    def fit(self, data):
        magnitude = data[0]
        sorted_data = np.sort(magnitude)
        lc_length = len(sorted_data)

        F_60_index = math.ceil(0.60 * lc_length)
        F_40_index = math.ceil(0.40 * lc_length)
        F_5_index = math.ceil(0.05 * lc_length)
        F_95_index = math.ceil(0.95 * lc_length)

        F_40_60 = sorted_data[F_60_index] - sorted_data[F_40_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        F_mid20 = F_40_60 / F_5_95

        self._value = F_mid20


class FluxPercentileRatioMid35(Feature):

    def fit(self, data):
        magnitude = data[0]
        sorted_data = np.sort(magnitude)
        lc_length = len(sorted_data)

        F_325_index = math.ceil(0.325 * lc_length)
        F_675_index = math.ceil(0.675 * lc_length)
        F_5_index = math.ceil(0.05 * lc_length)
        F_95_index = math.ceil(0.95 * lc_length)

        F_325_675 = sorted_data[F_675_index] - sorted_data[F_325_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        F_mid35 = F_325_675 / F_5_95

        self._value = F_mid35


class FluxPercentileRatioMid50(Feature):

    def fit(self, data):
        magnitude = data[0]
        sorted_data = np.sort(magnitude)
        lc_length = len(sorted_data)

        F_25_index = math.ceil(0.25 * lc_length)
        F_75_index = math.ceil(0.75 * lc_length)
        F_5_index = math.ceil(0.05 * lc_length)
        F_95_index = math.ceil(0.95 * lc_length)

        F_25_75 = sorted_data[F_75_index] - sorted_data[F_25_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        F_mid50 = F_25_75 / F_5_95

        self._value = F_mid50


class FluxPercentileRatioMid65(Feature):

    def fit(self, data):
        magnitude = data[0]
        sorted_data = np.sort(magnitude)
        lc_length = len(sorted_data)

        F_175_index = math.ceil(0.175 * lc_length)
        F_825_index = math.ceil(0.825 * lc_length)
        F_5_index = math.ceil(0.05 * lc_length)
        F_95_index = math.ceil(0.95 * lc_length)

        F_175_825 = sorted_data[F_825_index] - sorted_data[F_175_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        F_mid65 = F_175_825 / F_5_95

        self._value = F_mid65


class FluxPercentileRatioMid80(Feature):

    def fit(self, data):
        magnitude = data[0]
        sorted_data = np.sort(magnitude)
        lc_length = len(sorted_data)

        F_10_index = math.ceil(0.10 * lc_length)
        F_90_index = math.ceil(0.90 * lc_length)
        F_5_index = math.ceil(0.05 * lc_length)
        F_95_index = math.ceil(0.95 * lc_length)

        F_10_90 = sorted_data[F_90_index] - sorted_data[F_10_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        F_mid80 = F_10_90 / F_5_95

        self._value = F_mid80


class PercentDifferenceFluxPercentile(Feature):

    def fit(self, data):
        magnitude = data[0]
        median_data = np.median(magnitude)

        sorted_data = np.sort(magnitude)
        lc_length = len(sorted_data)
        F_5_index = math.ceil(0.05 * lc_length)
        F_95_index = math.ceil(0.95 * lc_length)
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]

        percent_difference = F_5_95 / median_data

        self._value = percent_difference


class PercentAmplitude(Feature):

    def fit(self, data):
        magnitude = data[0]
        median_data = np.median(magnitude)
        distance_median = np.abs(magnitude - median_data)
        max_distance = np.max(distance_median)

        percent_amplitude = max_distance / median_data

        self._value = percent_amplitude


class LinearTrend(Feature):

    Data = ['magnitude', 'time']

    def fit(self, data):
        magnitude = data[0]
        time = data[1]
        regression_slope = stats.linregress(time, magnitude)[0]

        self._value = regression_slope


class Eta_color(Feature):

    Data = ['magnitude', 'time', 'magnitude2']

    def fit(self, data):
        aligned_magnitude = data[4]
        aligned_magnitude2 = data[5]
        aligned_time = data[6]
        N = len(aligned_magnitude)
        B_Rdata = aligned_magnitude - aligned_magnitude2

        w = 1.0 / np.power(aligned_time[1:] - aligned_time[:-1], 2)
        w_mean = np.mean(w)

        N = len(aligned_time)
        sigma2 = np.var(B_Rdata)

        S1 = sum(w * (B_Rdata[1:] - B_Rdata[:-1]) ** 2)
        S2 = sum(w)

        eta_B_R = (w_mean * np.power(aligned_time[N - 1] -
                   aligned_time[0], 2) * S1 / (sigma2 * S2 * N ** 2))

        self._value = eta_B_R


class Eta_e(Feature):

    Data = ['magnitude', 'time']

    def fit(self, data):

        magnitude = data[0]
        time = data[1]
        w = 1.0 / np.power(np.subtract(time[1:], time[:-1]), 2)
        w_mean = np.mean(w)

        N = len(time)
        sigma2 = np.var(magnitude)

        S1 = sum(w * (magnitude[1:] - magnitude[:-1]) ** 2)
        S2 = sum(w)

        eta_e = (w_mean * np.power(time[N - 1] -
                 time[0], 2) * S1 / (sigma2 * S2 * N ** 2))

        self._value = eta_e


class Mean(Feature):

    def fit(self, data):
        magnitude = data[0]
        B_mean = np.mean(magnitude)

        self._value = B_mean


class Q31(Feature):

    def fit(self, data):
        magnitude = data[0]
        self._value = np.percentile(magnitude, 75) -\
                      np.percentile(magnitude, 25)


class Q31_color(Feature):

    Data = ['magnitude', 'time', 'magnitude2']

    def fit(self, data):
        aligned_magnitude = data[4]
        aligned_magnitude2 = data[5]
        N = len(aligned_magnitude)
        b_r = aligned_magnitude[:N] - aligned_magnitude2[:N]

        self._value = np.percentile(b_r, 75) - np.percentile(b_r, 25)


class AndersonDarling(Feature):

    def fit(self, data):

        magnitude = data[0]
        ander = stats.anderson(magnitude)[0]
        self._value = 1 / (1.0 + np.exp(-10 * (ander - 0.3)))


class PeriodLS(Feature):
    Data = ['magnitude', 'time']

    def __init__(self, ofac=6., **kwargs):
        super(PeriodLS, self).__init__(**kwargs)
        self.ofac = ofac

    def fit(self, data):

        magnitude = data[0]
        time = data[1]

        fx, fy, nout, jmax, prob = lomb.fasper(time, magnitude, self.ofac, 100.)
        period = fx[jmax]
        T = 1.0 / period
        new_time = np.mod(time, 2 * T) / (2 * T)

        self.prob = prob
        self.new_time = new_time
        self.period = period

        self._value = T


class Period_fit(Feature):

    Data = ['magnitude', 'time']
    depends = {"PeriodLS"}

    def __init__(self, **kwargs):
        super(Period_fit, self).__init__(**kwargs)
        self.periodls = kwargs["PeriodLS"]

    def fit(self, data):
        self._value = self.periodls.prob


class Psi_CS(Feature):

    Data = ['magnitude', 'time']
    depends = {"PeriodLS"}

    def __init__(self, **kwargs):
        super(Psi_CS, self).__init__(**kwargs)
        self.periodls = kwargs["PeriodLS"]

    def fit(self, data):
        magnitude = data[0]
        time = data[1]
        folded_data = magnitude[np.argsort(self.periodls.new_time)]
        sigma = np.std(folded_data)
        N = len(folded_data)
        m = np.mean(folded_data)
        s = np.cumsum(folded_data - m) * 1.0 / (N * sigma)
        R = np.max(s) - np.min(s)
        self._value = R


class Psi_eta(Feature):

    Data = ['magnitude', 'time']
    depends = {"PeriodLS"}

    def __init__(self, **kwargs):
        super(Psi_eta, self).__init__(**kwargs)
        self.periodls = kwargs["PeriodLS"]

    def fit(self, data):

        # folded_time = np.sort(new_time)
        magnitude = data[0]
        folded_data = magnitude[np.argsort(self.periodls.new_time)]

        # w = 1.0 / np.power(folded_time[1:]-folded_time[:-1] ,2)
        # w_mean = np.mean(w)

        # N = len(folded_time)
        # sigma2=np.var(folded_data)

        # S1 = sum(w*(folded_data[1:]-folded_data[:-1])**2)
        # S2 = sum(w)

        # Psi_eta = w_mean * np.power(folded_time[N-1]-folded_time[0],2) * S1 /
        # (sigma2 * S2 * N**2)

        N = len(folded_data)
        sigma2 = np.var(folded_data)

        Psi_eta = (1.0 / ((N - 1) * sigma2) *
                   np.sum(np.power(folded_data[1:] - folded_data[:-1], 2)))

        self._value = Psi_eta


class CAR_sigma(Feature):

    Data = ['magnitude', 'time', 'error']

    def CAR_Lik(self, parameters, t, x, error_vars):

        sigma = parameters[0]
        tau = parameters[1]
        # b = parameters[1] #comment it to do 2 pars estimation
        # tau = params(1,1);
        # sigma = sqrt(2*var(x)/tau);

        b = np.mean(x) / tau
        epsilon = 1e-300
        cte_neg = -np.infty
        num_datos = np.size(x)

        Omega = []
        x_hat = []
        a = []
        x_ast = []

        # Omega = np.zeros((num_datos,1))
        # x_hat = np.zeros((num_datos,1))
        # a = np.zeros((num_datos,1))
        # x_ast = np.zeros((num_datos,1))

        # Omega[0]=(tau*(sigma**2))/2.
        # x_hat[0]=0.
        # a[0]=0.
        # x_ast[0]=x[0] - b*tau

        Omega.append((tau * (sigma ** 2)) / 2.)
        x_hat.append(0.)
        a.append(0.)
        x_ast.append(x[0] - b * tau)

        loglik = 0.

        for i in range(1, num_datos):

            a_new = np.exp(-(t[i] - t[i - 1]) / tau)
            x_ast.append(x[i] - b * tau)
            x_hat.append(
                a_new * x_hat[i - 1] +
                (a_new * Omega[i - 1] / (Omega[i - 1] + error_vars[i - 1])) *
                (x_ast[i - 1] - x_hat[i - 1]))

            Omega.append(
                Omega[0] * (1 - (a_new ** 2)) + ((a_new ** 2)) * Omega[i - 1] *
                (1 - (Omega[i - 1] / (Omega[i - 1] + error_vars[i - 1]))))

            # x_ast[i]=x[i] - b*tau
            # x_hat[i]=a_new*x_hat[i-1] + (a_new*Omega[i-1]/(Omega[i-1] +
            # error_vars[i-1]))*(x_ast[i-1]-x_hat[i-1])
            # Omega[i]=Omega[0]*(1-(a_new**2)) + ((a_new**2))*Omega[i-1]*
            # ( 1 - (Omega[i-1]/(Omega[i-1]+ error_vars[i-1])))

            loglik_inter = np.log(
                ((2 * np.pi * (Omega[i] + error_vars[i])) ** -0.5) *
                (np.exp(-0.5 * (((x_hat[i] - x_ast[i]) ** 2) /
                 (Omega[i] + error_vars[i]))) + epsilon))

            loglik = loglik + loglik_inter

            if(loglik <= cte_neg):
                print('CAR lik se fue a inf')
                return None

        # the minus one is to perfor maximization using the minimize function
        return -loglik

    def calculateCAR(self, time, data, error):

        x0 = [10, 0.5]
        bnds = ((0, 100), (0, 100))
        # res = minimize(self.CAR_Lik, x0, args=(LC[:,0],LC[:,1],LC[:,2]) ,
        # method='nelder-mead',bounds = bnds)

        res = minimize(self.CAR_Lik, x0, args=(time, data, error),
                       method='nelder-mead', bounds=bnds)
        # options={'disp': True}
        sigma = res.x[0]
        tau = res.x[1]
        self.tau = tau
        return sigma

    # def getAtt(self):
    #     return CAR_sigma.tau

    def fit(self, data):
        # LC = np.hstack((self.time , data.reshape((self.N,1)), self.error))

        N = len(data[0])
        magnitude = data[0].reshape((N, 1))
        time = data[1].reshape((N, 1))
        error = data[2].reshape((N, 1)) ** 2

        a = self.calculateCAR(time, magnitude, error)

        self._value = a


class CAR_tau(Feature):

    Data = ['magnitude', 'time', 'error']
    depends = {'CAR_sigma'}

    def __init__(self, **kwargs):
        super(CAR_tau, self).__init__(**kwargs)
        self.carsigma = kwargs["CAR_sigma"]

    def fit(self, data):
        self._value = self.carsigma.tau


class CAR_mean(Feature):

    Data = ['magnitude', 'time', 'error']
    depends = {'CAR_sigma'}

    def __init__(self, **kwargs):
        super(CAR_mean, self).__init__(**kwargs)
        self.carsigma = kwargs["CAR_sigma"]

    def fit(self, data):
        magnitude = data[0]
        self._value = np.mean(magnitude) / self.carsigma.tau


class Freq1_harmonics_amplitude_0(Feature):
    Data = ['magnitude', 'time']

    def fit(self, data):
        magnitude = data[0]
        time = data[1]

        time = time - np.min(time)

        A = []
        PH = []
        scaledPH = []

        def model(x, a, b, c, Freq):
            return a*np.sin(2*np.pi*Freq*x)+b*np.cos(2*np.pi*Freq*x)+c

        for i in range(3):

            wk1, wk2, nout, jmax, prob = lomb.fasper(time, magnitude, 6., 100.)

            fundamental_Freq = wk1[jmax]

            # fit to a_i sin(2pi f_i t) + b_i cos(2 pi f_i t) + b_i,o

            # a, b are the parameters we care about
            # c is a constant offset
            # f is the fundamental Frequency
            def yfunc(Freq):
                def func(x, a, b, c):
                    return a*np.sin(2*np.pi*Freq*x)+b*np.cos(2*np.pi*Freq*x)+c
                return func

            Atemp = []
            PHtemp = []
            popts = []

            for j in range(4):
                popt, pcov = curve_fit(yfunc((j+1)*fundamental_Freq), time, magnitude)
                Atemp.append(np.sqrt(popt[0]**2+popt[1]**2))
                PHtemp.append(np.arctan(popt[1] / popt[0]))
                popts.append(popt)

            A.append(Atemp)
            PH.append(PHtemp)

            for j in range(4):
                magnitude = np.array(magnitude) - model(time, popts[j][0], popts[j][1], popts[j][2], (j+1)*fundamental_Freq)

        for ph in PH:
            scaledPH.append(np.array(ph) - ph[0])

        self.A = A
        self.PH = PH
        self.scaledPH = scaledPH

        self._value = A[0][0]


class Freq1_harmonics_amplitude_1(Feature):
    Data = ['magnitude', 'time']
    depends = {'Freq1_harmonics_amplitude_0'}

    def __init__(self, **kwargs):
        super(Freq1_harmonics_amplitude_1, self).__init__(**kwargs)
        self.fr1_h_amp_0 = kwargs["Freq1_harmonics_amplitude_0"]

    def fit(self, data):
        self._value = self.fr1_h_amp_0.A[0][1]


class Freq1_harmonics_amplitude_2(Feature):
    Data = ['magnitude', 'time']
    depends = {'Freq1_harmonics_amplitude_0'}

    def __init__(self, **kwargs):
        super(Freq1_harmonics_amplitude_2, self).__init__(**kwargs)
        self.fr1_h_amp_0 = kwargs["Freq1_harmonics_amplitude_0"]

    def fit(self, data):
        self._value = self.fr1_h_amp_0.A[0][2]


class Freq1_harmonics_amplitude_3(Feature):
    Data = ['magnitude', 'time']
    depends = {'Freq1_harmonics_amplitude_0'}

    def __init__(self, **kwargs):
        super(Freq1_harmonics_amplitude_3, self).__init__(**kwargs)
        self.fr1_h_amp_0 = kwargs["Freq1_harmonics_amplitude_0"]

    def fit(self, data):
        self._value = self.fr1_h_amp_0.A[0][3]


class Freq2_harmonics_amplitude_0(Feature):
    Data = ['magnitude', 'time']
    depends = {'Freq1_harmonics_amplitude_0'}

    def __init__(self, **kwargs):
        super(Freq2_harmonics_amplitude_0, self).__init__(**kwargs)
        self.fr1_h_amp_0 = kwargs["Freq1_harmonics_amplitude_0"]

    def fit(self, data):
        self._value = self.fr1_h_amp_0.A[1][0]


class Freq2_harmonics_amplitude_1(Feature):
    Data = ['magnitude', 'time']
    depends = {'Freq1_harmonics_amplitude_0'}

    def __init__(self, **kwargs):
        super(Freq2_harmonics_amplitude_1, self).__init__(**kwargs)
        self.fr1_h_amp_0 = kwargs["Freq1_harmonics_amplitude_0"]

    def fit(self, data):
        self._value = self.fr1_h_amp_0.A[1][1]


class Freq2_harmonics_amplitude_2(Feature):
    Data = ['magnitude', 'time']
    depends = {'Freq1_harmonics_amplitude_0'}

    def __init__(self, **kwargs):
        super(Freq2_harmonics_amplitude_2, self).__init__(**kwargs)
        self.fr1_h_amp_0 = kwargs["Freq1_harmonics_amplitude_0"]

    def fit(self, data):
        self._value = self.fr1_h_amp_0.A[1][2]


class Freq2_harmonics_amplitude_3(Feature):
    Data = ['magnitude', 'time']
    depends = {'Freq1_harmonics_amplitude_0'}

    def __init__(self, **kwargs):
        super(Freq2_harmonics_amplitude_3, self).__init__(**kwargs)
        self.fr1_h_amp_0 = kwargs["Freq1_harmonics_amplitude_0"]

    def fit(self, data):
        self._value = self.fr1_h_amp_0.A[1][3]


class Freq3_harmonics_amplitude_0(Feature):
    Data = ['magnitude', 'time']
    depends = {'Freq1_harmonics_amplitude_0'}

    def __init__(self, **kwargs):
        super(Freq3_harmonics_amplitude_0, self).__init__(**kwargs)
        self.fr1_h_amp_0 = kwargs["Freq1_harmonics_amplitude_0"]

    def fit(self, data):
        self._value = self.fr1_h_amp_0.A[2][0]


class Freq3_harmonics_amplitude_1(Feature):
    Data = ['magnitude', 'time']
    depends = {'Freq1_harmonics_amplitude_0'}

    def __init__(self, **kwargs):
        super(Freq3_harmonics_amplitude_1, self).__init__(**kwargs)
        self.fr1_h_amp_0 = kwargs["Freq1_harmonics_amplitude_0"]

    def fit(self, data):
        self._value = self.fr1_h_amp_0.A[2][1]


class Freq3_harmonics_amplitude_2(Feature):
    Data = ['magnitude', 'time']
    depends = {'Freq1_harmonics_amplitude_0'}

    def __init__(self, **kwargs):
        super(Freq3_harmonics_amplitude_2, self).__init__(**kwargs)
        self.fr1_h_amp_0 = kwargs["Freq1_harmonics_amplitude_0"]

    def fit(self, data):
        self._value = self.fr1_h_amp_0.A[2][2]


class Freq3_harmonics_amplitude_3(Feature):
    Data = ['magnitude', 'time']
    depends = {'Freq1_harmonics_amplitude_0'}

    def __init__(self, **kwargs):
        super(Freq3_harmonics_amplitude_3, self).__init__(**kwargs)
        self.fr1_h_amp_0 = kwargs["Freq1_harmonics_amplitude_0"]

    def fit(self, data):
        self._value = self.fr1_h_amp_0.A[2][3]


class Freq1_harmonics_rel_phase_0(Feature):
    Data = ['magnitude', 'time']
    depends = {'Freq1_harmonics_amplitude_0'}

    def __init__(self, **kwargs):
        super(Freq1_harmonics_rel_phase_0, self).__init__(**kwargs)
        self.fr1_h_amp_0 = kwargs["Freq1_harmonics_amplitude_0"]

    def fit(self, data):
        self._value = self.fr1_h_amp_0.scaledPH[0][0]


class Freq1_harmonics_rel_phase_1(Feature):
    Data = ['magnitude', 'time']
    depends = {'Freq1_harmonics_amplitude_0'}

    def __init__(self, **kwargs):
        super(Freq1_harmonics_rel_phase_1, self).__init__(**kwargs)
        self.fr1_h_amp_0 = kwargs["Freq1_harmonics_amplitude_0"]

    def fit(self, data):
        self._value = self.fr1_h_amp_0.scaledPH[0][1]


class Freq1_harmonics_rel_phase_2(Feature):
    Data = ['magnitude', 'time']
    depends = {'Freq1_harmonics_amplitude_0'}

    def __init__(self, **kwargs):
        super(Freq1_harmonics_rel_phase_2, self).__init__(**kwargs)
        self.fr1_h_amp_0 = kwargs["Freq1_harmonics_amplitude_0"]

    def fit(self, data):
        self._value = self.fr1_h_amp_0.scaledPH[0][2]


class Freq1_harmonics_rel_phase_3(Feature):
    Data = ['magnitude', 'time']
    depends = {'Freq1_harmonics_amplitude_0'}

    def __init__(self, **kwargs):
        super(Freq1_harmonics_rel_phase_3, self).__init__(**kwargs)
        self.fr1_h_amp_0 = kwargs["Freq1_harmonics_amplitude_0"]

    def fit(self, data):
        self._value = self.fr1_h_amp_0.scaledPH[0][3]


class Freq2_harmonics_rel_phase_0(Feature):
    Data = ['magnitude', 'time']
    depends = {'Freq1_harmonics_amplitude_0'}

    def __init__(self, **kwargs):
        super(Freq2_harmonics_rel_phase_0, self).__init__(**kwargs)
        self.fr1_h_amp_0 = kwargs["Freq1_harmonics_amplitude_0"]

    def fit(self, data):
        self._value = self.fr1_h_amp_0.scaledPH[1][0]


class Freq2_harmonics_rel_phase_1(Feature):
    Data = ['magnitude', 'time']
    depends = {'Freq1_harmonics_amplitude_0'}

    def __init__(self, **kwargs):
        super(Freq2_harmonics_rel_phase_1, self).__init__(**kwargs)
        self.fr1_h_amp_0 = kwargs["Freq1_harmonics_amplitude_0"]

    def fit(self, data):
        self._value = self.fr1_h_amp_0.scaledPH[1][1]


class Freq2_harmonics_rel_phase_2(Feature):
    Data = ['magnitude', 'time']
    depends = {'Freq1_harmonics_amplitude_0'}

    def __init__(self, **kwargs):
        super(Freq2_harmonics_rel_phase_2, self).__init__(**kwargs)
        self.fr1_h_amp_0 = kwargs["Freq1_harmonics_amplitude_0"]

    def fit(self, data):
        self._value = self.fr1_h_amp_0.scaledPH[1][2]


class Freq2_harmonics_rel_phase_3(Feature):
    Data = ['magnitude', 'time']
    depends = {'Freq1_harmonics_amplitude_0'}

    def __init__(self, **kwargs):
        super(Freq2_harmonics_rel_phase_3, self).__init__(**kwargs)
        self.fr1_h_amp_0 = kwargs["Freq1_harmonics_amplitude_0"]

    def fit(self, data):
        self._value = self.fr1_h_amp_0.scaledPH[1][3]


class Freq3_harmonics_rel_phase_0(Feature):
    Data = ['magnitude', 'time']
    depends = {'Freq1_harmonics_amplitude_0'}

    def __init__(self, **kwargs):
        super(Freq3_harmonics_rel_phase_0, self).__init__(**kwargs)
        self.fr1_h_amp_0 = kwargs["Freq1_harmonics_amplitude_0"]

    def fit(self, data):
        self._value = self.fr1_h_amp_0.scaledPH[2][0]


class Freq3_harmonics_rel_phase_1(Feature):
    Data = ['magnitude', 'time']
    depends = {'Freq1_harmonics_amplitude_0'}

    def __init__(self, **kwargs):
        super(Freq3_harmonics_rel_phase_1, self).__init__(**kwargs)
        self.fr1_h_amp_0 = kwargs["Freq1_harmonics_amplitude_0"]

    def fit(self, data):
        self._value = self.fr1_h_amp_0.scaledPH[2][1]


class Freq3_harmonics_rel_phase_2(Feature):
    Data = ['magnitude', 'time']
    depends = {'Freq1_harmonics_amplitude_0'}

    def __init__(self, **kwargs):
        super(Freq3_harmonics_rel_phase_2, self).__init__(**kwargs)
        self.fr1_h_amp_0 = kwargs["Freq1_harmonics_amplitude_0"]

    def fit(self, data):
        self._value = self.fr1_h_amp_0.scaledPH[2][2]


class Freq3_harmonics_rel_phase_3(Feature):
    Data = ['magnitude', 'time']
    depends = {'Freq1_harmonics_amplitude_0'}

    def __init__(self, **kwargs):
        super(Freq3_harmonics_rel_phase_3, self).__init__(**kwargs)
        self.fr1_h_amp_0 = kwargs["Freq1_harmonics_amplitude_0"]

    def fit(self, data):
        self._value = self.fr1_h_amp_0.scaledPH[2][3]


class Gskew(Feature):
    """Median-based measure of the skew"""

    def fit(self, data):
        magnitude = np.array(data[0])
        median_mag = np.median(magnitude)
        F_3_value = np.percentile(magnitude, 3)
        F_97_value = np.percentile(magnitude, 97)

        self._value = (np.median(magnitude[magnitude <= F_3_value]) +
                       np.median(magnitude[magnitude >= F_97_value])
                       - 2*median_mag)


class StructureFunction_index_21(Feature):
    Data = ['magnitude', 'time']

    def fit(self, data):
        magnitude = data[0]
        time = data[1]

        Nsf = 100
        Np = 100
        sf1 = np.zeros(Nsf)
        sf2 = np.zeros(Nsf)
        sf3 = np.zeros(Nsf)
        f = interp1d(time, magnitude)

        time_int = np.linspace(np.min(time), np.max(time), Np)
        mag_int = f(time_int)

        for tau in np.arange(1, Nsf):
            sf1[tau-1] = np.mean(np.power(np.abs(mag_int[0:Np-tau] - mag_int[tau:Np]) , 1.0))
            sf2[tau-1] = np.mean(np.abs(np.power(np.abs(mag_int[0:Np-tau] - mag_int[tau:Np]) , 2.0)))
            sf3[tau-1] = np.mean(np.abs(np.power(np.abs(mag_int[0:Np-tau] - mag_int[tau:Np]) , 3.0)))
        sf1_log = np.log10(np.trim_zeros(sf1))
        sf2_log = np.log10(np.trim_zeros(sf2))
        sf3_log = np.log10(np.trim_zeros(sf3))

        m_21, b_21 = np.polyfit(sf1_log, sf2_log, 1)
        m_31, b_31 = np.polyfit(sf1_log, sf3_log, 1)
        m_32, b_32 = np.polyfit(sf2_log, sf3_log, 1)

        self.m_21 = m_21
        self.m_31 = m_31
        self.m_32 = m_32

        self._value = m_21


class StructureFunction_index_31(Feature):
    Data = ['magnitude', 'time']
    depends = {'StructureFunction_index_21'}

    def __init__(self, **kwargs):
        super(StructureFunction_index_31, self).__init__(**kwargs)
        self.strfunc_i21 = kwargs["StructureFunction_index_21"]

    def fit(self, data):
        self._value = self.strfunc_i21.m_31


class StructureFunction_index_32(Feature):
    Data = ['magnitude', 'time']
    depends = {'StructureFunction_index_21'}

    def __init__(self, **kwargs):
        super(StructureFunction_index_32, self).__init__(**kwargs)
        self.strfunc_i21 = kwargs["StructureFunction_index_21"]

    def fit(self, data):
        self._value = self.strfunc_i21.m_32



