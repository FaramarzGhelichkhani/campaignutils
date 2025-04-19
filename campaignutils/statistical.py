import pandas as pd
import numpy as np
from scipy.stats import kstest, ks_2samp, ttest_ind, levene, ttest_rel

class StatisticalTests:

    @classmethod
    def to_array(cls, data, feature):
        """
        Convert input data to a numpy array.
        Handles both DataFrame and array-like inputs.
        """
        if isinstance(data, pd.DataFrame):
            return data[feature].values.tolist()
        elif isinstance(data, (np.ndarray, list)):
            return np.array(data)
        else:
            raise ValueError("Input data should be a pandas DataFrame or array-like (list/numpy).")

    @classmethod
    def ks2samp_test(cls, data1, data2, feature=None, alternative='two-sided'):
        """
        Perform the Kolmogorov-Smirnov test between two samples.
        """
        sample1 = cls.to_array(data1, feature)
        sample2 = cls.to_array(data2, feature)

        ks_statistic, p_value = ks_2samp(sample1, sample2, alternative=alternative)

        return ks_statistic, p_value
    
    @classmethod
    def ks_test(cls, data, feature=None, cdf='norm'):
        """
        Perform the one-sample Kolmogorov-Smirnov test against a specified cumulative distribution function (CDF).
        The default is the normal distribution.
        """
        sample = cls.to_array(data, feature)

        # You can use 'norm' or any other valid scipy distribution
        if cdf == 'norm':
            ks_statistic, p_value = kstest(sample, 'norm', args=(np.mean(sample), np.std(sample)))
        else:
            ks_statistic, p_value = kstest(sample, cdf)

        return ks_statistic, p_value

    @classmethod
    def t_test_ind(cls, data1, data2, feature=None, alternative='two-sided',  equal_var=True):
        """
        Perform an independent t-test between two samples.
        """
        sample1 = cls.to_array(data1, feature)
        sample2 = cls.to_array(data2, feature)

        t_statistic, p_value = ttest_ind(sample1, sample2, alternative=alternative, equal_var=equal_var) 

        return t_statistic, p_value

    @classmethod
    def levene_test(cls, data1, data2, feature=None, center='median'):
        """
        Perform Levene's test for equality of variances.
        """
        sample1 = cls.to_array(data1, feature)
        sample2 = cls.to_array(data2, feature)

        stat, p_value = levene(sample1, sample2, center=center)

        return stat, p_value

    @classmethod
    def t_test_rel(cls, data1, data2, feature=None,  alternative='greater'):
        sample1 = cls.to_array(data1, feature)
        sample2 = cls.to_array(data2, feature)
    
        t_statistic, p_value = ttest_rel(sample1, sample2, alternative=alternative) 

        return t_statistic, p_value
