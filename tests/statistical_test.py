import pandas as pd
import numpy as np

from campaignutils.statistical import StatisticalTests

# Sample data setup
df1 = pd.DataFrame({'value': np.random.normal(loc=0.0, scale=1.0, size=100)})
df2 = pd.DataFrame({'value': np.random.normal(loc=0.5, scale=1.0, size=100)})

def test_ks_test():
    ks_stat, p_value = StatisticalTests.ks_test(df1, df2, 'value')
    assert isinstance(ks_stat, float)
    assert isinstance(p_value, float)

def test_ks_1samp():
    ks_stat, p_value = StatisticalTests.ks_1samp(df1, 'value')
    assert isinstance(ks_stat, float)
    assert isinstance(p_value, float)

def test_t_test():
    t_stat, p_value = StatisticalTests.t_test(df1, df2, 'value')
    assert isinstance(t_stat, float)
    assert isinstance(p_value, float)

def test_levene_test():
    lev_stat, p_value = StatisticalTests.levene_test(df1, df2, 'value')
    assert isinstance(lev_stat, float)
    assert isinstance(p_value, float)
    