"""Calendar, missingness and label invariance for temporal co-clustering."""
import numpy as np
import pandas as pd
from fx_forecasting.evaluation.clustering_evolution import (
    analyze_evolution, pair_frequencies, align_partition, describe_path,
)


def test_pair_frequency_uses_joint_eligibility_and_full_calendar_windows():
    labels = np.array([[0, 0, -1], [1, 0, -1], [0, -1, 0], [1, 1, 0]])
    numerator, denominator = pair_frequencies(labels, 3)
    assert not denominator[:2].any()
    assert (numerator[2, 0, 1], denominator[2, 0, 1]) == (1, 2)
    assert (numerator[3, 0, 1], denominator[3, 0, 1]) == (1, 2)
    assert (numerator[2, 0, 2], denominator[2, 0, 2]) == (1, 1)
    assert (numerator[2, 1, 2], denominator[2, 1, 2]) == (0, 0)


def test_pair_frequencies_ignore_label_names_and_future_changes():
    rng = np.random.default_rng(4)
    labels = rng.integers(0, 3, (80, 8));labels[30:35, 2] = -1
    changed = labels.copy()
    for row in changed:
        present = row >= 0
        row[present] = rng.permutation(3)[row[present]]
    a, b = pair_frequencies(labels), pair_frequencies(changed)
    for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
    changed[70:] = 0
    c = pair_frequencies(changed)
    for x,y in zip(a,c):np.testing.assert_array_equal(x[:70],y[:70])


def test_display_matching_removes_pure_relabeling_without_changing_partition():
    old=np.array([0,0,1,1,-1]);new=np.array([1,1,0,0,1])
    matched=align_partition(new,old,2)
    np.testing.assert_array_equal(matched[:4],old[:4])
    np.testing.assert_array_equal(new[:,None]==new[None,:],matched[:,None]==matched[None,:])


def test_temporal_metrics_exclude_missing_currencies_and_report_singletons():
    x=np.array([[0,0,1,-1],[0,1,1,-1]])
    m=describe_path(x,pd.period_range('2000-01',periods=2,freq='M'),'r2k_w6',2)
    assert m.iloc[-1].n_common_lag1==3
    assert m.iloc[-1].switch_rate==1/3
    assert m.iloc[-1].singleton_assets==1
    assert m.iloc[-1].largest_share==2/3


def test_evolution_prefix_and_currency_axis_are_independent_of_future():
    dates=pd.period_range('2000-01',periods=86,freq='M');rng=np.random.default_rng(6)
    fx=pd.DataFrame(np.exp(rng.normal(0,.03,(86,7)).cumsum(axis=0)),index=dates,columns=list('ABCDEFG'))
    a,meta,_=analyze_evolution(fx.iloc[:78],k_values=(2,),summary_start='2001-01',audit_initialization=False)
    changed=fx.iloc[:,::-1].copy();changed.iloc[78:]*=50
    b,later,_=analyze_evolution(changed,k_values=(2,),summary_start='2001-01',audit_initialization=False)
    assert meta['currencies']==later['currencies']
    for method in a:np.testing.assert_array_equal(a[method],b[method][:len(a[method])])
    assert all((x[:47]==-1).all() for name,x in a.items() if name.startswith(('returns','adjusted')))
