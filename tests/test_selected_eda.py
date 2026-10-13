"""Check selected-sample regime boundaries and causal report treatments."""

import json
import unittest

import numpy as np
import pandas as pd

from fx_forecasting.config import PROJECT_ROOT
from fx_forecasting.data.selected_eda import calculate_selected_eda


class SelectedEDATests(unittest.TestCase):
    def setUp(self):
        self.choices = json.loads((PROJECT_ROOT / 'config/research_sample.json').read_text())
        self.index = pd.period_range('1994-01', '2016-12', freq='M')
        time = np.arange(len(self.index))
        self.fx = pd.DataFrame({'CN': np.exp(.003 * time + .01 * np.sin(time))}, index=self.index)
        self.levels = pd.DataFrame(True, index=self.index, columns=['CN'])
        self.macro = {
            'policy': pd.DataFrame({'CN': time / 10, 'US': time / 20}, index=self.index),
            'cpi': pd.DataFrame({'CN': np.exp(.008 * time), 'US': np.exp(.002 * time + .000001 * time**2)}, index=self.index),
            'reer': pd.DataFrame({'CN': np.exp(.001 * time)}, index=self.index),
            'neer': pd.DataFrame({'US': np.exp(.003 * time)}, index=self.index),
            'usd_credit': pd.DataFrame({'3P': np.exp(np.arange(92) * .02)}, index=pd.period_range('1994Q1', periods=92, freq='Q')),
        }

    def test_targets_and_fx_history_do_not_bridge_a_regime(self):
        self.levels.loc['2010-06', 'CN'] = False
        _, returns, features, targets, _ = calculate_selected_eda(self.fx, self.levels, self.macro, self.choices)
        self.assertTrue(targets[3].loc['2010-04':'2010-06', 'CN'].isna().all())
        self.assertTrue(returns.loc['2010-06':'2010-07', 'CN'].isna().all())
        self.assertTrue(features['momentum_3m'].loc['2010-07':'2010-09', 'CN'].isna().all())
        self.assertAlmostEqual(features['momentum_3m'].loc['2010-10', 'CN'],
                               100 * np.log(self.fx.loc['2010-10', 'CN'] / self.fx.loc['2010-07', 'CN']))
        self.assertTrue(features['volatility_12m'].loc['2011-06', 'CN'] != features['volatility_12m'].loc['2011-06', 'CN'])
        self.assertTrue(pd.notna(features['volatility_12m'].loc['2011-07', 'CN']))
        self.assertTrue(targets[12].tail(12).isna().all().all())

    def test_carry_limit_and_derived_inflation_are_applied_before_lag(self):
        self.macro['policy'].loc['2010-01':'2010-02', 'CN'] = np.nan
        self.macro['cpi'].loc['2010-01', 'US'] = np.nan
        _, _, features, _, flags = calculate_selected_eda(self.fx, self.levels, self.macro, self.choices)
        expected_rate = self.macro['policy'].loc['2009-12', 'CN'] - self.macro['policy'].loc['2010-01', 'US']
        self.assertAlmostEqual(features['rate_diff'].loc['2010-02', 'CN'], expected_rate)
        self.assertTrue(flags['policy_carried_domestic'].loc['2010-02', 'CN'])
        self.assertTrue(pd.isna(features['rate_diff'].loc['2010-03', 'CN']))
        self.assertFalse(flags['policy_carried_domestic'].loc['2010-03', 'CN'])
        previous_us = 100 * np.log(self.macro['cpi'].loc['2009-12', 'US'] / self.macro['cpi'].loc['2008-12', 'US'])
        domestic = 100 * np.log(self.macro['cpi'].loc['2010-01', 'CN'] / self.macro['cpi'].loc['2009-01', 'CN'])
        self.assertAlmostEqual(features['inflation_diff'].loc['2010-03', 'CN'], domestic - previous_us)
        self.assertTrue(flags['inflation_carried_us'].loc['2010-03', 'CN'])
        self.assertTrue(pd.isna(self.macro['cpi'].loc['2010-01', 'US']))

    def test_macro_lags_credit_alignment_and_future_perturbation(self):
        _, _, features, _, flags = calculate_selected_eda(self.fx, self.levels, self.macro, self.choices)
        stock = self.macro['usd_credit']['3P']
        for month in ('2010-07', '2010-08', '2010-09'):
            self.assertAlmostEqual(features['credit_stock'].loc[month, 'CN'], np.log(stock.loc['2010Q1']))
            self.assertAlmostEqual(features['credit_growth'].loc[month, 'CN'], 100 * np.log(stock.loc['2010Q1'] / stock.loc['2009Q1']))
        # Hand calculation: t-2 reference less the previous 60 reference months.
        logreer = np.log(self.macro['reer']['CN'])
        expected = 100 * (logreer.loc['2010-06'] - logreer.loc['2005-06':'2010-05'].mean())
        self.assertAlmostEqual(features['reer_gap'].loc['2010-08', 'CN'], expected)
        for key in ('policy', 'cpi', 'reer', 'neer'):
            self.macro[key].loc['2010-09':] *= 100
        self.macro['usd_credit'].loc['2010Q3':] *= 100
        _, _, new_features, _, new_flags = calculate_selected_eda(self.fx, self.levels, self.macro, self.choices)
        for key in features:
            pd.testing.assert_frame_equal(features[key].loc[:'2010-08'], new_features[key].loc[:'2010-08'])
        for key in flags:
            pd.testing.assert_frame_equal(flags[key].loc[:'2010-08'], new_flags[key].loc[:'2010-08'])


if __name__ == '__main__':
    unittest.main()
