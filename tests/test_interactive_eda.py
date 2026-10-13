"""Check forecast timing, missingness and independent benchmark calculations."""

import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from fx_forecasting.data.interactive_eda import (
    exploratory_sample, make_features, random_walk_metrics, safe_corr, window_mask,
)


class InteractiveEDATests(unittest.TestCase):
    def setUp(self):
        self.index=pd.period_range("1994-01","2015-12",freq="M")
        t=np.arange(len(self.index))
        self.fx=pd.DataFrame({"BR":np.exp(.005*t+.03*np.sin(t))},index=self.index)
        self.macro={
            "reer":pd.DataFrame({"BR":100+t},index=self.index),
            "policy":pd.DataFrame({"BR":t/10,"US":np.zeros(len(t))},index=self.index),
            "cpi":pd.DataFrame({"BR":np.exp(.01*t),"US":np.exp(.002*t)},index=self.index),
            "neer":pd.DataFrame({"US":100+t},index=self.index),
            "usd_credit":pd.DataFrame({"3P":np.exp(np.arange(88)*.02)},
                index=pd.period_range("1994Q1",periods=88,freq="Q")),
        }

    def test_horizons_use_future_calendar_endpoints_and_truncate(self):
        _,_,targets,_,_=make_features(self.fx,self.macro)
        for h,y in targets.items():
            expected=100*np.log(self.fx.iloc[100+h,0]/self.fx.iloc[100,0])
            self.assertAlmostEqual(y.iloc[100,0],expected)
            self.assertTrue(y.tail(h).isna().all().all())

    def test_missing_fx_is_never_filled(self):
        self.fx.iloc[100,0]=np.nan
        returns,features,_,_,_=make_features(self.fx,self.macro)
        self.assertTrue(returns.iloc[100:102].isna().all().all())
        self.assertTrue(features["volatility_12m"].iloc[101,0]!=features["volatility_12m"].iloc[101,0])

    def test_macro_lags_and_credit_quarter_alignment(self):
        _,f,_,_,credit=make_features(self.fx,self.macro)
        self.assertAlmostEqual(f["rate_diff"].iloc[100,0],9.9)
        # July/August/September 2010 use 2010Q1, not a future quarter.
        q=self.macro["usd_credit"]["3P"]
        expected=100*np.log(q.loc["2010Q1"]/q.loc["2009Q1"])
        for month in ("2010-07","2010-08","2010-09"):
            self.assertAlmostEqual(credit.loc[month],expected)
        self.macro["cpi"].loc["2010-01","US"]=np.nan
        _,f,_,_,_=make_features(self.fx,self.macro)
        self.assertTrue(np.isnan(f["inflation_diff"].loc["2010-03","BR"]))
        self.assertTrue(np.isnan(f["inflation_diff"].loc["2011-03","BR"]))

    def test_training_correlations_do_not_cross_2010_boundary(self):
        for h in (1,3,12):
            mask=window_mask(self.index,h,"train")
            self.assertEqual(self.index[mask][-1]+h,pd.Period("2009-12","M"))

    def test_pair_counts_and_constant_series(self):
        x=pd.Series(np.arange(50,dtype=float)); y=x.copy();y.iloc[:5]=np.nan
        rho,n=safe_corr(x,y)
        self.assertAlmostEqual(rho,1);self.assertEqual(n,45)
        rho,n=safe_corr(x*0,y)
        self.assertTrue(np.isnan(rho));self.assertEqual(n,45)

    def test_benchmark_against_hand_calculation_and_future_perturbation(self):
        y=pd.DataFrame({"BR":np.arange(len(self.index),dtype=float)/100},index=self.index)
        meta=pd.DataFrame({"Reference area":["Brazil"]},index=["BR"])
        result,paths=random_walk_metrics({3:y},meta)
        origin=pd.Period("2010-01","M"); position=self.index.get_loc(origin)
        actual=y.iloc[position,0]
        prediction=y.iloc[:position-2,0].mean()  # last realised origin is t−3
        saving=actual**2-(actual-prediction)**2
        self.assertAlmostEqual(paths[(3,"BR")].iloc[0],saving)
        post=y.loc["2010-01":,"BR"]
        self.assertAlmostEqual(result.iloc[0].rw_rmse,np.sqrt(np.mean(post**2)))
        changed=y.copy();changed.loc["2013-01":,"BR"]+=10000
        _,new_paths=random_walk_metrics({3:changed},meta)
        pd.testing.assert_series_equal(paths[(3,"BR")].loc[:"2012-12"],new_paths[(3,"BR")].loc[:"2012-12"])

    def test_euro_representative_and_numeraire(self):
        ids=["eur-national","eur-area","usd","brl"]
        metadata=pd.DataFrame({"REF_AREA":["FR","XM","US","BR"],
            "Reference area":["France","Euro area","United States","Brazil"],
            "CURRENCY":["EUR","EUR","USD","BRL"]},index=ids)
        datasets={"fx":SimpleNamespace(metadata=metadata,name="WS_XRU"),
                  "reer":SimpleNamespace(metadata=metadata)}
        sample=pd.DataFrame(1.,index=self.index,columns=ids)
        meta,_,fx=exploratory_sample(datasets,{"fx":sample})
        self.assertEqual(set(meta.index),{"XM","BR"})
        self.assertTrue(fx.loc[:"1998-12","XM"].isna().all())
        self.assertEqual(fx.loc["1999-01","XM"],1)

    def test_global_risk_is_robust_to_one_extreme_currency(self):
        control=pd.concat([self.fx.rename(columns={"BR":str(i)}) for i in range(5)],axis=1)
        extreme=control.copy();extreme.iloc[100:110,0]*=10000
        _,_,_,risk_control,_=make_features(control,self.macro)
        _,_,_,risk_extreme,_=make_features(extreme,self.macro)
        pd.testing.assert_series_equal(risk_control,risk_extreme)


if __name__ == "__main__":
    unittest.main()
