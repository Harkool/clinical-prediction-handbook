"""Independent target checks and failed-fit policies for the Python main course."""
import csv
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from cph import course_core as core
from cph import course_validation as validation
from cph.metrics import decision_curve, evaluate, roc_data


ROOT=Path(__file__).absolute().parents[1]


class MainCourseValidationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "examples/reference/main_case.csv").open(encoding="utf-8-sig",newline="") as stream:
            cls.records=list(csv.DictReader(stream))
        cls.model=core.fit_case(cls.records)
        cls.columns=core.as_columns(cls.records)
        cls.y=np.asarray(cls.columns['y'],float)
        cls.p=core.predict_case(cls.model,cls.columns)

    def test_apparent_probability_targets_match_independent_metrics(self):
        evaluation=evaluate(self.y,self.p,self.columns['id'])
        performance=validation.performance(self.y,self.p)
        self.assertEqual(performance['status'],'PASS')
        self.assertAlmostEqual(performance['estimates']['C'],roc_data(evaluation)['auc'],places=12)
        self.assertAlmostEqual(performance['estimates']['calibration_slope'],1.,places=7)
        self.assertAlmostEqual(performance['estimates']['calibration_intercept'],0.,places=7)
        self.assertAlmostEqual(performance['estimates']['OE'],1.,places=7)
        thresholds=[.05,.10,.20]
        expected=[row['net_benefit'] for row in decision_curve(evaluation,thresholds)]
        np.testing.assert_allclose(validation.net_benefit(self.y,self.p,thresholds),expected,atol=1e-12,rtol=0)

    def test_smooth_natural_cubic_span_matches_the_independent_historical_r_curve(self):
        with (ROOT / "examples/reference/calibration_curve.csv").open(encoding="utf-8-sig",newline="") as stream:
            reference=list(csv.DictReader(stream))
        grid=np.array([float(row['prediction']) for row in reference])
        expected=np.array([float(row['apparent']) for row in reference])
        knots=np.quantile(np.log(self.p/(1-self.p)),[.05,.5,.95])
        actual=validation.calibration_fit(self.y,self.p,grid,knots)
        np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=0)

    def test_whole_fit_bootstrap_and_nested_budget_are_reproducible_and_explicit(self):
        first=validation.bootstrap_optimism(self.records,self.model,B=3,seed=51)
        second=validation.bootstrap_optimism(self.records,self.model,B=3,seed=51)
        self.assertEqual(first,second)
        self.assertEqual(first['status'],'PASS')
        self.assertEqual(first['B'],3)
        self.assertEqual(len(first['replicates']),3)
        self.assertEqual(first['failed_replicates'],0)
        self.assertIn('PCG64',first['rng'])
        double=validation.development_intervals(self.records,B_outer=2,B_inner=2,seed=52)
        self.assertEqual(double['status'],'PASS')
        self.assertEqual(len(double['replicates']),2)
        self.assertEqual((double['B_outer'],double['B_inner']),(2,2))
        for row in double['table']:
            self.assertLessEqual(row['lower'],row['upper'])
            self.assertGreaterEqual(row['mean_inner_optimism_MCSE'],0)
        json.dumps(double,allow_nan=False)

    def test_failed_development_replicate_invalidates_all_corrections(self):
        original=validation._fit_beta
        counter={'full':0}
        def injected(X,y,initial=None,offset=None):
            if X.shape[1]==19:
                counter['full']+=1
                if counter['full']==1:
                    raise RuntimeError('deliberate resampling fit failure')
            return original(X,y,initial=initial,offset=offset)
        with patch.object(validation,'_fit_beta',side_effect=injected):
            report=validation.bootstrap_optimism(self.records,self.model,B=3,seed=53)
        self.assertEqual(report['status'],'NOT_ESTIMABLE')
        self.assertEqual((report['valid_replicates'],report['failed_replicates']),(2,1))
        self.assertTrue(all(row['corrected'] is None for row in report['table']))
        self.assertTrue(all(row['corrected_NB'] is None for row in report['dca']))
        self.assertIn('deliberate',report['ledger'][0]['reason'])

    def test_conditional_percentiles_never_delete_failed_replicates(self):
        original=validation._performance;counter={'calls':0}
        def injected(y,p):
            counter['calls']+=1
            if counter['calls']==2:
                raise ValueError('deliberate calibration non-estimability')
            return original(y,p)
        with patch.object(validation,'_performance',side_effect=injected):
            report=validation.fixed_model_intervals(self.y,self.p,B=3,seed=54)
        self.assertEqual(report['status'],'NOT_ESTIMABLE')
        self.assertTrue(all(row['lower'] is None and row['upper'] is None for row in report['table']))
        self.assertTrue(all(row['valid_replicates']==2 for row in report['table']))
        self.assertIn('no development',report['scope'])

    def test_curve_pointwise_range_is_not_an_interval_for_the_corrected_curve(self):
        report=validation.calibration_artifacts(self.records,self.model,B=3,conditional_B=4,seed=55,conditional_seed=56)
        self.assertTrue(report['success'])
        self.assertEqual(len(report['curve']),80)
        self.assertEqual(len(report['log']),7)
        self.assertEqual(report['recipe']['development_B'],3)
        self.assertEqual(report['recipe']['conditional_B'],4)
        self.assertEqual(report['recipe']['corrected_curve_interval'],'none; corrected point estimate and optimism MCSE')
        self.assertTrue(all(row['interval_target']=='fixed_model_apparent_curve_pointwise' for row in report['curve']))
        self.assertEqual(sum(row['n'] for row in report['groups']),3200)
        json.dumps(report,allow_nan=False)

    def test_small_subgroups_and_small_budget_are_explicitly_not_estimable(self):
        groups=validation.subgroup_performance(self.records,self.p,B=3,seed=57)
        self.assertEqual(groups['status'],'DESCRIPTIVE')
        self.assertTrue(all(row['interval_status']=='NOT_ESTIMABLE' for row in groups['table']))
        self.assertTrue(all(row['lower'] is None for row in groups['table']))
        tiny={'y':[0,0,0],'male':[0,0,0],'center':[1,1,1]}
        sparse=validation.subgroup_performance(tiny,[.1,.2,.3],B=3,seed=58)
        self.assertTrue(all(row['C'] is None for row in sparse['table']))
        self.assertTrue(all(row['lower'] is None for row in sparse['table']))

    def test_parameter_and_probability_failures_cannot_be_accepted(self):
        for bad in (0,1,True,2.5):
            with self.subTest(B=bad),self.assertRaises(ValueError):
                validation.fixed_model_intervals(self.y,self.p,B=bad)
        for confidence in (0,1,True,float('nan'),'0.95'):
            with self.subTest(confidence=confidence),self.assertRaises(ValueError):
                validation.fixed_model_intervals(self.y,self.p,B=3,confidence=confidence)
        for p in ([.2]*4,[0.,.2,.5,.8]):
            self.assertEqual(validation.performance([0,1,0,1],p)['status'],'NOT_ESTIMABLE')
        with self.assertRaises(ValueError):
            validation.net_benefit(self.y,self.p,[.1,.1])


if __name__=='__main__':
    unittest.main()
