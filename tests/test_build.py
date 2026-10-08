from __future__ import annotations

import io
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from ndxbots.config import load_settings
from ndxbots.strategy.build import run_build


class BuildOutputTests(unittest.TestCase):
    def _run(self, scored, targets, signal_date):
        with tempfile.TemporaryDirectory(prefix='.build-test-', dir=Path(__file__).resolve().parent) as folder:
            settings = replace(load_settings(), data_root=Path(folder))
            factors_dir = settings.data_root / 'factors'
            factors_dir.mkdir()
            pd.DataFrame({'date': [signal_date], 'code': ['US.QQQ']}).to_parquet(
                factors_dir / 'daily.parquet', index=False)
            panel_dir = settings.data_root / 'panel'
            panel_dir.mkdir()
            pd.DataFrame({'date': [signal_date], 'code': ['US.QQQ'], 'close': [100.]}).to_parquet(
                panel_dir / 'daily.parquet', index=False)
            simulation = {'curve': pd.DataFrame({'date': [signal_date]}), 'targets': targets}
            with patch('ndxbots.strategy.build.load_settings', return_value=settings), \
                 patch('ndxbots.strategy.build.build_score_table', return_value=scored), \
                 patch('ndxbots.strategy.build.run_backtest', return_value=simulation), \
                 patch('ndxbots.strategy.build._print_regime'), redirect_stdout(io.StringIO()):
                run_build()
            out = settings.data_root / 'strategy'
            return pd.read_csv(out / 'observation_pool.csv'), pd.read_csv(out / 'exec_pool.csv')

    def test_execution_includes_retained_outside_pool_and_zero_exit(self):
        day = pd.Timestamp('2026-09-18')
        scored = pd.DataFrame({'date': [day]*2, 'code': ['US.WATCH','US.RETAIN'],
                              'side': ['long','short'], 'in_pool': [True,False],
                              'in_hold': [False,True], 'pool_rank': [1.,float('nan')],
                              'short_pool_rank': [float('nan'),12.], 'score': [.9,.3],
                              'side_score': [.9,.7], 'combo_rank': [1.,12.]})
        targets = pd.DataFrame({'date': [day]*2, 'code': ['US.RETAIN','US.EXIT'],
                               'side': ['short','long'], 'target_quantity': [-2.,0.],
                               'current_quantity': [-2.,3.], 'delta_quantity': [0.,-3.],
                               'target_weight': [-.25,0.]})
        watch, execution = self._run(scored,targets,day)
        self.assertEqual(watch.code.tolist(),['US.WATCH'])
        self.assertEqual(execution.code.tolist(),['US.RETAIN','US.EXIT'])
        self.assertEqual(execution.loc[execution.code.eq('US.EXIT'),'target_quantity'].iloc[0],0)
        self.assertTrue(execution.model_source.eq('backtest').all())

    def test_missing_latest_stock_rows_do_not_export_stale_watchlist(self):
        previous = pd.Timestamp('2026-09-17')
        latest = pd.Timestamp('2026-09-18')
        scored = pd.DataFrame({'date': [previous], 'code': ['US.STALE'],
                              'side': ['long'], 'in_pool': [True], 'in_hold': [True],
                              'pool_rank': [1.], 'short_pool_rank': [float('nan')],
                              'score': [.9], 'side_score': [.9], 'combo_rank': [1.]})
        targets = pd.DataFrame({'date': [latest], 'code': ['US.STALE'],
                               'side': ['long'], 'target_quantity': [0.],
                               'current_quantity': [3.], 'delta_quantity': [-3.],
                               'target_weight': [0.]})
        watch, execution = self._run(scored,targets,latest)
        self.assertTrue(watch.empty)
        self.assertEqual(execution.code.tolist(),['US.STALE'])
        self.assertEqual(pd.Timestamp(execution.date.iloc[0]),latest)


if __name__ == '__main__':
    unittest.main()
