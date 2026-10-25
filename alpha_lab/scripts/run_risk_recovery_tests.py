"""在恢复分支验证新增行为及已有风险/撮合测试。"""
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
import risk_recovery_portfolio as engine
import test_risk_recovery as recovery
import test_risk_budget as risk
import test_causal_portfolio_adversarial as causal
risk.run_backtest = engine.run_backtest
causal.run_backtest = engine.run_backtest
causal.AccountingError = engine.AccountingError
for module, name in [(recovery, 'recovery_unit.log'), (risk, 'recovery_risk.log'), (causal, 'recovery_causal.log')]:
    with (Path('/tmp') / name).open('w') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    if not result.wasSuccessful():
        raise SystemExit(f'测试失败，见/tmp/{name}')
print('6 recovery + 6 risk + 25 causal tests passed')
