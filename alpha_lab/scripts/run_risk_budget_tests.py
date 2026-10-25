"""在新撮合分支运行风险行为测试和原25项因果对抗测试。"""
from pathlib import Path
import sys
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests')]
import test_risk_budget as new
import test_causal_portfolio_adversarial as old
import risk_budget_portfolio as engine
old.run_backtest = engine.run_backtest
old.AccountingError = engine.AccountingError
for module, name in [(new, 'risk_budget_unit.log'), (old, 'risk_budget_adversarial.log')]:
    with (Path('/tmp') / name).open('w') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    if not result.wasSuccessful():
        raise SystemExit(f'失败记录: /tmp/{name}')
print('6 risk tests and 25 causal adversarial tests passed')
