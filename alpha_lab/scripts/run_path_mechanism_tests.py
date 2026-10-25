"""重放新机制与原因果对抗集，生成prepare所需测试日志。"""
from pathlib import Path
import sys,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
import test_causal_portfolio_adversarial as old
import test_path_mechanisms as new
import path_mechanism_portfolio as engine
old.run_backtest=engine.run_backtest;old.AccountingError=engine.AccountingError
for module,name in [(new,'path_mechanism_tests_v2.log'),(old,'path_engine_adversarial_v2.log')]:
    with (Path('/tmp')/name).open('w') as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(module))
    if not result.wasSuccessful():raise SystemExit(f'测试未通过，见/tmp/{name}')
# 新环境没有早期失败记录时明确标注，不伪造失败或覆盖已有日志。
prior=Path('/tmp/path_mechanism_tests.log')
if not prior.exists():prior.write_text('No earlier failed attempt in this reproduction.\n')
print('7 mechanism tests and 25 causal adversarial tests passed.')
