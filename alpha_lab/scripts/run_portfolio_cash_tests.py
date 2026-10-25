"""分别在组合现金与提前重入分支运行行为和因果对抗检验。"""
from pathlib import Path
import argparse
import sys
import unittest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/root/autodl-tmp/alpha_lab'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    sys.path[:0]=[str(args.root/'src'),str(args.root/'tests')]
    import test_portfolio_cash as cash
    import test_cash_reentry as recovery
    import test_causal_portfolio_adversarial as old
    import portfolio_cash_engine as primary
    import portfolio_cash_reentry_engine as alternate
    for name,engine,count in [('cash',primary,31),('reentry',alternate,32)]:
        old.run_backtest=engine.run_backtest;old.AccountingError=engine.AccountingError;cash.engine=engine
        suites=[unittest.defaultTestLoader.loadTestsFromModule(m) for m in [cash,old]]
        if name=='reentry':suites.append(unittest.defaultTestLoader.loadTestsFromTestCase(recovery.RecoveryEntryTests))
        with (args.output/f'{name}_tests.log').open('w') as stream:
            result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.TestSuite(suites))
        assert result.wasSuccessful() and result.testsRun==count
        print(name,count,'passed')


if __name__=='__main__':main()
