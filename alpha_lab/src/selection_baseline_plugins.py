"""基线开发插件示例；identity用于验证候选路径，不代表新增策略。"""


def identity(scores,*,context,params):
    """原样返回父基线分数。新策略应放在独立模块，并只使用当日或之前信息。"""
    return scores.copy()
