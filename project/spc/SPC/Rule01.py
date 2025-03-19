"""
异常模式1:超出控制限
"""

import pandas as pd
from SPC import Rule
pd.options.mode.chained_assignment = None

class Rule01(Rule):
    rule_id = 1

    def execute(self, df: pd.DataFrame):
        """ 执行判异规则.

                   :param df: 数据dateframe
        """

        df[self.rule_id] = False

        for i in range(len(df)):
            # 检查判异规则一 .
            if df['value'][i] < df['LCL'][i] or df['value'][i] > df['UCL'][i]:
                df[self.rule_id][i] = True
                df["rule_message"][i].append(self.rule_id)
                df["SIGNAL"][i] = True