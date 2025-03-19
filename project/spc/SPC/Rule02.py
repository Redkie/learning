import pandas as pd
from SPC import Rule
pd.options.mode.chained_assignment = None
class Rule02(Rule):
    rule_id = 2

    def execute(self, df: pd.DataFrame):
        """ 执行判异规则.
            :param df: 数据dateframe
        """


        df[self.rule_id] = False

        # 异常模式2：连续9个点在同一侧
        for i in range(8, len(df)):
            if all(x > df['CL'][i] for x in df['value'][i-8:i+1]) or all(x < df['CL'][i] for x in df['value'][i-8:i+1]):
                df[self.rule_id][i-8:i+1] = True
                df["SIGNAL"][i-8:i+1] = True
                # for idn in range(i - 8, i + 1):
                # df["rule_message"][idn].append(self.rule_id)

        # 添加判异错误列表
        for i in range(len(df)):
            if df[self.rule_id][i]:
                df['rule_message'][i].append(self.rule_id)