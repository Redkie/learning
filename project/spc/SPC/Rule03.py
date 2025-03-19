import pandas as pd
from SPC import Rule
import numpy as np
pd.options.mode.chained_assignment = None # default='warn'

class Rule03(Rule):
    rule_id = 3 # 判异规则3

    def execute(self, df: pd.DataFrame):
        """ 执行判异规则.

            :param df: 数据df
        """
def execute(self, df: pd.DataFrame):

    df[self.rule_id] = False
    for i in range(13, len(df)):
        points = df['value'][i-13:i+1].tolist()
        # print(points)
        if all((points[index] > df['CL'][i]) != (points[index + 1] > df['CL'][i]) for index in range(len(points) - 1)):
            df[self.rule_id][i-13:i+1] = True
            df["SIGNAL"][i-13:i+1] = True
            # for idn in range(i - 13, i + 1):
            # df["rule_message"][idn].append(self.rule_id)
    for i in range(len(df)):
        if df[self.rule_id][i]:
            df['rule_message'][i].append(self.rule_id)