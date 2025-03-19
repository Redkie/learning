import pandas as pd
from SPC import Rule
import numpy as np
pd.options.mode.chained_assignment = None # default='warn'
class Rule07(Rule):
    rule_id = 7

    def execute(self, df: pd.DataFrame):

        df[self.rule_id] = False

        for i in range(15, len(df)):
            points = df['value'][i - 14:i + 1]
            if all(df['-1s'][0] < point < df['+1s'][0] for point in points):
                df[self.rule_id][i - 14:i + 1] = True
                df["SIGNAL"][i - 14:i + 1] = True

        for i in range(len(df)):
            if df[self.rule_id][i]:
                df['rule_message'][i].append(self.rule_id)