import pandas as pd
from SPC import Rule
import numpy as np
pd.options.mode.chained_assignment = None # default='warn'


class Rule05(Rule):
    rule_id = 5

    def execute(self, df: pd.DataFrame):
        df[self.rule_id] = False

        # Iterate the rows.
        for i in range(2, len(df)):

            if i >= 2:
                points =df['value'][i - 2:i + 1]
                if ((points.iloc[0] > df['+2s'][0]) + (points.iloc[1] > df['+2s'][0]) + (points.iloc[2] > df['+2s'][0])) >= 2 or ((points.iloc[0] < df['-2s'][0]) + (points.iloc[1] < df['-2s'][0]) + (points.iloc[2] < df['-2s'][0])) >= 2:
                    df[self.rule_id][i - 2:i + 1] = True
                    df["SIGNAL"][i - 2:i + 1] = True
                    # for idn in range(i - 5, i + 1):
                    # df["rule_message"][idn].append(self.rule_id)

        for i in range(len(df)):
            if df[self.rule_id][i]:
                df['rule_message'][i].append(self.rule_id)