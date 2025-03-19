import pandas as pd
from SPC import Rule
import numpy as np
pd.options.mode.chained_assignment = None # default='warn'
    
class Rule06(Rule):
    rule_id = 6

    def execute(self, df: pd.DataFrame):

        df[self.rule_id] = False


        for i in range(4, len(df)):
            points = df['value'][i - 4:i + 1]
            outside_c_up = sum((point > df['+1s'][0]) for point in points)
            outside_c_low = sum((point < df['-1s'][0]) for point in points)
            outside_c = outside_c_up if outside_c_up > outside_c_low else outside_c_low
            if outside_c >= 4:
                df[self.rule_id][i - 4:i + 1] = True
                df["SIGNAL"][i - 4:i + 1] = True

        for i in range(len(df)):
            if df[self.rule_id][i]:
                df['rule_message'][i].append(self.rule_id)