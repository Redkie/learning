import pandas as pd
from SPC import Rule
pd.options.mode.chained_assignment = None
class Rule08(Rule):
    rule_id = 8

    def execute(self, df: pd.DataFrame):
        """ Rule execution.

            :param df: The dataframe.
        """

        df[self.rule_id] = False

        for i in range(7, len(df)):

            points = df['value'][i - 7:i + 1]
            if (all(not (df['-1s'][0] < point < df['+1s'][0]) for point in points)
                        and any(point < df['CL'][0] for point in points)
                        and any(point > df['CL'][0] for point in points)):
                df[self.rule_id][i - 14:i + 1] = True
                df["SIGNAL"][i - 14:i + 1] = True

        for i in range(len(df)):
            if df[self.rule_id][i]:
                df['rule_message'][i].append(self.rule_id)