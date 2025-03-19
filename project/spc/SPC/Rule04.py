"""
异常模式4:连续9点落在中心线同一侧
"""

import pandas as pd
from SPC import Rule
pd.options.mode.chained_assignment = None

class Rule04(Rule):
    rule_id = 4

    def execute(self, df: pd.DataFrame):
        """ 执行判异规则.

                   :param df: 数据dateframe
        """
        df[self.rule_id] = False
        
        # 初始化计数器
        count = 0
        prev_side = None
        
        for i in range(len(df)):
            # 判断当前点在中心线哪一侧
            if df['value'][i] > df['CL'][i]:
                current_side = 'above'
            elif df['value'][i] < df['CL'][i]:
                current_side = 'below'
            else:
                current_side = None
                
            # 如果与前一侧相同则计数加1，否则重置
            if current_side == prev_side and current_side is not None:
                count += 1
            else:
                count = 1
                prev_side = current_side
                
            # 检查是否达到9点
            if count >= 9:
                df[self.rule_id][i] = True
                df["rule_message"][i].append(self.rule_id)
                df["SIGNAL"][i] = True
