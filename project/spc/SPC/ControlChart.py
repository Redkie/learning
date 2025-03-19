import pandas as pd
import numpy as np
import warnings
from SPC import Rule
from abc import ABC, abstractmethod
from scipy.stats import jarque_bera
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson
warnings.filterwarnings("ignore")

class ControlChart(ABC):
    _limits = False # 用于表示+2s、+1s、-1s、-2s的控制界限指示器。
    _dateformat = '%Y-%m-%d %H:%M:%S' # 日期格式。
    _dates = [] # 日期。
    _rules = [] # 规则。

    def __init__(self, number_of_charts: int):
        """ 初始化。

            :param number_of_charts: 图表的数量。
        """
        # 图表的类型。
        self._number_of_charts = number_of_charts
        self._rules = []
        self._dates = [] # 日期。

    @abstractmethod
    def plot(self):
        """ 绘制图表。
        """
        pass



    @abstractmethod
    def data(self, index: int):
        """ 返回数据。

            :param index: 数据的索引。
        """
        pass

    @property
    def limits(self):
        """ 返回控制界限指示器。
        """
        return self._limits

    @limits.setter
    def limits(self, limits):
        """ 设置控制界限指示器。

            :param limits: 控制界限指示器。
        """
        self._limits = limits

    @property
    def number_of_charts(self):
        """ 返回图表的编号。
        """
        return self._number_of_charts

    @property
    def dates(self):
        """ 返回日期。
        """
        return self._dates

    @dates.setter
    def dates(self, dates: list):
        """ 设置x轴的日期。

            :param dates: 日期。
        """
        self._dates = dates

    @property
    def dateformat(self) -> str:
        """ 返回日期格式。
        """
        return self._dateformat

    @dateformat.setter
    def dateformat(self, dateformat: str):
        """ 设置日期格式。

            :param dateformat: 日期格式。
        """
        self._dateformat = dateformat

    @abstractmethod
    def stable(self):
        """ 返回稳定指示器。
        """
        pass

    def append_rule(self, rule: Rule):
        """ 添加规则。

            :param rule: 规则。
        """
        # 添加规则。
        self._rules.append(rule)

    def append_rules(self, rules: list[Rule]):
        """ 添加规则。

            :param rules: 规则。
        """
        # 设置规则。
        self._rules = rules

    def execute_rules(self, df: pd.DataFrame):
        """ 规则执行。

            :param df: 数据框。
        """
        # 创建信号列。
        df["SIGNAL"] = False
        # 创建错误信息列
        df['rule_message'] = [[] for _ in range(len(df))] # 初始化为空列表
        for rule in self._rules:
            rule.execute(df)

    @staticmethod
    def _normally_distributed( data: list, significance_level: float):
        """ 检查数据是否符合正态分布。
            当数据没有显示出非正态性证据时返回真。
            当数据不符合正态分布时返回假。

            :param data: 值。
            :param significance_level: 显著性水平。
        """
        data = pd.DataFrame(data)
        if (len(data) > 5000):
            stat, p = jarque_bera(data)
        elif (len(data)>200):
            stat, p = normaltest(data)
        else:
            stat, p = shapiro(data)

        if p > significance_level:
            return True
        else:
            return False
    def _check_ad(self,data:list):
        ad, _, _ = anderson(data, dist='norm')
        return ad