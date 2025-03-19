import gc
import io
from matplotlib.figure import Figure
from matplotlib import figure
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import poisson, norm
from scipy.optimize import fsolve
import numpy as np
import pandas as pd
from datetime import datetime
from SPC import ControlChart
import matplotlib
matplotlib.use('Agg')
# 中文展示
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
class CControlChart(ControlChart):
    def __init__(self, data_list:list, edit_time_list:list, upper:int =1, groupSize:int = 1, title_name:str =""):
        """ 初始化。

            :param data_list: 样本中的缺陷数量。
            :param groupSize: 样本大小。C图需要分组，因此分组大小为1即可
            :param edit_time_list 时间戳数组
            :param xlabel: x轴标签。
            :param ylabel: y轴标签。
            :param upper : 上限
        """
        # 初始化基类。
        super().__init__(1) # C图。 编号为1

        # 记录参数。
        self.leng_origi = len(data_list) # 总的数据的长度
        self._xlabel = "time"
        self._ylabel = "value"
        self._title_name = title_name
        self.n = groupSize # 分组大小
        self.upper = upper # 上限
        self.p_initial = 0.01
        self.cpu = 0 # 计算得到的cpu值


        # 记录缺陷数量。
        self.data_list = data_list
        self.edit_time_list = edit_time_list
        self.group_num = len(self.data_list)//self.n
        self.grouped_data = []
        self.grouped_time = []
        # 分组
        for i in range(self.group_num):
            data = self.data_list[i*self.n :(i+1)*self.n]
            data = sum(data) / len(data)
            self.grouped_data.append(data)

            time = self.edit_time_list[i*self.n:(i+1)*self.n]
            time = time[0]
            self. grouped_time.append(time)
        self.grouped_data = np.array(self.grouped_data)
        self.grouped_time = np.array(self.grouped_time)
        # print(self.grouped_time,self.grouped_data)
        # 初始化数组。
        self.cl_c = np.zeros((self.group_num, 1))
        self.ucl_c = np.zeros((self.group_num, 1))
        self.lcl_c = np.zeros((self.group_num, 1))
        self.two_sigma_plus = np.zeros((self.group_num, 1))
        self.one_sigma_plus = np.zeros((self.group_num, 1))
        self.two_sigma_min = np.zeros((self.group_num, 1))
        self.one_sigma_min = np.zeros((self.group_num, 1))

        # 计算UCL, CL, LCL。
        self.cl_c[:] = self.grouped_data.mean()
        self.ucl_c[:] = self.grouped_data.mean() + 3 * np.sqrt(self.grouped_data.mean())
        self.lcl_c[:] = self.grouped_data.mean() - 3 * np.sqrt(self.grouped_data.mean())

        # 计算一倍和两倍标准差。
        self.two_sigma_plus[:] = self.grouped_data.mean() + 3 * np.sqrt(self.grouped_data.mean()) * 2/3
        self.one_sigma_plus[:] = self.grouped_data.mean() + 3 * np.sqrt(self.grouped_data.mean()) * 1/3
        self.one_sigma_min[:] = self.grouped_data.mean() - 3 * np.sqrt(self.grouped_data.mean()) * 1/3
        self.two_sigma_min[:] = self.grouped_data.mean() - 3 * np.sqrt(self.grouped_data.mean()) * 2/3
        # print(self.calculate_poisson_solution())
        self.cpu = self.calculate_poisson_solution()

    def calculate_poisson_solution(self):
        """计算泊松分布解的实现，p_initial为自定义的初始猜测值"""
        # 初始化累积概率 eta
        eta = None
        # 根据是否给定上下限来计算累积概率
        if self.upper is not None : # 只有上限
            eta = poisson.cdf(self.upper, self.grouped_data.mean())

        else: # 没有给定上下限
            raise ValueError("Must provide either an upper limit, a lower limit, or both.")

        # 定义目标函数，用于求解超越方程
        def func(p, eta):
            return (norm.cdf(1.5 + p) - norm.cdf(1.5 - p)) - eta

        # 使用fsolve求解超越方程
        p_solution = fsolve(func, [self.p_initial], args=(eta,))[0]

        # 根据计算出的 p_solution 计算工序能力值
        CPU = p_solution / 3 - 0.5
        return round(CPU,4)


    def plot(self):
        """ 创建图表。
        """
        time_01 = datetime.now()
        fig = Figure(figsize=(15, 10))
        ax = fig.add_subplot(111)
        # x轴可以是数字或日期时间。

        format=super().dateformat
        # x_values_C = [datetime.fromtimestamp(i).strftime(f'{format}') for i in self.grouped_time]
        x_values_C = [datetime.strptime(i, '%Y-%m-%d %H:%M:%S').strftime(format) for i in self.grouped_time]
        # x_values_C = [datetime.strptime(d, format).date() for d in super().dates]
        # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(super().dateformat))
        ax.set_xticks(ticks=range(len(x_values_C)), labels=x_values_C, rotation=45, fontsize=8)
        fig.autofmt_xdate(rotation=45) # 旋转x轴标签以便更好地显示
        # ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter(format))
        # C图。
        ax.plot(x_values_C, self.grouped_data, marker="o",color="blue", label="c")
        ax.plot(x_values_C, self.ucl_c, color="r", label="UCL")
        # 获取数据。
        df = self.data(0)

        # 绘制信号。
        for i in np.where(df["SIGNAL"])[0]:
            ax.plot(x_values_C[i], self.grouped_data[i], marker="s", color="r")
            # print(df['rule_message'][i])
            text = str(df['rule_message'][i])
            text = text.replace("[", "").replace("]", "")
            # ax.text(x_values_C[i], self.grouped_data[i], text , va='bottom')
            ax.annotate(text, (x_values_C[i], self.grouped_data[i]), textcoords="offset points", xytext=(0, 10), ha='center')
        # +2s, +1s的界限指示器。
        if super().limits:
            ax.plot(x_values_C, self.two_sigma_plus, color="pink", linestyle='dashed', label="+2s")
            ax.plot(x_values_C, self.one_sigma_plus, color="r", linestyle='dashed', label="+1s")

        ax.plot(x_values_C, self.cl_c, color="green", label="CL")

        # -1s, -2s的界限指示器。
        if super().limits:
            ax.plot(x_values_C, self.one_sigma_min, color="r", linestyle='dashed', label="-1s")
            ax.plot(x_values_C, self.two_sigma_min, color="pink", linestyle='dashed', label="-2s")

        ax.plot(x_values_C, self.lcl_c, color="r", label="LCL")
        ax.set_title(f'CPU:{self.cpu}'+"\n"+self._title_name, color='green')
        # plt.text(6,-6,f'CPU:{self.cpu}', color='b')
        # 添加图例。
        ax.legend(loc='upper right')

        # 设置x轴标签。
        ax.set_xlabel(self._xlabel)
        # 设置y轴标签。
        ax.set_ylabel(self._ylabel)
        fig.tight_layout()
        # plt.savefig('a.png',dpi=300)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        fig.clf()
        plt.close('all')
        print(f" generating picture {datetime.now() - time_01} seconds.")
        buf.seek(0)
        return buf


    def data(self, index:int):
        """ 返回数据。

            :param index: 数据的索引(0 = C图)
        """
        df = pd.DataFrame(np.column_stack(
            [self.grouped_data, self.grouped_time, self.ucl_c, self.two_sigma_plus, self.one_sigma_plus, self.cl_c,
             self.one_sigma_min, self.two_sigma_min, self.lcl_c]),
                          columns=['value', 'time', 'UCL', '+2s', '+1s', 'CL', '-1s', '-2s', 'LCL'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df['UCL'] = pd.to_numeric(df['UCL'], errors='coerce')
        df['LCL'] = pd.to_numeric(df['LCL'], errors='coerce')
        df['CL'] = pd.to_numeric(df['CL'], errors='coerce')
        df['+1s'] = pd.to_numeric(df['+1s'], errors='coerce')
        df['+2s'] = pd.to_numeric(df['+2s'], errors='coerce')
        df['-1s'] = pd.to_numeric(df['-1s'], errors='coerce')
        df['-2s'] = pd.to_numeric(df['-2s'], errors='coerce')
        # df.astype({'value': 'float64', 'UCL': 'float64', '+2s': 'float64', '+1s': 'float64', 'CL': 'float64', 'LCL': 'float',
        # '-2s': 'float64', '-1s': 'float64'})

        if index == 0: # C图。
            # print(df.dtypes)
            self.execute_rules(df)
            # print(self._rules)

            # 检查x轴是数字还是日期时间。
            if (len(super().dates) != 0):
                df['日期'] = super().dates
                df= df.set_index('日期')

            return df

        raise ValueError

    def stable(self):
        """ 返回稳定指示器。
        """
        # 执行规则。
        df = self.data(0)

        if True in df["SIGNAL"].values:
            return False

        return True



if __name__ == '__main__':
    from Rule01 import Rule01
    from Rule02 import Rule02
    from Rule03 import Rule03
    from Rule04 import Rule04
    from Rule05 import Rule05
    from Rule06 import Rule06
    from Rule07 import Rule07
    from Rule08 import Rule08

    while True:
        n = 1
        c = [6.000,6.000,7.000,6.000,6.000]
        dates = ["2024-04-23 10:21:19","2024-04-23 10:21:20","2024-04-23 10:21:29","2024-04-23 10:21:39","2024-04-23 10:21:59"]
        chart = CControlChart(data_list=c, groupSize=n, edit_time_list=dates, upper=10, title_name="fuu")
        # chart.dates = dates
        chart.dateformat = "%Y-%m-%d %H:%M:%S"
        # stages=chart.stages(data=chart.c, max_stages=2)
        # if stages is not None:
        # chart.split(stages)
        # chart.split([10])
        chart.limits=True

        chart.append_rules([Rule01(), Rule02(), Rule03(), Rule04(), Rule05(), Rule06(), Rule07(), Rule08()])
        chart.plot()

        df1 = chart.data(0)
        print(df1)
        print("stable={0}".format(chart.stable()))
        input()
