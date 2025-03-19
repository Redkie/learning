

from SPC.quality_indicators import QualityIndicators
import re
import pandas as pd
import numpy as np
from scipy.special import gamma
from scipy.stats import norm, probplot, shapiro, anderson
import matplotlib.pyplot as plt
import os
from datetime import datetime
import datetime
import io
from matplotlib.figure import Figure
import statsmodels.api as sm

from matplotlib import figure, gridspec
from datetime import datetime
from SPC import ControlChart
import matplotlib

matplotlib.use('Agg')
# 中文展示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)


class Quan_SixChart(ControlChart):
    def __init__(self, data_list: list, edit_time_list: list, groupSize: int = 2, title_name: str = "", **kwargs):
        """ 初始化。

            :param data_list: 样本中数据列表
            :param groupSize: 样本大小。
            :param edit_time_list 时间数组

            :param upper : 上限
            :param lower :下限
            :param title_name : 表名称
        """
        # 初始化基类。
        super().__init__(9)  # SIXPACK图。 编号为9

        # 记录参数。
        self.para_result = None
        self.leng_origi = len(data_list)  # 总的数据的长度
        self._title_name = title_name
        self.n = int(groupSize)  # 分组大小
        self.usl = kwargs.get('usl')  # 规格上限
        self.lsl = kwargs.get('lsl')  # 规格下限
        self.sl = kwargs.get('sl')  # 规格中限
        self.xi_ucl = kwargs.get('xi_ucl')
        self.xi_lcl = kwargs.get('xi_lcl')
        self.xi_cl = kwargs.get('xi_cl')
        self.mrs_ucl = kwargs.get('mrs_ucl')
        self.mrs_lcl = kwargs.get('mrs_lcl')
        self.mrs_cl = kwargs.get('mrs_cl')

        self.cpu = 0  # 计算得到的cpu值
        self.is_stable = True
        self.is_normal = ControlChart._normally_distributed(data_list)
        self.is_box = False
        self.is_john = False
        self.xla = False
        # 记录缺陷数量。
        self.data_list = data_list
        # 对传进来的时间戳数据进行判断
        format1 = super().dateformat

        if all(isinstance(t, int) for t in edit_time_list):
            self.edit_time_list = pd.to_datetime(edit_time_list, unit='ms').strftime(format1)
        else:
            self.edit_time_list = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S').strftime(format1)
                                   if t else str(i) for i, t in enumerate(edit_time_list)]

        self.df = pd.DataFrame({
            'value': np.array(self.data_list, dtype=float),
            'time': self.edit_time_list
        })

        # use box_cox
        if all(self.df['value'] > 0) and self.is_normal is False:
            # 满足box-cox要求全为正数分布且数据不符合正太分布

            tra_data, lam = ControlChart.myboxcox(self.df["value"])
            if lam == 'failed':
                pass
            else:
                # 先判断上下限是否都大于0,能否也box-cox转化
                if self.usl is not None and self.lsl is not None:
                    if self.usl > 0 and self.lsl > 0:
                        self.is_box = True
                        if lam != 0:
                            self.usl = (self.usl ** lam - 1) / lam
                            self.lsl = (self.lsl ** lam - 1) / lam
                        else:
                            self.usl = np.log(self.usl)
                            self.lsl = np.log(self.lsl)
                if self.usl is not None and self.lsl is None:  # 只有规格上限
                    if self.usl > 0:
                        if lam != 0:
                            self.usl = (self.usl ** lam - 1) / lam
                        else:
                            self.usl = np.log(self.usl)
                if self.usl is None and self.lsl is not None:
                    if self.lsl > 0:
                        if lam != 0:
                            self.lsl = (self.lsl ** lam - 1) / lam
                        else:
                            self.lsl = np.log(self.lsl)

                self.df['box_value'] = tra_data

        if self.n == 1:
            self.Rs = []

            for i in range(self.leng_origi - 1):
                self.Rs.append(abs(self.df['value'][i + 1] - self.df['value'][i]))
            self.rs_mean = sum(self.Rs) / len(self.Rs)

    def subgroup_data(self):
        """按固定子组容量分子组"""
        # 先按时间字段排序，为后面做准备
        # self.df = self.df.sort_values(by='time')

        subgroups = []

        # 计算每个分组应有的子组数量，舍弃最后不足 n 的子组
        num_subgroups = len(self.df) // self.n
        # 根据子组容量 n 对分组内的数据进行分组
        for i in range(num_subgroups):
            # 按子组提取数据
            subgroup = self.df.iloc[i * self.n:(i + 1) * self.n].copy()
            # 使用 .loc 来设置新的列，确保修改的是原始 DataFrame
            subgroup.loc[:, 'group_id'] = i + 1  # 使用 .loc 来设置 'group_id'
            subgroup.loc[:, 'subgroup_index'] = range(1, self.n + 1)  # 从 1 到 n
            # 将子组添加到列表中
            subgroups.append(subgroup)

        self.df = pd.concat(subgroups)

    def process_grouped_data(self):
        """处理分组数据，计算统计指标"""
        # 创建一个空列表来存储分组的汇总结果
        results = []
        quality_indicators = QualityIndicators()

        # 应用 calculate_sigma_within 函数
        stats = quality_indicators.calculate_sigma_within(self.df, self.usl, self.lsl)
        results.append(stats)

        # 将所有分组的汇总结果合并为一个 DataFrame
        results = pd.concat(results, ignore_index=True)
        # print(results)
        self.para_result = results

    def format_value(self, value):
        if isinstance(value, (int, float)):  # 检查是否为数字类型
            return f"{value:.2f}"  # 格式化为浮点数
        return "*"  # 不可格式化的值，返回默认字符串

    def calculate_D3_and_D4(self):
        d2_data = {
            2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326,
            6: 2.534, 7: 2.704, 8: 2.847, 9: 2.97,
            10: 3.078, 11: 3.173, 12: 3.258, 13: 3.336,
            14: 3.407, 15: 3.472, 16: 3.532, 17: 3.588,
            18: 3.64, 19: 3.689, 20: 3.735, 21: 3.778,
            22: 3.819, 23: 3.858, 24: 3.895, 25: 3.931,
            26: 3.964, 27: 3.997, 28: 4.027, 29: 4.057,
            30: 4.086, 31: 4.113, 32: 4.139, 33: 4.165,
            34: 4.189, 35: 4.213, 36: 4.236, 37: 4.259,
            38: 4.28, 39: 4.301, 40: 4.322, 41: 4.341,
            42: 4.361, 43: 4.379, 44: 4.398, 45: 4.415,
            46: 4.433, 47: 4.45, 48: 4.466, 49: 4.482,
            50: 4.498
        }
        d3_data = {
            2: 0.8525, 3: 0.8884, 4: 0.8798, 5: 0.8641,
            6: 0.848, 7: 0.8332, 8: 0.8193, 9: 0.8078,
            10: 0.7971, 11: 0.7873, 12: 0.7785, 13: 0.7704,
            14: 0.763, 15: 0.7562, 16: 0.7499, 17: 0.7441,
            18: 0.7386, 19: 0.7335, 20: 0.7287, 21: 0.7242,
            22: 0.7199, 23: 0.7159, 24: 0.7121, 25: 0.7084
        }
        # 检查样本大小是否在有效范围内
        if self.n < 2:
            raise ValueError("Sample size n must be greater than or equal to 2.")

        # 查找D2值
        if self.n in d2_data:
            d2 = d2_data[self.n]
        else:
            d2 = 3.4873 + 0.0250141 * self.n - 0.00009823 * self.n ** 2

        # 查找D3值
        if self.n in d3_data:
            d3 = d3_data[self.n]
        else:
            d3 = 0.80818 - 0.0051871 * self.n + 0.00005098 * self.n ** 2 - 0.00000019 * self.n ** 3
        A2 = 3 / (np.sqrt(self.n) * d2)

        if self.n < 7:
            D3 = 0
            D4 = 1 + 3 * d3 / d2
        else:
            D3 = 1 - 3 * d3 / d2
            D4 = 1 + 3 * d3 / d2
        return A2, D3, D4

    def calculate_B(self):
        if self.n < 2:
            raise ValueError("Sample size n must be greater than or equal to 2.")
        c2 = np.sqrt(2 / (self.n - 1)) * gamma(self.n / 2) / gamma((self.n - 1) / 2)
        c3 = np.sqrt(1 - c2 ** 2)  # 使用 ** 运算符进行幂运算
        # print("c2", c2, "\n c3", c3)
        if self.n < 6:
            Bl = 0
            Bu = 1 + 3 * c3 / c2
            As = 3 / (c2 * np.sqrt(self.n))
        else:
            Bl = 1 - 3 * c3 / c2
            Bu = 1 + 3 * c3 / c2
            As = 3 / (c2 * np.sqrt(self.n))
        return Bl, Bu, As

    def calculate_control_limits(self, grouped):
        if self.is_normal:
            # 计算均值控制图的控制限
            # 判断标准差取R_bar还是std_bar
            Bl, Bu, As = self.calculate_B()
            A2, D3, D4 = self.calculate_D3_and_D4()
            overall_std = grouped['std'].mean()  # 使用子组标准差的平均值
            overall_mean = grouped['mean'].mean()
            mean_range = grouped['range'].mean()
            if self.n > 8:
                # xs  ---> x
                UCL_mean = overall_mean + As * overall_std
                LCL_mean = overall_mean - As * overall_std

            else:
                # xr ----->x
                UCL_mean = overall_mean + A2 * mean_range
                LCL_mean = overall_mean - A2 * mean_range
            # 计算极差控制图的控制限
            mean_range = grouped['range'].mean()
            A2, D3, D4 = self.calculate_D3_and_D4()
            UCL_range = D4 * mean_range
            LCL_range = D3 * mean_range

            # 计算标准偏差控制图的控制限
            UCL_std = Bu * overall_std
            LCL_std = Bl * overall_std
        else:
            # 使用分位数计算中心线和上下控制限
            quantile_values = grouped['mean'].quantile([0.00135, 0.5, 0.99865])
            overall_mean = quantile_values[0.5]
            UCL_mean = quantile_values[0.99865]
            LCL_mean = quantile_values[0.00135]

            # 计算极差控制图的控制限
            quantile_values_range = grouped['range'].quantile([0.00135, 0.5, 0.99865])
            mean_range = quantile_values_range[0.5]
            UCL_range = quantile_values_range[0.99865]
            LCL_range = quantile_values_range[0.00135]

            # 计算标准偏差控制图的控制限
            quantile_values_std = grouped['std'].quantile([0.00135, 0.5, 0.99865])
            overall_std = quantile_values_std[0.5]
            UCL_std = quantile_values_std[0.99865]
            LCL_std = quantile_values_std[0.00135]

        return overall_mean, overall_std, mean_range, UCL_mean, LCL_mean, UCL_range, LCL_range, UCL_std, LCL_std

    # 绘制能力直方图
    def draw_NDC(self, ax):
        # 假设 df 是包含数据的 DataFrame

        # 将 'value' 列中的数据转换为 float 类型
        self.df.loc[:, 'value'] = self.df['value'].astype(float)

        # 计算数据的均值和标准差
        mean = self.df['value'].mean()
        std = self.df['value'].std()

        # 生成正态分布数据
        x = np.linspace(mean - 3 * std, mean + 3 * std, 100)
        y = norm.pdf(x, mean, std)
        std_in = round(self.para_result['sigma_within'].values[0], 5)
        x_in = np.linspace(mean - 3 * std_in, mean + 3 * std_in, 100)
        y_in = norm.pdf(x_in, mean, std_in)
        # 绘制直方图
        ax.hist(self.df['value'], bins=20, density=True, alpha=0.6, color='g', edgecolor='black')

        # 绘制正态分布曲线
        ax.plot(x, y, 'r', linewidth=2, label="整体")
        ax.plot(x_in, y_in, 'b--', linewidth=2, label='组内')
        # 绘制规格下限和规格上限的竖线
        if self.usl is not None:
            ax.axvline(x=self.usl, color='b', linestyle='--', label=f'上规范Tu: {self.usl:.2f}')
        if self.lsl is not None:
            ax.axvline(x=self.lsl, color='b', linestyle='--', label=f'下规范Tl: {self.lsl:.2f}')

        ax.set_title('能力直方图')
        ax.set_xlabel('样本值')
        ax.set_ylabel('频率')
        ax.grid(True)
        ax.legend(fontsize=8, frameon=False)

        # 添加文本到图的右侧
        text_x = 1.05  # 确保文本在图的右侧，这里假设x轴的最大值为1
        ax.text(text_x, 0.65, '异常模式1：超出控制限', fontsize=13, verticalalignment='center', transform=ax.transAxes)
        ax.text(text_x, 0.45, '异常模式2：连续9个点落在中心线同一侧', fontsize=13, verticalalignment='center',
                transform=ax.transAxes)
        ax.text(text_x, 0.25, '异常模式3：连续6个点递增或递减', fontsize=13, verticalalignment='center',
                transform=ax.transAxes)
        ax.text(text_x, 0.05, '异常模式4：连续14个点中相邻点交替上下', fontsize=13, verticalalignment='center',
                transform=ax.transAxes)

        ax.text(text_x, 1.0, f'样本均值 {mean:.4f}', fontsize=13, verticalalignment='center', transform=ax.transAxes)
        ax.text(text_x, 0.8, f'样本N {self.n}', fontsize=13, verticalalignment='center', transform=ax.transAxes)

    # 绘制均值-标准偏差图和均值-极差图
    def draw_MSD_and_RChart(self, ax1, ax2):

        # 分组并计算均值、标准偏差和极差
        grouped = self.df.groupby('group_id')['value'].agg(['mean', 'std', lambda x: x.max() - x.min()])
        grouped.columns = ['mean', 'std', 'range']
        # print(grouped)
        # 提取每个分组的第一个 result_time 并格式化为年月日
        if self.xla is False:
            first_result_times = self.df.groupby('group_id')['time'].first().reset_index(drop=True)
        else:
            first_result_times = range(len(grouped['mean']))

        # print(first_result_times)
        # 计算控制限
        overall_mean, overall_std, mean_range, UCL_mean, LCL_mean, UCL_range, LCL_range, UCL_std, LCL_std = self.calculate_control_limits(
            grouped)
        # 是否有传入控制限
        # X
        if self.xi_ucl is not None and self.xi_lcl is not None:
            if self.xi_cl is not None:
                overall_mean = self.xi_cl
            else:
                overall_mean = (self.xi_ucl + self.xi_lcl) / 2

        if self.xi_ucl is not None:
            UCL_mean = self.xi_ucl

        if self.xi_lcl is not None:
            LCL_mean = self.xi_lcl
        # 绘制均值控制图
        # 均值控制图的判异
        signal_df, message1 = self.data(grouped['mean'], overall_mean, UCL_mean, LCL_mean)
        mask = (signal_df['value'] > self.usl) | (signal_df['value'] < self.lsl)
        signal_df['SIGNAL'] |= mask
        indices = str(signal_df[(signal_df['value'] > self.usl) | (signal_df['value'] < self.lsl)].index.to_list())
        indices = indices.replace("[", "").replace("]", "")
        ax1.plot(first_result_times, signal_df['value'], marker='o', color="blue", linestyle='-')
        for i in np.where(signal_df["SIGNAL"])[0]:
            ax1.plot(first_result_times[i], signal_df['value'][i], marker="s", color="r")
            text = str(signal_df['rule_message'][i])
            text = text.replace("[", "").replace("]", "")
            ax1.annotate(text, (first_result_times[i], signal_df['value'][i]), textcoords="offset points",
                         xytext=(0, 10),
                         ha='center')
        # +2s, +1s的界限指示器。
        if super().limits:
            ax1.plot(first_result_times, signal_df['+2s'], color="pink", linestyle='dashed', label="+2s")
            ax1.plot(first_result_times, signal_df['+1s'], color="pink", linestyle='dashed', label="+1s")
            ax1.plot(first_result_times, signal_df['-2s'], color="pink", linestyle='dashed', label="-2s")
            ax1.plot(first_result_times, signal_df['-1s'], color="pink", linestyle='dashed', label="-1s")

        ax1.axhline(overall_mean, color='green', linestyle='--', label=f'中心限:{overall_mean:.4f}')
        ax1.axhline(UCL_mean, color='orange', linestyle='--', label=f'上限:{UCL_mean:.4f}')
        ax1.axhline(LCL_mean, color='orange', linestyle='--', label=f'下限:{LCL_mean:.4f}')
        if self.usl is not None and self.lsl is not None:
            if self.sl is not None:
                ax1.axhline(self.sl, color='purple', linestyle='-', label=f'规格中心:{self.sl:.4f}')
            else:
                self.sl = (self.usl+self.lsl)/2
                ax1.axhline(self.sl, color='purple', linestyle='-', label=f'规格中心:{self.sl:.4f}')
        if self.usl is not None:
            ax1.axhline(self.usl, color='red', linestyle='-', label=f'规格上限:{self.usl:.4f}')
        if self.lsl is not None:
            ax1.axhline(self.lsl, color='red', linestyle='-', label=f'规格下限:{self.lsl:.4f}')

        if self.is_normal:
            ax1.set_title(f"均值控制图")
        else:
            ax1.set_title(f"均值控制图（分位数法）")
        ax1.set_xlabel('时间')
        ax1.set_ylabel('均值')
        ax1.legend(fontsize=8, frameon=False,  bbox_to_anchor=(-0.1, 0.8))
        ax1.grid(False)

        ax1.set_xticklabels(first_result_times, rotation=45, ha='right')

        # 根据容量n决定绘制标准偏差图还是极差控制图
        if self.n >= 8:
            # 标准偏差图
            # 判异
            message_title = "标准偏差控制图"
            if self.mrs_ucl is not None and self.mrs_lcl is not None:
                if self.mrs_cl is not None:
                    overall_std = self.mrs_cl
                else:
                    overall_std = (self.mrs_ucl + self.mrs_lcl) / 2

            if self.mrs_ucl is not None:
                UCL_std = self.mrs_ucl

            if self.mrs_lcl is not None:
                LCL_std = self.mrs_lcl
            signal_df, message2 = self.data(grouped['std'], overall_std, UCL_std, LCL_std)

            # 绘图

            ax2.plot(first_result_times, signal_df['value'], marker='o', color="blue", linestyle='-')
            for i in np.where(signal_df["SIGNAL"])[0]:
                ax2.plot(first_result_times[i], signal_df['value'][i], marker="s", color="r")
                text = str(signal_df['rule_message'][i])
                text = text.replace("[", "").replace("]", "")
                ax2.annotate(text, (first_result_times[i], signal_df['value'][i]), textcoords="offset points",
                             xytext=(0, 10),
                             ha='center')
            if super().limits:
                ax2.plot(first_result_times, signal_df['+2s'], color="pink", linestyle='dashed', label="+2s")
                ax2.plot(first_result_times, signal_df['+1s'], color="pink", linestyle='dashed', label="+1s")
                ax2.plot(first_result_times, signal_df['-2s'], color="pink", linestyle='dashed', label="-2s")
                ax2.plot(first_result_times, signal_df['-1s'], color="pink", linestyle='dashed', label="-1s")

            ax2.axhline(overall_std, color='green', linestyle='--', label=f'中心限:{overall_std:.4f}')
            ax2.axhline(UCL_std, color='orange', linestyle='--', label=f'上限:{UCL_std:.4f}')
            ax2.axhline(LCL_std, color='orange', linestyle='--', label=f'下限:{LCL_std:4f}')
            if self.is_normal:
                ax2.set_title('标准偏差控制图')
            else:
                ax2.set_title('标准偏差控制图（分位数法）')
            ax2.set_xlabel('时间')
            ax2.set_ylabel('标准偏差')
            ax2.legend(fontsize=8, frameon=False, bbox_to_anchor=(-0.1, 0.6))
            ax2.grid(False)
            ax2.set_xticklabels(first_result_times, rotation=45, ha='right')
        else:
            # 极差控制图
            message_title = "极差控制图"
            if self.mrs_ucl is not None and self.mrs_lcl is not None:
                if self.mrs_cl is not None:
                    mean_range = self.mrs_cl
                else:
                    mean_range = (self.mrs_ucl + self.mrs_lcl) / 2

            if self.mrs_ucl is not None:
                UCL_range = self.mrs_ucl

            if self.mrs_lcl is not None:
                LCL_range = self.mrs_lcl
            # 判异
            signal_df, message2 = self.data(grouped['range'], mean_range, UCL_range, LCL_range)
            ax2.plot(first_result_times, signal_df['value'], marker='o', color="blue", linestyle='-')
            for i in np.where(signal_df["SIGNAL"])[0]:
                ax2.plot(first_result_times[i], signal_df['value'][i], marker="s", color="r")
                text = str(signal_df['rule_message'][i])
                text = text.replace("[", "").replace("]", "")
                ax2.annotate(text, (first_result_times[i], signal_df['value'][i]), textcoords="offset points",
                             xytext=(0, 10),
                             ha='center')
            if super().limits:
                ax2.plot(first_result_times, signal_df['+2s'], color="pink", linestyle='dashed', label="+2s")
                ax2.plot(first_result_times, signal_df['+1s'], color="pink", linestyle='dashed', label="+1s")
                ax2.plot(first_result_times, signal_df['-2s'], color="pink", linestyle='dashed', label="-2s")
                ax2.plot(first_result_times, signal_df['-1s'], color="pink", linestyle='dashed', label="-1s")

            ax2.axhline(UCL_range, color='orange', linestyle='--', label=f'上限:{UCL_range:.4f}')
            ax2.axhline(LCL_range, color='orange', linestyle='--', label=f'下限:{LCL_range:.4f}')
            ax2.axhline(mean_range, color='green', linestyle='--', label=f'中心限:{mean_range:.4f}')
            if self.is_normal:
                ax2.set_title('极差控制图')
            else:
                ax2.set_title('极差控制图（分位数法）')
            ax2.set_xlabel('时间')
            ax2.set_ylabel('极差')
            ax2.legend(fontsize=8, frameon=False, bbox_to_anchor=(-0.1, 0.6))
            ax2.grid(False)
            ax2.set_xticklabels(first_result_times, rotation=45, ha='right')
        message = "均值控制图\n" + message1 + "\n" + message_title + "\n" + message2 + '\n' + "超出规范线的数据：\n" + indices
        # print(message)
        # print(signal_df)
        return message

    # 绘制单值移动极差控制图
    def draw_imrchart(self, ax1, ax2):
        CL_I = self.df['value'].mean()
        UCL_I = CL_I + 2.66 * self.df['rs_mean'][0]
        LCL_I = CL_I - 2.66 * self.df['rs_mean'][0]

        if self.xi_ucl is not None and self.xi_lcl is not None:
            if self.xi_cl is not None:
                CL_I = self.xi_cl
            else:
                CL_I = (self.xi_ucl + self.xi_lcl) / 2

        if self.xi_ucl is not None:
            UCL_I = self.xi_ucl

        if self.xi_lcl is not None:
            LCL_I = self.xi_lcl
        signal_df, message1 = self.data(self.df['value'], CL_I, UCL_I, LCL_I)
        ax1.set_facecolor('white')
        ax1.grid(True, color='gray', alpha=0.3, linestyle='-')
        mask = (signal_df['value'] > self.usl) | (signal_df['value'] < self.lsl)
        signal_df['SIGNAL'] |= mask
        indices = str(signal_df[(signal_df['value'] > self.usl) | (signal_df['value'] < self.lsl)].index.to_list())
        indices = indices.replace("[", "").replace("]", "")

        ax1.plot(self.df['time'], signal_df['value'], marker='o', color="blue", linestyle='-')

        for i in np.where(signal_df["SIGNAL"])[0]:
            ax1.plot(self.df['time'][i], signal_df['value'][i], marker="s", color="r")
            text = str(signal_df['rule_message'][i])
            text = text.replace("[", "").replace("]", "")
            ax1.annotate(text, (self.df['time'][i], signal_df['value'][i]), textcoords="offset points",
                         xytext=(0, 10), ha='center')
        # +2s, +1s的界限指示器。
        if super().limits:
            ax1.plot(self.df['time'], signal_df['+2s'], color="pink", linestyle='dashed', label="+2s")
            ax1.plot(self.df['time'], signal_df['+1s'], color="pink", linestyle='dashed', label="+1s")
            ax1.plot(self.df['time'], signal_df['-2s'], color="pink", linestyle='dashed', label="-2s")
            ax1.plot(self.df['time'], signal_df['-1s'], color="pink", linestyle='dashed', label="-1s")

        # 上下限需要根据传入参数进行解决
        if self.usl is not None and self.lsl is not None:
            if self.sl is not None:
                ax1.axhline(self.sl, color='purple', linestyle='-', label=f'规格中心:{self.sl}')
            else:
                self.sl = (self.usl+self.lsl)/2
                ax1.axhline(self.sl, color='purple', linestyle='-', label=f'规格中心:{self.sl:.4f}')
        if self.usl is not None:
            ax1.axhline(self.usl, color='red', linestyle='-', label=f'规格上限:{self.usl:.4f}')
        if self.lsl is not None:
            ax1.axhline(self.lsl, color='red', linestyle='-', label=f'规格下限:{self.lsl:.4f}')

        ax1.axhline(CL_I, color='green', linestyle='--', label=f'中心限:{CL_I:.4f}')
        ax1.axhline(UCL_I, color='orange', linestyle='--', label=f'上限:{UCL_I:.4f}')
        ax1.axhline(LCL_I, color='orange', linestyle='--', label=f'下限:{LCL_I:.4f}')

        ax1.set_xlabel('时间', fontsize=10, fontweight='bold')
        ax1.set_ylabel('单值', fontsize=10, fontweight='bold')
        ax1.set_xticklabels(self.df['time'], rotation=45, ha='right')
        ax1.set_title("单值控制图", fontsize=10, fontweight='bold')
        ax1.legend(fontsize=8, frameon=False, bbox_to_anchor=(-0.1, 0.8))

        # 移动极差控制图
        CL_R = self.df['rs_mean'][0]
        UCL_R = CL_R * 3.267
        LCL_R = 0
        if self.mrs_ucl is not None and self.mrs_lcl is not None:
            if self.mrs_cl is not None:
                CL_R = self.mrs_cl
            else:
                CL_R = (self.mrs_ucl + self.mrs_lcl) / 2

        if self.mrs_ucl is not None:
            UCL_R = self.mrs_ucl

        if self.mrs_lcl is not None:
            LCL_R = self.mrs_lcl
        # 横坐标变化了  不需要加时间 变成数字就好
        signal_df, message2 = self.data(self.Rs, CL_R, UCL_R, LCL_R)

        X_R = list(range(1, len(self.Rs) + 1))
        # print(X_R)

        ax2.set_facecolor('white')
        ax2.grid(True, color='gray', alpha=0.3, linestyle='-')

        ax2.plot(X_R, signal_df['value'], marker='o', color="blue", linestyle='-')

        for i in np.where(signal_df["SIGNAL"])[0]:
            ax2.plot(X_R[i], signal_df['value'][i], marker="s", color="r")
            text = str(signal_df['rule_message'][i])
            text = text.replace("[", "").replace("]", "")
            ax2.annotate(text, (X_R[i], signal_df['value'][i]), textcoords="offset points",
                         xytext=(0, 10), ha='center')
        # +2s, +1s的界限指示器。
        if super().limits:
            ax2.plot(X_R, signal_df['+2s'], color="pink", linestyle='dashed', label="+2s")
            ax2.plot(X_R, signal_df['+1s'], color="pink", linestyle='dashed', label="+1s")
            ax2.plot(X_R, signal_df['-2s'], color="pink", linestyle='dashed', label="-2s")
            ax2.plot(X_R, signal_df['-1s'], color="pink", linestyle='dashed', label="-1s")

        ax2.axhline(CL_R, color='green', linestyle='--', label=f'中心限:{CL_R:4f}')
        ax2.axhline(UCL_R, color='orange', linestyle='--', label=f'上限:{UCL_R:.4f}')
        ax2.axhline(LCL_R, color='orange', linestyle='--', label=f'下限:{LCL_R:.4f}')

        ax2.set_title("移动极差图", fontsize=10, fontweight='bold')
        ax2.set_ylabel('移动极差', fontsize=10, fontweight='bold')
        ax2.set_xlabel('样本编号', fontsize=10, fontweight='bold')
        ax2.legend(fontsize=8, frameon=False, bbox_to_anchor=(-0.1, 0.6))

        # 调整布局

        message = "单值控制图：\n" + message1 + "\n" + "极差控制图：" + "\n" + message2+ '\n' + "超出规范线的数据：\n" + indices
        return message

    # 绘制散点图
    def draw_Scatter_Group(self, ax):

        # plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
        # plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

        # 对每个group_id进行绘图
        for group, group_data in self.df.groupby('group_id'):
            # 将该组的所有值绘制在同一横坐标上，并设置颜色为相同的颜色
            ax.scatter([group] * len(group_data), group_data['value'], color='gray', alpha=0.5)

        # 添加标签和标题
        ax.set_xlabel('批次')
        ax.set_ylabel('样本值')
        ax.set_title('每组样本散点图')

    # 绘制能力图
    def draw_Capability_Chart(self, ax):
        final_df = self.para_result

        # 提取数据
        group_std = round(final_df['sigma_within'].values[0], 5)
        overall_std = round(final_df['overall_std'].values[0], 5)
        overall_mean = final_df['overall_mean'].values[0]
        group_ppm = final_df['group_ppm'].values[0]
        overall_ppm = final_df['overall_ppm'].values[0]
        cp = final_df['cp'].values[0]
        cpk = final_df['cpk'].values[0]
        pp = final_df['pp'].values[0]
        ppk = final_df['ppk'].values[0]

        # plt.rcParams['font.sans-serif'] = ['SimHei']
        # plt.rcParams['axes.unicode_minus'] = False

        if self.usl is None:
            self.usl = 100
        if self.lsl is None:
            self.lsl = 0

        right_width = max(overall_mean + 3 * overall_std, self.usl, overall_mean + 3 * group_std) + 1
        left_width = min(overall_mean - 3 * overall_std, self.lsl, overall_mean - 3 * group_std) - 1
        fig = ax.figure
        # 获取ax的gridspec位置
        subplot_spec = ax.get_subplotspec()
        # 在ax的位置上创建一个新的3x1的gridspec
        gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=subplot_spec, width_ratios=[3, 3])
        # 删除原有的ax
        fig.delaxes(ax)

        # 创建新的子图
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

        ax1.set_yticks([])
        ax1.set_xticks([])
        ax2.set_yticks([])

        # 整体图
        ax2.hlines(2, overall_mean - 3 * overall_std, overall_mean + 3 * overall_std, color='blue')
        ax2.scatter([overall_mean - 3 * overall_std, overall_mean + 3 * overall_std], [2, 2], color='red')
        ax2.scatter(overall_mean, 2, color='red')
        ax2.set_xlim(left_width, right_width)

        # 组内图
        ax2.hlines(1, overall_mean - 3 * group_std, overall_mean + 3 * group_std, color='blue')
        ax2.scatter([overall_mean - 3 * group_std, overall_mean + 3 * group_std], [1, 1], color='red')
        ax2.scatter(overall_mean, 1, color='red')
        ax2.set_xlim(left_width, right_width)

        # 规格图
        ax2.hlines(0, self.lsl, self.usl, color='green')
        ax2.scatter([self.lsl, self.usl], [0, 0], color='red')
        ax2.scatter((self.lsl + self.usl) / 2, 0, color='red')
        ax2.set_xlim(left_width, right_width)

        # 添加注释文字
        textstr_overall = f'    整体\n标准差  {overall_std:.5f}\nPp      {self.format_value(pp)}\nPpk     {self.format_value(ppk)}\nPPM     {self.format_value(overall_ppm)}'
        ax2.text(1.5, 0.8, textstr_overall, fontsize=13, verticalalignment='center', transform=ax2.transAxes)

        textstr_within = f'    组内\n标准差  {group_std:.5f}\nCp      {self.format_value(cp)}\nCpk     {self.format_value(cpk)}\nPPM     {self.format_value(group_ppm)}'
        ax1.text(0, 0.8, textstr_within, fontsize=13, verticalalignment='center', transform=ax1.transAxes)

        ax2.text((self.lsl + self.usl) / 2, 0.1, '规格', fontsize=10, verticalalignment='bottom')
        ax2.text(overall_mean, 2.1, '整体', fontsize=10, verticalalignment='bottom')
        ax2.text(overall_mean, 1.1, '组内', fontsize=10, verticalalignment='bottom')
        ax2.text(0.5, 2.5, '能力图', fontsize=15, verticalalignment='bottom')

        for spine in ax1.spines.values():
            spine.set_visible(False)
        for spine in ax2.spines.values():
            spine.set_visible(False)

    # 绘制正态概率图
    def draw_normal_probability_plot(self, ax):

        """
        模拟Minitab的正态概率图，包括中线和曲线置信区间。

        参数：
        data: 输入样本数据
        confidence: 置信水平，默认95%
        """
        data = self.df['value']
        mean = np.mean(data)
        std = np.std(data)

        # 排序数据
        sorted_data = np.sort(data)
        n = len(sorted_data)
        _, p_value = shapiro(data)
        # 近视中位秩工序
        percentiles = (np.arange(1, n + 1) - 0.3) / (n + 0.4)
        # Hansen公式：     percentiles = (np.arange(1, n + 1) - 0.5) / (n)
        # 数学期望公式：     percentiles = (np.arange(1, n + 1)) / (n + 1)
        # 效果可以选择对应的公式来计算经验累积分布函数
        # 计算理论分位数（标准正态分布）
        theoretical_quantiles = norm.ppf(percentiles)
        theoretical_values = mean + theoretical_quantiles * std
        # 数据点，横坐标是实际数据，纵坐标是理论值
        ax.scatter(sorted_data, theoretical_values, label='Data Points', color='blue', alpha=0.7)

        # 设置y轴为正态概率分布，将理论值转换为累积概率值
        # percent_labels = [0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 91, 92, 93, 94, 95,
        #                   96, 97, 98, 99, 99.9]  # 累积概率
        # z_labels = norm.ppf(np.array(percent_labels) / 100)  # 累积概率对应的z值
        # value_labels = mean + z_labels * std  # z值通过样本均值和标准差转换为数据值
        # ax.set_yticks(value_labels)  # 累积概率映射到数据值

        # 绘制置信区间：先用理论值和实际值的映射关系训练线性回归模型，再绘制回归曲线及其直线区间曲线
        df_data = pd.DataFrame({'x': theoretical_values, 'y': sorted_data})
        X = sm.add_constant(df_data['x'])
        model = sm.OLS(df_data['y'], X).fit()

        # 绘制置信区间带
        range_data = max(sorted_data) - min(sorted_data)
        x_line_min = min(sorted_data) - range_data * 0.6
        x_line_max = max(sorted_data) + range_data * 0.6
        x_line = np.linspace(x_line_min, x_line_max, 500)
        X_vals = sm.add_constant(x_line)
        preds = model.get_prediction(X_vals)
        conf_int_vals = preds.conf_int(alpha=1 - 0.95)  # 置信区间曲线

        # 散点图的拟合曲线和置信区间
        ax.plot(x_line, model.predict(X_vals), color='red')  # 绘制回归曲线
        ax.plot(x_line, conf_int_vals[:, 0], color='red', )  # 绘制置信区间曲线
        ax.plot(x_line, conf_int_vals[:, 1], color='red', )

        # Anderson-Darling 正态性检验
        ad_statistic, critical_values, significance_level = anderson(data, dist='norm')

        ax.set_title(f"正态概率图 \n AD: {ad_statistic:.4f}   P:{p_value:.4f}")
        ax.set_yticks([])
        # 计算正态相关性系数 p 值
        alpha = 0.05
        # 绘制正态概率图
        ax.grid(True)

        # 添加文本到图的右侧
        text_x = 1.05  # 确保文本在图的右侧，这里假设x轴的最大值为1
        ax.text(text_x, 0.9, '异常模式5：连续3点中有2点落在同一侧B区以外', fontsize=13, verticalalignment='center',
                transform=ax.transAxes)
        ax.text(text_x, 0.7, '异常模式6：连续5点中4点落在同一侧C区以外', fontsize=13, verticalalignment='center',
                transform=ax.transAxes)
        ax.text(text_x, 0.5, '异常模式7：连续15点落在C区域', fontsize=13, verticalalignment='center',
                transform=ax.transAxes)
        ax.text(text_x, 0.3, '异常模式8：连续8点两侧有点且无一在C区域', fontsize=13, verticalalignment='center',
                transform=ax.transAxes)
        # text_x = 1.3
        # ax.text(text_x, 0.15, f'正态相关性系数 p = {p_value:.4f}', fontsize=11, horizontalalignment='center',
        #         transform=ax.transAxes)
        # ax.text(text_x, 0.05, f'非正态界限 α = {alpha}', fontsize=11, horizontalalignment='center', transform=ax.transAxes)

        return p_value >= alpha

    # 绘制质量指标分析表
    def plot_quality_indicators(self):
        # 选择特定的 category 进行筛选

        # 创建一个新的DataFrame来存储结果
        results = {
            '参数': [],
            '目标值': [],
            '实际值': [],
            '分析结果': [],
            '颜色': []  # 新增颜色列
        }
        target_values = {
            'assess_within_sigma': 4,
            'assess_overall_std': 4,
            'assess_within_offset': 1.5,
            'assess_overall_offset': 1.5,
            'ppk/cpk': 1.0,
            'k_value': 0
        }
        param_names = {
            'assess_within_sigma': '组内（Tu-Tl）/σ 小于目标值异常',
            'assess_overall_std': '整体（Tu-Tl）/σ 小于目标值异常',
            'assess_within_offset': '组内|T0-u|/σ 大于目标值异常',
            'assess_overall_offset': '整体|T0-u|/σ 大于目标值异常',
            'ppk/cpk': 'PPK/CPK比值 接近1.0正常',
            'k_value': 'K值 1-Cpk/Cp 接近0最佳'
        }
        if self.usl is not None and self.lsl is not None:
            target_values['assess_within_sigma'] = 10.5
            target_values['assess_overall_std'] = 10.5

        elif self.usl is not None:
            param_names['assess_within_sigma'] = '组内（Tu-u）/σ 小于目标值异常'
            param_names['assess_overall_std'] = '整体（Tu-u）/σ 小于目标值异常'
        elif self.lsl is not None:
            param_names['assess_within_sigma'] = '组内（u-Tl）/σ 小于目标值异常'
            param_names['assess_overall_std'] = '整体（u-Tl）/σ 小于目标值异常'

        # 定义分析结果的判定函数
        def analyze_result(target, actual, param):
            if param == 'assess_within_sigma':
                if actual >= target:
                    return '组内标准偏差正常', 'green'
                else:
                    return '组内标准偏差偏大!', 'red'
            elif param == 'assess_overall_std':
                if actual >= target:
                    return '整体标准偏差正常', 'green'
                else:
                    return '整体标准偏差偏大!', 'red'
            elif param == 'assess_within_offset':
                if actual <= target:
                    return '组内均值与规范中心距离正常', 'green'
                else:
                    return '组内均值与规范中心偏离过大!', 'red'
            elif param == 'assess_overall_offset':
                if actual <= target:
                    return '整体均值与规范中心距离正常', 'green'
                else:
                    return '整体均值与规范中心偏离过大!', 'red'
            elif param == 'ppk/cpk':
                if abs(actual - target) <= 0.1:
                    return '受随机误差波动影响小，过程正常', 'green'
                else:
                    return '受特殊过程的影响很大!需对过程进行改进', 'red'
            elif param == 'k_value':
                if actual == '*':
                    return '只有单侧规范，不计算K值', 'black'
                else:
                    return '分布均值对规范中心的相对偏离度', 'black'

        # 遍历每个参数
        for param in ['assess_within_sigma', 'assess_overall_std', 'assess_within_offset', 'assess_overall_offset',
                      'ppk/cpk', 'k_value']:
            actual_value = self.para_result.loc[0, param]
            target_value = target_values.get(param, None)

            if target_value is not None:
                analysis_result, color = analyze_result(target_value, actual_value, param)
            else:
                analysis_result = 'No Target'
                color = 'black'

            results['参数'].append(param_names[param])
            results['目标值'].append(target_value)
            results['实际值'].append(actual_value)
            results['分析结果'].append(analysis_result)
            results['颜色'].append(color)

        # 将结果转换为DataFrame
        results_df = pd.DataFrame(results)

        return results_df

    def draw_six(self):
        time_01 = datetime.now()
        plt.close('all')  # 关闭所有打开的图形
        fig = Figure(figsize=(15, 12))  # 调整高度以容纳新的子图
        if self.is_box:
            fig.suptitle(f" {self._title_name}的Process Capability Sixpack报告 (Use Box-Cox Trans) ", fontsize=30)
        else:
            fig.suptitle(f" {self._title_name}的Process Capability Sixpack报告", fontsize=30)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(0.95, 0.98, current_time,
                 fontsize=20,
                 color='pink',
                 alpha=0.8,
                 rotation=30,
                 ha='right', va='top')
        ax1 = fig.add_subplot(4, 2, 1)  # 在4x2网格中的位置1
        ax2 = fig.add_subplot(4, 2, 2)  # 在4x2网格中的位置2
        ax3 = fig.add_subplot(4, 2, 3)  # 在4x2网格中的位置3
        ax4 = fig.add_subplot(4, 2, 4)  # 在4x2网格中的位置4
        ax5 = fig.add_subplot(4, 2, 5)  # 在4x2网格中的位置5
        ax6 = fig.add_subplot(4, 2, 6)  # 在4x2网格中的位置6
        ax7 = fig.add_subplot(4, 1, 4)  # 在4x1网格中的位置4，用于绘制质量指标分析表

        self.draw_Scatter_Group(ax5)
        self.draw_NDC(ax2)
        self.is_normal = self.draw_normal_probability_plot(ax4)
        self.draw_Capability_Chart(ax6)
        if self.n == 1:  # 画imr
            message = self.draw_imrchart(ax1, ax3)
        else:
            message = self.draw_MSD_and_RChart(ax1, ax3)

        results_df = self.plot_quality_indicators()

        column_names = ['参数', '目标值', '实际值', '分析结果']
        ax7.axis('off')
        table = ax7.table(cellText=results_df[column_names].values, colLabels=column_names, cellLoc='center',
                          loc='center')
        # 设置行高和列宽
        row_height = 0.13  # 调整行高
        col_widths = [0.4, 0.2, 0.2, 0.4]  # 调整列宽
        for (row, col), cell in table.get_celld().items():
            if row == 0:  # 表头行
                cell.set_height(row_height * 1.5)  # 表头行高度可以稍微大一些
            else:
                cell.set_height(row_height)
            cell.set_width(col_widths[col])
            if row > 0 and col == 2 and row < 6:  # 实际值列
                cell.get_text().set_color(results_df.iloc[row - 1]['颜色'])
        ax7.set_title('质量指标分析')
        text_x = 1.25  # 确保文本在图的右侧，这里假设x轴的最大值为1
        ax7.text(text_x, 0.9, f'Tu:上规范', fontsize=12, horizontalalignment='center', transform=ax7.transAxes)
        ax7.text(text_x, 0.75, f'Tl:下规范', fontsize=12, horizontalalignment='center', transform=ax7.transAxes)
        ax7.text(text_x, 0.6, f'T0:规范中心', fontsize=12, horizontalalignment='center', transform=ax7.transAxes)
        ax7.text(text_x, 0.45, f'σ:标准偏差', fontsize=12, horizontalalignment='center', transform=ax7.transAxes)
        ax7.text(text_x, 0.3, f'u:均值', fontsize=12, horizontalalignment='center', transform=ax7.transAxes)
        ax7.text(text_x, 0.15, f'k值:相对偏离度', fontsize=12, horizontalalignment='center', transform=ax7.transAxes)
        ax7.text(text_x, 0.05, f'*:不可计算', fontsize=12, horizontalalignment='center', transform=ax7.transAxes)

        fig.tight_layout()
        fig.savefig('b.png', dpi=300)

        buf = io.BytesIO()
        # fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        # 在SPC模块中找到图表生成代码（假设使用matplotlib），通常类似：
        # fig.savefig(buf, 'png', dpi=100)
        # 修改为：
        fig.savefig(buf, format='jpg', dpi=100,
                    bbox_inches='tight', facecolor='white')

        plt.close('all')
        buf.seek(0)
        print(f" generating SixPackChart  {datetime.now() - time_01} seconds.")

        return buf, message

    def plot(self):
        """主方法，负责调用其他方法

        """
        # 分组
        self.subgroup_data()
        # print(self.df.dtypes)
        if self.n == 1:
            self.df['rs_mean'] = self.rs_mean
        # print(self.df)
        # 计算分组的一些参数信息
        self.process_grouped_data()
        # print(self.para_result.dtypes)
        # print(self.para_result)
        # 画图
        buf, message = self.draw_six()
        return buf, message

    def data(self, value, CL, UCL, LCL):
        """ 返回数据。


        """

        rule_json = {
            1: "检验1: 位于控制限以外。\n 检验出下列点不合格",
            2: "检验2: 连续9点落在中心线的同一侧。\n 检验出下列点不合格",
            3: "检验3 连续6点递增或递减。\n 检验出下列点不合格",
            4: "检验4 连续14点中相邻点交替上下。\n 检验出下列点不合格",
            5: "检验5 连续3点中有两点落在中心线同一侧的B区以外。\n 检验出下列点不合格",
            6: "检验6 连续5点中有4点落在中心线同一侧的C区以外。\n 检验出下列点不合格",
            7: "检验7 连续15点落在中心线两侧的C区内。\n 检验出下列点不合格",
            8: "检验8 连续8点落在中心线两侧且五一点在C区内。\n 检验出下列点不合格"
        }  # 记录检验的异常信息
        message = {
            1: "",
            2: "",
            3: "",
            4: "",
            5: "",
            6: "",
            7: "",
            8: ""

        }
        signal_df = None
        try:
            # 清空字典:

            one_sigma_plus = [(UCL - CL) / 3 + CL] * len(value)
            two_sigma_plus = [2 * (UCL - CL) / 3 + CL] * len(value)

            one_sigma_min = [CL - (CL - LCL) / 3] * len(value)
            two_sigma_min = [CL - 2 * (CL - LCL) / 3] * len(value)

            LCL = [LCL] * len(value)
            UCL = [UCL] * len(value)
            CL = [CL] * len(value)

            signal_df = pd.DataFrame(np.column_stack(
                [value, UCL, two_sigma_plus, one_sigma_plus, CL,
                 one_sigma_min, two_sigma_min, LCL]),
                columns=['value', 'UCL', '+2s', '+1s', 'CL', '-1s', '-2s', 'LCL'])
            signal_df['value'] = pd.to_numeric(signal_df['value'], errors='coerce')
            signal_df['UCL'] = pd.to_numeric(signal_df['UCL'], errors='coerce')
            signal_df['LCL'] = pd.to_numeric(signal_df['LCL'], errors='coerce')
            signal_df['CL'] = pd.to_numeric(signal_df['CL'], errors='coerce')
            signal_df['+1s'] = pd.to_numeric(signal_df['+1s'], errors='coerce')
            signal_df['+2s'] = pd.to_numeric(signal_df['+2s'], errors='coerce')
            signal_df['-1s'] = pd.to_numeric(signal_df['-1s'], errors='coerce')
            signal_df['-2s'] = pd.to_numeric(signal_df['-2s'], errors='coerce')
            self.execute_rules(signal_df)

            # print(signal_df)
            if any(signal_df['SIGNAL']):
                self.is_stable = False
            for i in range(len(signal_df)):
                for j in signal_df['rule_message'][i]:
                    if j:
                        if message[j]:
                            message[j] += ',' + str(i)
                        else:
                            message[j] = str(i)

            # 获取两个字典中共同的键
            common_keys = set(rule_json.keys()) & set(message.keys())

            # 格式化输出
            formatted_output = []

            for key in common_keys:
                # 格式化字符串并添加到列表中
                if message[key] != "":
                    formatted_output.append(f"{rule_json[key]} : {message[key]}")

            # 将格式化后的字符串列表用换行符连接起来
            message = "\n".join(formatted_output)

            # print(self._rules)
            return signal_df, message
        except Exception as e:
            raise e

    def stable(self):
        """ 返回稳定指示器。
        """
        # 执行规则。
        if self.is_stable:
            return "0"
        else:
            return "1"


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
        json1 = {
            "data": [
            {
              "edit_time": "2025-03-09 13:14:55",
              "data": 35.86
            },
            {
              "edit_time": "2025-03-09 13:49:03",
              "data": 35.63
            },
            {
              "edit_time": "2025-03-09 13:55:12",
              "data": 35.39
            },
            {
              "edit_time": "2025-03-09 13:58:10",
              "data": 35.78
            },
            {
              "edit_time": "2025-03-09 13:58:10",
              "data": 34.53
            },
            {
              "edit_time": "2025-03-09 14:04:29",
              "data": 38.44
            },
            {
              "edit_time": "2025-03-09 14:10:30",
              "data": 36.17
            },
            {
              "edit_time": "2025-03-09 14:10:30",
              "data": 36.17
            },
            {
              "edit_time": "2025-03-09 14:13:30",
              "data": 35.08
            },
            {
              "edit_time": "2025-03-09 14:16:35",
              "data": 36.48
            },
            {
              "edit_time": "2025-03-09 14:19:40",
              "data": 35.55
            },
            {
              "edit_time": "2025-03-09 14:22:45",
              "data": 37.42
            },
            {
              "edit_time": "2025-03-09 14:25:53",
              "data": 36.09
            },
            {
              "edit_time": "2025-03-09 14:28:58",
              "data": 35.7
            },
            {
              "edit_time": "2025-03-09 14:28:58",
              "data": 35.7
            },
            {
              "edit_time": "2025-03-09 14:32:02",
              "data": 36.72
            },
            {
              "edit_time": "2025-03-09 14:35:14",
              "data": 38.05
            },
            {
              "edit_time": "2025-03-09 15:18:26",
              "data": 35.31
            },
            {
              "edit_time": "2025-03-09 15:21:30",
              "data": 35.23
            },
            {
              "edit_time": "2025-03-09 15:21:30",
              "data": 35.7
            }

          ],
            "test": [
                1

            ],
            "chart_title": "f222",

            "uspec": 10,
            "group_size":10
        }
        data1 = json1.get('data')

        data_list = [item.get('data') for item in data1]
        time_list = [item.get('edit_time') for item in data1]

        print(data_list)
        sorted_data = np.sort(data_list)
        n = len(sorted_data)
        _, p_value = shapiro(data_list)
        # 近视中位秩工序
        percentiles = (np.arange(1, n + 1) - 0.3) / (n + 0.4)
        for i in range(len(percentiles)):
            print(percentiles[i])


        analysis = Quan_SixChart(data_list=data_list
            , edit_time_list=time_list
            , lsl=0,usl=38, groupSize=10, title_name="value")
        # analysis.limits = True
        analysis.append_rules([Rule01(), Rule02(), Rule03(), Rule04(), Rule05(), Rule06(), Rule07(), Rule08()])
        analysis.plot()
        input()