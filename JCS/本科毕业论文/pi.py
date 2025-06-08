import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 创建结果保存目录
if not os.path.exists('results'):
    os.makedirs('results')

# 1. 从Excel文件读取数据
file_path = 'DACNEW.xlsx'

# 读取环境因子表格
env_df = pd.read_excel(file_path, sheet_name='解释变量表格（环境因子）', index_col=0)
print("环境因子表格:")
print(env_df.head())

# 读取物种丰度表格
species_df = pd.read_excel(file_path, sheet_name='响应变量表格（丰度）', index_col=0)
print("\n物种丰度表格:")
print(species_df.head())

# 2. 数据预处理
# 处理环境因子中的无效0值
zero_cols = ['T', 'pH', 'DO', 'Z']
env_df[zero_cols] = env_df[zero_cols].replace(0, np.nan)

# 合并两个表格（按时间索引）
merged_df = env_df.merge(species_df, left_index=True, right_index=True, how='inner')

# 3. 相关性分析
env_columns = env_df.columns.tolist()
species_columns = species_df.columns.tolist()

# 存储结果的数据结构
results = []
corr_matrix = pd.DataFrame(index=env_columns, columns=species_columns)
pvalue_matrix = pd.DataFrame(index=env_columns, columns=species_columns)

for env in env_columns:
    env_pvalues = []

    for species in species_columns:
        # 提取配对数据并删除缺失值
        paired = merged_df[[env, species]].dropna()
        if len(paired) < 3:  # 至少需要3个样本
            corr = np.nan
            p_value = np.nan
        else:
            # 计算皮尔逊相关性和p值
            corr, p_value = stats.pearsonr(paired[env], paired[species])

        # 存储结果
        corr_matrix.loc[env, species] = corr
        pvalue_matrix.loc[env, species] = p_value
        env_pvalues.append(p_value)

    # 计算该环境因子的平均p值（忽略NaN）
    valid_pvalues = [p for p in env_pvalues if not np.isnan(p)]
    if valid_pvalues:
        avg_p = np.mean(valid_pvalues)
        median_p = np.median(valid_pvalues)
        min_p = min(valid_pvalues)
        max_p = max(valid_pvalues)
    else:
        avg_p = median_p = min_p = max_p = np.nan

    results.append({
        '环境因子': env,
        '平均p值': avg_p,
        '中位数p值': median_p,
        '最小p值': min_p,
        '最大p值': max_p
    })

# 转换为DataFrame并排序
results_df = pd.DataFrame(results)
sorted_results = results_df.sort_values(by='平均p值', ascending=False)

# 4. 保存分析结果
# 保存排序后的环境因子p值结果
sorted_results.to_csv('results/环境因子平均p值排序.csv', index=False, encoding='utf-8-sig')

# 保存完整的相关系数矩阵
corr_matrix.to_csv('results/环境因子-物种相关系数矩阵.csv', encoding='utf-8-sig')

# 保存完整的p值矩阵
pvalue_matrix.to_csv('results/环境因子-物种p值矩阵.csv', encoding='utf-8-sig')

print("\n平均p值最大的前5个环境因子:")
print(sorted_results.head(5))

# 5. 数据可视化
# 5.1 绘制环境因子平均p值排序图
plt.figure(figsize=(12, 8))
sns.barplot(x='平均p值', y='环境因子', data=sorted_results, palette='viridis')
plt.title('环境因子与物种丰度相关性的平均p值排序')
plt.xlabel('平均p值')
plt.ylabel('环境因子')
plt.tight_layout()
plt.savefig('results/环境因子平均p值排序.png', dpi=300)
plt.close()

# 5.2 绘制热图 - 相关系数矩阵
plt.figure(figsize=(18, 12))
sns.heatmap(corr_matrix.astype(float), annot=True, fmt=".2f", cmap='coolwarm',
            center=0, vmin=-1, vmax=1, linewidths=0.5)
plt.title('环境因子与物种丰度的相关系数热图')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('results/环境因子-物种相关系数热图.png', dpi=300)
plt.close()

# 5.3 绘制热图 - p值矩阵
plt.figure(figsize=(18, 12))
sns.heatmap(pvalue_matrix.astype(float), annot=True, fmt=".2f", cmap='viridis_r',
            vmin=0, vmax=0.2, linewidths=0.5)
plt.title('环境因子与物种丰度相关性的p值热图')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('results/环境因子-物种p值热图.png', dpi=300)
plt.close()

# 5.4 绘制平均p值最大的5个环境因子与物种的相关性
top_envs = sorted_results.head(5)['环境因子'].tolist()
top_corr = corr_matrix.loc[top_envs]

plt.figure(figsize=(15, 8))
sns.heatmap(top_corr.astype(float), annot=True, fmt=".2f", cmap='coolwarm',
            center=0, vmin=-1, vmax=1, linewidths=0.5)
plt.title('平均p值最大的5个环境因子与物种丰度的相关性')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('results/高p值环境因子-物种相关性.png', dpi=300)
plt.close()

print("\n分析完成! 结果已保存到 'results' 目录中。")