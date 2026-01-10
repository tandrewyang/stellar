# HLSMAC地图优化算法实验报告

本目录包含了HLSMAC地图优化算法的完整实验报告和相关数据。

## 文件说明

### 主要报告
- **`optimization_report.md`**: 完整的实验报告，包括：
  - 第一部分：优化算法详细介绍
  - 第二部分：每个地图在不同优化算法中的最佳表现表格
  - 第三部分：各地图优化设计和提升原因分析
  - 第四部分：胜率曲线图说明
  - 总结

### 数据文件
- **`training_results_data.json`**: 所有地图在所有算法下的训练结果数据（JSON格式）
- **`*_win_rate_history.json`**: 各个地图的胜率历史数据，可用于绘制曲线图
  - `adcc_RewardShaping_win_rate_history.json`: ADCC地图使用RewardShaping算法的胜率历史
  - `gmzz_HierarchicalArchitecture_win_rate_history.json`: GMZZ地图使用HierarchicalArchitecture算法的胜率历史
  - `jctq_TargetedOptimization_win_rate_history.json`: JCTQ地图使用TargetedOptimization算法的胜率历史
  - `jdsr_HierarchicalArchitecture_win_rate_history.json`: JDSR地图使用HierarchicalArchitecture算法的胜率历史
  - `sdjx_CurriculumLearning_win_rate_history.json`: SDJX地图使用CurriculumLearning算法的胜率历史
  - `wwjz_HierarchicalArchitecture_win_rate_history.json`: WWJZ地图使用HierarchicalArchitecture算法的胜率历史

### 辅助文件
- **`win_rate_curves_data.md`**: 胜率历史数据的表格形式（用于快速查看）
- **`generate_plots.py`**: 生成胜率曲线图的Python脚本（需要matplotlib）

## 使用方法

### 查看报告
直接打开 `optimization_report.md` 文件查看完整报告。

### 绘制胜率曲线图

#### 方法1：使用英文版本（推荐，避免中文显示问题）
```bash
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC/report
python3 generate_plots.py
```
这个版本使用英文标签，避免中文字体显示问题。

**新功能**: 图表现在会同时显示Baseline（dTAPE）和最佳算法的对比：
- Baseline使用灰色虚线表示
- 最佳算法使用彩色实线表示
- 可以在同一张图上直观地看到性能提升

#### 方法2：使用中文版本（需要中文字体）
```bash
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC/report
python3 generate_plots_chinese.py
```
如果中文显示为方块（□），请参考 `FONT_FIX.md` 安装中文字体。

**新功能**: 图表现在会同时显示Baseline（dTAPE）和最佳算法的对比：
- Baseline使用灰色虚线表示
- 最佳算法使用彩色实线表示
- 可以在同一张图上直观地看到性能提升

**需要先安装matplotlib：**
```bash
pip install matplotlib
```

**如果中文显示为方块，安装中文字体：**
```bash
# Ubuntu/Debian
sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei

# CentOS/RHEL
sudo yum install wqy-microhei-fonts wqy-zenhei-fonts
```

#### 方法2：使用Excel或其他工具
1. 打开对应的 `*_win_rate_history.json` 文件
2. 提取 `t_env_history` 和 `win_rate_history` 数据
3. 将训练步数转换为训练进度百分比：`progress = t_env / 2005000 * 100`
4. 使用Excel或其他绘图工具绘制曲线图

#### 方法3：查看表格数据
打开 `win_rate_curves_data.md` 文件查看表格形式的胜率历史数据。

## 报告结构

### 第一部分：优化算法详细介绍
详细介绍了7种优化算法：
1. dTAPE (Baseline) - 基线算法
2. TransformerMixer - Transformer架构的Mixer网络
3. HierarchicalArchitecture - 分层架构
4. EnhancedStateRepresentation - 增强状态表示
5. CurriculumLearning - 课程学习
6. RewardShaping - 奖励塑形
7. TargetedOptimization - 地图特定优化

### 第二部分：最佳表现表格
展示了每个地图在不同优化算法中的最佳表现，包括：
- Baseline胜率
- 最佳算法
- 最佳胜率
- 提升幅度

### 第三部分：详细分析
针对每个地图，详细分析了：
- 优化设计
- 提升原因
- 为什么该算法在该地图上表现最好

### 第四部分：胜率曲线图
说明了如何生成和使用胜率曲线图。

### 总结
总结了主要发现和优化建议。

## 主要发现

1. **TargetedOptimization在困难地图上表现突出**: JCTQ地图从0%提升到93.75%
2. **RewardShaping在奖励敏感地图上效果显著**: ADCC地图从0%提升到100%
3. **HierarchicalArchitecture适合需要战略思考的地图**: JDSR和WWJZ地图都达到了100%胜率
4. **CurriculumLearning在复杂战术地图上有效**: 多个地图都达到了100%胜率
5. **不同优化算法适用于不同类型的地图**: 需要根据地图特点选择合适的优化方法

## 数据格式

### training_results_data.json
```json
{
  "map_name": {
    "algorithm_name": {
      "test_win_rate": 0.9375,
      "test_reward": 10.5,
      "completion": 99.89,
      "t_env": 2004850
    }
  }
}
```

### *_win_rate_history.json
```json
{
  "map_name": "jctq",
  "algorithm": "TargetedOptimization",
  "win_rate_history": [0.0, 0.1, 0.2, ...],
  "t_env_history": [10000, 20000, 30000, ...]
}
```

## 注意事项

1. 所有胜率数据都是测试胜率（test_battle_won_mean）
2. 训练进度基于总训练步数2,005,000计算
3. 部分地图的某些算法可能没有完成训练，因此数据可能缺失
4. 图表生成需要matplotlib库，如果不可用，可以使用其他工具根据JSON数据绘制
5. **中文显示问题**: 如果图表中中文显示为方块（□），请：
   - 使用英文版本：`generate_plots.py`
   - 或参考 `FONT_FIX.md` 安装中文字体后使用 `generate_plots_chinese.py`

## 更新日志

- 2025-12-13: 初始报告生成

