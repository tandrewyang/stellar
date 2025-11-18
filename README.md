# HLSMAC 多智能体强化学习训练框架 

基于dTAPE算法在HLSMAC（High-Level Strategic Multi-Agent Challenge）12个地图上的训练和评测框架。

## 项目结构

```
StarCraft2_HLSMAC/
├── RLalgs/                    # 算法实现
│   └── dTAPE/                 # dTAPE算法实现
│       └── src/
│           ├── config/         # 配置文件
│           │   ├── algs/      # 算法配置
│           │   │   ├── d_tape.yaml          # 原始dTAPE配置
│           │   │   └── d_tape_improved.yaml # 优化版本配置
│           │   └── envs/      # 环境配置
│           │       └── sc2te.yaml  # HLSMAC环境配置
│           └── ...
├── smac/                      # SMAC环境代码
│   └── smac/
│       └── env/
│           └── sc2_tactics/   # HLSMAC环境实现
├── Tactics_Maps/              # HLSMAC地图文件
│   └── HLSMAC_Maps/
├── train_hlsmac.sh            # 训练所有12个地图
├── train_single_map.sh        # 训练单个地图
├── evaluate_hlsmac.sh         # 评测单个地图模型
├── evaluate_all_maps.sh       # 评测所有地图模型
└── README.md                  # 本文档
```

## HLSMAC 12个地图

| 地图ID | 中文名称 | 英文名称 | 地图文件 |
|--------|---------|---------|---------|
| adcc | 暗度陈仓 | Advancing Secretly by an Unknown Path | adcc_te.SC2Map |
| dhls | 调虎离山 | Luring the Tiger Out of His Den | dhls_te.SC2Map |
| fkwz | 反客为主 | Turning from the Guest into the Host | fkwz_te.SC2Map |
| gmzz | 关门捉贼 | Catching the Thief by Closing His Escape Route | gmzz_te.SC2Map |
| jctq | 金蝉脱壳 | Slipping Away by Casting Off a Cloak | jctq_te.SC2Map |
| jdsr | 借刀杀人 | Killing Someone with a Borrowed Knife | jdsr_te.SC2Map |
| sdjx | 声东击西 | Making a Feint to the East and Attacking in the West | sdjx_te.SC2Map |
| swct | 上屋抽梯 | Removing the Ladder After the Enemy Has Climbed Up | swct_te.SC2Map |
| tlhz | 偷梁换柱 | Stealing the Beams and Pillars | tlhz_te.SC2Map |
| wwjz | 围魏救赵 | Relieving the State of Zhao by Besieging the State of Wei | wwjz_te.SC2Map |
| wzsy | 无中生有 | Creating Something Out of Nothing | wzsy_te.SC2Map |
| yqgz | 欲擒故纵 | Letting the Enemy Off in Order to Catch Him | yqgz_te.SC2Map |

## 环境配置

### 1. 安装StarCraft II

**Linux平台：**
```bash
# 下载SC2 4.10版本（约1.6GB）
cd /share/project/ytz
wget --continue "http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip" -O SC2.4.10.zip

# 解压（密码: iagreetotheeula）
unzip -P iagreetotheeula SC2.4.10.zip

# 设置环境变量（已确认安装路径）
export SC2PATH="/share/project/ytz/StarCraftII"
echo 'export SC2PATH="/share/project/ytz/StarCraftII"' >> ~/.bashrc
```

**Windows平台：**
- 从战网客户端下载安装星际争霸II（版本5.0.15）
- 下载链接：https://download.battlenet.com.cn/zh-cn/?product=bnetdesk

### 2. 安装Python依赖

```bash
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/dTAPE
pip install -r requirements.txt  # 如果有requirements.txt
# 或手动安装：
pip install torch sacred pyyaml numpy pysc2
```

### 3. 配置地图路径

确保HLSMAC地图文件在正确位置：
```bash
# 地图应该在SC2PATH/Maps/目录下
cp -r Tactics_Maps/HLSMAC_Maps/* $SC2PATH/Maps/
```

### 4. 设置Python路径

```bash
# 添加项目路径到PYTHONPATH
export PYTHONPATH="/share/project/ytz/RLproject/StarCraft2_HLSMAC:$PYTHONPATH"
export PYTHONPATH="/share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/dTAPE/src:$PYTHONPATH"
export PYTHONPATH="/share/project/ytz/RLproject/StarCraft2_HLSMAC/smac:$PYTHONPATH"
```

## 使用方法

### 训练

#### 训练单个地图

```bash
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC

# 训练指定地图（使用原始dTAPE算法）
bash train_single_map.sh adcc 0 42

# 训练指定地图（使用优化版本）
ALG_CONFIG=d_tape_improved bash train_single_map.sh adcc 0 42

# 参数说明：
# - 第一个参数：地图名称（adcc, dhls, fkwz等）
# - 第二个参数：GPU ID（默认0）
# - 第三个参数：随机种子（默认42）
```

#### 训练所有12个地图

```bash
# 使用原始dTAPE算法
bash train_hlsmac.sh

# 使用优化版本
ALG_CONFIG=d_tape_improved bash train_hlsmac.sh

# 自定义参数
GPU_ID=0 SEED=42 T_MAX=2005000 ALG_CONFIG=d_tape_improved bash train_hlsmac.sh
```

### 评测

#### 评测单个地图模型

```bash
# 评测指定地图的模型
bash evaluate_hlsmac.sh adcc ../../results/adcc_d_tape/models/episode_2000000.pt 0

# 参数说明：
# - 第一个参数：地图名称
# - 第二个参数：模型checkpoint路径
# - 第三个参数：GPU ID（可选，默认0）
```

#### 评测所有地图模型

```bash
# 评测所有地图的最新模型
bash evaluate_all_maps.sh

# 指定结果目录和算法配置
RESULTS_DIR=../../results ALG_CONFIG=d_tape_improved bash evaluate_all_maps.sh

# 评测指定训练步数的模型
LOAD_STEP=2000000 bash evaluate_all_maps.sh
```

### 查看训练日志

```bash
# 查看TensorBoard日志
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC
tensorboard --logdir=results/train_logs

# 查看文本日志
tail -f results/train_logs/adcc_d_tape/train.log
```

## 算法优化说明

### 原始dTAPE算法

- **配置文件**: `RLalgs/dTAPE/src/config/algs/d_tape.yaml`
- **特点**: 
  - 使用QMIX mixer
  - 信息瓶颈通信机制
  - OW-QMIX损失函数

### 优化版本 (d_tape_improved)

**配置文件**: `RLalgs/dTAPE/src/config/algs/d_tape_improved.yaml`

**主要改进点：**

1. **增强的Mixer网络**
   - `mixing_embed_dim`: 32 → 64（增强表达能力）
   - `hypernet_layers`: 2 → 3（更深的网络）
   - `hypernet_embed`: 64 → 128（更强的特征提取）

2. **改进的RNN隐藏状态**
   - `central_rnn_hidden_dim`: 64 → 128
   - 更强的状态表示能力

3. **优化的学习策略**
   - `lr`: 0.001 → 0.0008（更稳定的训练）
   - `td_lambda`: 0.6 → 0.7（更重视长期回报）
   - `epsilon_anneal_time`: 100000 → 150000（增加探索时间）

4. **损失函数平衡**
   - `w`: 0.5 → 0.6（平衡central和qmix损失）
   - `central_mixing_embed_dim`: 256 → 512

5. **通信机制增强**
   - `comm_embed_dim`: 3 → 4（增加通信维度）

## 结果文件结构

训练完成后，结果文件结构如下：

```
results/
├── adcc_d_tape_seed42/
│   ├── models/
│   │   ├── episode_500000.pt
│   │   ├── episode_1000000.pt
│   │   └── episode_2000000.pt
│   ├── logs/
│   └── sacred/
├── train_logs/
│   └── adcc_d_tape/
│       └── train.log
└── evaluation_20250118_120000/
    ├── adcc_eval.log
    └── ...
```

## 模型加载和评测

### Python脚本示例

```python
import sys
sys.path.insert(0, '/share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/dTAPE/src')

from run import REGISTRY as run_REGISTRY
import yaml

# 加载配置
with open('src/config/algs/d_tape_improved.yaml', 'r') as f:
    alg_config = yaml.safe_load(f)

with open('src/config/envs/sc2te.yaml', 'r') as f:
    env_config = yaml.safe_load(f)

# 设置参数
config = {
    **alg_config,
    **env_config,
    'env_args': {
        **env_config.get('env_args', {}),
        'map_name': 'adcc'
    },
    'checkpoint_path': 'results/adcc_d_tape_improved/models/episode_2000000.pt',
    'evaluate': True,
    'test_nepisode': 32
}

# 运行评测
run_REGISTRY['episode'](None, config, None)
```

## 常见问题

### 0. Sacred Git 信息收集错误

**错误**: `ValueError: Reference at 'refs/heads/master' does not exist`

**原因**: Git 仓库未初始化或没有提交，导致 Sacred 无法获取 Git 信息

**解决**: ✅ 已在 `main.py` 中添加错误处理，自动捕获并忽略 Git 错误

如果仍有问题，可以手动初始化 Git 仓库：
```bash
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC
git init
git add .
git commit -m "Initial commit"
```

### 1. 地图参数未找到错误

**错误**: `ValueError: Map parameters for 'adcc' not found in map_param_registry`

**原因**: 地图注册表中使用 `adcc_te` 格式，但训练脚本使用短名称 `adcc`

**解决**: ✅ 已在 `maps/__init__.py` 中添加别名映射，自动将短名称转换为完整名称

### 2. PySC2 地图未找到错误

**错误**: `pysc2.maps.lib.NoMapError: Map doesn't exist: adcc`

**原因**: PySC2 的 maps 注册表中没有注册 HLSMAC 地图

**解决**: ✅ 已在 `sc2_tactics_env.py` 中添加回退机制，如果 PySC2 找不到地图，直接使用地图文件路径

### 3. Protobuf 版本兼容性问题

**错误**: `TypeError: Descriptors cannot be created directly`

**原因**: protobuf 版本 >= 4.0 与 s2clientprotocol 不兼容

**解决**:
```bash
# 降级 protobuf 到 3.20.3
pip install protobuf==3.20.3 --force-reinstall

# 或使用修复脚本
bash fix_protobuf.sh
```

### 1. SC2PATH未设置

```bash
export SC2PATH="/path/to/StarCraftII"
```

### 2. 找不到地图文件

确保地图文件在 `$SC2PATH/Maps/` 目录下：
```bash
ls $SC2PATH/Maps/*.SC2Map
```

### 3. 模块导入错误

确保PYTHONPATH设置正确：
```bash
export PYTHONPATH="/share/project/ytz/RLproject/StarCraft2_HLSMAC:$PYTHONPATH"
export PYTHONPATH="/share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/dTAPE/src:$PYTHONPATH"
export PYTHONPATH="/share/project/ytz/RLproject/StarCraft2_HLSMAC/smac:$PYTHONPATH"
```

### 4. GPU内存不足

在配置文件中减小 `batch_size` 和 `batch_size_run`：
```yaml
batch_size: 64  # 从128减小到64
batch_size_run: 1
```

## 参考文献

1. Hong, Xingxing et al. "HLSMAC: A New StarCraft Multi-Agent Challenge for High-Level Strategic Decision-Making." ArXiv abs/2509.12927 (2025).

2. Samvelyan, Mikayel et al. "The StarCraft Multi-Agent Challenge." ArXiv abs/1902.04043 (2019).

3. dTAPE算法论文: https://arxiv.org/pdf/2312.15667

## 联系方式

如有问题，请参考：
- HLSMAC GitHub: [待补充]
- dTAPE GitHub: https://github.com/LxzGordon/TAPE/tree/main/deterministic

# stellar
# stellar
# stellar
