# Baseline 修复总结报告

## 修复时间
2025-11-18

## 修复范围
除 dTAPE 外的所有 10 个 baseline 算法

---

## 一、已完成的修复

### 1. ✅ Sacred Git 错误修复
**修复位置**: 每个 baseline 的 `src/main.py`

**修复内容**: 添加了 Git 错误处理代码，自动捕获并忽略 Git 相关错误

**涉及算法**: 
- DOP-master
- FOP-main
- maven_code
- pymarl2-master
- ResQ
- ResZ
- RODE-main
- ROMA-master
- sTAPE
- wqmix-master

**修复代码**:
```python
# Patch sacred dependencies to avoid Git errors
try:
    from sacred import dependencies
    original_get_commit = dependencies.get_commit_if_possible
    def safe_get_commit(main_file, save_git_info):
        try:
            return original_get_commit(main_file, save_git_info)
        except (ValueError, Exception):
            return None, None, False
    dependencies.get_commit_if_possible = safe_get_commit
except ImportError:
    pass
```

---

### 2. ✅ 默认地图配置修复
**修复位置**: 每个 baseline 的 `src/config/envs/sc2te.yaml`

**修复内容**: 将默认地图从 `"3m"`（SMAC地图）改为 `"adcc"`（HLSMAC地图）

**涉及算法**: 所有 10 个 baseline

**修复前**:
```yaml
env_args:
  map_name: "3m"  # ❌ SMAC地图
```

**修复后**:
```yaml
env_args:
  map_name: "adcc"  # ✅ HLSMAC地图
```

---

### 3. ✅ 训练脚本创建
**修复位置**: 每个 baseline 目录下的 `train_single_map.sh`

**修复内容**: 为缺少训练脚本的 baseline 创建了统一的训练脚本

**涉及算法**:
- DOP-master: ✅ 已创建
- FOP-main: ✅ 已创建
- maven_code: ✅ 已创建
- pymarl2-master: ✅ 已创建
- ResQ: ✅ 已创建
- ResZ: ✅ 已创建
- RODE-main: ✅ 已创建
- ROMA-master: ✅ 已创建
- sTAPE: ✅ 已创建
- wqmix-master: ✅ 已创建

**脚本功能**:
- 自动设置 SC2PATH（如果未设置）
- 支持指定地图、GPU、种子等参数
- 自动创建结果目录
- 统一的训练命令格式

**使用示例**:
```bash
cd RLalgs/ResQ
bash train_single_map.sh adcc 0 42
```

---

## 二、共享修复（所有baseline自动受益）

### 1. ✅ 地图别名映射
**修复位置**: `smac/smac/env/sc2_tactics/maps/__init__.py`

**修复内容**: 添加了地图名称别名映射，自动将短名称（如 `adcc`）转换为完整名称（如 `adcc_te`）

**受益算法**: 所有 baseline（包括 dTAPE）

---

### 2. ✅ PySC2 地图回退机制
**修复位置**: `smac/smac/env/sc2_tactics/sc2_tactics_env.py`

**修复内容**: 添加了地图文件路径回退机制，如果 PySC2 注册表中找不到地图，直接使用文件路径

**受益算法**: 所有 baseline（包括 dTAPE）

---

## 三、各算法默认配置

| 算法 | 默认算法配置 | 训练脚本路径 |
|------|------------|------------|
| DOP-master | `dop` | `RLalgs/DOP-master/train_single_map.sh` |
| FOP-main | `fop` | `RLalgs/FOP-main/train_single_map.sh` |
| maven_code | `coma` | `RLalgs/maven_code/train_single_map.sh` |
| pymarl2-master | `riit` | `RLalgs/pymarl2-master/train_single_map.sh` |
| ResQ | `ResQ` | `RLalgs/ResQ/train_single_map.sh` |
| ResZ | `qmix` | `RLalgs/ResZ/train_single_map.sh` |
| RODE-main | `qmix` | `RLalgs/RODE-main/train_single_map.sh` |
| ROMA-master | `qmix_smac` | `RLalgs/ROMA-master/train_single_map.sh` |
| sTAPE | `s_tape` | `RLalgs/sTAPE/train_single_map.sh` |
| wqmix-master | `ow_qmix` | `RLalgs/wqmix-master/train_single_map.sh` |

---

## 四、使用说明

### 训练单个地图

```bash
# 进入算法目录
cd RLalgs/<algorithm_name>

# 运行训练
bash train_single_map.sh <map_name> <gpu_id> <seed>

# 示例：使用ResQ训练adcc地图
cd RLalgs/ResQ
bash train_single_map.sh adcc 0 42
```

### 自定义算法配置

```bash
# 使用环境变量指定算法配置
ALG_CONFIG=fop bash train_single_map.sh adcc 0 42
```

---

## 五、验证清单

### 每个baseline应满足：
- [x] `src/main.py` 包含 Sacred Git 错误修复
- [x] `src/config/envs/sc2te.yaml` 默认地图为 `adcc`
- [x] 存在 `train_single_map.sh` 训练脚本
- [x] 训练脚本包含 SC2PATH 自动设置
- [x] 训练脚本可执行（chmod +x）

### 共享修复验证：
- [x] `maps/__init__.py` 包含地图别名映射
- [x] `sc2_tactics_env.py` 包含 PySC2 地图回退机制

---

## 六、注意事项

1. **dTAPE 未修改**: 按照要求，dTAPE 的所有修复保持不变
2. **共享修复**: 地图别名映射和 PySC2 回退机制在共享文件中，所有 baseline 自动受益
3. **训练脚本**: 所有新创建的训练脚本都遵循相同的格式，便于统一管理
4. **默认配置**: 每个算法的默认配置可能不同，使用前请查看对应的算法配置文件

---

## 七、后续建议

1. **批量训练脚本**: 可以为每个 baseline 创建类似 `train_hlsmac.sh` 的批量训练脚本
2. **评测脚本**: 为每个 baseline 创建评测脚本
3. **依赖检查**: 检查每个 baseline 的 Python 依赖是否完整
4. **测试验证**: 对每个 baseline 进行单地图训练测试，确保修复有效

---

## 八、修复统计

- **修复的 baseline 数量**: 10 个
- **修复的文件数量**: 
  - main.py: 10 个
  - sc2te.yaml: 10 个
  - train_single_map.sh: 10 个（新建）
- **共享修复文件**: 2 个（所有 baseline 受益）

**总计**: 32 个文件修复/创建

