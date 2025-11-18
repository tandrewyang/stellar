# Baseline 训练问题检查报告

## 检查时间
2025-11-18

## 检查范围
所有11个baseline算法：
1. dTAPE
2. DOP-master
3. FOP-main
4. maven_code
5. pymarl2-master
6. ResQ
7. ResZ
8. RODE-main
9. ROMA-master
10. sTAPE
11. wqmix-master

---

## 一、环境配置问题

### 1.1 环境配置文件存在但默认地图错误
**问题**: 所有baseline的 `sc2te.yaml` 配置文件中，默认地图都是 `"3m"`（SMAC地图），而不是HLSMAC地图

**影响**: 
- 如果直接运行训练脚本而不指定地图，会使用错误的默认地图
- 需要手动指定 `env_args.map_name` 参数

**涉及算法**: 所有11个baseline

**示例**:
```yaml
# 当前配置
env: sc2_tactics
env_args:
  map_name: "3m"  # ❌ 应该是HLSMAC地图，如 "adcc"
```

---

## 二、训练脚本问题

### 2.1 缺少统一的HLSMAC训练脚本
**问题**: 
- 大部分baseline只有Docker运行脚本（DOP, FOP, ROMA, RODE）
- 没有针对HLSMAC的统一训练脚本
- 每个baseline的训练方式不一致

**影响**: 
- 无法批量训练所有12个HLSMAC地图
- 需要为每个baseline单独编写训练脚本

**涉及算法**:
- DOP-master: 只有Docker脚本
- FOP-main: 只有Docker脚本
- ROMA-master: 只有Docker脚本
- RODE-main: 只有Docker脚本
- pymarl2-master: 有通用训练脚本，但需要手动配置

### 2.2 SC2PATH环境变量检查缺失
**问题**: 大部分训练脚本没有检查SC2PATH环境变量

**影响**: 
- 如果SC2PATH未设置，训练会失败
- 错误信息不明确

**涉及算法**: 除dTAPE外，其他baseline的训练脚本都缺少此检查

---

## 三、环境注册问题

### 3.1 环境注册正确
**状态**: ✅ 所有baseline都已正确注册HLSMAC环境

**检查结果**:
```python
# 所有baseline的 envs/__init__.py 都包含：
from smac.env import MultiAgentEnv, StarCraft2Env, SC2TacticsEnv
from smac.env.sc2_tactics.sc2_tactics_env import SC2TacticsEnv_NEW
```

---

## 四、依赖和兼容性问题

### 4.1 Sacred框架使用不一致
**问题**: 
- dTAPE使用Sacred框架（已修复Git错误）
- 其他baseline可能使用不同的实验管理框架
- 需要检查每个baseline的依赖

**需要检查**:
- pymarl2-master: 可能使用不同的框架
- ResQ/ResZ: 需要检查依赖
- wqmix-master: 需要检查依赖

### 4.2 Protobuf版本兼容性
**问题**: 所有baseline都可能遇到protobuf版本问题

**影响**: 
- 如果protobuf版本 >= 4.0，会导致 `TypeError: Descriptors cannot be created directly`
- 需要为每个baseline单独降级protobuf

**解决方案**: 已在dTAPE中修复，其他baseline需要类似处理

---

## 五、地图名称映射问题

### 5.1 地图别名映射缺失
**问题**: 
- 地图注册表中使用 `adcc_te` 格式
- 训练脚本使用短名称 `adcc`
- 只有dTAPE已修复别名映射

**影响**: 
- 其他baseline如果使用短名称会报错
- 需要手动使用完整名称（如 `adcc_te`）

**涉及算法**: 除dTAPE外，其他10个baseline

---

## 六、PySC2地图注册问题

### 6.1 PySC2地图注册缺失
**问题**: 
- PySC2的 `maps.get()` 无法找到HLSMAC地图
- 只有dTAPE已添加文件路径回退机制

**影响**: 
- 其他baseline会报错 `pysc2.maps.lib.NoMapError: Map doesn't exist: adcc`
- 需要修改环境代码添加回退机制

**涉及算法**: 除dTAPE外，其他10个baseline

---

## 七、具体算法问题

### 7.1 dTAPE
**状态**: ✅ 已修复大部分问题
- ✅ Git错误已修复
- ✅ Protobuf版本已修复
- ✅ 地图别名映射已添加
- ✅ PySC2地图回退已添加
- ✅ SC2PATH自动设置已添加

**剩余问题**: 无

### 7.2 DOP-master
**问题**:
- ❌ 只有Docker训练脚本，无直接训练脚本
- ❌ 缺少SC2PATH检查
- ❌ 地图别名映射缺失
- ❌ PySC2地图回退缺失

### 7.3 FOP-main
**问题**:
- ❌ 只有Docker训练脚本，无直接训练脚本
- ❌ 缺少SC2PATH检查
- ❌ 地图别名映射缺失
- ❌ PySC2地图回退缺失

### 7.4 maven_code
**问题**:
- ❌ 缺少SC2PATH检查
- ❌ 地图别名映射缺失
- ❌ PySC2地图回退缺失

### 7.5 pymarl2-master
**问题**:
- ⚠️ 有通用训练脚本，但需要手动配置参数
- ❌ 缺少SC2PATH检查
- ❌ 地图别名映射缺失
- ❌ PySC2地图回退缺失

### 7.6 ResQ
**问题**:
- ❌ 缺少训练脚本
- ❌ 缺少SC2PATH检查
- ❌ 地图别名映射缺失
- ❌ PySC2地图回退缺失

### 7.7 ResZ
**问题**:
- ❌ 缺少训练脚本
- ❌ 缺少SC2PATH检查
- ❌ 地图别名映射缺失
- ❌ PySC2地图回退缺失

### 7.8 RODE-main
**问题**:
- ❌ 只有Docker训练脚本，无直接训练脚本
- ❌ 缺少SC2PATH检查
- ❌ 地图别名映射缺失
- ❌ PySC2地图回退缺失

### 7.9 ROMA-master
**问题**:
- ❌ 只有Docker训练脚本，无直接训练脚本
- ❌ 缺少SC2PATH检查
- ❌ 地图别名映射缺失
- ❌ PySC2地图回退缺失

### 7.10 sTAPE
**问题**:
- ❌ 缺少训练脚本（只有runalgo.sh）
- ❌ 缺少SC2PATH检查
- ❌ 地图别名映射缺失
- ❌ PySC2地图回退缺失

### 7.11 wqmix-master
**问题**:
- ❌ 缺少训练脚本
- ❌ 缺少SC2PATH检查
- ❌ 地图别名映射缺失
- ❌ PySC2地图回退缺失

---

## 八、优先级问题总结

### 高优先级（必须修复）
1. **PySC2地图注册问题** - 影响所有baseline运行
2. **地图别名映射** - 影响地图名称使用
3. **SC2PATH检查** - 影响环境初始化

### 中优先级（建议修复）
4. **默认地图配置** - 影响直接运行训练
5. **统一训练脚本** - 影响批量训练

### 低优先级（可选）
6. **Sacred框架兼容** - 仅影响实验管理
7. **Docker脚本** - 如果使用Docker环境

---

## 九、修复建议

### 通用修复（适用于所有baseline）
1. 在 `maps/__init__.py` 中添加地图别名映射（参考dTAPE）
2. 在 `sc2_tactics_env.py` 中添加PySC2地图回退机制（参考dTAPE）
3. 在训练脚本中添加SC2PATH自动设置（参考dTAPE）
4. 修改 `sc2te.yaml` 默认地图为HLSMAC地图

### 针对特定baseline
1. **Docker脚本baseline** (DOP, FOP, ROMA, RODE): 创建非Docker训练脚本
2. **缺少脚本baseline** (ResQ, ResZ, wqmix): 创建训练脚本
3. **Sacred框架baseline**: 修复Git错误（参考dTAPE）

---

## 十、测试建议

### 测试步骤
1. 选择一个baseline（如pymarl2-master）
2. 应用所有通用修复
3. 测试单个地图训练（如adcc）
4. 如果成功，批量应用到其他baseline
5. 测试所有12个HLSMAC地图

### 验证清单
- [ ] SC2PATH自动设置
- [ ] 地图别名映射工作
- [ ] PySC2地图回退工作
- [ ] 训练可以启动
- [ ] 环境可以初始化
- [ ] 可以运行一个完整的episode

