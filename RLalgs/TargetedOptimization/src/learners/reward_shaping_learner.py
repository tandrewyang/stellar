"""
奖励塑形Learner：通过改进奖励信号提升学习效率
"""
import copy
from torch.autograd import Variable
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from utils.rl_utils import build_td_lambda_targets
import torch as th
import numpy as np
from torch.optim import RMSprop, Adam
from collections import deque
from controllers import REGISTRY as mac_REGISTRY
import torch.distributions as D
from utils.th_utils import get_parameters_num
from learners.max_q_learner import MAXQLearner


class RewardShapingLearner(MAXQLearner):
    """
    奖励塑形：通过改进奖励信号提升学习效率
    """
    def __init__(self, mac, scheme, logger, args):
        super(RewardShapingLearner, self).__init__(mac, scheme, logger, args)
        
        # 奖励塑形参数
        self.reward_shaping_enabled = getattr(args, 'reward_shaping_enabled', True)
        self.shaping_type = getattr(args, 'reward_shaping_type', 'potential_based')
        
        # 获取地图名称
        try:
            self.map_name = getattr(args.env_args, 'map_name', '')
        except:
            self.map_name = ''
        
        # 潜在函数奖励塑形
        self.potential_function = None
        if self.shaping_type == 'potential_based':
            self.potential_function = self._create_potential_function(args)
        
        # 奖励塑形权重
        self.shaping_weight = getattr(args, 'reward_shaping_weight', 0.1)
        self.shaping_decay = getattr(args, 'reward_shaping_decay', 0.99)
        self.current_shaping_weight = self.shaping_weight
        
        # jctq (金蝉脱壳) 特定参数
        self.survival_reward_weight = getattr(args, 'survival_reward_weight', 2.0)
        self.escape_reward_weight = getattr(args, 'escape_reward_weight', 1.5)
        self.disperse_reward_weight = getattr(args, 'disperse_reward_weight', 0.8)
        self.kill_reward_weight = getattr(args, 'kill_reward_weight', 0.3)
        self.time_critical_weight = getattr(args, 'time_critical_weight', 1.2)
        
        # swct (上屋抽梯) 特定参数
        self.lure_reward_weight = getattr(args, 'lure_reward_weight', 1.2)
        self.forcefield_reward_weight = getattr(args, 'forcefield_reward_weight', 2.0)
        self.warp_prism_reward_weight = getattr(args, 'warp_prism_reward_weight', 1.5)
        self.tactical_positioning_weight = getattr(args, 'tactical_positioning_weight', 1.0)
        self.phase_based_reward = getattr(args, 'phase_based_reward', False)
        
        # dhls (调虎离山) 特定参数 - 分兵拖延+基地攻击策略
        self.dhls_survival_reward_weight = getattr(args, 'survival_reward_weight', 3.0)  # 生存奖励权重
        self.dhls_delay_intercept_weight = getattr(args, 'delay_intercept_weight', 5.0)  # 拖延拦截奖励权重（大幅增加）
        self.dhls_base_attack_weight = getattr(args, 'base_attack_weight', 6.0)  # 基地攻击奖励权重（大幅增加）
        self.dhls_tactical_coordination_weight = getattr(args, 'tactical_coordination_weight', 5.5)  # 战术协调奖励权重（大幅增加）
        self.dhls_time_window_weight = getattr(args, 'time_window_weight', 5.0)  # 时间差奖励权重（大幅增加）
        self.dhls_division_reward_weight = getattr(args, 'division_reward_weight', 4.0)  # 分兵奖励权重（大幅增加）
        self.dhls_enemy_kill_weight = getattr(args, 'enemy_kill_weight', 4.5)  # 敌方单位消灭奖励权重（新增）
        self.dhls_nydus_usage_weight = getattr(args, 'nydus_usage_weight', 3.5)  # 虫洞使用奖励权重（新增）
        
        # yqgz (欲擒故纵) 特定参数 - 少量送死引诱+大量部队攻击
        self.yqgz_survival_reward_weight = getattr(args, 'survival_reward_weight', 3.0)  # 生存奖励权重
        self.yqgz_sacrifice_lure_weight = getattr(args, 'sacrifice_lure_weight', 6.0)  # 送死引诱奖励权重（大幅增加）
        self.yqgz_main_force_attack_weight = getattr(args, 'main_force_attack_weight', 7.0)  # 大量部队攻击奖励权重（大幅增加）
        self.yqgz_distance_advantage_weight = getattr(args, 'distance_advantage_weight', 5.5)  # 距离优势奖励权重（大幅增加）
        self.yqgz_tactical_coordination_weight = getattr(args, 'tactical_coordination_weight', 6.5)  # 战术协调奖励权重（大幅增加）
        self.yqgz_enemy_advance_weight = getattr(args, 'enemy_advance_weight', 4.5)  # 敌人前进奖励权重（大幅增加）
        self.yqgz_enemy_kill_weight = getattr(args, 'enemy_kill_weight', 5.0)  # 敌方单位消灭奖励权重（新增）
        self.yqgz_close_range_attack_weight = getattr(args, 'close_range_attack_weight', 6.0)  # 近距离攻击奖励权重（新增）
        
        # tlhz (偷梁换柱) 特定参数 - 孵化巢暴露+集中攻击策略
        self.survival_reward_weight = getattr(args, 'survival_reward_weight', 2.0)  # 生存奖励权重
        self.build_hatchery_weight = getattr(args, 'build_hatchery_weight', 3.0)  # 建造孵化巢奖励权重
        self.expose_weight = getattr(args, 'expose_weight', 2.8)  # 暴露奖励权重（孵化巢在光子炮台下）
        self.shatter_detection_weight = getattr(args, 'shatter_detection_weight', 3.5)  # 被击碎检测奖励权重
        self.concentrated_attack_weight = getattr(args, 'concentrated_attack_weight', 3.5)  # 集中攻击奖励权重
        self.timing_window_weight = getattr(args, 'timing_window_weight', 4.0)  # 时间窗口奖励权重（被击碎那一刻集中攻击）
        self.train_weight = getattr(args, 'train_weight', 2.0)  # 训练奖励权重（训练单位）
        self.damage_reward_factor = getattr(args, 'damage_reward_factor', 2.0)  # 伤害奖励放大因子（TLHZ专用）
        
        # fkwz (反客为主) 特定参数 - 主动进攻（建造+资源管理）
        self.warpgate_train_reward_weight = getattr(args, 'warpgate_train_reward_weight', 2.2)  # 折跃门训练奖励权重
        self.warp_prism_reward_weight = getattr(args, 'warp_prism_reward_weight', 2.0)  # 折跃棱镜奖励权重
        self.resource_management_weight = getattr(args, 'resource_management_weight', 1.8)  # 资源管理奖励权重
        self.active_offensive_weight = getattr(args, 'active_offensive_weight', 2.5)  # 主动进攻奖励权重
        self.unit_advantage_weight = getattr(args, 'unit_advantage_weight', 1.6)  # 单位优势奖励权重
        self.damage_reward_factor = getattr(args, 'damage_reward_factor', 0.9)  # 伤害奖励放大因子
        self.passive_to_active_conversion_weight = getattr(args, 'passive_to_active_conversion_weight', 3.0)  # 从被动到主动的转换奖励权重
        
    def _create_potential_function(self, args):
        """
        创建潜在函数（用于基于潜在函数的奖励塑形）
        """
        # 这里可以实现一个简单的潜在函数
        # 例如：基于状态特征的线性函数
        # 实际应用中可以使用神经网络
        return None
    
    def shape_reward(self, rewards, states, next_states, t_env, batch=None):
        """
        对奖励进行塑形
        """
        if not self.reward_shaping_enabled:
            return rewards
        
        shaped_rewards = rewards.clone()
        
        # jctq (金蝉脱壳) - 逃脱生存策略
        if self.shaping_type == 'escape_survival' or self.map_name == 'jctq':
            shaped_rewards = self._apply_escape_survival_shaping(rewards, states, next_states, batch)
        
        # swct (上屋抽梯) - 诱敌阻挡撤退策略
        elif self.shaping_type == 'lure_block_retreat' or self.map_name == 'swct':
            shaped_rewards = self._apply_lure_block_retreat_shaping(rewards, states, next_states, batch, t_env)
        
        # dhls (调虎离山) - 机制感知（虫洞机制）
        elif self.shaping_type == 'mechanism_aware' or self.map_name == 'dhls':
            shaped_rewards = self._apply_mechanism_aware_shaping(rewards, states, next_states, batch, t_env)
        
        # yqgz (欲擒故纵) - 大规模协调
        elif self.shaping_type == 'large_scale_coordination' or self.map_name == 'yqgz':
            shaped_rewards = self._apply_large_scale_coordination_shaping(rewards, states, next_states, batch, t_env)
        
        # tlhz (偷梁换柱) - 长期规划（建造机制）
        elif self.shaping_type == 'long_term_planning' or self.map_name == 'tlhz':
            shaped_rewards = self._apply_long_term_planning_shaping(rewards, states, next_states, batch, t_env)
        
        # fkwz (反客为主) - 主动进攻（建造+资源管理）
        elif self.shaping_type == 'active_offensive' or self.map_name == 'fkwz':
            shaped_rewards = self._apply_active_offensive_shaping(rewards, states, next_states, batch, t_env)
        
        elif self.shaping_type == 'potential_based':
            # 基于潜在函数的奖励塑形
            # φ(s') - φ(s)，其中φ是潜在函数
            if self.potential_function is not None:
                # 这里可以实现具体的潜在函数计算
                # 简化版本：基于状态的简单函数
                potential_current = self._compute_potential(states)
                potential_next = self._compute_potential(next_states)
                potential_bonus = potential_next - potential_current
                shaped_rewards = rewards + self.current_shaping_weight * potential_bonus
            else:
                # 简化版本：基于奖励的启发式塑形
                shaped_rewards = self._heuristic_shaping(rewards, states)
        
        elif self.shaping_type == 'dense_reward':
            # 密集奖励：为中间步骤添加奖励
            shaped_rewards = self._add_dense_rewards(rewards, states)
        
        elif self.shaping_type == 'curiosity':
            # 好奇心驱动：鼓励探索
            shaped_rewards = self._add_curiosity_bonus(rewards, states)
        
        # 更新塑形权重（逐渐减少）
        self.current_shaping_weight *= self.shaping_decay
        
        return shaped_rewards
    
    def _compute_potential(self, states):
        """
        计算潜在函数值（简化版本）
        """
        # 这里可以实现更复杂的潜在函数
        # 例如：基于状态特征的神经网络
        # 简化版本：返回零（不改变奖励）
        return th.zeros_like(states[:, :, 0:1])
    
    def _heuristic_shaping(self, rewards, states):
        """
        启发式奖励塑形
        """
        # 例如：鼓励接近目标、避免危险等
        # 这里可以根据具体任务实现
        return rewards
    
    def _add_dense_rewards(self, rewards, states):
        """
        添加密集奖励
        """
        # 为中间步骤添加小的奖励信号
        # 例如：每步给予小的生存奖励
        dense_bonus = 0.01 * th.ones_like(rewards)
        return rewards + self.current_shaping_weight * dense_bonus
    
    def _add_curiosity_bonus(self, rewards, states):
        """
        添加好奇心奖励（鼓励探索）
        """
        # 简化版本：基于状态的新颖性
        # 实际可以使用预测误差作为好奇心信号
        return rewards
    
    def _apply_escape_survival_shaping(self, rewards, states, next_states, batch=None):
        """
        jctq (金蝉脱壳) - 逃脱生存奖励塑形
        核心策略：逃脱优先 + 生存优先 + 分散隐藏
        """
        shaped_rewards = rewards.clone()
        batch_size, seq_len, n_agents = rewards.shape
        
        # 1. 存活奖励：每步给予存活奖励（基于奖励信号推断，如果奖励>0说明可能存活）
        # 简化实现：如果原始奖励不是负的（不是死亡），给予存活奖励
        survival_bonus = th.where(rewards >= 0, 
                                  self.survival_reward_weight * 0.1,  # 存活奖励
                                  th.zeros_like(rewards))  # 死亡无奖励
        shaped_rewards = shaped_rewards + survival_bonus
        
        # 2. 逃脱奖励：鼓励单位分散（通过状态差异推断）
        # 简化实现：如果状态变化大（可能是在移动/逃脱），给予逃脱奖励
        # states shape: [batch_size, seq_len, state_dim]
        state_diff = th.abs(next_states - states).sum(dim=-1, keepdim=False)  # [batch_size, seq_len]
        state_diff = state_diff.unsqueeze(-1)  # [batch_size, seq_len, 1]
        state_diff_max = state_diff.max() if state_diff.numel() > 0 else 1.0
        escape_bonus = self.escape_reward_weight * 0.05 * th.clamp(state_diff / (state_diff_max + 1e-6), 0, 1)
        escape_bonus = escape_bonus.expand(batch_size, seq_len, n_agents)  # [batch_size, seq_len, n_agents]
        shaped_rewards = shaped_rewards + escape_bonus
        
        # 3. 分散奖励：鼓励单位分散（通过状态空间分布推断）
        # 简化实现：基于状态的标准差，标准差大说明分散
        # Compute variance across state dimensions to measure dispersion
        state_var = th.var(states, dim=-1, keepdim=False)  # [batch_size, seq_len]
        state_var = state_var.unsqueeze(-1)  # [batch_size, seq_len, 1]
        state_var_max = state_var.max() if state_var.numel() > 0 else 1.0
        disperse_bonus = self.disperse_reward_weight * 0.03 * th.clamp(state_var / (state_var_max + 1e-6), 0, 1)
        disperse_bonus = disperse_bonus.expand(batch_size, seq_len, n_agents)  # [batch_size, seq_len, n_agents]
        shaped_rewards = shaped_rewards + disperse_bonus
        
        # 4. 降低击杀奖励权重（避免暴露位置）
        # 如果原始奖励很高（可能是击杀），降低其权重
        kill_penalty = th.where(rewards > 0.5, 
                               (1.0 - self.kill_reward_weight) * rewards * 0.5,  # 降低击杀奖励
                               th.zeros_like(rewards))
        shaped_rewards = shaped_rewards - kill_penalty
        
        # 5. 时间紧迫性：越接近结束，存活奖励越高
        # 简化实现：基于序列位置，后面的步骤奖励更高
        time_factor = th.linspace(0, 1, seq_len, device=rewards.device).view(1, -1, 1)  # [1, seq_len, 1]
        time_bonus = self.time_critical_weight * 0.05 * time_factor.expand(batch_size, seq_len, n_agents)
        shaped_rewards = shaped_rewards + time_bonus
        
        return shaped_rewards
    
    def _apply_lure_block_retreat_shaping(self, rewards, states, next_states, batch, t_env):
        """
        swct (上屋抽梯) - 诱敌阻挡撤退奖励塑形
        核心策略：强烈生存奖励 + 机制使用奖励 + 战术执行奖励
        5 vs 11劣势，需要更强的生存和机制使用奖励
        """
        shaped_rewards = rewards.clone()
        batch_size, seq_len, n_agents = rewards.shape
        
        # ========== 1. 强烈生存奖励（核心）==========
        # 每步存活都有奖励，存活越久奖励越高
        # 这是5 vs 11劣势地图的关键
        survival_bonus = th.ones_like(rewards) * 0.15  # 每步基础生存奖励
        shaped_rewards = shaped_rewards + survival_bonus
        
        # 存活时间奖励：越接近结束还存活，奖励越高
        time_progress = th.linspace(0, 1, seq_len, device=rewards.device).view(1, -1, 1)
        time_progress = time_progress.expand(batch_size, seq_len, n_agents)
        survival_time_bonus = 0.2 * time_progress  # 随时间线性增加
        shaped_rewards = shaped_rewards + survival_time_bonus
        
        # ========== 2. 力场阻挡奖励（机制使用）==========
        # 检测大幅状态变化（可能是力场生效阻挡敌人）
        state_change = th.abs(next_states - states).sum(dim=-1, keepdim=False)  # [batch_size, seq_len]
        state_change = state_change.unsqueeze(-1)  # [batch_size, seq_len, 1]
        state_change_max = state_change.max() if state_change.numel() > 0 else 1.0
        
        # 大幅状态变化（可能是力场阻挡成功）
        forcefield_threshold = state_change_max * 0.4  # 40%阈值
        forcefield_bonus = self.forcefield_reward_weight * 0.25 * th.where(
            state_change > forcefield_threshold,
            th.ones_like(state_change),
            th.zeros_like(state_change)
        )
        forcefield_bonus = forcefield_bonus.expand(batch_size, seq_len, n_agents)
        shaped_rewards = shaped_rewards + forcefield_bonus
        
        # ========== 3. 传送撤退奖励（机制使用）==========
        # 检测大幅位置变化（可能是传送）
        if states.shape[-1] >= 2:
            pos_diff = next_states[:, :, :2] - states[:, :, :2]  # [batch_size, seq_len, 2]
            pos_change = th.norm(pos_diff, dim=-1, keepdim=False)  # [batch_size, seq_len]
            pos_change = pos_change.unsqueeze(-1)  # [batch_size, seq_len, 1]
            pos_change_max = pos_change.max() if pos_change.numel() > 0 else 1.0
            
            # 大幅位置变化（可能是传送撤退）
            warp_threshold = pos_change_max * 0.3  # 30%阈值
            warp_bonus = self.warp_prism_reward_weight * 0.2 * th.where(
                pos_change > warp_threshold,
                th.ones_like(pos_change),
                th.zeros_like(pos_change)
            )
            warp_bonus = warp_bonus.expand(batch_size, seq_len, n_agents)
            shaped_rewards = shaped_rewards + warp_bonus
        else:
            warp_bonus = th.zeros_like(rewards)
        
        # ========== 4. 敌人伤害奖励 ==========
        # 即使不能获胜，也要奖励对敌人造成伤害
        # 如果原始奖励为正（可能是击杀或伤害），给予额外奖励
        damage_bonus = th.where(
            rewards > 0,
            rewards * 0.5,  # 伤害奖励放大50%
            th.zeros_like(rewards)
        )
        shaped_rewards = shaped_rewards + damage_bonus
        
        # ========== 5. 诱敌奖励（移动且获得奖励）==========
        state_change_expanded = state_change.expand(batch_size, seq_len, n_agents)
        lure_bonus = self.lure_reward_weight * 0.1 * th.where(
            (state_change_expanded > state_change_max * 0.2) & (rewards > 0),
            th.ones_like(rewards),
            th.zeros_like(rewards)
        )
        shaped_rewards = shaped_rewards + lure_bonus
        
        # ========== 6. 战术位置奖励 ==========
        if states.shape[-1] >= 2:
            pos_states = states[:, :, :2]  # [batch_size, seq_len, 2]
            state_center = pos_states.mean(dim=-1, keepdim=False)  # [batch_size, seq_len]
            state_center = state_center.unsqueeze(-1)  # [batch_size, seq_len, 1]
            state_center = state_center.expand(batch_size, seq_len, 2)  # [batch_size, seq_len, 2]
            dist_to_center = th.norm(pos_states - state_center, dim=-1, keepdim=False)  # [batch_size, seq_len]
            dist_to_center = dist_to_center.unsqueeze(-1)  # [batch_size, seq_len, 1]
            dist_max = dist_to_center.max() if dist_to_center.numel() > 0 else 1.0
            position_bonus = self.tactical_positioning_weight * 0.08 * th.exp(-dist_to_center / (dist_max + 1e-6))
            position_bonus = position_bonus.expand(batch_size, seq_len, n_agents)
            shaped_rewards = shaped_rewards + position_bonus
        
        # ========== 7. 分阶段奖励（增强版）==========
        if self.phase_based_reward:
            # 阶段1（前30%）：诱敌深入 + 生存
            phase1_mask = th.linspace(0, 1, seq_len, device=rewards.device).view(1, -1, 1) < 0.3
            phase1_mask = phase1_mask.expand(batch_size, seq_len, n_agents)
            phase1_bonus = (lure_bonus + survival_bonus) * phase1_mask * 1.5  # 增强
            shaped_rewards = shaped_rewards + phase1_bonus
            
            # 阶段2（中40%）：力场阻挡 + 生存
            phase2_mask = (th.linspace(0, 1, seq_len, device=rewards.device).view(1, -1, 1) >= 0.3) & \
                          (th.linspace(0, 1, seq_len, device=rewards.device).view(1, -1, 1) < 0.7)
            phase2_mask = phase2_mask.expand(batch_size, seq_len, n_agents)
            phase2_bonus = (forcefield_bonus + survival_bonus) * phase2_mask * 1.5  # 增强
            shaped_rewards = shaped_rewards + phase2_bonus
            
            # 阶段3（后30%）：撤退 + 生存
            phase3_mask = th.linspace(0, 1, seq_len, device=rewards.device).view(1, -1, 1) >= 0.7
            phase3_mask = phase3_mask.expand(batch_size, seq_len, n_agents)
            phase3_bonus = (warp_bonus + survival_bonus + survival_time_bonus) * phase3_mask * 1.5  # 增强
            shaped_rewards = shaped_rewards + phase3_bonus
        
        return shaped_rewards
    
    def _apply_mechanism_aware_shaping(self, rewards, states, next_states, batch=None, t_env=0):
        """
        dhls (调虎离山) - 重新设计的奖励塑形函数
        战术核心：利用虫洞机制快速传送，分兵拖延敌方军队，主队趁机攻击敌方基地
        16 vs 11，200步，Z vs T，有NydusCanal（虫洞）机制
        
        新设计思路（更有效）：
        1. 强烈生存奖励（每步都有，确保存活）
        2. 虫洞使用奖励（检测大幅位置变化，可能是虫洞传送）- 新增
        3. 分兵奖励（少量单位远离主队，用于拖延）
        4. 拖延拦截奖励（少量单位在敌方军队附近，进行拦截）
        5. 基地攻击奖励（主队攻击敌方基地，通过奖励变化和状态变化推断）
        6. 时间差奖励（在拖延期间对基地的攻击给予额外奖励）
        7. 战术协调奖励（同时进行拖延和基地攻击的协调行为）
        8. 获胜检测奖励（大幅奖励获胜）
        9. 敌人伤害奖励（放大伤害奖励）
        """
        shaped_rewards = rewards.clone()
        batch_size, seq_len, n_agents = rewards.shape
        
        # 时间进度
        time_progress = th.linspace(0, 1, seq_len, device=rewards.device).view(1, -1, 1)
        time_progress = time_progress.expand(batch_size, seq_len, n_agents)
        
        # ========== 0. 获胜检测奖励（最重要，大幅奖励获胜）==========
        if batch is not None:
            terminated = batch.get("terminated", None)
            if terminated is not None:
                episode_end_reward = rewards[:, -1:, :]
                win_bonus = th.zeros_like(rewards)
                if episode_end_reward.shape[1] > 0:
                    win_bonus[:, -1:, :] = th.where(
                        episode_end_reward > 0,
                        th.ones_like(episode_end_reward) * 20.0,  # 大幅增加获胜奖励（从10增加到20）
                        th.zeros_like(episode_end_reward)
                    )
                shaped_rewards = shaped_rewards + win_bonus
        
        # ========== 1. 强烈生存奖励（核心，每步都有）==========
        survival_bonus = th.ones_like(rewards) * 0.6  # 增加基础生存奖励（从0.5增加到0.6）
        shaped_rewards = shaped_rewards + survival_bonus
        
        survival_time_bonus = 0.6 * time_progress  # 增加时间奖励（从0.5增加到0.6）
        shaped_rewards = shaped_rewards + survival_time_bonus
        
        # ========== 2. 虫洞使用奖励（新增，检测大幅位置变化）==========
        if states.shape[-1] >= 2 and next_states.shape[-1] >= 2:
            pos_states = states[:, :, :2]
            pos_next_states = next_states[:, :, :2]
            
            # 计算每个单位的位置变化
            pos_change = th.norm(pos_next_states - pos_states, dim=-1, keepdim=False)  # [batch_size, seq_len]
            pos_change = pos_change.unsqueeze(-1)  # [batch_size, seq_len, 1]
            pos_change_max = pos_change.max() if pos_change.numel() > 0 else 1.0
            
            # 大幅位置变化（可能是虫洞传送）
            nydus_threshold = pos_change_max * 0.5  # 50%阈值，检测大幅位置变化
            nydus_bonus = self.dhls_delay_intercept_weight * 0.8 * th.where(
                pos_change > nydus_threshold,
                th.ones_like(pos_change),
                th.zeros_like(pos_change)
            )
            nydus_bonus = nydus_bonus.expand(batch_size, seq_len, n_agents)
            
            # 前期到中期（0-70%）强烈增强虫洞使用奖励
            early_mid_phase = time_progress < 0.7
            nydus_bonus = nydus_bonus * (1.0 + 2.0 * early_mid_phase.float())  # 增强200%
            shaped_rewards = shaped_rewards + nydus_bonus
        
        # ========== 3. 分兵奖励（检测单位分散，少量单位远离主队）==========
        if states.shape[-1] >= 2:
            pos_states = states[:, :, :2]
            center = pos_states.mean(dim=-1, keepdim=True)
            distances_to_center = th.norm(pos_states.unsqueeze(-2) - center, dim=-1)
            
            distance_std = th.std(distances_to_center, dim=-1, keepdim=True)
            distance_std_max = distance_std.max() if distance_std.numel() > 0 else 1.0
            
            division_bonus = self.dhls_division_reward_weight * 0.5 * th.clamp(
                distance_std / (distance_std_max + 1e-6), 0, 1
            )
            division_bonus = division_bonus.expand(batch_size, seq_len, n_agents)
            
            # 前期（前60%）增强分兵奖励
            early_phase = time_progress < 0.6
            division_bonus = division_bonus * (1.0 + 1.0 * early_phase.float())
            shaped_rewards = shaped_rewards + division_bonus
        
        # ========== 4. 拖延拦截奖励（少量单位在敌方军队附近）==========
        if states.shape[-1] >= 2 and next_states.shape[-1] >= 2:
            pos_states = states[:, :, :2]
            center = pos_states.mean(dim=-1, keepdim=True)
            distances_to_center = th.norm(pos_states.unsqueeze(-2) - center, dim=-1)
            
            mean_distance = distances_to_center.mean(dim=-1, keepdim=True)
            std_distance = distances_to_center.std(dim=-1, keepdim=True)
            is_delaying = distances_to_center > (mean_distance + std_distance * 0.6)  # 调整阈值
            
            n_delaying = is_delaying.sum(dim=-1, keepdim=True).float()
            
            # 拖延拦截奖励：少量单位（2-5个）远离主队时给予奖励
            delay_bonus = self.dhls_delay_intercept_weight * 0.8 * th.where(
                (n_delaying >= 2) & (n_delaying <= 5),  # 调整范围（2-5个）
                th.ones_like(n_delaying),
                th.zeros_like(n_delaying)
            )
            delay_bonus = delay_bonus.expand(batch_size, seq_len, n_agents)
            
            # 前期到中期（前70%）强烈增强拖延奖励
            early_mid_phase = time_progress < 0.7
            delay_bonus = delay_bonus * (1.0 + 2.0 * early_mid_phase.float())  # 增强200%
            shaped_rewards = shaped_rewards + delay_bonus
        
        # ========== 5. 基地攻击奖励（检测对敌方基地的攻击）==========
        if seq_len > 1:
            reward_change = rewards[:, 1:, :] - rewards[:, :-1, :]
            reward_change = th.cat([reward_change, th.zeros_like(reward_change[:, :1, :])], dim=1)
            
            # 基地攻击奖励：获得正奖励时给予额外奖励
            base_attack_bonus = self.dhls_base_attack_weight * 1.0 * th.where(
                reward_change > 0,
                th.clamp(reward_change, 0, 1) + 0.8,  # 增加基础奖励（从0.5增加到0.8）
                th.zeros_like(reward_change)
            )
            
            # 后期（后50%）强烈增强基地攻击奖励
            late_phase = time_progress >= 0.5
            base_attack_bonus = base_attack_bonus * (1.0 + 2.5 * late_phase.float())  # 增强250%（从200%增加到250%）
            shaped_rewards = shaped_rewards + base_attack_bonus
        
        # ========== 6. 时间差奖励（在拖延期间对基地的攻击给予额外奖励）==========
        if seq_len > 1 and states.shape[-1] >= 2:
            pos_states = states[:, :, :2]
            center = pos_states.mean(dim=-1, keepdim=True)
            distances_to_center = th.norm(pos_states.unsqueeze(-2) - center, dim=-1)
            mean_distance = distances_to_center.mean(dim=-1, keepdim=True)
            std_distance = distances_to_center.std(dim=-1, keepdim=True)
            is_delaying = distances_to_center > (mean_distance + std_distance * 0.6)
            
            reward_change = rewards[:, 1:, :] - rewards[:, :-1, :]
            reward_change = th.cat([reward_change, th.zeros_like(reward_change[:, :1, :])], dim=1)
            
            # 时间差奖励：同时有单位在拖延和攻击基地
            time_window_bonus = self.dhls_time_window_weight * 1.0 * th.where(
                is_delaying & (reward_change > 0),
                th.ones_like(reward_change),
                th.zeros_like(reward_change)
            )
            
            # 中期到后期（40%-100%）强烈增强时间差奖励
            mid_late_phase = time_progress >= 0.4
            time_window_bonus = time_window_bonus * (1.0 + 2.0 * mid_late_phase.float())  # 增强200%（从150%增加到200%）
            shaped_rewards = shaped_rewards + time_window_bonus
        
        # ========== 7. 战术协调奖励（同时进行拖延和基地攻击的协调行为）==========
        if seq_len > 1 and states.shape[-1] >= 2:
            pos_states = states[:, :, :2]
            center = pos_states.mean(dim=-1, keepdim=True)
            distances_to_center = th.norm(pos_states.unsqueeze(-2) - center, dim=-1)
            mean_distance = distances_to_center.mean(dim=-1, keepdim=True)
            std_distance = distances_to_center.std(dim=-1, keepdim=True)
            is_delaying = distances_to_center > (mean_distance + std_distance * 0.6)
            is_main_force = ~is_delaying
            
            reward_change = rewards[:, 1:, :] - rewards[:, :-1, :]
            reward_change = th.cat([reward_change, th.zeros_like(reward_change[:, :1, :])], dim=1)
            
            main_force_attacking = is_main_force & (reward_change > 0)
            has_delaying = is_delaying.any(dim=-1, keepdim=True)
            
            # 战术协调奖励：同时有单位在拖延和主队在攻击基地
            coordination_bonus = self.dhls_tactical_coordination_weight * 1.2 * th.where(
                main_force_attacking & has_delaying.expand_as(main_force_attacking),
                th.ones_like(reward_change),
                th.zeros_like(reward_change)
            )
            
            # 中期到后期（50%-100%）强烈增强协调奖励
            mid_late_phase = time_progress >= 0.5
            coordination_bonus = coordination_bonus * (1.0 + 2.2 * mid_late_phase.float())  # 增强220%（从180%增加到220%）
            shaped_rewards = shaped_rewards + coordination_bonus
        
        # ========== 8. 敌人伤害奖励（大幅放大）==========
        damage_bonus = th.where(
            rewards > 0,
            rewards * 2.0,  # 大幅增加伤害奖励放大（从1.5增加到2.0）
            th.zeros_like(rewards)
        )
        shaped_rewards = shaped_rewards + damage_bonus
        
        return shaped_rewards
    
    def _apply_large_scale_coordination_shaping(self, rewards, states, next_states, batch=None, t_env=0):
        """
        yqgz (欲擒故纵) - 重新设计的奖励塑形函数
        战术核心：先用少量的部队送死引诱敌人前进，然后大量的部队趁距离接近消灭敌人
        24 vs 8，150步，Z vs T，我方24个zergling，敌方8个（marine + siegeTank）
        
        新设计思路（更有效）：
        1. 强烈生存奖励（每步都有，确保存活）
        2. 引诱行为奖励（少量单位主动接近敌人，通过位置变化检测）- 改进
        3. 敌人接近奖励（检测敌人距离接近，通过状态变化推断）
        4. 集中攻击奖励（大量单位同时攻击，通过奖励变化和单位数量推断）- 改进
        5. 距离优势奖励（敌人接近时集中攻击，给予额外奖励）
        6. 战术协调奖励（引诱+集中攻击的协调行为）
        7. 获胜检测奖励（大幅奖励获胜）
        8. 敌人伤害奖励（放大伤害奖励）
        """
        shaped_rewards = rewards.clone()
        batch_size, seq_len, n_agents = rewards.shape
        
        # 时间进度
        time_progress = th.linspace(0, 1, seq_len, device=rewards.device).view(1, -1, 1)
        time_progress = time_progress.expand(batch_size, seq_len, n_agents)
        
        # ========== 0. 获胜检测奖励（最重要，大幅奖励获胜）==========
        if batch is not None:
            terminated = batch.get("terminated", None)
            if terminated is not None:
                episode_end_reward = rewards[:, -1:, :]
                win_bonus = th.zeros_like(rewards)
                if episode_end_reward.shape[1] > 0:
                    win_bonus[:, -1:, :] = th.where(
                        episode_end_reward > 0,
                        th.ones_like(episode_end_reward) * 20.0,  # 大幅增加获胜奖励（从10增加到20）
                        th.zeros_like(episode_end_reward)
                    )
                shaped_rewards = shaped_rewards + win_bonus
        
        # ========== 1. 强烈生存奖励（核心，每步都有）==========
        survival_bonus = th.ones_like(rewards) * 0.6  # 增加基础生存奖励（从0.5增加到0.6）
        shaped_rewards = shaped_rewards + survival_bonus
        
        survival_time_bonus = 0.6 * time_progress  # 增加时间奖励（从0.5增加到0.6）
        shaped_rewards = shaped_rewards + survival_time_bonus
        
        # ========== 2. 引诱行为奖励（改进：检测少量单位主动接近敌人）==========
        if states.shape[-1] >= 2 and next_states.shape[-1] >= 2:
            pos_states = states[:, :, :2]
            pos_next_states = next_states[:, :, :2]
            
            # 计算主队中心
            center = pos_states.mean(dim=-1, keepdim=True)
            distances_to_center = th.norm(pos_states.unsqueeze(-2) - center, dim=-1)
            
            # 识别远离主队的单位（用于引诱）
            mean_distance = distances_to_center.mean(dim=-1, keepdim=True)
            std_distance = distances_to_center.std(dim=-1, keepdim=True)
            is_sacrificing = distances_to_center > (mean_distance + std_distance * 0.7)  # 调整阈值
            
            # 计算引诱单位的数量（应该是少量，2-8个）
            n_sacrificing = is_sacrificing.sum(dim=-1, keepdim=True).float()
            
            # 检测引诱单位的位置变化（主动接近敌人）
            pos_change = th.norm(pos_next_states - pos_states, dim=-1, keepdim=False)  # [batch_size, seq_len]
            pos_change = pos_change.unsqueeze(-1)  # [batch_size, seq_len, 1]
            pos_change_max = pos_change.max() if pos_change.numel() > 0 else 1.0
            
            # 引诱行为奖励：少量单位（2-8个）远离主队且位置变化大（主动接近敌人）
            is_luring = (n_sacrificing >= 2) & (n_sacrificing <= 8) & (pos_change > pos_change_max * 0.3)
            sacrifice_bonus = self.yqgz_sacrifice_lure_weight * 0.9 * th.where(
                is_luring,
                th.ones_like(n_sacrificing),
                th.zeros_like(n_sacrificing)
            )
            sacrifice_bonus = sacrifice_bonus.expand(batch_size, seq_len, n_agents)
            
            # 前期（前60%）强烈增强引诱奖励
            early_phase = time_progress < 0.6
            sacrifice_bonus = sacrifice_bonus * (1.0 + 2.5 * early_phase.float())  # 增强250%（从200%增加到250%）
            shaped_rewards = shaped_rewards + sacrifice_bonus
        
        # ========== 3. 敌人接近奖励（检测敌人距离接近）==========
        if states.shape[-1] >= 2 and next_states.shape[-1] >= 2:
            # 通过状态变化推断：如果状态变化大，可能是敌人接近了
            state_change = th.norm(next_states - states, dim=-1, keepdim=False)
            state_change = state_change.unsqueeze(-1)
            state_change_max = state_change.max() if state_change.numel() > 0 else 1.0
            
            # 敌人接近奖励：状态变化大时给予奖励
            enemy_advance_bonus = self.yqgz_enemy_advance_weight * 0.3 * th.clamp(
                state_change / (state_change_max + 1e-6), 0, 1
            )
            enemy_advance_bonus = enemy_advance_bonus.expand(batch_size, seq_len, n_agents)
            
            # 前期到中期（0-70%）增强敌人接近奖励
            early_mid_phase = time_progress < 0.7
            enemy_advance_bonus = enemy_advance_bonus * (1.0 + 1.0 * early_mid_phase.float())  # 增强100%（从60%增加到100%）
            shaped_rewards = shaped_rewards + enemy_advance_bonus
        
        # ========== 4. 集中攻击奖励（改进：检测大量单位同时攻击）==========
        if seq_len > 1 and states.shape[-1] >= 2:
            pos_states = states[:, :, :2]
            center = pos_states.mean(dim=-1, keepdim=True)
            distances_to_center = th.norm(pos_states.unsqueeze(-2) - center, dim=-1)
            mean_distance = distances_to_center.mean(dim=-1, keepdim=True)
            std_distance = distances_to_center.std(dim=-1, keepdim=True)
            is_sacrificing = distances_to_center > (mean_distance + std_distance * 0.7)
            is_main_force = ~is_sacrificing  # 主队单位
            
            reward_change = rewards[:, 1:, :] - rewards[:, :-1, :]
            reward_change = th.cat([reward_change, th.zeros_like(reward_change[:, :1, :])], dim=1)
            
            # 主队攻击奖励：主队单位获得正奖励时给予奖励
            main_force_attacking = is_main_force & (reward_change > 0)
            n_main_force_attacking = main_force_attacking.sum(dim=-1, keepdim=True).float()
            n_main_force = (~is_sacrificing).sum(dim=-1, keepdim=True).float()
            
            # 集中攻击奖励：大量主队单位（超过70%）同时攻击时给予额外奖励 - 更精确
            has_concentrated_attack = (n_main_force_attacking > n_main_force * 0.7)  # 超过70%的主队在攻击（从60%增加到70%）
            main_force_attack_bonus = self.yqgz_main_force_attack_weight * 1.5 * th.where(
                main_force_attacking & has_concentrated_attack.expand_as(main_force_attacking),
                th.clamp(reward_change, 0, 2) + 1.2,  # 增加基础奖励（从0.8增加到1.2）
                th.where(
                    main_force_attacking,
                    th.clamp(reward_change, 0, 1) + 0.5,  # 少量攻击也有奖励（从0.3增加到0.5）
                    th.zeros_like(reward_change)
                )
            )
            
            # 后期（后40%）强烈增强集中攻击奖励
            late_phase = time_progress >= 0.6
            main_force_attack_bonus = main_force_attack_bonus * (1.0 + 4.0 * late_phase.float())  # 增强400%（从300%增加到400%）
            shaped_rewards = shaped_rewards + main_force_attack_bonus
        
        # ========== 5. 距离优势奖励（敌人接近时集中攻击）==========
        if seq_len > 1 and states.shape[-1] >= 2:
            pos_states = states[:, :, :2]
            center = pos_states.mean(dim=-1, keepdim=True)
            distances_to_center = th.norm(pos_states.unsqueeze(-2) - center, dim=-1)
            mean_distance = distances_to_center.mean(dim=-1, keepdim=True)
            std_distance = distances_to_center.std(dim=-1, keepdim=True)
            is_sacrificing = distances_to_center > (mean_distance + std_distance * 0.7)
            is_main_force = ~is_sacrificing
            
            reward_change = rewards[:, 1:, :] - rewards[:, :-1, :]
            reward_change = th.cat([reward_change, th.zeros_like(reward_change[:, :1, :])], dim=1)
            
            has_sacrificing = is_sacrificing.any(dim=-1, keepdim=True)
            main_force_attacking = is_main_force & (reward_change > 0)
            
            # 距离优势奖励：同时有单位在引诱和主队在集中攻击
            distance_bonus = self.yqgz_distance_advantage_weight * 1.1 * th.where(
                main_force_attacking & has_sacrificing.expand_as(main_force_attacking),
                th.ones_like(reward_change),
                th.zeros_like(reward_change)
            )
            
            # 中期到后期（40%-100%）强烈增强距离优势奖励
            mid_late_phase = time_progress >= 0.4
            distance_bonus = distance_bonus * (1.0 + 2.2 * mid_late_phase.float())  # 增强220%（从180%增加到220%）
            shaped_rewards = shaped_rewards + distance_bonus
        
        # ========== 6. 战术协调奖励（引诱+集中攻击的协调行为）==========
        if seq_len > 1 and states.shape[-1] >= 2:
            pos_states = states[:, :, :2]
            center = pos_states.mean(dim=-1, keepdim=True)
            distances_to_center = th.norm(pos_states.unsqueeze(-2) - center, dim=-1)
            mean_distance = distances_to_center.mean(dim=-1, keepdim=True)
            std_distance = distances_to_center.std(dim=-1, keepdim=True)
            is_sacrificing = distances_to_center > (mean_distance + std_distance * 0.7)
            is_main_force = ~is_sacrificing
            
            reward_change = rewards[:, 1:, :] - rewards[:, :-1, :]
            reward_change = th.cat([reward_change, th.zeros_like(reward_change[:, :1, :])], dim=1)
            
            # 少量单位在引诱（2-8个）
            n_sacrificing = is_sacrificing.sum(dim=-1, keepdim=True).float()
            has_few_sacrificing = (n_sacrificing >= 2) & (n_sacrificing <= 8)
            
            # 大量主队在集中攻击（超过60%的主队单位获得正奖励）
            main_force_attacking = is_main_force & (reward_change > 0)
            n_main_force_attacking = main_force_attacking.sum(dim=-1, keepdim=True).float()
            n_main_force = (~is_sacrificing).sum(dim=-1, keepdim=True).float()
            has_many_attacking = (n_main_force_attacking > n_main_force * 0.6)  # 超过60%的主队在攻击
            
            # 战术协调奖励：少量单位引诱 + 大量主队集中攻击
            coordination_bonus = self.yqgz_tactical_coordination_weight * 1.2 * th.where(
                has_few_sacrificing.expand_as(main_force_attacking) & has_many_attacking.expand_as(main_force_attacking),
                th.ones_like(reward_change),
                th.zeros_like(reward_change)
            )
            
            # 中期到后期（50%-100%）强烈增强协调奖励
            mid_late_phase = time_progress >= 0.5
            coordination_bonus = coordination_bonus * (1.0 + 2.5 * mid_late_phase.float())  # 增强250%（从200%增加到250%）
            shaped_rewards = shaped_rewards + coordination_bonus
        
        # ========== 7. 敌人伤害奖励（大幅放大）==========
        damage_bonus = th.where(
            rewards > 0,
            rewards * 2.2,  # 大幅增加伤害奖励放大（从1.8增加到2.2）
            th.zeros_like(rewards)
        )
        shaped_rewards = shaped_rewards + damage_bonus
        
        return shaped_rewards
    
    def _apply_long_term_planning_shaping(self, rewards, states, next_states, batch=None, t_env=0):
        """
        tlhz (偷梁换柱) - 孵化巢暴露+集中攻击策略奖励塑形
        战术核心：把工蜂做成孵化巢暴露在光子炮台下，在被击碎那一刻将所有兵力全部集中攻击光子炮台
        4 vs 2，300步，需要强烈的建造、暴露、被击碎检测、集中攻击奖励
        
        重新设计思路（参考DHLS和YQGZ的优化经验）：
        1. 强烈生存奖励（每步都有，确保存活）- 大幅增强
        2. 建造孵化巢奖励（工蜂建造孵化巢）- 更精确检测，大幅增强
        3. 暴露奖励（孵化巢在光子炮台附近/攻击范围内）- 更精确检测，大幅增强
        4. 被击碎检测（孵化巢血量大幅减少，检测被击碎的那一刻）- 更精确检测，大幅增强
        5. 集中攻击奖励（所有兵力集中攻击光子炮台，超过75%单位同时攻击）- 更精确检测，大幅增强
        6. 时间窗口奖励（被击碎那一刻集中攻击）- 更精确检测，大幅增强
        7. 敌方单位消灭奖励（检测敌方单位数量减少）- 新增
        8. 训练奖励（训练单位，为集中攻击做准备）- 大幅增强
        9. 获胜检测奖励（检测episode结束时的获胜状态）- 大幅增强到20.0
        """
        shaped_rewards = rewards.clone()
        batch_size, seq_len, n_agents = rewards.shape
        
        # 计算时间进度（0-1）
        time_progress = th.linspace(0, 1, seq_len, device=rewards.device).view(1, -1, 1)
        time_progress = time_progress.expand(batch_size, seq_len, n_agents)
        
        # ========== 0. 获胜检测奖励（最重要，大幅奖励获胜）==========
        if batch is not None:
            terminated = batch.get("terminated", None)
            if terminated is not None:
                episode_end_reward = rewards[:, -1:, :]
                win_bonus = th.zeros_like(rewards)
                if episode_end_reward.shape[1] > 0:
                    win_bonus[:, -1:, :] = th.where(
                        episode_end_reward > 0,
                        th.ones_like(episode_end_reward) * 20.0,  # 大幅增加到20.0（从10.0增加）
                        th.zeros_like(episode_end_reward)
                    )
                shaped_rewards = shaped_rewards + win_bonus
        
        # ========== 1. 强烈生存奖励（核心，每步都有）- 大幅增强 ==========
        survival_bonus = th.ones_like(rewards) * self.survival_reward_weight * 0.15  # 使用配置权重
        shaped_rewards = shaped_rewards + survival_bonus
        
        # 存活时间奖励：越接近结束还存活，奖励越高
        survival_time_bonus = self.survival_reward_weight * 0.1 * time_progress
        shaped_rewards = shaped_rewards + survival_time_bonus
        
        # ========== 2. 建造孵化巢奖励（前期核心）- 更精确检测 ==========
        state_change = th.abs(next_states - states).sum(dim=-1, keepdim=False)
        state_change = state_change.unsqueeze(-1)
        state_change_max = state_change.max() if state_change.numel() > 0 else 1.0
        
        # 更精确的建造检测：大幅状态变化（可能是建造孵化巢完成）
        build_threshold = state_change_max * 0.25  # 降低阈值，更敏感
        build_detected = th.where(
            state_change > build_threshold,
            th.ones_like(state_change),
            th.zeros_like(state_change)
        )
        build_bonus = self.build_hatchery_weight * 0.6 * build_detected
        build_bonus = build_bonus.expand(batch_size, seq_len, n_agents)
        
        # 前期（前30%）强烈建造奖励
        early_phase = time_progress < 0.3
        build_bonus = build_bonus * (1.0 + 2.5 * early_phase.float())  # 前期增强250%
        shaped_rewards = shaped_rewards + build_bonus
        
        # ========== 3. 暴露奖励（孵化巢在光子炮台附近）- 更精确检测 ==========
        if states.shape[-1] >= 2:
            pos_states = states[:, :, :2]
            
            # 计算单位到地图中心的距离（假设敌人在中心附近）
            distances_to_center = th.norm(pos_states, dim=-1, keepdim=True)
            distance_max = distances_to_center.max() if distances_to_center.numel() > 0 else 1.0
            
            # 更精确的暴露检测：单位在中心附近（可能是暴露在光子炮台下）
            expose_bonus = self.expose_weight * 0.4 * (1.0 - th.clamp(
                distances_to_center / (distance_max + 1e-6), 0, 1
            ))
            expose_bonus = expose_bonus.expand(batch_size, seq_len, n_agents)
            
            # 前期到中期（10%-50%）增强暴露奖励
            early_mid_phase = (time_progress >= 0.1) & (time_progress < 0.5)
            expose_bonus = expose_bonus * (1.0 + 1.5 * early_mid_phase.float())  # 增强150%
            shaped_rewards = shaped_rewards + expose_bonus
        
        # ========== 4. 被击碎检测（孵化巢血量大幅减少）- 更精确检测 ==========
        if seq_len > 1 and states.shape[-1] > 0:
            state_norm = th.norm(states, dim=-1, keepdim=False)
            state_norm = state_norm.unsqueeze(-1)
            state_norm_prev = state_norm[:, :-1, :]
            state_norm_prev = th.cat([state_norm_prev[:, :1, :], state_norm_prev], dim=1)
            state_norm_max = state_norm.max() if state_norm.numel() > 0 else 1.0
            state_norm_decrease = (state_norm_prev - state_norm) / (state_norm_max + 1e-6)
            
            # 更精确的被击碎检测：状态大幅减少（超过15%）
            is_shattered = state_norm_decrease > 0.15  # 提高阈值，更精确
            shatter_bonus = self.shatter_detection_weight * 0.8 * th.where(
                is_shattered,
                th.ones_like(state_norm_decrease),
                th.zeros_like(state_norm_decrease)
            )
            shatter_bonus = shatter_bonus.expand(batch_size, seq_len, n_agents)
            
            # 中期到后期（30%-80%）增强被击碎检测奖励
            mid_late_phase = (time_progress >= 0.3) & (time_progress < 0.8)
            shatter_bonus = shatter_bonus * (1.0 + 2.0 * mid_late_phase.float())  # 增强200%
            shaped_rewards = shaped_rewards + shatter_bonus
        
        # ========== 5. 集中攻击奖励（所有兵力集中攻击光子炮台）- 更精确检测 ==========
        if states.shape[-1] >= 2 and seq_len > 1:
            pos_states = states[:, :, :2]
            
            # 计算单位间距离的方差（方差小说明集中）
            pos_std = th.std(pos_states, dim=-1, keepdim=False)
            pos_std = pos_std.unsqueeze(-1)
            pos_std_max = pos_std.max() if pos_std.numel() > 0 else 1.0
            concentration = 1.0 - th.clamp(pos_std / (pos_std_max + 1e-6), 0, 1)
            
            # 检测攻击行为：奖励变化为正（攻击敌人）
            reward_change = rewards[:, 1:, :] - rewards[:, :-1, :]
            reward_change = th.cat([reward_change, th.zeros_like(reward_change[:, :1, :])], dim=1)
            is_attacking = reward_change > 0.01  # 更精确的阈值
            
            # 更精确的集中攻击检测：超过75%的单位同时攻击且位置集中
            attacking_ratio = is_attacking.float().mean(dim=-1, keepdim=True)  # [batch_size, seq_len, 1]
            is_concentrated_attack = (attacking_ratio > 0.75) & (concentration.expand_as(attacking_ratio) > 0.7)
            
            concentrated_attack_bonus = self.concentrated_attack_weight * 1.2 * th.where(
                is_concentrated_attack.expand_as(is_attacking),
                th.ones_like(is_attacking),
                th.zeros_like(is_attacking)
            )
            
            # 后期（后40%）强烈集中攻击奖励
            late_phase = time_progress >= 0.6
            concentrated_attack_bonus = concentrated_attack_bonus * (1.0 + 3.0 * late_phase.float())  # 后期增强300%
            shaped_rewards = shaped_rewards + concentrated_attack_bonus
        
        # ========== 6. 时间窗口奖励（被击碎那一刻集中攻击）- 更精确检测 ==========
        if seq_len > 1 and states.shape[-1] >= 2:
            pos_states = states[:, :, :2]
            pos_std = th.std(pos_states, dim=-1, keepdim=False)
            pos_std = pos_std.unsqueeze(-1)
            pos_std_max = pos_std.max() if pos_std.numel() > 0 else 1.0
            concentration = 1.0 - th.clamp(pos_std / (pos_std_max + 1e-6), 0, 1)
            
            if states.shape[-1] > 0:
                state_norm = th.norm(states, dim=-1, keepdim=False)
                state_norm = state_norm.unsqueeze(-1)
                state_norm_prev = state_norm[:, :-1, :]
                state_norm_prev = th.cat([state_norm_prev[:, :1, :], state_norm_prev], dim=1)
                state_norm_max = state_norm.max() if state_norm.numel() > 0 else 1.0
                state_norm_decrease = (state_norm_prev - state_norm) / (state_norm_max + 1e-6)
                is_shattered = state_norm_decrease > 0.15  # 更精确的阈值
                
                reward_change = rewards[:, 1:, :] - rewards[:, :-1, :]
                reward_change = th.cat([reward_change, th.zeros_like(reward_change[:, :1, :])], dim=1)
                is_attacking = reward_change > 0.01
                attacking_ratio = is_attacking.float().mean(dim=-1, keepdim=True)
                is_concentrated_attack = (attacking_ratio > 0.75) & (concentration.expand_as(attacking_ratio) > 0.7)
                
                # 时间窗口奖励：被击碎那一刻集中攻击 - 大幅增强
                timing_bonus = self.timing_window_weight * 1.5 * th.where(
                    is_shattered.expand_as(is_attacking) & is_concentrated_attack.expand_as(is_attacking),
                    th.ones_like(reward_change),
                    th.zeros_like(reward_change)
                )
                
                # 中期到后期（40%-90%）强烈增强时间窗口奖励
                mid_late_phase = (time_progress >= 0.4) & (time_progress < 0.9)
                timing_bonus = timing_bonus * (1.0 + 3.5 * mid_late_phase.float())  # 增强350%
                shaped_rewards = shaped_rewards + timing_bonus
        
        # ========== 7. 敌方单位消灭奖励（新增）- 检测敌方单位数量减少 ==========
        if seq_len > 1 and states.shape[-1] > 0:
            # 通过状态范数变化检测敌方单位减少（状态范数可能包含敌方信息）
            state_norm = th.norm(states, dim=-1, keepdim=False)
            state_norm = state_norm.unsqueeze(-1)
            state_norm_prev = state_norm[:, :-1, :]
            state_norm_prev = th.cat([state_norm_prev[:, :1, :], state_norm_prev], dim=1)
            state_norm_max = state_norm.max() if state_norm.numel() > 0 else 1.0
            
            # 检测大额正奖励（可能是消灭敌方单位）
            large_positive_reward = rewards > 0.1  # 大额正奖励阈值
            enemy_kill_bonus = self.shatter_detection_weight * 0.5 * th.where(
                large_positive_reward,
                th.ones_like(rewards),
                th.zeros_like(rewards)
            )
            
            # 后期（后50%）增强敌方单位消灭奖励
            late_phase = time_progress >= 0.5
            enemy_kill_bonus = enemy_kill_bonus * (1.0 + 2.0 * late_phase.float())
            shaped_rewards = shaped_rewards + enemy_kill_bonus
        
        # ========== 8. 训练奖励（训练单位，为集中攻击做准备）- 大幅增强 ==========
        if states.shape[-1] > 0:
            state_norm = th.norm(states, dim=-1, keepdim=False)
            state_norm = state_norm.unsqueeze(-1)
            state_norm_max = state_norm.max() if state_norm.numel() > 0 else 1.0
            
            if seq_len > 1:
                state_norm_prev = state_norm[:, :-1, :]
                state_norm_prev = th.cat([th.zeros_like(state_norm_prev[:, :1, :]), state_norm_prev], dim=1)
                unit_increase = (state_norm - state_norm_prev) / (state_norm_max + 1e-6)
                
                # 训练奖励：单位数量增加
                train_bonus = self.train_weight * 0.4 * th.clamp(unit_increase, 0, 1)
                train_bonus = train_bonus.expand(batch_size, seq_len, n_agents)
                
                # 中期（20%-60%）增强训练奖励
                mid_phase = (time_progress >= 0.2) & (time_progress < 0.6)
                train_bonus = train_bonus * (1.0 + 1.5 * mid_phase.float())  # 增强150%
                shaped_rewards = shaped_rewards + train_bonus
        
        # ========== 9. 敌人伤害奖励（大幅放大）==========
        damage_bonus = th.where(
            rewards > 0,
            rewards * self.damage_reward_factor,  # 使用配置的伤害奖励放大因子
            th.zeros_like(rewards)
        )
        shaped_rewards = shaped_rewards + damage_bonus
        
        return shaped_rewards
    
    def _apply_active_offensive_shaping(self, rewards, states, next_states, batch=None, t_env=0):
        """
        fkwz (反客为主) - 主动进攻奖励塑形（建造+资源管理+单位优势）
        战术核心：从被动转为主动，通过建造、训练、资源管理来获得优势
        5 vs 7，300步，需要建造、训练、资源管理和主动进攻
        
        重新设计思路（参考DHLS和YQGZ的优化经验）：
        1. 强烈生存奖励（每步都有，确保存活）- 大幅增强（当前奖励为负，需要更强）
        2. 前期（0-30%）：强烈建造和训练奖励，建立单位优势 - 更精确检测，大幅增强
        3. 中期（30-70%）：资源管理奖励，确保资源有效使用，继续训练 - 大幅增强
        4. 后期（70-100%）：主动进攻奖励，利用单位优势攻击 - 更精确检测，大幅增强
        5. 单位数量优势奖励（鼓励从5增加到9，超过敌方7）- 更精确检测，大幅增强
        6. 折跃棱镜使用奖励（支持建造和训练）- 大幅增强
        7. 从被动到主动的转换奖励（检测从防守转为进攻）- 更精确检测，大幅增强
        8. 敌方单位消灭奖励（检测敌方单位数量减少）- 新增
        9. 获胜检测奖励（检测episode结束时的获胜状态）- 大幅增强到20.0
        """
        shaped_rewards = rewards.clone()
        batch_size, seq_len, n_agents = rewards.shape
        
        # 计算时间进度（0-1）
        time_progress = th.linspace(0, 1, seq_len, device=rewards.device).view(1, -1, 1)
        time_progress = time_progress.expand(batch_size, seq_len, n_agents)
        
        # ========== 0. 获胜检测奖励（最重要，大幅奖励获胜）==========
        if batch is not None:
            terminated = batch.get("terminated", None)
            if terminated is not None:
                episode_end_reward = rewards[:, -1:, :]
                win_bonus = th.zeros_like(rewards)
                if episode_end_reward.shape[1] > 0:
                    win_bonus[:, -1:, :] = th.where(
                        episode_end_reward > 0,
                        th.ones_like(episode_end_reward) * 20.0,  # 大幅增加到20.0（从10.0增加）
                        th.zeros_like(episode_end_reward)
                    )
                shaped_rewards = shaped_rewards + win_bonus
        
        # ========== 1. 强烈生存奖励（核心，每步都有）- 大幅增强（当前奖励为负）==========
        # 使用配置权重，但增加基础值（因为当前奖励为负）
        survival_bonus = th.ones_like(rewards) * 0.8  # 大幅增加基础生存奖励（从0.5增加到0.8）
        shaped_rewards = shaped_rewards + survival_bonus
        
        survival_time_bonus = 0.6 * time_progress  # 大幅增加存活时间奖励（从0.5增加到0.6）
        shaped_rewards = shaped_rewards + survival_time_bonus
        
        # ========== 2. 建造奖励（前期强烈，0-30%）- 更精确检测 ==========
        state_change = th.abs(next_states - states).sum(dim=-1, keepdim=False)
        state_change = state_change.unsqueeze(-1)
        state_change_max = state_change.max() if state_change.numel() > 0 else 1.0
        
        # 更精确的建造检测：状态大幅变化（可能是建造完成）
        build_threshold = state_change_max * 0.25  # 降低阈值，更敏感
        build_detected = th.where(
            state_change > build_threshold,
            th.ones_like(state_change),
            th.zeros_like(state_change)
        )
        
        build_bonus = self.warpgate_train_reward_weight * 0.7 * build_detected
        build_bonus = build_bonus.expand(batch_size, seq_len, n_agents)
        
        # 前期（0-30%）强烈奖励建造
        early_phase = time_progress < 0.3
        build_bonus = build_bonus * (1.0 + 3.0 * early_phase.float())  # 前期增强300%
        shaped_rewards = shaped_rewards + build_bonus
        
        # ========== 3. 训练奖励（前期到中期强烈，0-60%）- 更精确检测 ==========
        train_bonus = th.zeros_like(rewards)
        if states.shape[-1] > 0:
            state_norm = th.norm(states, dim=-1, keepdim=False)
            state_norm = state_norm.unsqueeze(-1)
            state_norm_max = state_norm.max() if state_norm.numel() > 0 else 1.0
            
            if seq_len > 1:
                state_norm_prev = state_norm[:, :-1, :]
                state_norm_prev = th.cat([th.zeros_like(state_norm_prev[:, :1, :]), state_norm_prev], dim=1)
                unit_increase = (state_norm - state_norm_prev) / (state_norm_max + 1e-6)
                
                # 训练奖励：单位数量增加 - 大幅增强
                train_bonus = self.warpgate_train_reward_weight * 0.8 * th.clamp(unit_increase, 0, 1)
                train_bonus = train_bonus.expand(batch_size, seq_len, n_agents)
                
                # 前期到中期（0-60%）强烈奖励训练
                train_phase = time_progress < 0.6
                train_bonus = train_bonus * (1.0 + 2.5 * train_phase.float())  # 前期到中期增强250%
                shaped_rewards = shaped_rewards + train_bonus
        
        # ========== 4. 单位数量优势奖励（核心：从5增加到9，超过敌方7）- 更精确检测 ==========
        if states.shape[-1] > 0:
            state_norm = th.norm(states, dim=-1, keepdim=False)
            state_norm = state_norm.unsqueeze(-1)
            state_norm_max = state_norm.max() if state_norm.numel() > 0 else 1.0
            
            # 更精确的单位数量优势检测：状态范数增加表示单位数量增加
            unit_advantage_ratio = th.clamp(state_norm / (state_norm_max + 1e-6), 0, 1)
            # 当单位数量优势明显时给予奖励
            unit_advantage_bonus = self.unit_advantage_weight * 0.4 * unit_advantage_ratio
            unit_advantage_bonus = unit_advantage_bonus.expand(batch_size, seq_len, n_agents)
            
            # 中期到后期（30-100%）奖励单位优势
            advantage_phase = time_progress >= 0.3
            unit_advantage_bonus = unit_advantage_bonus * (1.0 + 2.0 * advantage_phase.float())  # 增强200%
            shaped_rewards = shaped_rewards + unit_advantage_bonus
        
        # ========== 5. 资源管理奖励（中期强烈，30-70%）==========
        optimal_change = state_change_max * 0.25
        efficiency_score = 1.0 - th.abs(state_change - optimal_change) / (state_change_max + 1e-6)
        resource_bonus = self.resource_management_weight * 0.25 * th.clamp(efficiency_score, 0, 1)
        resource_bonus = resource_bonus.expand(batch_size, seq_len, n_agents)
        
        # 中期（30-70%）强烈奖励资源管理
        mid_phase = (time_progress >= 0.3) & (time_progress < 0.7)
        resource_bonus = resource_bonus * (1.0 + 1.5 * mid_phase.float())  # 增强150%
        shaped_rewards = shaped_rewards + resource_bonus
        
        # ========== 6. 主动进攻奖励（后期强烈，70-100%）- 更精确检测，大幅增强 ==========
        # 检测对敌人造成伤害（正奖励）
        damage_bonus = th.where(
            rewards > 0,
            rewards * self.damage_reward_factor,  # 使用配置的伤害奖励放大因子
            th.zeros_like(rewards)
        )
        
        # 后期（70-100%）强烈奖励主动进攻
        late_phase = time_progress >= 0.7
        damage_bonus = damage_bonus * (1.0 + 3.0 * late_phase.float())  # 后期增强300%
        shaped_rewards = shaped_rewards + damage_bonus
        
        # ========== 7. 折跃棱镜使用奖励（支持建造和训练）- 大幅增强 ==========
        if states.shape[-1] >= 2 and next_states.shape[-1] >= 2:
            pos_states = states[:, :, :2]
            pos_next_states = next_states[:, :, :2]
            position_change = th.norm(pos_next_states - pos_states, dim=-1, keepdim=False)
            position_change = position_change.unsqueeze(-1)
            position_change_max = position_change.max() if position_change.numel() > 0 else 1.0
            
            # 更精确的折跃棱镜移动检测：位置变化
            prism_threshold = position_change_max * 0.15  # 降低阈值，更敏感
            prism_detected = th.where(
                position_change > prism_threshold,
                th.ones_like(position_change),
                th.zeros_like(position_change)
            )
            
            prism_bonus = self.warp_prism_reward_weight * 0.4 * prism_detected
            prism_bonus = prism_bonus.expand(batch_size, seq_len, n_agents)
            
            # 前期到中期（0-60%）奖励折跃棱镜使用
            prism_phase = time_progress < 0.6
            prism_bonus = prism_bonus * (1.0 + 1.5 * prism_phase.float())  # 增强150%
            shaped_rewards = shaped_rewards + prism_bonus
        
        # ========== 8. 从被动到主动的转换奖励（检测从防守转为进攻）- 更精确检测 ==========
        if seq_len > 1:
            rewards_prev = rewards[:, :-1, :]
            rewards_prev = th.cat([th.zeros_like(rewards_prev[:, :1, :]), rewards_prev], dim=1)
            reward_increase = rewards - rewards_prev
            
            # 更精确的转换检测：奖励从负转为正，或大幅增加
            is_transition = (rewards_prev < 0) & (rewards > 0) | (reward_increase > 0.1)
            transition_bonus = self.passive_to_active_conversion_weight * 0.4 * th.where(
                is_transition,
                th.ones_like(rewards),
                th.zeros_like(rewards)
            )
            
            # 中期到后期（40-90%）奖励从被动转为主动
            transition_phase = (time_progress >= 0.4) & (time_progress < 0.9)
            transition_bonus = transition_bonus * (1.0 + 2.0 * transition_phase.float())  # 增强200%
            shaped_rewards = shaped_rewards + transition_bonus
        
        # ========== 9. 敌方单位消灭奖励（新增）- 检测敌方单位数量减少 ==========
        if seq_len > 1:
            # 检测大额正奖励（可能是消灭敌方单位）
            large_positive_reward = rewards > 0.1  # 大额正奖励阈值
            enemy_kill_bonus = self.active_offensive_weight * 0.5 * th.where(
                large_positive_reward,
                th.ones_like(rewards),
                th.zeros_like(rewards)
            )
            
            # 后期（后50%）增强敌方单位消灭奖励
            late_phase = time_progress >= 0.5
            enemy_kill_bonus = enemy_kill_bonus * (1.0 + 2.5 * late_phase.float())  # 增强250%
            shaped_rewards = shaped_rewards + enemy_kill_bonus
        
        return shaped_rewards
    
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # 对奖励进行塑形
        if self.reward_shaping_enabled:
            batch_rewards = batch["reward"][:, :-1]
            batch_states = batch["state"][:, :-1]
            batch_next_states = batch["state"][:, 1:]
            
            shaped_rewards = self.shape_reward(
                batch_rewards, 
                batch_states, 
                batch_next_states,
                t_env,
                batch=batch
            )
            
            # 更新batch中的奖励
            batch.data.transition_data["reward"][:, :-1] = shaped_rewards
        
            # 记录奖励塑形信息
            self.log_shaping_info(t_env)
        
        # 调用父类的train方法
        return super(RewardShapingLearner, self).train(batch, t_env, episode_num)
    
    def log_shaping_info(self, t_env):
        """
        记录奖励塑形信息
        """
        if self.reward_shaping_enabled:
            self.logger.log_stat(
                "reward_shaping/weight",
                self.current_shaping_weight,
                t_env
            )

