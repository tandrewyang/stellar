"""
课程学习Learner：逐步增加任务难度
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


class CurriculumLearner(MAXQLearner):
    """
    课程学习：逐步增加任务难度
    """
    def __init__(self, mac, scheme, logger, args):
        super(CurriculumLearner, self).__init__(mac, scheme, logger, args)
        
        # 课程学习参数
        self.curriculum_enabled = getattr(args, 'curriculum_enabled', True)
        self.curriculum_schedule = getattr(args, 'curriculum_schedule', 'linear')
        self.curriculum_start_step = getattr(args, 'curriculum_start_step', 0)
        self.curriculum_end_step = getattr(args, 'curriculum_end_step', 1000000)
        
        # 难度级别（0-1，0最简单，1最难）
        self.current_difficulty = 0.0
        self.min_difficulty = getattr(args, 'curriculum_min_difficulty', 0.0)
        self.max_difficulty = getattr(args, 'curriculum_max_difficulty', 1.0)
        
        # 课程学习指标
        self.episode_rewards = deque(maxlen=100)
        self.episode_wins = deque(maxlen=100)
        
    def update_curriculum_difficulty(self, t_env):
        """
        根据训练进度更新课程难度
        """
        if not self.curriculum_enabled:
            self.current_difficulty = self.max_difficulty
            return
        
        if t_env < self.curriculum_start_step:
            self.current_difficulty = self.min_difficulty
        elif t_env >= self.curriculum_end_step:
            self.current_difficulty = self.max_difficulty
        else:
            # 线性调度
            if self.curriculum_schedule == 'linear':
                progress = (t_env - self.curriculum_start_step) / (
                    self.curriculum_end_step - self.curriculum_start_step
                )
                self.current_difficulty = (
                    self.min_difficulty + 
                    (self.max_difficulty - self.min_difficulty) * progress
                )
            # 自适应调度（基于性能）
            elif self.curriculum_schedule == 'adaptive':
                if len(self.episode_wins) > 10:
                    recent_win_rate = np.mean(list(self.episode_wins)[-10:])
                    if recent_win_rate > 0.8:
                        # 性能好，增加难度
                        self.current_difficulty = min(
                            self.current_difficulty + 0.01,
                            self.max_difficulty
                        )
                    elif recent_win_rate < 0.2:
                        # 性能差，降低难度
                        self.current_difficulty = max(
                            self.current_difficulty - 0.01,
                            self.min_difficulty
                        )
    
    def apply_curriculum_to_batch(self, batch):
        """
        将课程学习应用到batch上（例如：过滤简单样本、调整奖励等）
        """
        if not self.curriculum_enabled:
            return batch
        
        # 根据难度过滤或调整batch
        # 这里可以根据具体需求实现，例如：
        # 1. 只使用难度低于current_difficulty的样本
        # 2. 调整奖励以反映难度
        # 3. 混合不同难度的样本
        
        return batch
    
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # 更新课程难度
        self.update_curriculum_difficulty(t_env)
        
        # 应用课程学习到batch
        batch = self.apply_curriculum_to_batch(batch)
        
        # 记录课程信息
        self.log_curriculum_info(t_env)
        
        # 调用父类的train方法
        return super(CurriculumLearner, self).train(batch, t_env, episode_num)
    
    def log_curriculum_info(self, t_env):
        """
        记录课程学习信息
        """
        if self.curriculum_enabled:
            self.logger.log_stat(
                "curriculum/difficulty", 
                self.current_difficulty, 
                t_env
            )
            self.logger.log_stat(
                "curriculum/progress",
                (t_env - self.curriculum_start_step) / max(
                    self.curriculum_end_step - self.curriculum_start_step, 1
                ),
                t_env
            )

