from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smac.env.sc2_tactics.maps import get_map_params
from smac.env.sc2_tactics.utils import common_utils
import numpy as np
import enum
from absl import logging
from pysc2.lib import protocol
from s2clientprotocol import common_pb2 as sc_common
from s2clientprotocol import sc2api_pb2 as sc_pb
from s2clientprotocol import raw_pb2 as r_pb
import smac.env.sc2_tactics.sc2_tactics_env as te

# 动作表
actions = {
    "move": 16,
    "attack": 23,
    "stop": 4,
    "BurrowDown": 2095,  # WidowMineBurrow
    "BurrowUp": 2097,    # WidowMineUnburrow
}

class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3

class SC2TacticsPZYYEnv(te.SC2TacticsEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # ==========================================
        # 调试配置 (Debug Config)
        # ==========================================
        self._move_amount = 4   # 增大步长，防止移动看起来像转身
        self.DEBUG_FORCE_MOVE = False # 【重要】初次运行建议设为 True，测试单位能否物理移动
        self.DEBUG_PRINT_IDS = True   # 【重要】初次运行设为 True，查看单位真实ID
        # ==========================================

        self.n_actions += 1
        self.n_actions_no_attack += 1
        
        # 预设ID，稍后会在 reset 中验证
        self.widow_mine_id = 498
        self.widow_mine_burrowed_id = 500
        self.marine_id = 48
        
        print(f"--- PZYY Env Loaded. Move Amount: {self._move_amount} ---")

    def reset(self):
        """重置环境时，顺便打印一次单位ID，确保地图单位是对的"""
        res = super().reset()
        
        if self.DEBUG_PRINT_IDS and self._episode_count == 0:
            print("\n" + "="*30)
            print("【环境自检】正在检查地图单位 ID...")
            found_marine = False
            found_mine = False
            
            for a_id, unit in self.agents.items():
                print(f"Agent {a_id}: Type={unit.unit_type}, Pos=({unit.pos.x:.1f}, {unit.pos.y:.1f})")
                if unit.unit_type == 48: found_marine = True
                if unit.unit_type in [498, 500]: found_mine = True
            
            print("-" * 20)
            if not found_marine:
                print("⚠️ 警告: 没找到 ID=48 的枪兵！你的地图可能使用了变种单位(如战役版枪兵)。请根据上方打印的 Type 修改代码 self.marine_id。")
            if not found_mine:
                print("⚠️ 警告: 没找到 ID=498/500 的寡妇雷！请检查上方 Type。")
            print("="*30 + "\n")
            
        return res
    
    def get_unit_type_id(self, unit, ally):
        """
        覆盖父类方法：手动映射单位 ID 到小的连续整数 (0, 1, 2...)
        解决 ID 跨度过大导致的 IndexError
        """
        # 你的单位 ID 常量
        MARINE_ID = 48
        WIDOW_MINE_ID = 498
        WIDOW_MINE_BURROWED_ID = 500

        if ally:  # 处理我方单位 (Player 1)
            if unit.unit_type == MARINE_ID:
                type_id = 0  # 枪兵映射为 0
            elif unit.unit_type == WIDOW_MINE_ID:
                type_id = 1  # 寡妇雷映射为 1
            elif unit.unit_type == WIDOW_MINE_BURROWED_ID:
                type_id = 2  # 埋地寡妇雷映射为 2
            else:
                # 遇到未知单位，给一个稍微大一点的 ID 或者 0，防止报错
                # 也可以打印一下警告
                type_id = 0 
                if self.debug:
                    print(f"Warning: Unknown ally unit type {unit.unit_type}")
        
        else:  # 处理敌方单位 (Player 2)
            # 敌方是 Zerg: Zergling (105), Overseer (129) 等
            # 你需要根据实际情况映射，或者直接复用父类逻辑(如果敌方ID比较连续)
            # 这里给一个简单的映射示例：
            if unit.unit_type == 105: # Zergling
                type_id = 0
            elif unit.unit_type == 129: # Overseer
                type_id = 1
            else:
                # 如果敌方单位类型很多，或者你不在意敌方具体类型，
                # 可以简单地用 unit.unit_type - min_enemy_id
                # 或者直接返回 0
                type_id = 0
        
        return type_id

    def get_avail_agent_actions(self, agent_id):
        unit = self.get_unit_by_id(agent_id)
        if unit.health <= 0:
            return [1] + [0] * (self.n_actions - 1)

        avail_actions = [0] * self.n_actions
        avail_actions[1] = 1 # Stop always allowed

        # 移动逻辑
        # 注意：这里使用 self.marine_id 变量，如果 ID 不对，请修改 __init__
        if unit.unit_type != self.widow_mine_burrowed_id:
            # 只要不是埋地的寡妇雷，理论上都能动
            if self.can_move(unit, Direction.NORTH): avail_actions[2] = 1
            if self.can_move(unit, Direction.SOUTH): avail_actions[3] = 1
            if self.can_move(unit, Direction.EAST):  avail_actions[4] = 1
            if self.can_move(unit, Direction.WEST):  avail_actions[5] = 1

        # 埋地/出地逻辑
        if unit.unit_type == self.widow_mine_id: 
            avail_actions[6] = 1 # Burrow
        elif unit.unit_type == self.widow_mine_burrowed_id: 
            avail_actions[6] = 1 # Unburrow

        # 攻击逻辑
        shoot_range = self.unit_shoot_range(agent_id)
        # 修正：寡妇雷埋地后也是有射程的 (5)
        if unit.unit_type == self.widow_mine_burrowed_id:
            shoot_range = 5.0
            
        for t_id, t_unit in self.enemies.items():
            if t_unit.health > 0:
                dist = self.distance(unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y)
                if dist <= shoot_range:
                    avail_actions[t_id + self.n_actions_no_attack] = 1
        
        return avail_actions

    def get_agent_action(self, a_id, action):
        unit = self.get_unit_by_id(a_id)
        if unit is None: return None
        tag = unit.tag
        x, y = unit.pos.x, unit.pos.y

        # ==========================================
        # 强力调试模式：无视网络，强行移动
        # ==========================================
        if self.DEBUG_FORCE_MOVE:
            # 强制所有单位向地图中心 (例如 20, 20) 移动
            # 如果这样能动，说明是 RL 没训练好；如果这样都不动，说明地图有问题
            target_pos = sc_common.Point2D(x=x + self._move_amount, y=y) # 向东走
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=target_pos,
                unit_tags=[tag], queue_command=False)
            return sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        # ==========================================

        avail_actions = self.get_avail_agent_actions(a_id)
        # 如果网络选了不可用的动作，这里会报错。
        # 如果报错，说明 obs 或者 mask 逻辑有问题。
        if avail_actions[action] == 0:
             # 如果选了无效动作，为了不崩，强制转为 Stop (仅调试用，正式训练应 assert)
             # assert False, f"Agent {a_id} chose invalid action {action}"
             action = 1 

        if action == 0: return None
        
        if action == 1: # Stop
            cmd = r_pb.ActionRawUnitCommand(ability_id=actions["stop"], unit_tags=[tag], queue_command=False)
            
        elif action == 2: # North
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(x=x, y=y + self._move_amount),
                unit_tags=[tag], queue_command=False)
        elif action == 3: # South
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(x=x, y=y - self._move_amount),
                unit_tags=[tag], queue_command=False)
        elif action == 4: # East
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(x=x + self._move_amount, y=y),
                unit_tags=[tag], queue_command=False)
        elif action == 5: # West
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["move"],
                target_world_space_pos=sc_common.Point2D(x=x - self._move_amount, y=y),
                unit_tags=[tag], queue_command=False)
                
        elif action == 6: # Burrow
            ab_id = actions["BurrowDown"] if unit.unit_type == self.widow_mine_id else actions["BurrowUp"]
            cmd = r_pb.ActionRawUnitCommand(ability_id=ab_id, unit_tags=[tag], queue_command=False)
                
        else: # Attack
            target_id = action - self.n_actions_no_attack
            target_unit = self.enemies[target_id]
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["attack"],
                target_unit_tag=target_unit.tag,
                unit_tags=[tag],
                queue_command=False,
            )

        return sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))

