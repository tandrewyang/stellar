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

actions = {
    "move": 16,
    "attack": 23,
    "stop": 4,
}

class Direction(enum.IntEnum):
    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3

class SC2TacticsLDTJEnv(te.SC2TacticsEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 调试开关
        self.DEBUG_PRINT_IDS = True
        
        # 单位 ID 常量定义
        self.MUTALISK_ID = 108
        self.SPORE_CRAWLER_ID = 98
        self.WIDOW_MINE_ID = 498
        self.SIEGE_TANK_SIEGED_ID = 33
        
        print("----------------------")
        print("You created a LDTJ (Mutalisk vs Mine/Tank) env!")
        print("----------------------")

    def reset(self):
        res = super().reset()
        if self.DEBUG_PRINT_IDS and self._episode_count == 0:
            print("\n" + "="*30)
            print("【环境自检】正在检查 LDTJ 单位 ID...")
            for a_id, unit in self.agents.items():
                print(f"Agent {a_id}: Type={unit.unit_type}, Pos=({unit.pos.x:.1f}, {unit.pos.y:.1f})")
            for e_id, unit in self.enemies.items():
                print(f"Enemy {e_id}: Type={unit.unit_type}")
            print("="*30 + "\n")
        return res

    def get_unit_type_id(self, unit, ally):
        """
        关键函数：手动映射 ID，防止 IndexError
        """
        if ally: # 我方 (Zerg)
            if unit.unit_type == self.MUTALISK_ID: # 异龙 (108)
                return 0
            elif unit.unit_type == self.SPORE_CRAWLER_ID: # 孢子爬虫 (98)
                return 1
        else: # 敌方 (Terran)
            if unit.unit_type == self.WIDOW_MINE_ID: # 寡妇雷 (498)
                return 0
            elif unit.unit_type == self.SIEGE_TANK_SIEGED_ID: # 架起的坦克 (33)
                return 1
        
        # 默认返回 0，防止未知单位报错
        return 0

    def get_avail_agent_actions(self, agent_id):
        """
        限制动作：孢子爬虫不能移动
        """
        unit = self.get_unit_by_id(agent_id)
        if unit.health <= 0:
            return [1] + [0] * (self.n_actions - 1)

        avail_actions = [0] * self.n_actions
        avail_actions[1] = 1 # Stop always allowed

        # 移动逻辑：只有异龙能动
        if unit.unit_type == self.MUTALISK_ID:
            if self.can_move(unit, Direction.NORTH): avail_actions[2] = 1
            if self.can_move(unit, Direction.SOUTH): avail_actions[3] = 1
            if self.can_move(unit, Direction.EAST):  avail_actions[4] = 1
            if self.can_move(unit, Direction.WEST):  avail_actions[5] = 1
        # 孢子爬虫 (ID 98) 是建筑，avail_actions[2-5] 保持为 0

        # 攻击逻辑
        shoot_range = self.unit_shoot_range(agent_id)
        for t_id, t_unit in self.enemies.items():
            if t_unit.health > 0:
                dist = self.distance(unit.pos.x, unit.pos.y, t_unit.pos.x, t_unit.pos.y)
                if dist <= shoot_range:
                    avail_actions[t_id + self.n_actions_no_attack] = 1
        
        return avail_actions

    def get_agent_action(self, a_id, action):
        """执行动作"""
        avail_actions = self.get_avail_agent_actions(a_id)
        # 安全检查：如果神经网络输出了非法动作（比如让爬虫移动），强制变为 Stop
        if avail_actions[action] == 0:
            action = 1 

        unit = self.get_unit_by_id(a_id)
        tag = unit.tag
        x, y = unit.pos.x, unit.pos.y

        if action == 0: return None # No-Op
        
        cmd = None
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
                
        else: # Attack
            target_id = action - self.n_actions_no_attack
            target_unit = self.enemies[target_id]
            cmd = r_pb.ActionRawUnitCommand(
                ability_id=actions["attack"],
                target_unit_tag=target_unit.tag,
                unit_tags=[tag],
                queue_command=False,
            )

        if cmd:
            return sc_pb.Action(action_raw=r_pb.ActionRaw(unit_command=cmd))
        return None

