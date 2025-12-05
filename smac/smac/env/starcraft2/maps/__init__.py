from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smac.env.starcraft2.maps import smac_maps


def get_map_params(map_name):
    map_param_registry = smac_maps.get_smac_map_registry()
    
    # If map not found in standard SMAC maps, try tactics maps
    if map_name not in map_param_registry:
        try:
            from smac.env.sc2_tactics.maps import get_map_params as get_tactics_map_params
            return get_tactics_map_params(map_name)
        except (ImportError, KeyError, ValueError):
            pass
    
    # If still not found, raise KeyError
    if map_name not in map_param_registry:
        raise KeyError(f"Map '{map_name}' not found in map registry")
    
    return map_param_registry[map_name]
