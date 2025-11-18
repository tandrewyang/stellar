from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smac.env.sc2_tactics.maps import sc2_tactics_maps

# Map name aliases: short names -> full names with _te suffix
MAP_NAME_ALIASES = {
    "adcc": "adcc_te",
    "dhls": "dhls_te",
    "fkwz": "fkwz_te",
    "gmzz": "gmzz_te",
    "jctq": "jctq_te",
    "jdsr": "jdsr_te",
    "sdjx": "sdjx_te",
    "swct": "swct_te",
    "tlhz": "tlhz_te",
    "wwjz": "wwjz_te",
    "wzsy": "wzsy_te",
    "yqgz": "yqgz_te",
}

def get_map_params(map_name):
    map_param_registry = sc2_tactics_maps.get_tactics_map_registry()
    
    # Try to resolve alias first
    actual_map_name = MAP_NAME_ALIASES.get(map_name, map_name)
    
    # Try the actual name, then try with _te suffix if not found
    map_param = map_param_registry.get(actual_map_name, {})
    if not map_param and not map_name.endswith("_te"):
        # Try with _te suffix
        map_param = map_param_registry.get(f"{map_name}_te", {})
    
    map_param = map_param.copy() if map_param else {}
    
    if not map_param:
        available = [k for k in map_param_registry.keys() if map_name in k.lower()][:5]
        raise ValueError(
            f"Map parameters for '{map_name}' not found in map_param_registry. "
            f"Available maps with similar names: {available}"
        )
    return map_param

#map_param_registry = sc2_thirty_six_tactics_maps.get_tactics_map_registry()
#return map_param_registry[map_name]