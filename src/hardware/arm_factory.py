"""Build inert adapters from model/role configuration; never connect on import."""
from collections.abc import Mapping

from contracts.robot_contract import normalize_robot_contract


def normalize_hardware_config(config, *, base_dir=None):
    if not isinstance(config, Mapping) or set(config) - {'type', 'arms'} or not config.get('arms'):
        raise ValueError('Hardware config requires type and arms')
    backend = config.get('type')
    if backend not in ('arx', 'b601'):
        raise ValueError('Hardware type must be arx or b601')
    if not isinstance(config['arms'], Mapping) or set(config['arms']) - {'left', 'right'}:
        raise ValueError('Hardware arms must name left and/or right')
    arms, ports = {}, set()
    for role in ('left', 'right'):
        if role not in config['arms']:
            continue
        arm = dict(config['arms'][role])
        if backend == 'arx':
            allowed = {'model', 'can_interface', 'command_timeout_s', 'max_feedback_age_s'}
            port_key = 'can_interface'
            supported = ('L5', 'Trossen_L5')
        else:
            allowed = {'model', 'serial_port', 'hardware_yaml', 'home_qpos',
                       'command_timeout_s', 'max_feedback_age_s'}
            port_key = 'serial_port'
            supported = ('B601_DM',)
        if set(arm) - allowed:
            raise ValueError(f'Unknown hardware options for {role}')
        model = arm.get('model')
        if model not in supported:
            raise ValueError(f'{role}: explicitly choose one of {supported}')
        if backend == 'b601':
            if not isinstance(arm.get('hardware_yaml'), str) or not arm['hardware_yaml'].strip():
                raise ValueError(f'{role} needs a DM hardware_yaml')
            home = arm.get('home_qpos')
            if not isinstance(home, (list, tuple)) or len(home) != 7:
                raise ValueError(f'{role} home_qpos must be 7 joint radians')
        port = arm.get(port_key)
        if not isinstance(port, str) or not port.strip() or port in ports:
            raise ValueError('Each arm needs a distinct port')
        ports.add(port)
        arms[role] = arm
    return {'type': backend, 'arms': arms}


def hardware_robot_contract(config):
    return normalize_robot_contract({'robot_contract_version': 1,
        'arms': {role: {'model': arm['model']} for role, arm in config['arms'].items()}})


def build_arms(config, *, base_dir=None):
    config = normalize_hardware_config(config, base_dir=base_dir)
    # Validate every arm's configuration before connecting any hardware.
    if config['type'] == 'arx':
        from hardware.backends.arx import ARXAdapter as Adapter
    else:
        from hardware.backends.b601 import B601Adapter as Adapter
    return {role: Adapter(role=role, **arm) for role, arm in config['arms'].items()}
