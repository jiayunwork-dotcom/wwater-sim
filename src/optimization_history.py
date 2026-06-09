"""
优化历史管理模块
提供优化结果的持久化存储、加载和删除功能
"""

import json
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
import numpy as np

from .nsga2_optimizer import (
    OptimizationConfig,
    OptimizationVariable,
    OptimizationObjective,
    Individual,
    OptimizationResult,
    DEFAULT_VARIABLES,
    DEFAULT_OBJECTIVES,
)


HISTORY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "optimization_history")


def _ensure_history_dir():
    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR, exist_ok=True)


def _individual_to_dict(ind: Individual, variables: List[OptimizationVariable]) -> Dict[str, Any]:
    d = {
        'variables': ind.variables.tolist() if isinstance(ind.variables, np.ndarray) else list(ind.variables),
        'objectives': ind.objectives.tolist() if isinstance(ind.objectives, np.ndarray) else list(ind.objectives),
        'constraints': ind.constraints.tolist() if isinstance(ind.constraints, np.ndarray) and len(ind.constraints) > 0 else [],
        'constraint_violation': float(ind.constraint_violation),
        'rank': ind.rank,
        'crowding_distance': float(ind.crowding_distance),
        'is_feasible': ind.is_feasible,
        'converged': ind.converged,
        'effluent_quality': ind.effluent_quality if ind.effluent_quality else {},
    }
    if ind.sludge_result is not None:
        d['sludge_result'] = {
            'daily_sludge_kg': ind.sludge_result.daily_sludge_kg,
            'MLSS_gL': ind.sludge_result.MLSS_gL,
            'total_biomass_kg': ind.sludge_result.total_biomass_kg,
            'XBH_kg': ind.sludge_result.XBH_kg,
            'XBA_kg': ind.sludge_result.XBA_kg,
            'XP_kg': ind.sludge_result.XP_kg,
            'XI_kg': ind.sludge_result.XI_kg,
            'XS_kg': ind.sludge_result.XS_kg,
            'sludge_concentration_mgL': ind.sludge_result.sludge_concentration_mgL,
            'waste_flow_m3d': getattr(ind.sludge_result, 'waste_flow_m3d', 0.0),
        }
    else:
        d['sludge_result'] = None

    if ind.energy_result is not None:
        d['energy_result'] = {
            'total_kwh_d': ind.energy_result.total_kwh_d,
            'unit_kwh_m3': ind.energy_result.unit_kwh_m3,
            'aeration_kwh_d': ind.energy_result.aeration_kwh_d,
            'return_pump_kwh_d': ind.energy_result.return_pump_kwh_d,
            'internal_pump_kwh_d': ind.energy_result.internal_pump_kwh_d,
            'mixing_kwh_d': ind.energy_result.mixing_kwh_d,
            'other_kwh_d': ind.energy_result.other_kwh_d,
        }
    else:
        d['energy_result'] = None

    return d


def _dict_to_individual(d: Dict[str, Any]) -> Individual:
    ind = Individual(
        variables=np.array(d['variables']),
        objectives=np.array(d['objectives']) if d.get('objectives') else np.array([]),
        constraints=np.array(d.get('constraints', [])),
        constraint_violation=d.get('constraint_violation', 0.0),
        rank=d.get('rank', 0),
        crowding_distance=d.get('crowding_distance', 0.0),
        is_feasible=d.get('is_feasible', True),
        converged=d.get('converged', False),
        effluent_quality=d.get('effluent_quality', {}),
    )
    if d.get('sludge_result') is not None:
        from .analysis import SludgeProductionResult
        sr = d['sludge_result']
        ind.sludge_result = SludgeProductionResult(
            daily_sludge_kg=sr['daily_sludge_kg'],
            MLSS_gL=sr['MLSS_gL'],
            total_biomass_kg=sr['total_biomass_kg'],
            XBH_kg=sr['XBH_kg'],
            XBA_kg=sr['XBA_kg'],
            XP_kg=sr['XP_kg'],
            XI_kg=sr['XI_kg'],
            XS_kg=sr['XS_kg'],
            sludge_concentration_mgL=sr['sludge_concentration_mgL'],
            waste_flow_m3d=sr.get('waste_flow_m3d', 0.0),
        )

    if d.get('energy_result') is not None:
        from .analysis import EnergyConsumptionResult
        er = d['energy_result']
        ind.energy_result = EnergyConsumptionResult(
            total_kwh_d=er['total_kwh_d'],
            unit_kwh_m3=er['unit_kwh_m3'],
            aeration_kwh_d=er['aeration_kwh_d'],
            return_pump_kwh_d=er['return_pump_kwh_d'],
            internal_pump_kwh_d=er['internal_pump_kwh_d'],
            mixing_kwh_d=er['mixing_kwh_d'],
            other_kwh_d=er['other_kwh_d'],
        )

    return ind


def _config_to_dict(config: OptimizationConfig) -> Dict[str, Any]:
    return {
        'population_size': config.population_size,
        'max_generations': config.max_generations,
        'crossover_probability': config.crossover_probability,
        'mutation_probability': config.mutation_probability,
        'crossover_distribution_index': config.crossover_distribution_index,
        'mutation_distribution_index': config.mutation_distribution_index,
        'constraint_standard': config.constraint_standard,
        'use_pareto': config.use_pareto,
        'objective_weights': config.objective_weights,
        'variables': [
            {
                'name': v.name,
                'display_name': v.display_name,
                'unit': v.unit,
                'min_value': v.min_value,
                'max_value': v.max_value,
                'default_value': v.default_value,
                'description': v.description,
            }
            for v in config.variables
        ],
        'objectives': [
            {
                'name': o.name,
                'display_name': o.display_name,
                'unit': o.unit,
                'direction': o.direction,
                'description': o.description,
            }
            for o in config.objectives
        ],
    }


def _dict_to_config(d: Dict[str, Any]) -> OptimizationConfig:
    config = OptimizationConfig(
        population_size=d.get('population_size', 50),
        max_generations=d.get('max_generations', 100),
        crossover_probability=d.get('crossover_probability', 0.9),
        mutation_probability=d.get('mutation_probability', 0.1),
        crossover_distribution_index=d.get('crossover_distribution_index', 20.0),
        mutation_distribution_index=d.get('mutation_distribution_index', 20.0),
        constraint_standard=d.get('constraint_standard', '一级A'),
        use_pareto=d.get('use_pareto', True),
        objective_weights=d.get('objective_weights', {}),
    )
    config.variables = [
        OptimizationVariable(**v) for v in d.get('variables', [])
    ]
    config.objectives = [
        OptimizationObjective(**o) for o in d.get('objectives', [])
    ]
    return config


@dataclass
class OptimizationHistoryRecord:
    record_id: str
    timestamp: str
    config: OptimizationConfig
    pareto_front: List[Individual]
    best_fitness_history: List[float]
    avg_fitness_history: List[float]
    total_evaluations: int = 0
    was_aborted: bool = False
    objective_names: List[str] = field(default_factory=list)
    population_size_snapshot: int = 50
    max_generations_snapshot: int = 100


def save_optimization_result(result: OptimizationResult) -> str:
    _ensure_history_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    record_id = f"opt_{timestamp}"

    obj_names = [o.display_name for o in result.config.objectives]
    pareto_dicts = [_individual_to_dict(ind, result.config.variables) for ind in result.pareto_front]

    data = {
        'record_id': record_id,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'config': _config_to_dict(result.config),
        'pareto_front': pareto_dicts,
        'best_fitness_history': result.best_fitness_history,
        'avg_fitness_history': result.avg_fitness_history,
        'total_evaluations': result.total_evaluations,
        'was_aborted': result.was_aborted,
        'objective_names': obj_names,
        'population_size_snapshot': result.config.population_size,
        'max_generations_snapshot': result.config.max_generations,
        'pareto_count': len(result.pareto_front),
    }

    filepath = os.path.join(HISTORY_DIR, f"{record_id}.json")
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return record_id


def list_optimization_history() -> List[Dict[str, Any]]:
    _ensure_history_dir()
    records = []
    for filename in sorted(os.listdir(HISTORY_DIR), reverse=True):
        if filename.endswith('.json'):
            filepath = os.path.join(HISTORY_DIR, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                records.append({
                    'record_id': data.get('record_id', filename.replace('.json', '')),
                    'timestamp': data.get('timestamp', ''),
                    'objective_names': data.get('objective_names', []),
                    'population_size_snapshot': data.get('population_size_snapshot', 50),
                    'max_generations_snapshot': data.get('max_generations_snapshot', 100),
                    'pareto_count': data.get('pareto_count', 0),
                    'total_evaluations': data.get('total_evaluations', 0),
                    'was_aborted': data.get('was_aborted', False),
                })
            except Exception:
                continue
    return records


def load_optimization_history(record_id: str) -> Optional[OptimizationHistoryRecord]:
    filepath = os.path.join(HISTORY_DIR, f"{record_id}.json")
    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        config = _dict_to_config(data['config'])
        pareto_front = [_dict_to_individual(ind_d) for ind_d in data.get('pareto_front', [])]

        return OptimizationHistoryRecord(
            record_id=data['record_id'],
            timestamp=data['timestamp'],
            config=config,
            pareto_front=pareto_front,
            best_fitness_history=data.get('best_fitness_history', []),
            avg_fitness_history=data.get('avg_fitness_history', []),
            total_evaluations=data.get('total_evaluations', 0),
            was_aborted=data.get('was_aborted', False),
            objective_names=data.get('objective_names', []),
            population_size_snapshot=data.get('population_size_snapshot', 50),
            max_generations_snapshot=data.get('max_generations_snapshot', 100),
        )
    except Exception:
        return None


def delete_optimization_history(record_id: str) -> bool:
    filepath = os.path.join(HISTORY_DIR, f"{record_id}.json")
    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False
