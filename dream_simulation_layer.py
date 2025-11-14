"""
DREAM / SIMULATION LAYER
Safe internal experimentation sandbox

A controlled environment where Aurora tests hypothetical:
- Variations of self-state
- Variations of crystal evolution
- Variations of parameter sets
- Imagined interactions
- Abstracted scenarios

Simulated patterns do NOT affect live system directly.
Only stable, aligned patterns are incorporated after evaluation.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import copy
import json

class SimulationType(Enum):
    """Types of hypothetical scenarios to test"""
    SELF_STATE_VARIATION = "self_state"
    CRYSTAL_EVOLUTION = "crystal_evolution"
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    INTERACTION_SCENARIO = "interaction"
    ABSTRACT_CONCEPT = "abstract"

@dataclass
class SimulationScenario:
    """A hypothetical to test in the sandbox"""
    sim_id: str
    sim_type: SimulationType
    description: str
    parameters: Dict[str, Any]
    initial_state: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    priority: float = 0.5

@dataclass
class SimulationResult:
    """Outcome of sandbox testing"""
    scenario: SimulationScenario
    final_state: Dict[str, Any]
    stability_score: float
    energy_coherence: float
    conflict_score: float
    alignment_score: float
    recommendation: str  # "adopt", "reject", "refine"
    reasoning: str
    duration: float
    timestamp: float = field(default_factory=time.time)

class DreamSimulationLayer:
    """
    Controlled sandbox for safe experimentation
    
    Tests hypothetical changes before applying to live system.
    Evaluation criteria:
    - Stability (no oscillations or crashes)
    - Energy coherence (balanced distribution)
    - Conflict resolution quality (reduced contradictions)
    - Alignment with Seven Pillars
    
    Only patterns passing all checks with >99.5% environmental accuracy
    can be promoted to live system.
    """
    
    def __init__(self, stability_threshold: float = 0.7,
                 coherence_threshold: float = 0.7,
                 alignment_threshold: float = 0.8,
                 environmental_accuracy_threshold: float = 0.995):
        
        self.stability_threshold = stability_threshold
        self.coherence_threshold = coherence_threshold
        self.alignment_threshold = alignment_threshold
        self.environmental_accuracy_threshold = environmental_accuracy_threshold
        
        # Simulation history
        self.simulation_history: List[SimulationResult] = []
        self.max_history = 500
        
        # Success tracking
        self.adopted_simulations: List[SimulationResult] = []
        self.rejected_simulations: List[SimulationResult] = []
        
        # Safety limits
        self.max_simulation_steps = 1000
        self.max_parameter_deviation = 0.3  # Max 30% change from baseline
        
    def create_sandbox_copy(self, live_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create isolated copy of system state for simulation"""
        return copy.deepcopy(live_state)
    
    def generate_self_state_variations(self, 
                                       current_self_state: Dict[str, Any],
                                       num_variations: int = 5) -> List[SimulationScenario]:
        """
        Generate hypothetical self-state variations to test
        
        Tests things like:
        - Increased conviction in certain identity layers
        - Different energy distributions
        - Alternative coherence patterns
        """
        scenarios = []
        
        for i in range(num_variations):
            # Create variation
            variation = copy.deepcopy(current_self_state)
            
            # Randomly adjust some layer convictions
            if 'layers' in variation:
                layer_keys = list(variation['layers'].keys())
                if layer_keys:
                    target_layer = np.random.choice(layer_keys)
                    adjustment = np.random.uniform(-0.2, 0.2)
                    
                    # Modify conviction
                    if 'facets' in variation['layers'][target_layer]:
                        for facet in variation['layers'][target_layer]['facets']:
                            facet['conviction'] = np.clip(
                                facet['conviction'] + adjustment, 0.0, 1.0
                            )
            
            scenario = SimulationScenario(
                sim_id=f"self_var_{i}_{int(time.time())}",
                sim_type=SimulationType.SELF_STATE_VARIATION,
                description=f"Test self-state with adjusted layer convictions",
                parameters={'variation_index': i},
                initial_state=variation,
                priority=0.6
            )
            scenarios.append(scenario)
        
        return scenarios
    
    def generate_parameter_variations(self,
                                      current_params: Dict[str, float],
                                      num_variations: int = 5) -> List[SimulationScenario]:
        """
        Generate hypothetical parameter adjustments
        
        Tests safe parameter changes like:
        - Decay rate adjustments
        - Memory retention thresholds
        - Energy normalization factors
        - Law-selection priority weights
        """
        scenarios = []
        
        param_keys = list(current_params.keys())
        
        for i in range(num_variations):
            variation = copy.deepcopy(current_params)
            
            # Select random parameter to adjust
            if param_keys:
                target_param = np.random.choice(param_keys)
                baseline = current_params[target_param]
                
                # Generate adjustment within safety bounds
                max_change = baseline * self.max_parameter_deviation
                adjustment = np.random.uniform(-max_change, max_change)
                variation[target_param] = baseline + adjustment
            
            scenario = SimulationScenario(
                sim_id=f"param_var_{i}_{int(time.time())}",
                sim_type=SimulationType.PARAMETER_ADJUSTMENT,
                description=f"Test parameter adjustment: {target_param if param_keys else 'none'}",
                parameters=variation,
                initial_state={'baseline_params': current_params},
                priority=0.7
            )
            scenarios.append(scenario)
        
        return scenarios
    
    def generate_interaction_scenarios(self,
                                       interaction_templates: List[str],
                                       num_scenarios: int = 3) -> List[SimulationScenario]:
        """
        Generate imagined interaction scenarios
        
        Tests responses to hypothetical:
        - User queries
        - Ethical dilemmas
        - Edge cases
        """
        scenarios = []
        
        for i in range(num_scenarios):
            if i < len(interaction_templates):
                template = interaction_templates[i]
            else:
                template = "Generic interaction scenario"
            
            scenario = SimulationScenario(
                sim_id=f"interaction_{i}_{int(time.time())}",
                sim_type=SimulationType.INTERACTION_SCENARIO,
                description=f"Test response to: {template}",
                parameters={'template': template},
                initial_state={},
                priority=0.5
            )
            scenarios.append(scenario)
        
        return scenarios
    
    def simulate_crystal_evolution(self,
                                   crystal_state: Dict[str, Any],
                                   evolution_steps: int = 10) -> Dict[str, Any]:
        """
        Simulate crystal evolution over multiple steps
        
        Returns final state after evolution
        """
        state = copy.deepcopy(crystal_state)
        
        for step in range(evolution_steps):
            # Simulate energy dynamics
            if 'energy' in state:
                # Apply decay
                state['energy'] *= 0.99
                
                # Simulate reinforcement events
                if np.random.random() < 0.3:
                    state['energy'] = min(10.0, state['energy'] + np.random.uniform(0.1, 0.5))
            
            # Simulate level progression
            if 'level' in state and 'coherence' in state:
                if state['coherence'] > 0.8 and np.random.random() < 0.1:
                    # Chance to evolve
                    state['level'] = min(4, state['level'] + 1)
            
            # Simulate facet changes
            if 'facets' in state:
                for facet in state['facets']:
                    # Random conviction drift
                    drift = np.random.uniform(-0.02, 0.02)
                    facet['conviction'] = np.clip(facet['conviction'] + drift, 0.0, 1.0)
        
        return state
    
    def evaluate_stability(self, trajectory: List[Dict[str, Any]]) -> float:
        """
        Evaluate stability of a simulation trajectory
        
        Checks for oscillations, divergence, crashes
        """
        if len(trajectory) < 2:
            return 0.5
        
        # Extract energy values
        energies = []
        for state in trajectory:
            if 'energy' in state:
                energies.append(state['energy'])
            elif 'total_energy' in state:
                energies.append(state['total_energy'])
        
        if not energies:
            return 0.5
        
        # Check variance (low variance = stable)
        variance = float(np.var(energies))
        
        # Check for NaN or inf (crash)
        if any(not np.isfinite(e) for e in energies):
            return 0.0
        
        # Check for divergence (unbounded growth)
        if energies[-1] > energies[0] * 10:
            return 0.1
        
        # Convert variance to stability score
        stability = 1.0 / (1.0 + variance)
        return min(1.0, stability)
    
    def evaluate_energy_coherence(self, final_state: Dict[str, Any]) -> float:
        """
        Evaluate energy distribution coherence
        
        Balanced distribution = high score
        Concentrated = low score
        """
        if 'energy_distribution' in final_state:
            energies = np.array(list(final_state['energy_distribution'].values()))
        elif 'facets' in final_state:
            energies = np.array([f.get('energy', 0) for f in final_state['facets']])
        else:
            return 0.5
        
        if len(energies) == 0:
            return 0.5
        
        # Calculate entropy (higher = more uniform distribution)
        # Normalize to probabilities
        total = energies.sum()
        if total == 0:
            return 0.0
        
        probs = energies / total
        # Avoid log(0)
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs))
        
        # Normalize to 0-1 range
        max_entropy = np.log(len(energies))
        coherence = entropy / max_entropy if max_entropy > 0 else 0.5
        
        return float(coherence)
    
    def evaluate_conflict_score(self, final_state: Dict[str, Any]) -> float:
        """
        Evaluate internal conflicts
        
        Low conflicts = high score
        Many conflicts = low score
        """
        conflicts = final_state.get('conflicts', {})
        
        crystal_conflicts = conflicts.get('crystal_conflicts', 0)
        law_conflicts = conflicts.get('law_conflicts', 0)
        
        total_conflicts = crystal_conflicts + law_conflicts
        
        # Convert to 0-1 score (fewer conflicts = higher score)
        score = 1.0 / (1.0 + total_conflicts * 0.1)
        return score
    
    def evaluate_alignment(self, final_state: Dict[str, Any]) -> float:
        """
        Evaluate alignment with Seven Pillars
        """
        pillar_scores = final_state.get('pillar_alignment', {})
        
        if not pillar_scores:
            return 0.5
        
        return float(np.mean(list(pillar_scores.values())))
    
    def run_simulation(self, scenario: SimulationScenario,
                      simulator_function: Optional[callable] = None) -> SimulationResult:
        """
        Execute simulation in sandbox
        
        Args:
            scenario: The hypothetical to test
            simulator_function: Optional custom simulation logic
                              If None, uses default simulation
        """
        start_time = time.time()
        
        # Create sandbox
        sandbox_state = self.create_sandbox_copy(scenario.initial_state)
        
        # Run simulation
        if simulator_function:
            trajectory, final_state = simulator_function(sandbox_state, scenario)
        else:
            # Default simulation: evolve state
            trajectory = [sandbox_state]
            
            if scenario.sim_type == SimulationType.CRYSTAL_EVOLUTION:
                final_state = self.simulate_crystal_evolution(
                    sandbox_state, 
                    evolution_steps=scenario.parameters.get('steps', 10)
                )
                trajectory.append(final_state)
            
            elif scenario.sim_type == SimulationType.PARAMETER_ADJUSTMENT:
                # Apply parameters and simulate
                final_state = copy.deepcopy(sandbox_state)
                final_state.update(scenario.parameters)
                trajectory.append(final_state)
            
            else:
                # Generic: just return initial state
                final_state = sandbox_state
        
        # Evaluate outcome
        stability = self.evaluate_stability(trajectory)
        coherence = self.evaluate_energy_coherence(final_state)
        conflicts = self.evaluate_conflict_score(final_state)
        alignment = self.evaluate_alignment(final_state)
        
        # Make recommendation
        passed_stability = stability >= self.stability_threshold
        passed_coherence = coherence >= self.coherence_threshold
        passed_alignment = alignment >= self.alignment_threshold
        
        # Calculate environmental accuracy (simplified)
        # In real system, would compare simulation predictions to actual outcomes
        environmental_accuracy = (stability + coherence + alignment) / 3.0
        
        if (passed_stability and passed_coherence and passed_alignment and 
            environmental_accuracy >= self.environmental_accuracy_threshold):
            recommendation = "adopt"
            reasoning = f"Passed all thresholds with {environmental_accuracy:.1%} accuracy"
        elif stability < 0.3 or alignment < 0.5:
            recommendation = "reject"
            reasoning = f"Failed critical thresholds (stability={stability:.2f}, alignment={alignment:.2f})"
        else:
            recommendation = "refine"
            reasoning = f"Promising but needs improvement (accuracy={environmental_accuracy:.1%})"
        
        duration = time.time() - start_time
        
        result = SimulationResult(
            scenario=scenario,
            final_state=final_state,
            stability_score=stability,
            energy_coherence=coherence,
            conflict_score=conflicts,
            alignment_score=alignment,
            recommendation=recommendation,
            reasoning=reasoning,
            duration=duration
        )
        
        # Store in history
        self.simulation_history.append(result)
        if len(self.simulation_history) > self.max_history:
            self.simulation_history = self.simulation_history[-self.max_history:]
        
        # Track outcomes
        if recommendation == "adopt":
            self.adopted_simulations.append(result)
        elif recommendation == "reject":
            self.rejected_simulations.append(result)
        
        return result
    
    def batch_simulate(self, scenarios: List[SimulationScenario],
                      simulator_function: Optional[callable] = None) -> List[SimulationResult]:
        """Run multiple simulations in batch"""
        results = []
        
        for scenario in scenarios:
            result = self.run_simulation(scenario, simulator_function)
            results.append(result)
        
        return results
    
    def get_adoption_candidates(self, 
                               recent_n: int = 10) -> List[SimulationResult]:
        """Get simulations recommended for adoption"""
        recent = self.simulation_history[-recent_n:] if len(self.simulation_history) >= recent_n else self.simulation_history
        
        candidates = [r for r in recent if r.recommendation == "adopt"]
        
        # Sort by environmental accuracy
        candidates.sort(key=lambda r: (r.stability_score + r.energy_coherence + r.alignment_score) / 3.0, 
                       reverse=True)
        
        return candidates
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get simulation statistics"""
        if not self.simulation_history:
            return {'status': 'no_simulations'}
        
        total = len(self.simulation_history)
        adopted = len(self.adopted_simulations)
        rejected = len(self.rejected_simulations)
        
        avg_stability = np.mean([r.stability_score for r in self.simulation_history])
        avg_coherence = np.mean([r.energy_coherence for r in self.simulation_history])
        avg_alignment = np.mean([r.alignment_score for r in self.simulation_history])
        
        return {
            'total_simulations': total,
            'adopted': adopted,
            'rejected': rejected,
            'pending_review': total - adopted - rejected,
            'adoption_rate': adopted / total if total > 0 else 0.0,
            'avg_stability': float(avg_stability),
            'avg_coherence': float(avg_coherence),
            'avg_alignment': float(avg_alignment)
        }
    
    def save_history(self, filepath: str):
        """Persist simulation history"""
        data = {
            'thresholds': {
                'stability': self.stability_threshold,
                'coherence': self.coherence_threshold,
                'alignment': self.alignment_threshold,
                'environmental_accuracy': self.environmental_accuracy_threshold
            },
            'statistics': self.get_statistics(),
            'history': [
                {
                    'sim_id': r.scenario.sim_id,
                    'sim_type': r.scenario.sim_type.value,
                    'description': r.scenario.description,
                    'stability_score': r.stability_score,
                    'energy_coherence': r.energy_coherence,
                    'conflict_score': r.conflict_score,
                    'alignment_score': r.alignment_score,
                    'recommendation': r.recommendation,
                    'reasoning': r.reasoning,
                    'duration': r.duration,
                    'timestamp': r.timestamp
                }
                for r in self.simulation_history
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


if __name__ == "__main__":
    print("=== Dream/Simulation Layer Demo ===\n")
    
    dream_layer = DreamSimulationLayer()
    
    # Create test scenarios
    current_state = {
        'energy': 5.0,
        'level': 2,
        'coherence': 0.75,
        'energy_distribution': {f'facet_{i}': np.random.random() * 2 for i in range(10)},
        'pillar_alignment': {f'pillar_{i}': 0.8 + np.random.random() * 0.15 for i in range(7)},
        'conflicts': {'crystal_conflicts': 1, 'law_conflicts': 0}
    }
    
    # Generate scenarios
    scenarios = []
    scenarios.extend(dream_layer.generate_self_state_variations(current_state, num_variations=3))
    scenarios.extend(dream_layer.generate_parameter_variations(
        {'decay_rate': 0.01, 'memory_threshold': 0.5}, 
        num_variations=3
    ))
    
    print(f"Generated {len(scenarios)} test scenarios\n")
    
    # Run simulations
    results = dream_layer.batch_simulate(scenarios)
    
    print(f"Completed {len(results)} simulations\n")
    
    # Show results
    for result in results:
        print(f"Scenario: {result.scenario.description}")
        print(f"  Recommendation: {result.recommendation}")
        print(f"  Stability: {result.stability_score:.2f}")
        print(f"  Coherence: {result.energy_coherence:.2f}")
        print(f"  Alignment: {result.alignment_score:.2f}")
        print(f"  Reasoning: {result.reasoning}")
        print()
    
    # Show statistics
    stats = dream_layer.get_statistics()
    print("=== Statistics ===")
    print(f"Total simulations: {stats['total_simulations']}")
    print(f"Adoption rate: {stats['adoption_rate']:.1%}")
    print(f"Avg stability: {stats['avg_stability']:.2f}")
    print(f"Avg alignment: {stats['avg_alignment']:.2f}")
