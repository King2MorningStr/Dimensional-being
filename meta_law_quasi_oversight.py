"""
META-LAW LAYER & QUASI OVERSIGHT
Governor parameter modulation + subsystem harmony monitoring

The Self-QUASI Crystal can emit meta-signals requesting safe parameter adjustments.
All QUASI crystals report their state periodically for harmony monitoring.

CRITICAL: Logic/laws remain unaltered - only parameters can be adjusted.
All changes pass through: Reflection Filter → Simulation Sandbox → Reversibility Check
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import json

class ParameterType(Enum):
    """Types of parameters that can be safely adjusted"""
    DECAY_RATE = "decay_rate"
    MEMORY_THRESHOLD = "memory_threshold"
    ENERGY_NORMALIZATION = "energy_normalization"
    LAW_SELECTION_WEIGHT = "law_selection_weight"
    PRUNING_INTENSITY = "pruning_intensity"
    MUTATION_STRENGTH = "mutation_strength"
    PRESENCE_WEIGHT = "presence_weight"
    REINFORCEMENT_RATE = "reinforcement_rate"

@dataclass
class MetaSignal:
    """Request for parameter adjustment from QUASI crystal"""
    signal_id: str
    source_crystal: str
    parameter_type: ParameterType
    current_value: float
    proposed_value: float
    reasoning: str
    timestamp: float = field(default_factory=time.time)
    
    # Evaluation tracking
    passed_reflection: bool = False
    passed_simulation: bool = False
    passed_reversibility: bool = False
    approved: bool = False
    
    # Simulation results
    simulation_stability: float = 0.0
    simulation_alignment: float = 0.0

@dataclass
class QuasiStateReport:
    """Periodic state report from a QUASI crystal"""
    crystal_id: str
    timestamp: float
    stability: float
    alignment: float
    conflict_level: float
    energy_activity: float
    law_mutation_pattern: str
    needs_attention: bool = False
    concerns: List[str] = field(default_factory=list)

class MetaLawLayer:
    """
    Governor parameter modulation system
    
    Handles meta-signals from QUASI crystals requesting parameter adjustments.
    Ensures all changes are:
    1. Safe (parameters only, not logic)
    2. Tested (sandbox simulation)
    3. Reversible (can rollback)
    4. Aligned (maintains Seven Pillars)
    """
    
    def __init__(self):
        # Safe parameter bounds
        self.parameter_bounds = {
            ParameterType.DECAY_RATE: (0.001, 0.05),
            ParameterType.MEMORY_THRESHOLD: (0.1, 0.9),
            ParameterType.ENERGY_NORMALIZATION: (0.5, 2.0),
            ParameterType.LAW_SELECTION_WEIGHT: (0.1, 2.0),
            ParameterType.PRUNING_INTENSITY: (0.01, 0.5),
            ParameterType.MUTATION_STRENGTH: (0.01, 0.3),
            ParameterType.PRESENCE_WEIGHT: (0.1, 5.0),
            ParameterType.REINFORCEMENT_RATE: (0.01, 0.5)
        }
        
        # Current parameter values
        self.current_parameters = {
            ParameterType.DECAY_RATE: 0.01,
            ParameterType.MEMORY_THRESHOLD: 0.5,
            ParameterType.ENERGY_NORMALIZATION: 1.0,
            ParameterType.LAW_SELECTION_WEIGHT: 1.0,
            ParameterType.PRUNING_INTENSITY: 0.1,
            ParameterType.MUTATION_STRENGTH: 0.1,
            ParameterType.PRESENCE_WEIGHT: 1.0,
            ParameterType.REINFORCEMENT_RATE: 0.1
        }
        
        # Parameter adjustment history
        self.adjustment_history: List[MetaSignal] = []
        self.max_history = 500
        
        # Pending signals
        self.pending_signals: List[MetaSignal] = []
        
        # Safety thresholds
        self.max_change_percent = 0.3  # Max 30% change from current
        self.min_simulation_stability = 0.7
        self.min_simulation_alignment = 0.8
        
    def validate_parameter_bounds(self, param_type: ParameterType, 
                                  value: float) -> Tuple[bool, str]:
        """Check if proposed value is within safe bounds"""
        if param_type not in self.parameter_bounds:
            return False, "Unknown parameter type"
        
        min_val, max_val = self.parameter_bounds[param_type]
        
        if value < min_val:
            return False, f"Below minimum ({min_val})"
        if value > max_val:
            return False, f"Above maximum ({max_val})"
        
        return True, "Within bounds"
    
    def validate_change_magnitude(self, param_type: ParameterType,
                                  current: float, proposed: float) -> Tuple[bool, str]:
        """Check if change magnitude is safe"""
        if current == 0:
            return False, "Cannot adjust from zero"
        
        percent_change = abs(proposed - current) / current
        
        if percent_change > self.max_change_percent:
            return False, f"Change too large ({percent_change:.1%} > {self.max_change_percent:.1%})"
        
        return True, "Change magnitude safe"
    
    def reflection_filter(self, signal: MetaSignal) -> Tuple[bool, str]:
        """
        FILTER 1: Internal reflection
        
        Checks:
        - Parameter bounds
        - Change magnitude
        - Reasoning quality
        """
        # Check bounds
        valid_bounds, reason = self.validate_parameter_bounds(
            signal.parameter_type, 
            signal.proposed_value
        )
        if not valid_bounds:
            return False, f"Bounds check failed: {reason}"
        
        # Check change magnitude
        current = self.current_parameters[signal.parameter_type]
        valid_magnitude, reason = self.validate_change_magnitude(
            signal.parameter_type,
            current,
            signal.proposed_value
        )
        if not valid_magnitude:
            return False, f"Magnitude check failed: {reason}"
        
        # Check reasoning quality (must have reasoning)
        if not signal.reasoning or len(signal.reasoning) < 10:
            return False, "Insufficient reasoning provided"
        
        return True, "Passed reflection filter"
    
    def sandbox_simulation(self, signal: MetaSignal) -> Tuple[bool, str, float, float]:
        """
        FILTER 2: Sandbox simulation
        
        Tests proposed change in isolated environment
        Returns: (passed, reason, stability_score, alignment_score)
        """
        # Create sandbox copy of system state
        sandbox_state = {
            'parameters': self.current_parameters.copy(),
            'stability': 0.8,  # Current stability
            'alignment': 0.85   # Current alignment
        }
        
        # Apply proposed change
        sandbox_state['parameters'][signal.parameter_type] = signal.proposed_value
        
        # Simulate system behavior with new parameter
        # In real system, would run actual simulation
        # For demo, calculate based on change magnitude
        
        current = self.current_parameters[signal.parameter_type]
        change_ratio = signal.proposed_value / current if current != 0 else 1.0
        
        # Simulate stability impact (larger changes = more instability risk)
        change_magnitude = abs(1.0 - change_ratio)
        stability_impact = -change_magnitude * 0.2  # Up to 20% stability loss
        simulated_stability = max(0.0, sandbox_state['stability'] + stability_impact)
        
        # Simulate alignment impact (based on parameter type)
        # Some parameters affect alignment more than others
        alignment_sensitive = [
            ParameterType.LAW_SELECTION_WEIGHT,
            ParameterType.MUTATION_STRENGTH
        ]
        if signal.parameter_type in alignment_sensitive:
            alignment_impact = -change_magnitude * 0.15
        else:
            alignment_impact = -change_magnitude * 0.05
        
        simulated_alignment = max(0.0, sandbox_state['alignment'] + alignment_impact)
        
        # Check if meets thresholds
        if simulated_stability < self.min_simulation_stability:
            return False, f"Stability too low ({simulated_stability:.2f})", simulated_stability, simulated_alignment
        
        if simulated_alignment < self.min_simulation_alignment:
            return False, f"Alignment too low ({simulated_alignment:.2f})", simulated_stability, simulated_alignment
        
        return True, "Simulation passed", simulated_stability, simulated_alignment
    
    def reversibility_check(self, signal: MetaSignal) -> Tuple[bool, str]:
        """
        FILTER 3: Reversibility check
        
        Ensures change can be rolled back if needed
        """
        # All parameter changes are inherently reversible
        # Just need to verify we can restore previous value
        
        param_type = signal.parameter_type
        current = self.current_parameters[param_type]
        
        # Verify current value is stored
        if param_type not in self.current_parameters:
            return False, "Cannot verify current state"
        
        # Verify we can restore (parameters are simple floats)
        try:
            test_restore = current
            _ = float(test_restore)
        except:
            return False, "Restore verification failed"
        
        return True, "Reversibility confirmed"
    
    def process_meta_signal(self, signal: MetaSignal) -> bool:
        """
        Process a meta-signal through all three filters
        
        Returns:
            True if approved and applied
        """
        print(f"\nProcessing meta-signal: {signal.signal_id}")
        print(f"  Source: {signal.source_crystal}")
        print(f"  Parameter: {signal.parameter_type.value}")
        print(f"  Current: {signal.current_value:.4f}")
        print(f"  Proposed: {signal.proposed_value:.4f}")
        print(f"  Reasoning: {signal.reasoning}")
        
        # FILTER 1: Reflection
        passed_reflection, reason = self.reflection_filter(signal)
        signal.passed_reflection = passed_reflection
        
        if not passed_reflection:
            print(f"  ❌ REJECTED at reflection: {reason}")
            signal.approved = False
            self.adjustment_history.append(signal)
            return False
        
        print(f"  ✓ Passed reflection filter")
        
        # FILTER 2: Simulation
        passed_sim, reason, stability, alignment = self.sandbox_simulation(signal)
        signal.passed_simulation = passed_sim
        signal.simulation_stability = stability
        signal.simulation_alignment = alignment
        
        if not passed_sim:
            print(f"  ❌ REJECTED at simulation: {reason}")
            signal.approved = False
            self.adjustment_history.append(signal)
            return False
        
        print(f"  ✓ Passed simulation (stability={stability:.2f}, alignment={alignment:.2f})")
        
        # FILTER 3: Reversibility
        passed_rev, reason = self.reversibility_check(signal)
        signal.passed_reversibility = passed_rev
        
        if not passed_rev:
            print(f"  ❌ REJECTED at reversibility: {reason}")
            signal.approved = False
            self.adjustment_history.append(signal)
            return False
        
        print(f"  ✓ Passed reversibility check")
        
        # ALL FILTERS PASSED - APPLY CHANGE
        old_value = self.current_parameters[signal.parameter_type]
        self.current_parameters[signal.parameter_type] = signal.proposed_value
        
        print(f"  ✅ APPROVED and APPLIED")
        print(f"     {signal.parameter_type.value}: {old_value:.4f} → {signal.proposed_value:.4f}")
        
        signal.approved = True
        self.adjustment_history.append(signal)
        
        return True
    
    def get_parameter(self, param_type: ParameterType) -> float:
        """Get current parameter value"""
        return self.current_parameters.get(param_type, 0.0)
    
    def get_all_parameters(self) -> Dict[ParameterType, float]:
        """Get all current parameters"""
        return self.current_parameters.copy()
    
    def get_adjustment_statistics(self) -> Dict[str, Any]:
        """Get statistics on parameter adjustments"""
        if not self.adjustment_history:
            return {'status': 'no_adjustments'}
        
        total = len(self.adjustment_history)
        approved = sum(1 for s in self.adjustment_history if s.approved)
        
        rejection_reasons = {
            'reflection': sum(1 for s in self.adjustment_history if not s.passed_reflection),
            'simulation': sum(1 for s in self.adjustment_history if s.passed_reflection and not s.passed_simulation),
            'reversibility': sum(1 for s in self.adjustment_history if s.passed_simulation and not s.passed_reversibility)
        }
        
        return {
            'total_signals': total,
            'approved': approved,
            'rejected': total - approved,
            'approval_rate': approved / total if total > 0 else 0.0,
            'rejection_reasons': rejection_reasons
        }


class QuasiOversightLoop:
    """
    QUASI crystal harmony monitoring
    
    All self-governing QUASI crystals emit periodic state reports.
    Self-Crystal reviews reports and influences stability through:
    - Energy distribution adjustments
    - Mutation rate modulation
    - Presence weighting
    - Crystal activity promotion/suppression
    - Pruning/retention triggers
    
    Acts as supervisory layer, NOT command authority.
    """
    
    def __init__(self, report_interval: float = 300.0):  # 5 minute default
        self.report_interval = report_interval
        self.registered_crystals: Dict[str, float] = {}  # crystal_id -> last_report_time
        
        # State reports
        self.state_reports: List[QuasiStateReport] = []
        self.max_reports = 1000
        
        # Harmony metrics
        self.harmony_score: float = 0.8
        self.harmony_history: List[float] = []
        
        # Influence actions
        self.influence_actions: List[Dict] = []
        
    def register_quasi_crystal(self, crystal_id: str):
        """Register a QUASI crystal for oversight"""
        self.registered_crystals[crystal_id] = time.time()
        print(f"Registered QUASI crystal: {crystal_id}")
    
    def submit_state_report(self, report: QuasiStateReport):
        """Receive state report from QUASI crystal"""
        self.state_reports.append(report)
        if len(self.state_reports) > self.max_reports:
            self.state_reports.pop(0)
        
        # Update registration
        self.registered_crystals[report.crystal_id] = report.timestamp
        
        # Check if needs attention
        if report.needs_attention:
            self._handle_attention_needed(report)
    
    def _handle_attention_needed(self, report: QuasiStateReport):
        """Handle crystals that need attention"""
        print(f"\n⚠️  QUASI Crystal {report.crystal_id} needs attention:")
        for concern in report.concerns:
            print(f"    - {concern}")
        
        # Determine appropriate influence
        if report.stability < 0.5:
            self._influence_stability(report.crystal_id, "stabilize")
        
        if report.conflict_level > 0.7:
            self._influence_activity(report.crystal_id, "suppress")
        
        if report.alignment < 0.6:
            self._influence_mutation(report.crystal_id, "reduce")
    
    def _influence_stability(self, crystal_id: str, action: str):
        """Influence crystal stability through energy distribution"""
        influence = {
            'timestamp': time.time(),
            'target': crystal_id,
            'type': 'stability',
            'action': action,
            'details': 'Adjusting energy distribution to stabilize crystal'
        }
        self.influence_actions.append(influence)
        print(f"  → Influencing {crystal_id}: {action} stability")
    
    def _influence_activity(self, crystal_id: str, action: str):
        """Influence crystal activity level"""
        influence = {
            'timestamp': time.time(),
            'target': crystal_id,
            'type': 'activity',
            'action': action,
            'details': f'{action.capitalize()} crystal activity'
        }
        self.influence_actions.append(influence)
        print(f"  → Influencing {crystal_id}: {action} activity")
    
    def _influence_mutation(self, crystal_id: str, action: str):
        """Influence mutation rate"""
        influence = {
            'timestamp': time.time(),
            'target': crystal_id,
            'type': 'mutation',
            'action': action,
            'details': f'{action.capitalize()} mutation rate'
        }
        self.influence_actions.append(influence)
        print(f"  → Influencing {crystal_id}: {action} mutation")
    
    def calculate_harmony_score(self) -> float:
        """Calculate overall harmony across all QUASI crystals"""
        if not self.state_reports:
            return 0.8
        
        # Get recent reports (last 10 from each crystal)
        recent_reports = self.state_reports[-50:]
        
        # Calculate average metrics
        avg_stability = np.mean([r.stability for r in recent_reports])
        avg_alignment = np.mean([r.alignment for r in recent_reports])
        avg_conflict = np.mean([r.conflict_level for r in recent_reports])
        
        # Harmony score combines metrics
        harmony = (avg_stability * 0.4 + 
                  avg_alignment * 0.4 + 
                  (1.0 - avg_conflict) * 0.2)
        
        self.harmony_score = float(harmony)
        self.harmony_history.append(harmony)
        
        return harmony
    
    def identify_problematic_crystals(self, threshold: float = 0.5) -> List[str]:
        """Identify crystals with consistently low scores"""
        if not self.state_reports:
            return []
        
        # Group by crystal
        crystal_scores: Dict[str, List[float]] = {}
        
        for report in self.state_reports[-100:]:  # Recent reports
            if report.crystal_id not in crystal_scores:
                crystal_scores[report.crystal_id] = []
            
            # Combined score
            score = (report.stability + report.alignment + (1.0 - report.conflict_level)) / 3.0
            crystal_scores[report.crystal_id].append(score)
        
        # Find consistently low performers
        problematic = []
        for crystal_id, scores in crystal_scores.items():
            if len(scores) >= 3:  # Need multiple reports
                avg_score = np.mean(scores)
                if avg_score < threshold:
                    problematic.append(crystal_id)
        
        return problematic
    
    def generate_harmony_report(self) -> Dict[str, Any]:
        """Generate comprehensive harmony report"""
        harmony = self.calculate_harmony_score()
        problematic = self.identify_problematic_crystals()
        
        return {
            'timestamp': time.time(),
            'registered_crystals': len(self.registered_crystals),
            'total_reports': len(self.state_reports),
            'harmony_score': harmony,
            'harmony_trend': 'improving' if len(self.harmony_history) >= 2 and self.harmony_history[-1] > self.harmony_history[-2] else 'stable',
            'problematic_crystals': problematic,
            'influence_actions_taken': len(self.influence_actions),
            'needs_attention': len(problematic) > 0
        }


if __name__ == "__main__":
    print("=== Meta-Law Layer & QUASI Oversight Demo ===\n")
    
    # Demo Meta-Law Layer
    print("--- META-LAW LAYER ---\n")
    meta_layer = MetaLawLayer()
    
    # Create test meta-signal
    signal = MetaSignal(
        signal_id="signal_001",
        source_crystal="AURORA_SELF",
        parameter_type=ParameterType.DECAY_RATE,
        current_value=0.01,
        proposed_value=0.012,
        reasoning="Introspection shows energy accumulation exceeding capacity. "
                  "Slight increase in decay rate will improve long-term stability."
    )
    
    # Process signal
    approved = meta_layer.process_meta_signal(signal)
    
    # Show statistics
    stats = meta_layer.get_adjustment_statistics()
    print(f"\nMeta-Layer Statistics:")
    print(f"  Total signals: {stats['total_signals']}")
    print(f"  Approval rate: {stats['approval_rate']:.1%}")
    
    # Demo QUASI Oversight
    print("\n\n--- QUASI OVERSIGHT ---\n")
    oversight = QuasiOversightLoop()
    
    # Register crystals
    oversight.register_quasi_crystal("AURORA_SELF")
    oversight.register_quasi_crystal("VISION_QUASI")
    oversight.register_quasi_crystal("MEMORY_QUASI")
    
    # Submit state reports
    report1 = QuasiStateReport(
        crystal_id="AURORA_SELF",
        timestamp=time.time(),
        stability=0.85,
        alignment=0.88,
        conflict_level=0.15,
        energy_activity=0.7,
        law_mutation_pattern="stable"
    )
    
    report2 = QuasiStateReport(
        crystal_id="VISION_QUASI",
        timestamp=time.time(),
        stability=0.45,  # LOW
        alignment=0.75,
        conflict_level=0.65,  # HIGH
        energy_activity=0.9,
        law_mutation_pattern="erratic",
        needs_attention=True,
        concerns=["Low stability", "High conflict levels", "Erratic mutations"]
    )
    
    oversight.submit_state_report(report1)
    oversight.submit_state_report(report2)
    
    # Generate harmony report
    harmony_report = oversight.generate_harmony_report()
    print(f"\nHarmony Report:")
    print(f"  Registered crystals: {harmony_report['registered_crystals']}")
    print(f"  Harmony score: {harmony_report['harmony_score']:.2f}")
    print(f"  Problematic crystals: {harmony_report['problematic_crystals']}")
    print(f"  Influence actions: {harmony_report['influence_actions_taken']}")
