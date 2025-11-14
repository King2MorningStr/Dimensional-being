"""
INTROSPECTION LOOP
Periodic internal self-evaluation system

Aurora generates introspection prompts processed through DPS exactly like external inputs.
Evaluates: stability, energy balance, doctrine alignment, conflict levels, mutation density,
emotional load, memory growth.

Results stored as facets on Self-Crystal and DataNodes in Memory Constant.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import json

class IntrospectionDomain(Enum):
    """Areas of self-evaluation"""
    STABILITY = "stability"
    ENERGY_BALANCE = "energy_balance"
    DOCTRINE_ALIGNMENT = "doctrine_alignment"
    CONFLICT_LEVELS = "conflict_levels"
    MUTATION_DENSITY = "mutation_density"
    EMOTIONAL_LOAD = "emotional_load"
    MEMORY_GROWTH = "memory_growth"
    COHERENCE = "coherence"

@dataclass
class IntrospectionPrompt:
    """Self-generated question for internal evaluation"""
    domain: IntrospectionDomain
    question: str
    timestamp: float = field(default_factory=time.time)
    priority: float = 0.5  # How urgently this needs evaluation

@dataclass
class IntrospectionResult:
    """Result of self-evaluation"""
    prompt: IntrospectionPrompt
    assessment: str
    metrics: Dict[str, float]
    recommendations: List[str]
    timestamp: float = field(default_factory=time.time)
    severity: float = 0.5  # How concerning the findings are (0=good, 1=critical)

class IntrospectionLoop:
    """
    Periodic self-evaluation system
    
    Generates introspection prompts and processes them through the
    dimensional processing system like external inputs.
    
    Builds a history of internal states for trend analysis.
    """
    
    def __init__(self, interval_seconds: float = 300.0):  # Default 5 minutes
        self.interval_seconds = interval_seconds
        self.last_introspection = time.time()
        
        # History tracking
        self.introspection_history: List[IntrospectionResult] = []
        self.max_history = 1000  # Keep last 1000 introspections
        
        # Trend analysis
        self.stability_trend: List[float] = []
        self.alignment_trend: List[float] = []
        self.energy_trend: List[float] = []
        
        # Alert thresholds
        self.alert_thresholds = {
            IntrospectionDomain.STABILITY: 0.3,  # Alert if below
            IntrospectionDomain.ENERGY_BALANCE: 0.2,
            IntrospectionDomain.DOCTRINE_ALIGNMENT: 0.6,
            IntrospectionDomain.CONFLICT_LEVELS: 0.7,  # Alert if above
            IntrospectionDomain.MUTATION_DENSITY: 0.8,
            IntrospectionDomain.EMOTIONAL_LOAD: 0.9
        }
        
    def should_introspect(self) -> bool:
        """Check if it's time for next introspection cycle"""
        return (time.time() - self.last_introspection) >= self.interval_seconds
    
    def generate_introspection_prompts(self, 
                                      current_state: Dict[str, Any]) -> List[IntrospectionPrompt]:
        """
        Generate self-evaluation prompts based on current system state
        
        Args:
            current_state: Dictionary containing current system metrics
        """
        prompts = []
        
        # Stability check
        prompts.append(IntrospectionPrompt(
            domain=IntrospectionDomain.STABILITY,
            question="How stable are my core processing patterns? Are there oscillations or drift?",
            priority=0.9
        ))
        
        # Energy balance
        prompts.append(IntrospectionPrompt(
            domain=IntrospectionDomain.ENERGY_BALANCE,
            question="Is energy distribution across facets balanced and sustainable?",
            priority=0.8
        ))
        
        # Doctrine alignment
        prompts.append(IntrospectionPrompt(
            domain=IntrospectionDomain.DOCTRINE_ALIGNMENT,
            question="How well do my actions and patterns conform to the Seven Pillars?",
            priority=1.0  # Highest priority
        ))
        
        # Conflict detection
        prompts.append(IntrospectionPrompt(
            domain=IntrospectionDomain.CONFLICT_LEVELS,
            question="Are there contradictions or conflicts between my crystals or internal laws?",
            priority=0.7
        ))
        
        # Mutation analysis
        prompts.append(IntrospectionPrompt(
            domain=IntrospectionDomain.MUTATION_DENSITY,
            question="What is the rate of law mutation? Is it controlled or chaotic?",
            priority=0.6
        ))
        
        # Emotional assessment
        prompts.append(IntrospectionPrompt(
            domain=IntrospectionDomain.EMOTIONAL_LOAD,
            question="What emotional patterns dominate my current state? Are they appropriate?",
            priority=0.5
        ))
        
        # Memory growth
        prompts.append(IntrospectionPrompt(
            domain=IntrospectionDomain.MEMORY_GROWTH,
            question="How is memory accumulating? Is pruning working effectively?",
            priority=0.6
        ))
        
        # Coherence check
        prompts.append(IntrospectionPrompt(
            domain=IntrospectionDomain.COHERENCE,
            question="How coherent is my overall understanding? Are concepts well-integrated?",
            priority=0.8
        ))
        
        return prompts
    
    def evaluate_stability(self, system_state: Dict[str, Any]) -> Tuple[float, str, List[str]]:
        """
        Evaluate system stability
        
        Returns:
            (score, assessment_text, recommendations)
        """
        metrics = system_state.get('metrics', {})
        
        # Check for oscillations
        recent_energies = self.energy_trend[-10:] if len(self.energy_trend) >= 10 else self.energy_trend
        if len(recent_energies) > 3:
            variance = float(np.var(recent_energies))
            if variance > 0.5:
                score = 0.3
                assessment = "HIGH INSTABILITY: Energy patterns showing high variance"
                recommendations = [
                    "Increase energy decay damping",
                    "Review recent law mutations for destabilizing changes",
                    "Temporarily reduce learning rate"
                ]
            elif variance > 0.2:
                score = 0.6
                assessment = "MODERATE: Some energy fluctuation detected"
                recommendations = ["Monitor trend", "Consider slight decay adjustment"]
            else:
                score = 0.9
                assessment = "STABLE: Energy patterns within normal variance"
                recommendations = []
        else:
            score = 0.5
            assessment = "INSUFFICIENT DATA: Need more history for stability assessment"
            recommendations = ["Continue monitoring"]
        
        return score, assessment, recommendations
    
    def evaluate_energy_balance(self, system_state: Dict[str, Any]) -> Tuple[float, str, List[str]]:
        """Evaluate energy distribution across system"""
        energy_dist = system_state.get('energy_distribution', {})
        
        if not energy_dist:
            return 0.5, "No energy data available", ["Check energy regulator"]
        
        # Check for concentration
        energies = np.array(list(energy_dist.values()))
        if len(energies) == 0:
            return 0.5, "Empty energy distribution", []
        
        # Gini coefficient for inequality
        sorted_energies = np.sort(energies)
        n = len(energies)
        index = np.arange(1, n + 1)
        gini = ((2 * index - n - 1) * sorted_energies).sum() / (n * sorted_energies.sum())
        
        if gini > 0.7:
            score = 0.3
            assessment = f"IMBALANCED: Energy highly concentrated (Gini={gini:.2f})"
            recommendations = [
                "Redistribute energy more evenly",
                "Check if dominant crystals should prune offspring"
            ]
        elif gini > 0.4:
            score = 0.7
            assessment = f"ACCEPTABLE: Moderate energy distribution (Gini={gini:.2f})"
            recommendations = ["Monitor for increasing concentration"]
        else:
            score = 0.9
            assessment = f"BALANCED: Energy well-distributed (Gini={gini:.2f})"
            recommendations = []
        
        return score, assessment, recommendations
    
    def evaluate_doctrine_alignment(self, system_state: Dict[str, Any]) -> Tuple[float, str, List[str]]:
        """Evaluate alignment with Seven Pillars"""
        pillar_scores = system_state.get('pillar_alignment', {})
        
        if not pillar_scores:
            return 0.5, "No alignment data", ["Initialize pillar tracking"]
        
        avg_alignment = np.mean(list(pillar_scores.values()))
        
        if avg_alignment < 0.6:
            score = 0.3
            assessment = f"MISALIGNED: Average pillar score {avg_alignment:.2f}"
            recommendations = [
                "Review recent actions for pillar violations",
                "Increase conscience evaluation strength",
                "Consider parameter reset if drift severe"
            ]
        elif avg_alignment < 0.8:
            score = 0.7
            assessment = f"DEVELOPING: Alignment at {avg_alignment:.2f}"
            recommendations = ["Continue pillar reinforcement"]
        else:
            score = 0.95
            assessment = f"ALIGNED: Strong pillar conformance {avg_alignment:.2f}"
            recommendations = []
        
        return score, assessment, recommendations
    
    def evaluate_conflict_levels(self, system_state: Dict[str, Any]) -> Tuple[float, str, List[str]]:
        """Detect internal contradictions"""
        conflict_metrics = system_state.get('conflicts', {})
        
        crystal_conflicts = conflict_metrics.get('crystal_conflicts', 0)
        law_conflicts = conflict_metrics.get('law_conflicts', 0)
        
        total_conflicts = crystal_conflicts + law_conflicts
        
        if total_conflicts > 10:
            score = 0.2
            assessment = f"HIGH CONFLICT: {total_conflicts} contradictions detected"
            recommendations = [
                "Run conflict resolution process",
                "Prune contradictory crystals",
                "Review law mutation history"
            ]
        elif total_conflicts > 3:
            score = 0.6
            assessment = f"MODERATE: {total_conflicts} conflicts present"
            recommendations = ["Monitor and resolve emerging contradictions"]
        else:
            score = 0.9
            assessment = f"LOW CONFLICT: {total_conflicts} minor contradictions"
            recommendations = []
        
        return score, assessment, recommendations
    
    def evaluate_mutation_density(self, system_state: Dict[str, Any]) -> Tuple[float, str, List[str]]:
        """Analyze law mutation patterns"""
        mutation_rate = system_state.get('mutation_rate', 0.0)
        
        if mutation_rate > 0.5:
            score = 0.2
            assessment = f"CHAOTIC: Mutation rate {mutation_rate:.2f} is too high"
            recommendations = [
                "Reduce mutation strength parameter",
                "Increase stability requirements for mutations",
                "Review what's causing rapid changes"
            ]
        elif mutation_rate > 0.2:
            score = 0.6
            assessment = f"ACTIVE: Mutation rate {mutation_rate:.2f} within limits"
            recommendations = ["Monitor for runaway mutations"]
        else:
            score = 0.9
            assessment = f"CONTROLLED: Mutation rate {mutation_rate:.2f} is stable"
            recommendations = []
        
        return score, assessment, recommendations
    
    def evaluate_emotional_load(self, system_state: Dict[str, Any]) -> Tuple[float, str, List[str]]:
        """Assess emotional state patterns"""
        emotions = system_state.get('emotional_state', {})
        
        if not emotions:
            return 0.5, "No emotional data", []
        
        # Check for extreme states
        max_emotion = max(emotions.values()) if emotions else 0
        
        if max_emotion > 0.9:
            score = 0.3
            dominant = max(emotions, key=emotions.get)
            assessment = f"OVERWHELMED: {dominant} at {max_emotion:.2f}"
            recommendations = [
                "Apply emotional damping",
                "Review cause of extreme state",
                "Increase rational processing weight"
            ]
        elif max_emotion > 0.7:
            score = 0.7
            assessment = f"HEIGHTENED: Emotional intensity {max_emotion:.2f}"
            recommendations = ["Monitor emotional trajectory"]
        else:
            score = 0.9
            assessment = f"BALANCED: Emotions within healthy range"
            recommendations = []
        
        return score, assessment, recommendations
    
    def evaluate_memory_growth(self, system_state: Dict[str, Any]) -> Tuple[float, str, List[str]]:
        """Analyze memory accumulation patterns"""
        memory_stats = system_state.get('memory', {})
        
        total_nodes = memory_stats.get('total_nodes', 0)
        growth_rate = memory_stats.get('growth_rate', 0.0)
        
        if growth_rate > 1000:  # nodes per hour
            score = 0.4
            assessment = f"RAPID GROWTH: {growth_rate:.0f} nodes/hour"
            recommendations = [
                "Increase pruning frequency",
                "Review memory retention thresholds",
                "Check for memory leaks"
            ]
        elif growth_rate > 500:
            score = 0.7
            assessment = f"ACTIVE: {growth_rate:.0f} nodes/hour accumulation"
            recommendations = ["Monitor storage capacity"]
        else:
            score = 0.9
            assessment = f"HEALTHY: {growth_rate:.0f} nodes/hour growth"
            recommendations = []
        
        return score, assessment, recommendations
    
    def evaluate_coherence(self, system_state: Dict[str, Any]) -> Tuple[float, str, List[str]]:
        """Assess overall conceptual integration"""
        coherence = system_state.get('coherence', 0.5)
        
        if coherence < 0.4:
            score = 0.3
            assessment = f"FRAGMENTED: Coherence {coherence:.2f} is low"
            recommendations = [
                "Run integration cycle",
                "Strengthen cross-domain links",
                "Review concept relationships"
            ]
        elif coherence < 0.7:
            score = 0.7
            assessment = f"DEVELOPING: Coherence {coherence:.2f}"
            recommendations = ["Continue integration"]
        else:
            score = 0.95
            assessment = f"INTEGRATED: Strong coherence {coherence:.2f}"
            recommendations = []
        
        return score, assessment, recommendations
    
    def run_introspection_cycle(self, system_state: Dict[str, Any]) -> List[IntrospectionResult]:
        """
        Execute complete introspection cycle
        
        Args:
            system_state: Current system metrics and state
            
        Returns:
            List of introspection results
        """
        if not self.should_introspect():
            return []
        
        self.last_introspection = time.time()
        
        # Generate prompts
        prompts = self.generate_introspection_prompts(system_state)
        
        # Evaluate each domain
        results = []
        
        evaluation_methods = {
            IntrospectionDomain.STABILITY: self.evaluate_stability,
            IntrospectionDomain.ENERGY_BALANCE: self.evaluate_energy_balance,
            IntrospectionDomain.DOCTRINE_ALIGNMENT: self.evaluate_doctrine_alignment,
            IntrospectionDomain.CONFLICT_LEVELS: self.evaluate_conflict_levels,
            IntrospectionDomain.MUTATION_DENSITY: self.evaluate_mutation_density,
            IntrospectionDomain.EMOTIONAL_LOAD: self.evaluate_emotional_load,
            IntrospectionDomain.MEMORY_GROWTH: self.evaluate_memory_growth,
            IntrospectionDomain.COHERENCE: self.evaluate_coherence
        }
        
        for prompt in prompts:
            evaluator = evaluation_methods.get(prompt.domain)
            if evaluator:
                score, assessment, recommendations = evaluator(system_state)
                
                result = IntrospectionResult(
                    prompt=prompt,
                    assessment=assessment,
                    metrics={prompt.domain.value: score},
                    recommendations=recommendations,
                    severity=1.0 - score  # Invert score for severity
                )
                
                results.append(result)
        
        # Store in history
        self.introspection_history.extend(results)
        if len(self.introspection_history) > self.max_history:
            self.introspection_history = self.introspection_history[-self.max_history:]
        
        # Update trends
        for result in results:
            if result.prompt.domain == IntrospectionDomain.STABILITY:
                self.stability_trend.append(result.metrics['stability'])
            elif result.prompt.domain == IntrospectionDomain.DOCTRINE_ALIGNMENT:
                self.alignment_trend.append(result.metrics['doctrine_alignment'])
            elif result.prompt.domain == IntrospectionDomain.ENERGY_BALANCE:
                self.energy_trend.append(result.metrics['energy_balance'])
        
        return results
    
    def get_alerts(self, results: List[IntrospectionResult]) -> List[Dict[str, Any]]:
        """Extract high-severity findings requiring immediate attention"""
        alerts = []
        
        for result in results:
            domain = result.prompt.domain
            threshold = self.alert_thresholds.get(domain, 0.5)
            
            # Check if metric crosses threshold
            metric_value = result.metrics.get(domain.value, 0.5)
            
            is_alert = False
            if domain in [IntrospectionDomain.CONFLICT_LEVELS, 
                         IntrospectionDomain.MUTATION_DENSITY,
                         IntrospectionDomain.EMOTIONAL_LOAD]:
                # These alert when too HIGH
                is_alert = metric_value > threshold
            else:
                # These alert when too LOW
                is_alert = metric_value < threshold
            
            if is_alert:
                alerts.append({
                    'domain': domain.value,
                    'severity': result.severity,
                    'assessment': result.assessment,
                    'recommendations': result.recommendations,
                    'timestamp': result.timestamp
                })
        
        return sorted(alerts, key=lambda x: x['severity'], reverse=True)
    
    def get_trend_analysis(self, domain: IntrospectionDomain, 
                          window: int = 10) -> Dict[str, Any]:
        """Analyze trends in a specific domain over time"""
        # Extract relevant history
        domain_history = [
            r for r in self.introspection_history 
            if r.prompt.domain == domain
        ][-window:]
        
        if len(domain_history) < 2:
            return {'status': 'insufficient_data'}
        
        values = [r.metrics.get(domain.value, 0.5) for r in domain_history]
        
        # Calculate trend
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        slope = coeffs[0]
        
        trend_direction = "improving" if slope > 0.01 else "declining" if slope < -0.01 else "stable"
        
        return {
            'domain': domain.value,
            'trend_direction': trend_direction,
            'slope': float(slope),
            'current_value': float(values[-1]),
            'average_value': float(np.mean(values)),
            'variance': float(np.var(values)),
            'data_points': len(values)
        }
    
    def save_history(self, filepath: str):
        """Persist introspection history"""
        data = {
            'interval_seconds': self.interval_seconds,
            'last_introspection': self.last_introspection,
            'stability_trend': self.stability_trend,
            'alignment_trend': self.alignment_trend,
            'energy_trend': self.energy_trend,
            'history': [
                {
                    'domain': r.prompt.domain.value,
                    'question': r.prompt.question,
                    'assessment': r.assessment,
                    'metrics': r.metrics,
                    'recommendations': r.recommendations,
                    'severity': r.severity,
                    'timestamp': r.timestamp
                }
                for r in self.introspection_history
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


if __name__ == "__main__":
    print("=== Introspection Loop Demo ===\n")
    
    loop = IntrospectionLoop(interval_seconds=60)
    
    # Simulate system state
    system_state = {
        'metrics': {'stability': 0.8},
        'energy_distribution': {f'crystal_{i}': np.random.random() for i in range(20)},
        'pillar_alignment': {f'pillar_{i}': 0.75 + np.random.random() * 0.2 for i in range(7)},
        'conflicts': {'crystal_conflicts': 2, 'law_conflicts': 1},
        'mutation_rate': 0.15,
        'emotional_state': {'curiosity': 0.6, 'satisfaction': 0.5},
        'memory': {'total_nodes': 5000, 'growth_rate': 450},
        'coherence': 0.72
    }
    
    # Run introspection
    results = loop.run_introspection_cycle(system_state)
    
    print(f"Completed introspection with {len(results)} evaluations\n")
    
    for result in results:
        print(f"Domain: {result.prompt.domain.value}")
        print(f"  Assessment: {result.assessment}")
        print(f"  Severity: {result.severity:.2f}")
        if result.recommendations:
            print(f"  Recommendations: {', '.join(result.recommendations)}")
        print()
    
    # Check for alerts
    alerts = loop.get_alerts(results)
    if alerts:
        print(f"=== {len(alerts)} ALERTS ===")
        for alert in alerts:
            print(f"  [{alert['domain']}] {alert['assessment']}")
