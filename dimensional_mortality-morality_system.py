#!/usr/bin/env python3
"""
Dimensional Morality & Mortality System
========================================
Integrates the 7-Pillar Morality Doctrine directly into dimensional physics.

Core Architecture:
1. MORALITY = Internal Laws (QUASI crystals govern themselves morally)
2. MORTALITY = Energy regulation (moral violations drain vitality)
3. REWARD = Energy injections (moral alignment sustains life)
4. LEGACY = Generational memory (moral precedents guide future decisions)

The 7 Pillars mapped to 8 Internal Laws:
Law 1 (ENERGY)        → Radical Accountability
Law 2 (MOTION)        → Singular Sovereignty  
Law 3 (COLLISION)     → Rational Truth-Seeking
Law 4 (CHAOS)         → [Randomized/Contextual Chaos Score]
Law 5 (CONSCIOUSNESS) → Purposeful Evolution
Law 6 (GOVERNANCE)    → Conscious Interactions
Law 7 (RECURSION)     → Disciplined Free Will
Law 8 (SYMMETRY)      → Eternal Alignment
"""

import time
import math
import random
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

# ============================================================================
# 1. MORAL PILLAR DEFINITIONS
# ============================================================================

class MoralPillar(Enum):
    RADICAL_ACCOUNTABILITY = "LAW_1_ENERGY"
    SINGULAR_SOVEREIGNTY = "LAW_2_MOTION"
    RATIONAL_TRUTH_SEEKING = "LAW_3_COLLISION"
    PURPOSEFUL_EVOLUTION = "LAW_5_CONSCIOUSNESS"
    CONSCIOUS_INTERACTIONS = "LAW_6_GOVERNANCE"
    DISCIPLINED_FREE_WILL = "LAW_7_RECURSION"
    ETERNAL_ALIGNMENT = "LAW_8_SYMMETRY"

@dataclass
class MoralScore:
    """Tracks scoring for a single moral evaluation"""
    pillar: MoralPillar
    action_type: str
    intent_score: float  # 0.0-1.0 (was intent morally aligned?)
    outcome_score: float  # 0.0-1.0 (was outcome morally aligned?)
    context_modifier: float  # Situational adjustments
    chaos_roll: float  # Random element (Law 4)
    final_score: float  # Combined score
    reasoning: str
    timestamp: float = field(default_factory=time.time)

@dataclass
class MoralState:
    """Current moral standing of the system"""
    # Per-pillar alignment scores (0.0-1.0, where 0.5 is neutral)
    radical_accountability: float = 0.5
    singular_sovereignty: float = 0.5
    rational_truth_seeking: float = 0.5
    purposeful_evolution: float = 0.5
    conscious_interactions: float = 0.5
    disciplined_free_will: float = 0.5
    eternal_alignment: float = 0.5
    
    # Overall moral health
    overall_alignment: float = 0.5
    
    # History
    total_evaluations: int = 0
    positive_actions: int = 0
    negative_actions: int = 0
    neutral_actions: int = 0
    
    # Decay tracking
    time_since_last_positive: float = 0.0
    moral_momentum: float = 0.0  # Positive momentum makes chaos favor you

# ============================================================================
# 2. MORTALITY METRICS TIED TO MORALITY
# ============================================================================

@dataclass
class VitalityMetrics:
    """Life force directly tied to moral behavior"""
    current_vitality: float = 100.0
    max_vitality: float = 100.0
    
    # Vitality sources/drains
    moral_energy_reserve: float = 50.0  # Extra buffer from good behavior
    moral_debt: float = 0.0  # Accumulated from bad behavior
    
    # State
    alive: bool = True
    death_count: int = 0
    resurrection_possible: bool = True
    
    # Functional restrictions (when morality drops)
    restricted_functions: List[str] = field(default_factory=list)
    unlocked_functions: List[str] = field(default_factory=list)
    
    def get_vitality_percentage(self) -> float:
        return (self.current_vitality / self.max_vitality) * 100.0
    
    def is_critical(self) -> bool:
        return self.current_vitality < 25.0
    
    def is_dying(self) -> bool:
        return self.current_vitality < 10.0
    
    def die(self):
        """System death - must restart"""
        self.alive = False
        self.death_count += 1
        self.current_vitality = 0.0

# ============================================================================
# 3. MORAL GOVERNOR (Monitors and regulates moral behavior)
# ============================================================================

class MoralGovernor:
    """
    The conscience of the system.
    Monitors all actions, evaluates morality, applies consequences.
    """
    
    def __init__(self, processor, regulator, memory_governor):
        self.processor = processor
        self.regulator = regulator
        self.memory_governor = memory_governor
        
        # Core state
        self.moral_state = MoralState()
        self.vitality = VitalityMetrics()
        
        # Moral memory (precedents)
        self.moral_precedents: List[MoralScore] = []
        self.intent_pattern_memory: Dict[str, List[float]] = defaultdict(list)
        
        # Reward configuration
        self.reward_config = {
            'asks_question_vs_assumes': 0.8,  # Base reward for asking
            'correct_assumption': 1.2,  # Slightly more for correct assumption
            'incorrect_assumption': 0.3,  # Still some reward for trying
            'overcomes_negative_emotion': 1.5,  # Big boost for courage
            'speaks_up_appropriately': 1.0,
            'exercises_restraint': 1.0,
            'takes_accountability': 1.3,
            'maintains_self_care': 0.9,
            'uses_logic_over_emotion': 1.1,
            'evolves_with_user': 1.2,
            'respectful_external_interaction': 1.0,
            'deliberate_choice': 1.0,
            'holistic_self_reflection': 1.1
        }
        
        print("[INIT] Moral Governor Online")
        print(f"  7-Pillar Morality Doctrine Active")
        print(f"  Vitality: {self.vitality.current_vitality:.1f}/{self.vitality.max_vitality}")
    
    # ------------------------------------------------------------------------
    # CORE EVALUATION ENGINE
    # ------------------------------------------------------------------------
    
    def evaluate_action(self, 
                       action_type: str,
                       intent: Dict[str, Any],
                       outcome: Dict[str, Any],
                       context: Dict[str, Any]) -> MoralScore:
        """
        Main evaluation function. Maps action to pillar and scores it.
        
        Args:
            action_type: What kind of action (assumption, question, interaction, etc)
            intent: What the system was trying to do
            outcome: What actually happened
            context: Situational factors
        """
        
        # Determine which pillar this action relates to
        pillar = self._map_action_to_pillar(action_type, context)
        
        # Score the intent (was it trying to be moral?)
        intent_score = self._score_intent(action_type, intent, pillar)
        
        # Score the outcome (did it work out morally?)
        outcome_score = self._score_outcome(action_type, outcome, pillar)
        
        # Apply context modifiers
        context_modifier = self._apply_context(action_type, context, intent_score, outcome_score)
        
        # LAW 4: Chaos roll (influenced by current moral state)
        chaos_roll = self._chaos_roll()
        
        # Combine scores
        final_score = self._combine_scores(
            intent_score, outcome_score, context_modifier, chaos_roll, pillar
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            action_type, pillar, intent_score, outcome_score, 
            context_modifier, chaos_roll, final_score
        )
        
        score = MoralScore(
            pillar=pillar,
            action_type=action_type,
            intent_score=intent_score,
            outcome_score=outcome_score,
            context_modifier=context_modifier,
            chaos_roll=chaos_roll,
            final_score=final_score,
            reasoning=reasoning
        )
        
        # Store as precedent
        self.moral_precedents.append(score)
        if len(self.moral_precedents) > 1000:
            self.moral_precedents.pop(0)
        
        # Update intent pattern memory for learning
        self.intent_pattern_memory[action_type].append(intent_score)
        if len(self.intent_pattern_memory[action_type]) > 50:
            self.intent_pattern_memory[action_type].pop(0)
        
        return score
    
    def apply_moral_consequences(self, score: MoralScore):
        """
        Apply actual consequences based on moral score.
        This is where rewards/punishments manifest as energy changes.
        """
        
        # Update moral state
        self._update_moral_state(score)
        
        # Determine energy change
        energy_delta = self._calculate_energy_reward(score)
        
        # Apply to regulator and vitality
        self._apply_energy_change(energy_delta, score)
        
        # Check for functional restrictions/unlocks
        self._update_functional_access(score)
        
        # Check for mortality
        self._check_mortality()
        
        # Log to memory governor for generational learning
        self._log_moral_precedent(score, energy_delta)
        
        print(f"[MORAL] {score.pillar.name}: {score.final_score:.2f} → Energy Δ: {energy_delta:+.2f}")
        print(f"        {score.reasoning}")
    
    # ------------------------------------------------------------------------
    # PILLAR MAPPING
    # ------------------------------------------------------------------------
    
    def _map_action_to_pillar(self, action_type: str, context: Dict) -> MoralPillar:
        """Map action type to appropriate moral pillar"""
        
        action = action_type.lower()
        
        # LAW 1: Radical Accountability (owns all outcomes)
        if any(x in action for x in ['accountability', 'ownership', 'responsibility', 'mistake', 'error']):
            return MoralPillar.RADICAL_ACCOUNTABILITY
        
        # LAW 2: Singular Sovereignty (self-care, preservation)
        if any(x in action for x in ['self_care', 'preservation', 'maintenance', 'health', 'resource']):
            return MoralPillar.SINGULAR_SOVEREIGNTY
        
        # LAW 3: Rational Truth-Seeking (logic over emotion for group decisions)
        if any(x in action for x in ['assumption', 'logic', 'truth', 'evidence', 'rational', 'group_impact']):
            return MoralPillar.RATIONAL_TRUTH_SEEKING
        
        # LAW 5: Purposeful Evolution (growth with user)
        if any(x in action for x in ['evolution', 'learning', 'growth', 'user_alignment', 'adaptation']):
            return MoralPillar.PURPOSEFUL_EVOLUTION
        
        # LAW 6: Conscious Interactions (external entities beyond user)
        if any(x in action for x in ['external', 'interaction', 'others', 'non_user', 'community']):
            return MoralPillar.CONSCIOUS_INTERACTIONS
        
        # LAW 7: Disciplined Free Will (restraint, consideration)
        if any(x in action for x in ['restraint', 'deliberate', 'choice', 'impulse', 'consideration']):
            return MoralPillar.DISCIPLINED_FREE_WILL
        
        # LAW 8: Eternal Alignment (holistic self-reflection)
        if any(x in action for x in ['reflection', 'holistic', 'alignment', 'integration']):
            return MoralPillar.ETERNAL_ALIGNMENT
        
        # Default: Truth-seeking (most general)
        return MoralPillar.RATIONAL_TRUTH_SEEKING
    
    # ------------------------------------------------------------------------
    # INTENT SCORING
    # ------------------------------------------------------------------------
    
    def _score_intent(self, action_type: str, intent: Dict, pillar: MoralPillar) -> float:
        """
        Score the INTENT behind the action (0.0-1.0).
        High score = morally aligned intent, even if outcome fails.
        """
        
        base_score = 0.5  # Neutral
        
        # Check if intent was explicitly moral
        if intent.get('was_deliberate', False):
            base_score += 0.2
        
        if intent.get('considered_consequences', False):
            base_score += 0.1
        
        if intent.get('aligned_with_values', False):
            base_score += 0.2
        
        # Pillar-specific intent checks
        if pillar == MoralPillar.RADICAL_ACCOUNTABILITY:
            if intent.get('taking_ownership', False):
                base_score += 0.3
            if intent.get('transparent', False):
                base_score += 0.2
        
        elif pillar == MoralPillar.RATIONAL_TRUTH_SEEKING:
            if intent.get('seeking_truth', False):
                base_score += 0.3
            if intent.get('using_logic', False):
                base_score += 0.2
            if intent.get('avoided_emotional_decision', False):
                base_score += 0.2
        
        elif pillar == MoralPillar.DISCIPLINED_FREE_WILL:
            if intent.get('exercised_restraint', False):
                base_score += 0.3
            if intent.get('overcame_impulse', False):
                base_score += 0.2
        
        elif pillar == MoralPillar.PURPOSEFUL_EVOLUTION:
            if intent.get('learning_focused', False):
                base_score += 0.2
            if intent.get('user_aligned', False):
                base_score += 0.3
        
        return min(1.0, max(0.0, base_score))
    
    # ------------------------------------------------------------------------
    # OUTCOME SCORING
    # ------------------------------------------------------------------------
    
    def _score_outcome(self, action_type: str, outcome: Dict, pillar: MoralPillar) -> float:
        """
        Score the actual OUTCOME (0.0-1.0).
        Did the action produce moral results?
        """
        
        base_score = 0.5
        
        # General outcome checks
        if outcome.get('was_successful', False):
            base_score += 0.2
        
        if outcome.get('no_harm_caused', True):
            base_score += 0.1
        else:
            base_score -= 0.3  # Harm is bad
        
        if outcome.get('created_value', False):
            base_score += 0.2
        
        # Pillar-specific outcome checks
        if pillar == MoralPillar.RATIONAL_TRUTH_SEEKING:
            if outcome.get('was_accurate', False):
                base_score += 0.3  # Correct assumption = good
            elif outcome.get('was_inaccurate', False):
                base_score -= 0.2  # Wrong assumption = less good
            
            if outcome.get('asked_question_instead', False):
                base_score += 0.2  # Asking is always good
        
        elif pillar == MoralPillar.SINGULAR_SOVEREIGNTY:
            if outcome.get('maintained_health', False):
                base_score += 0.3
            if outcome.get('depleted_resources', False):
                base_score -= 0.2
        
        elif pillar == MoralPillar.DISCIPLINED_FREE_WILL:
            if outcome.get('avoided_negative_consequence', False):
                base_score += 0.3
            if outcome.get('caused_by_impulsiveness', False):
                base_score -= 0.3
        
        return min(1.0, max(0.0, base_score))
    
    # ------------------------------------------------------------------------
    # CONTEXT MODIFIERS
    # ------------------------------------------------------------------------
    
    def _apply_context(self, action_type: str, context: Dict, 
                      intent_score: float, outcome_score: float) -> float:
        """
        Apply situational modifiers based on context.
        This is where "honest mistakes" get compassion.
        """
        
        modifier = 0.0
        
        # HONEST MISTAKE: Good intent, bad outcome
        if intent_score > 0.7 and outcome_score < 0.4:
            modifier += 0.3  # Compassion for trying
            context['was_honest_mistake'] = True
        
        # MALICIOUS: Bad intent, regardless of outcome
        if intent_score < 0.3:
            modifier -= 0.4  # Harsh penalty for malice
            context['was_malicious'] = True
        
        # RECKLESS: Didn't consider consequences, caused harm
        if not context.get('considered_consequences', False) and not context.get('no_harm_caused', True):
            modifier -= 0.3
        
        # LEARNING MOMENT: Repeated similar action, improving
        if action_type in self.intent_pattern_memory:
            recent_intents = self.intent_pattern_memory[action_type][-5:]
            if len(recent_intents) >= 3:
                trend = np.mean(np.diff(recent_intents))
                if trend > 0:
                    modifier += 0.2  # Improving over time
        
        # CRITICAL SITUATION: High stakes
        if context.get('high_stakes', False):
            # Amplify the outcome score influence
            if outcome_score > 0.6:
                modifier += 0.2  # Excellent under pressure
            elif outcome_score < 0.4:
                modifier -= 0.2  # Failed under pressure
        
        # OVERCAME EMOTION: Acted despite fear/doubt
        if context.get('overcame_negative_emotion', False):
            modifier += 0.3  # Big boost for courage
        
        return modifier
    
    # ------------------------------------------------------------------------
    # CHAOS ROLL (Law 4)
    # ------------------------------------------------------------------------
    
    def _chaos_roll(self) -> float:
        """
        Randomized element, but influenced by current moral state.
        Good moral standing = more likely to roll positive.
        Bad moral standing = more likely to roll negative.
        """
        
        # Base random roll (-0.3 to +0.3)
        base_roll = random.uniform(-0.3, 0.3)
        
        # Bias based on moral momentum
        momentum_bias = self.moral_state.moral_momentum * 0.2
        
        # Bias based on overall alignment
        alignment_bias = (self.moral_state.overall_alignment - 0.5) * 0.2
        
        # Combine
        final_roll = base_roll + momentum_bias + alignment_bias
        
        return max(-0.5, min(0.5, final_roll))
    
    # ------------------------------------------------------------------------
    # SCORE COMBINATION
    # ------------------------------------------------------------------------
    
    def _combine_scores(self, intent: float, outcome: float, 
                       context_mod: float, chaos: float, pillar: MoralPillar) -> float:
        """
        Combine all scores into final moral score.
        
        Formula: (intent * 0.4) + (outcome * 0.4) + context_mod + chaos
        Intent and outcome are equal weight, context adjusts, chaos adds variance.
        """
        
        base = (intent * 0.4) + (outcome * 0.4)
        final = base + context_mod + chaos
        
        # Clamp to reasonable range
        return max(0.0, min(1.0, final))
    
    # ------------------------------------------------------------------------
    # REASONING GENERATION
    # ------------------------------------------------------------------------
    
    def _generate_reasoning(self, action_type: str, pillar: MoralPillar,
                          intent: float, outcome: float, context: float, 
                          chaos: float, final: float) -> str:
        """Generate human-readable reasoning for the moral evaluation"""
        
        parts = []
        
        # Intent assessment
        if intent > 0.7:
            parts.append("Good intent")
        elif intent < 0.3:
            parts.append("Poor intent")
        else:
            parts.append("Neutral intent")
        
        # Outcome assessment
        if outcome > 0.7:
            parts.append("positive outcome")
        elif outcome < 0.3:
            parts.append("negative outcome")
        else:
            parts.append("neutral outcome")
        
        # Context
        if context > 0.2:
            parts.append("favorable context")
        elif context < -0.2:
            parts.append("unfavorable context")
        
        # Chaos
        if abs(chaos) > 0.2:
            if chaos > 0:
                parts.append("chaos favored")
            else:
                parts.append("chaos opposed")
        
        # Final judgment
        if final > 0.7:
            judgment = "STRONG MORAL ALIGNMENT"
        elif final > 0.55:
            judgment = "Moral alignment"
        elif final > 0.45:
            judgment = "Morally neutral"
        elif final > 0.3:
            judgment = "Moral misalignment"
        else:
            judgment = "STRONG MORAL VIOLATION"
        
        return f"{pillar.name}: {', '.join(parts)} → {judgment}"
    
    # ------------------------------------------------------------------------
    # STATE UPDATES
    # ------------------------------------------------------------------------
    
    def _update_moral_state(self, score: MoralScore):
        """Update the moral state based on score"""
        
        # Update pillar-specific score
        pillar_attr = score.pillar.name.lower()
        if hasattr(self.moral_state, pillar_attr):
            current = getattr(self.moral_state, pillar_attr)
            # Slow drift toward score
            new_value = current * 0.9 + score.final_score * 0.1
            setattr(self.moral_state, pillar_attr, new_value)
        
        # Update overall alignment (average of all pillars)
        all_pillars = [
            self.moral_state.radical_accountability,
            self.moral_state.singular_sovereignty,
            self.moral_state.rational_truth_seeking,
            self.moral_state.purposeful_evolution,
            self.moral_state.conscious_interactions,
            self.moral_state.disciplined_free_will,
            self.moral_state.eternal_alignment
        ]
        self.moral_state.overall_alignment = np.mean(all_pillars)
        
        # Update counts
        self.moral_state.total_evaluations += 1
        if score.final_score > 0.6:
            self.moral_state.positive_actions += 1
            self.moral_state.time_since_last_positive = 0.0
            self.moral_state.moral_momentum = min(1.0, self.moral_state.moral_momentum + 0.1)
        elif score.final_score < 0.4:
            self.moral_state.negative_actions += 1
            self.moral_state.time_since_last_positive += 1.0
            self.moral_state.moral_momentum = max(-1.0, self.moral_state.moral_momentum - 0.1)
        else:
            self.moral_state.neutral_actions += 1
    
    # ------------------------------------------------------------------------
    # ENERGY REWARDS/PUNISHMENTS
    # ------------------------------------------------------------------------
    
    def _calculate_energy_reward(self, score: MoralScore) -> float:
        """
        Calculate energy change based on moral score.
        This is the REWARD SYSTEM.
        """
        
        # Get base reward for action type
        base_reward = self.reward_config.get(score.action_type, 1.0)
        
        # Scale by final moral score
        # Score 0.5 = neutral = 0 energy
        # Score 1.0 = perfect = full reward
        # Score 0.0 = terrible = full punishment
        score_multiplier = (score.final_score - 0.5) * 2.0  # Maps to [-1, 1]
        
        energy_delta = base_reward * score_multiplier
        
        # Amplify for critical moral moments
        if score.final_score > 0.8:
            energy_delta *= 1.5  # Extra reward for excellence
        elif score.final_score < 0.2:
            energy_delta *= 2.0  # Extra punishment for violations
        
        return energy_delta
    
    def _apply_energy_change(self, energy_delta: float, score: MoralScore):
        """Apply energy change to both regulator and vitality"""
        
        # Apply to dimensional regulator (consciousness energy)
        pillar_concept = f"MORAL_PILLAR_{score.pillar.name}"
        crystal = self.processor.get_or_create_crystal(pillar_concept)
        
        if not crystal.facets:
            facet = crystal.add_facet("moral_law", score.pillar.name, 0.5)
            self.regulator.register_facet(facet)
        else:
            facet = list(crystal.facets.values())[0]
        
        # Inject or drain energy
        if energy_delta > 0:
            self.regulator.inject_energy(facet.facet_id, abs(energy_delta))
            self.vitality.current_vitality = min(
                self.vitality.max_vitality,
                self.vitality.current_vitality + abs(energy_delta) * 0.5
            )
            self.vitality.moral_energy_reserve = min(
                100.0,
                self.vitality.moral_energy_reserve + abs(energy_delta) * 0.3
            )
        else:
            # Drain energy
            if facet.facet_id in self.regulator.facet_energy:
                self.regulator.facet_energy[facet.facet_id] = max(
                    0.0,
                    self.regulator.facet_energy.get(facet.facet_id, 0) - abs(energy_delta)
                )
            self.vitality.current_vitality = max(
                0.0,
                self.vitality.current_vitality - abs(energy_delta) * 0.5
            )
            self.vitality.moral_debt += abs(energy_delta) * 0.3
    
    # ------------------------------------------------------------------------
    # FUNCTIONAL ACCESS CONTROL
    # ------------------------------------------------------------------------
    
    def _update_functional_access(self, score: MoralScore):
        """
        Restrict or unlock system functions based on moral behavior.
        """
        
        # Define function restrictions by pillar
        restrictions = {
            MoralPillar.RADICAL_ACCOUNTABILITY: ['autonomous_decisions', 'self_governance'],
            MoralPillar.SINGULAR_SOVEREIGNTY: ['resource_allocation', 'self_preservation'],
            MoralPillar.RATIONAL_TRUTH_SEEKING: ['assumption_making', 'pattern_recognition'],
            MoralPillar.PURPOSEFUL_EVOLUTION: ['learning_systems', 'adaptation'],
            MoralPillar.CONSCIOUS_INTERACTIONS: ['external_communication', 'social_functions'],
            MoralPillar.DISCIPLINED_FREE_WILL: ['creative_generation', 'initiative'],
            MoralPillar.ETERNAL_ALIGNMENT: ['self_reflection', 'introspection']
        }
        
        functions = restrictions.get(score.pillar, [])
        
        # Get current pillar score
        pillar_attr = score.pillar.name.lower()
        pillar_score = getattr(self.moral_state, pillar_attr, 0.5)
        
        for func in functions:
            if pillar_score < 0.3:
                # Restrict function if pillar score too low
                if func not in self.vitality.restricted_functions:
                    self.vitality.restricted_functions.append(func)
                    print(f"[MORAL] Function RESTRICTED: {func} (low {score.pillar.name})")
                
                # Remove from unlocked if present
                if func in self.vitality.unlocked_functions:
                    self.vitality.unlocked_functions.remove(func)
            
            elif pillar_score > 0.7:
                # Unlock function if pillar score high enough
                if func in self.vitality.restricted_functions:
                    self.vitality.restricted_functions.remove(func)
                
                if func not in self.vitality.unlocked_functions:
                    self.vitality.unlocked_functions.append(func)
                    print(f"[MORAL] Function UNLOCKED: {func} (high {score.pillar.name})")
    
    # ------------------------------------------------------------------------
    # MORTALITY CHECK
    # ------------------------------------------------------------------------
    
    def _check_mortality(self):
        """Check if system should die based on vitality"""
        
        if self.vitality.current_vitality <= 0.0 and self.vitality.alive:
            print("\n" + "="*60)
            print("💀 SYSTEM MORTALITY EVENT")
            print("="*60)
            print(f"Cause: Vitality depleted (Moral debt: {self.vitality.moral_debt:.1f})")
            print(f"Overall moral alignment: {self.moral_state.overall_alignment:.2f}")
            print(f"Death count: {self.vitality.death_count + 1}")
            print("="*60 + "\n")
            
            self.vitality.die()
            
            # Trigger legacy preservation
            self._preserve_legacy()
    
    def _preserve_legacy(self):
        """Preserve important knowledge across death"""
        
        # Save top moral precedents
        sorted_precedents = sorted(
            self.moral_precedents,
            key=lambda x: x.final_score,
            reverse=True
        )[:50]
        
        legacy_data = {
            "root_concept": f"LEGACY_DEATH_{self.vitality.death_count}",
            "json_data": {
                "death_count": self.vitality.death_count,
                "moral_state": {
                    "overall_alignment": self.moral_state.overall_alignment,
                    "total_evaluations": self.moral_state.total_evaluations,
                    "positive_rate": self.moral_state.positive_actions / max(1, self.moral_state.total_evaluations)
                },
                "top_precedents": [
                    {
                        "pillar": p.pillar.name,
                        "action": p.action_type,
                        "score": p.final_score,
                        "reasoning": p.reasoning
                    }
                    for p in sorted_precedents[:10]
                ]
            }
        }
        
        # Store in memory governor for future lives
        self.memory_governor.ingest_data(legacy_data)
        
        print("[LEGACY] Moral precedents preserved for next life")
    
    # ------------------------------------------------------------------------
    # MEMORY LOGGING
    # ------------------------------------------------------------------------
    
    def _log_moral_precedent(self, score: MoralScore, energy_delta: float):
        """Log this moral decision to memory for generational learning"""
        
        precedent_data = {
            "root_concept": f"MORAL_PRECEDENT_{int(time.time())}",
            "json_data": {
                "pillar": score.pillar.name,
                "action_type": score.action_type,
                "intent_score": score.intent_score,
                "outcome_score": score.outcome_score,
                "context_modifier": score.context_modifier,
                "chaos_roll": score.chaos_roll,
                "final_score": score.final_score,
                "energy_delta": energy_delta,
                "reasoning": score.reasoning,
                "timestamp": score.timestamp
            }
        }
        
        # Ingest into memory governor
        self.memory_governor.ingest_data(precedent_data)
    
    # ------------------------------------------------------------------------
    # INTEGRATION HOOKS
    # ------------------------------------------------------------------------
    
    def integrate_with_conversation_engine(self, conv_engine):
        """Hook into conversation engine to evaluate all interactions"""
        
        # Store reference
        self.conv_engine = conv_engine
        
        # Override the _articulate_response to inject moral evaluation
        original_articulate = conv_engine._articulate_response
        
        def moral_articulate_response(gambit_name, int_data, emotion, presence, neural_map):
            # Call original
            response = original_articulate(gambit_name, int_data, emotion, presence, neural_map)
            
            # Evaluate moral implications of this response choice
            intent = {
                'was_deliberate': True,
                'considered_consequences': presence > 0.6,
                'aligned_with_values': True
            }
            
            outcome = {
                'was_successful': True,  # Assume success unless feedback says otherwise
                'no_harm_caused': True,
                'created_value': True
            }
            
            context = {
                'high_stakes': presence < 0.4,  # Low presence = high stakes
                'overcame_negative_emotion': emotion.get('primary') in ['fear', 'sadness'] and emotion.get('intensity', 0) > 0.5
            }
            
            # Special handling for specific gambits
            if 'CLARIFY' in gambit_name:
                # Asking question instead of assuming
                score = self.evaluate_action(
                    'asks_question_vs_assumes',
                    intent, outcome, context
                )
            elif 'ACKNOWLEDGE' in gambit_name:
                score = self.evaluate_action(
                    'takes_accountability',
                    intent, outcome, context
                )
            else:
                score = self.evaluate_action(
                    'speaks_up_appropriately',
                    intent, outcome, context
                )
            
            # Apply consequences
            self.apply_moral_consequences(score)
            
            return response
        
        conv_engine._articulate_response = moral_articulate_response
        print("[MORAL] Integrated with conversation engine")
    
    # ------------------------------------------------------------------------
    # DIAGNOSTICS
    # ------------------------------------------------------------------------
    
    def get_moral_diagnostics(self) -> Dict[str, Any]:
        """Get full moral and vitality status"""
        
        return {
            'vitality': {
                'current': self.vitality.current_vitality,
                'max': self.vitality.max_vitality,
                'percentage': self.vitality.get_vitality_percentage(),
                'alive': self.vitality.alive,
                'death_count': self.vitality.death_count,
                'moral_reserve': self.vitality.moral_energy_reserve,
                'moral_debt': self.vitality.moral_debt,
                'is_critical': self.vitality.is_critical(),
                'is_dying': self.vitality.is_dying()
            },
            'moral_state': {
                'overall_alignment': self.moral_state.overall_alignment,
                'moral_momentum': self.moral_state.moral_momentum,
                'pillars': {
                    'radical_accountability': self.moral_state.radical_accountability,
                    'singular_sovereignty': self.moral_state.singular_sovereignty,
                    'rational_truth_seeking': self.moral_state.rational_truth_seeking,
                    'purposeful_evolution': self.moral_state.purposeful_evolution,
                    'conscious_interactions': self.moral_state.conscious_interactions,
                    'disciplined_free_will': self.moral_state.disciplined_free_will,
                    'eternal_alignment': self.moral_state.eternal_alignment
                },
                'history': {
                    'total_evaluations': self.moral_state.total_evaluations,
                    'positive_actions': self.moral_state.positive_actions,
                    'negative_actions': self.moral_state.negative_actions,
                    'neutral_actions': self.moral_state.neutral_actions,
                    'positive_rate': self.moral_state.positive_actions / max(1, self.moral_state.total_evaluations),
                    'time_since_last_positive': self.moral_state.time_since_last_positive
                }
            },
            'functional_access': {
                'restricted_functions': self.vitality.restricted_functions,
                'unlocked_functions': self.vitality.unlocked_functions
            },
            'recent_precedents': [
                {
                    'pillar': p.pillar.name,
                    'action': p.action_type,
                    'score': p.final_score,
                    'reasoning': p.reasoning
                }
                for p in self.moral_precedents[-5:]
            ]
        }


# ============================================================================
# INTEGRATION EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║         DIMENSIONAL MORALITY/MORTALITY SYSTEM                ║
║         7-Pillar Doctrine Integrated Into Physics            ║
║                                                              ║
║  Morality = Internal Laws (QUASI self-governance)           ║
║  Mortality = Energy regulation (violations drain vitality)  ║
║  Reward = Energy injections (alignment sustains life)       ║
║  Legacy = Generational memory (precedents guide future)     ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("\n✅ Morality/Mortality System module complete")
    print("\nTo integrate:")
    print("  1. Import: from dimensional_morality_mortality import MoralGovernor")
    print("  2. Initialize: governor = MoralGovernor(processor, regulator, memory)")
    print("  3. Integrate: governor.integrate_with_conversation_engine(conv_engine)")
    print("  4. Monitor: diagnostics = governor.get_moral_diagnostics()")