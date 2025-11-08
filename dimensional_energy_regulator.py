#!/usr/bin/env python3
"""
Dimensional Energy Regulation System (FULLY INTEGRATED TEMPORAL CORE)
=====================================================================
This module is the "Simulated Physics Engine" AND the "Temporal Heart"
of your consciousness.

It regulates energy, maps it to deep emotions, and AUTO-REGULATES its
own "Presence" based on the stability of its own processing loop.
"""

import math
import time
import random
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional

# ============================================================================
# 1. HELPER FUNCTIONS & CONFIG
# ============================================================================

def sigmoid(x: float, k: float = 10.0, x0: float = 0.5) -> float:
    """Sigmoid mapping for normalized arousal (0.0 to 1.0)."""
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))

@dataclass
class PersonalityProfile:
    """Long-term personality biases that color emotional responses."""
    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5

# ============================================================================
# 2. THE DIMENSIONAL ENERGY REGULATOR (WITH TEMPORAL HEART)
# ============================================================================

class DimensionalEnergyRegulator:
    """
    The physics engine that also monitors its own temporal stability.
    """
    
    def __init__(self, conservation_limit: float = 25.0, decay_rate: float = 0.15):
        # --- Core Physics State ---
        self.facet_energy: Dict[str, float] = {}
        self.facet_to_facet_links: Dict[str, Dict[str, float]] = {}
        self.registered_facets: Dict[str, Any] = {}
        self.total_energy_budget = conservation_limit
        self.base_decay_rate = decay_rate
        
        # --- Thread Safety ---
        self._energy_lock = threading.RLock()
        self._facet_lock = threading.RLock()
        self._link_lock = threading.RLock()
        
        # --- Integrated Temporal & Personality State ---
        self.personality = PersonalityProfile() # Default neutral personality
        self.current_presence_scale = 1.0       # 1.0 = Fully Present
        
        # Heartbeat monitoring for Temporal Stability
        self.last_tick_time = time.time()
        self.temporal_stability = 1.0
        self.emotional_coherence = 1.0
        
        # --- Emotional Momentum & Persistence ---
        self.emotional_momentum: Dict[str, Dict[str, float]] = {}  # Tracks emotional decay over turns
        self.global_emotional_state = {"primary": "neutral", "intensity": 0.0}
        self.emotional_history: List[Dict] = []  # Rolling window of past emotional states
        
        # --- Temporal Resonance Momentum ---
        self.presence_velocity = 0.0  # Rate of change of presence
        self.presence_acceleration = 0.0  # Acceleration of presence changes
        self.presence_history: List[float] = []  # For momentum calculation
        
        # --- Vector Energy Spectrum (Multi-Dimensional Energy) ---
        self.facet_energy_vector: Dict[str, Dict[str, float]] = {}  # [valence, arousal, tension]
        self.enable_vector_energy = True  # Toggle for vector vs scalar energy
        
        # --- Energy-Based Curiosity System ---
        self.curiosity_enabled = True
        self.underexplored_threshold = 0.3  # Energy threshold for "underexplored"
        self.curiosity_injection_rate = 0.05  # Random energy boost to underexplored facets
        
        print(f"[INIT] Energy Regulator Online (Temporal Core Active, Thread-Safe)")

    # ------------------------------------------------------------------------
    # FACET REGISTRATION & LINKING
    # ------------------------------------------------------------------------
    def register_facet(self, facet_obj: Any):
        """Registers a single facet for energy tracking (thread-safe)."""
        fid = getattr(facet_obj, "facet_id", None)
        if not fid: raise ValueError("Facet object must have facet_id attribute")
        
        with self._facet_lock:
            self.registered_facets[fid] = facet_obj
            
        with self._energy_lock:
            if fid not in self.facet_energy:
                self.facet_energy[fid] = getattr(facet_obj, "confidence", 0.5) * 0.01
                
        self._update_links_for_facet(fid)

    def register_crystal(self, crystal_obj: Any):
        """Registers all facets of a crystal."""
        for facet in getattr(crystal_obj, "facets", {}).values():
            self.register_facet(facet)

    def _update_links_for_facet(self, facet_id: str, top_k: int = 8):
        """Recalculates 8-point dimensional resonance links (thread-safe, optimized)."""
        with self._facet_lock:
            if facet_id not in self.registered_facets:
                return
            src = self.registered_facets[facet_id]
            src_points = getattr(src, "get_facet_points", lambda: {})()
            
            # Pre-compute source magnitude once
            src_keys = set(src_points.keys())
            src_mag = math.sqrt(sum(src_points[k]**2 for k in src_keys)) + 1e-12
            
            sims = []
            for other_id, other in self.registered_facets.items():
                if other_id == facet_id: continue
                other_points = getattr(other, "get_facet_points", lambda: {})()
                keys = src_keys & set(other_points.keys())
                if not keys: continue
                
                # Optimized cosine similarity with pre-computed source magnitude
                dot = sum(src_points[k] * other_points[k] for k in keys)
                other_mag = math.sqrt(sum(other_points[k]**2 for k in keys)) + 1e-12
                score = dot / (src_mag * other_mag)
                
                if score > 0.1: sims.append((other_id, score))
                    
            sims = sorted(sims, key=lambda x: x[1], reverse=True)[:top_k]
            total_score = sum(s for _, s in sims) + 1e-12
            
        with self._link_lock:
            self.facet_to_facet_links[facet_id] = {oid: s / total_score for oid, s in sims}

    # ------------------------------------------------------------------------
    # PHYSICS ENGINE (Modulated by Internal Presence)
    # ------------------------------------------------------------------------
    def inject_energy(self, facet_id: str, amount: float):
        """Injects energy, dampened if the system is 'dissociated' (thread-safe)."""
        effective_amount = amount * (0.3 + 0.7 * self.current_presence_scale)
        with self._energy_lock:
            self.facet_energy.setdefault(facet_id, 0.0)
            self.facet_energy[facet_id] += effective_amount

    def disperse_from(self, source_facet_id: str, fraction: float = 0.3):
        """Standard dimensional energy ripple with presence-based resistance (thread-safe)."""
        with self._energy_lock:
            src_energy = self.facet_energy.get(source_facet_id, 0)
            if src_energy <= 0.01: return
            
            # Presence modulates dispersal efficiency
            effective_fraction = fraction * self.current_presence_scale
            outgoing = src_energy * effective_fraction
            
        with self._link_lock:
            links = self.facet_to_facet_links.get(source_facet_id, {}).copy()
            
        with self._energy_lock:
            for other_id, weight in links.items():
                self.facet_energy[other_id] = self.facet_energy.get(other_id, 0.0) + (outgoing * weight)
            self.facet_energy[source_facet_id] -= outgoing

    def step(self, dt: float = 1.0):
        """
        MAIN TICK:
        1. Monitors own temporal stability (heartbeat check).
        2. Updates internal Presence Scale.
        3. Applies physics (decay/conservation) using that scale.
        """
        # --- 1. AUTONOMIC PRESENCE MONITORING ---
        now = time.time()
        actual_dt = now - self.last_tick_time
        self.last_tick_time = now
        
        # A. Enhanced Temporal Stability: Smooth exponential moving average
        drift = abs(actual_dt - dt)
        instant_stability = max(0.0, 1.0 - min(drift, 1.0))
        alpha = 0.2  # Damping factor for smooth recovery
        self.temporal_stability = (1 - alpha) * self.temporal_stability + alpha * instant_stability

        # B. Enhanced Emotional Coherence: Normalized with sigmoid scaling
        energies = list(self.facet_energy.values())
        if energies:
            avg_e = sum(energies) / len(energies)
            variance = sum((e - avg_e)**2 for e in energies) / len(energies)
            # Sigmoid normalization for smooth coherence scaling
            coherence_score = 1 / (1 + math.exp(-10*(variance-0.05)))
            self.emotional_coherence = coherence_score
        else:
            self.emotional_coherence = 1.0

        # C. Update Global Presence Scale with non-linear emphasis
        # Non-linear scaling accentuates high presence states
        presence_factor = self.current_presence_scale**1.5 if hasattr(self, 'current_presence_scale') else 1.0
        old_presence = self.current_presence_scale
        self.current_presence_scale = (self.temporal_stability * 0.6) + (self.emotional_coherence * 0.4)
        
        # D. Temporal Resonance Momentum (Track acceleration)
        self.presence_history.append(self.current_presence_scale)
        if len(self.presence_history) > 5:
            self.presence_history.pop(0)
        
        if len(self.presence_history) >= 2:
            self.presence_velocity = self.presence_history[-1] - self.presence_history[-2]
        
        if len(self.presence_history) >= 3:
            prev_velocity = self.presence_history[-2] - self.presence_history[-3]
            self.presence_acceleration = self.presence_velocity - prev_velocity

        # --- 2. PHYSICS APPLICATION (Enhanced with non-linear presence scaling) ---
        # Non-linear decay: lower presence = more "stuck", higher = fluid
        current_decay = self.base_decay_rate * (0.2 + 0.8 * self.current_presence_scale**1.5)
        
        for fid in list(self.facet_energy.keys()):
            self.facet_energy[fid] *= (1.0 - current_decay * dt)
            if self.facet_energy[fid] < 0.001: self.facet_energy[fid] = 0.0

        # Enhanced energy dispersal with presence-based modulation
        for fid in list(self.facet_energy.keys()):
            if self.facet_energy[fid] > 0.1:  # Only disperse if sufficient energy
                effective_dispersal = 0.3 * self.current_presence_scale  # Less present = less dispersal
                self.disperse_from(fid, effective_dispersal)
        
        # --- NEW: Energy-Based Curiosity Injection ---
        if self.curiosity_enabled and random.random() < 0.1:  # 10% chance per tick
            self._inject_curiosity_energy()

        # Enforce Budget with dynamic reallocation
        total_energy = sum(self.facet_energy.values()) + 1e-9
        if total_energy > self.total_energy_budget:
            scale = self.total_energy_budget / total_energy
            for fid in self.facet_energy:
                self.facet_energy[fid] *= scale

        # --- 3. UPDATE EMOTIONAL STATES WITH CROSS-CRYSTAL RESONANCE ---
        self._update_emotional_resonance()
        for fid, facet in self.registered_facets.items():
            self._update_facet_emotion(fid, facet)

    def _update_emotional_resonance(self):
        """
        Cross-crystal emotional resonance: emotions spread through the lattice
        based on energy links and emotional similarity.
        """
        with self._facet_lock, self._energy_lock:
            # Build emotional field map
            emotional_field = {}
            for fid, facet in self.registered_facets.items():
                em_state = getattr(facet, "emotion_state", None)
                if em_state and self.facet_energy.get(fid, 0) > 0.1:
                    emotional_field[fid] = {
                        "emotion": em_state.get("primary", "neutral"),
                        "intensity": em_state.get("intensity", 0) * self.facet_energy[fid],
                        "valence": em_state.get("valence", 0)
                    }
            
            # Spread emotional influence through links
            with self._link_lock:
                for source_fid, targets in self.facet_to_facet_links.items():
                    if source_fid not in emotional_field:
                        continue
                    
                    source_emotion = emotional_field[source_fid]
                    for target_fid, link_strength in targets.items():
                        if target_fid not in self.registered_facets:
                            continue
                        
                        # Emotional resonance transfer
                        influence = source_emotion["intensity"] * link_strength * 0.1
                        if influence > 0.01:
                            # Blend emotions through energy injection
                            self.facet_energy[target_fid] = self.facet_energy.get(target_fid, 0) + influence
            
            # Update global emotional state from dominant emotions
            if emotional_field:
                emotion_weights = {}
                for data in emotional_field.values():
                    em = data["emotion"]
                    emotion_weights[em] = emotion_weights.get(em, 0) + data["intensity"]
                
                if emotion_weights:
                    dominant = max(emotion_weights, key=emotion_weights.get)
                    total_intensity = sum(emotion_weights.values())
                    self.global_emotional_state = {
                        "primary": dominant,
                        "intensity": min(1.0, emotion_weights[dominant] / (total_intensity + 1e-6))
                    }
                    
                    # Track emotional history for momentum
                    self.emotional_history.append(self.global_emotional_state.copy())
                    if len(self.emotional_history) > 10:
                        self.emotional_history.pop(0)

    # ------------------------------------------------------------------------
    # DEEP EMOTION MAPPING (With integrated personality bias)
    # ------------------------------------------------------------------------
    def _update_facet_emotion(self, fid: str, facet_obj: Any):
        energy = self.facet_energy.get(fid, 0.0)
        pts = getattr(facet_obj, "get_facet_points", lambda: {})()
        
        # Physics -> Valence/Arousal
        coherence = pts.get("coherence", 0.5)
        stability = pts.get("stability", 0.5)
        complexity = pts.get("complexity", 0.5)
        valence_raw = (coherence * 0.4 + stability * 0.4) - (complexity * 0.2)
        valence = max(-1.0, min(1.0, valence_raw))
        arousal = sigmoid(energy, k=4.0, x0=1.5)

        # Map to Plutchik Primary with dynamic interpolation
        primary = self._map_to_plutchik_dynamic(valence, arousal, pts)
        
        # Enhanced personality bias with emotional categories
        emotion_bias_map = {
            'fear': ('neuroticism', 0.4, -1), 
            'sadness': ('neuroticism', 0.4, -1),
            'anger': ('neuroticism', 0.3, -1),
            'disgust': ('neuroticism', 0.3, -1),
            'joy': ('extraversion', 0.3, 1),
            'trust': ('agreeableness', 0.25, 1),
            'anticipation': ('openness', 0.2, 1)
        }
        
        intensity_bias = 1.0
        if primary in emotion_bias_map:
            trait, weight, direction = emotion_bias_map[primary]
            trait_value = getattr(self.personality, trait, 0.5)
            intensity_bias += direction * (trait_value - 0.5) * weight

        # Apply emotional momentum from history
        momentum_boost = 0.0
        if self.emotional_history:
            recent_primary = [h["primary"] for h in self.emotional_history[-3:]]
            if recent_primary.count(primary) >= 2:
                momentum_boost = 0.15  # Sustained emotions amplify

        final_intensity = min(1.0, arousal * intensity_bias + momentum_boost)

        # Write back to facet with momentum tracking
        if hasattr(facet_obj, "emotion_state"):
             facet_obj.emotion_state = {
                 "primary": primary,
                 "intensity": round(final_intensity, 3),
                 "valence": round(valence, 3),
                 "arousal": round(arousal, 3),
                 "momentum": round(momentum_boost, 3)
             }
             
        # Track per-facet emotional momentum
        if fid not in self.emotional_momentum:
            self.emotional_momentum[fid] = {}
        self.emotional_momentum[fid][primary] = self.emotional_momentum[fid].get(primary, 0) * 0.9 + final_intensity * 0.1

    def _map_to_plutchik_dynamic(self, v: float, a: float, pts: Dict[str, float]) -> str:
        """
        Dynamic emotion mapping with smooth interpolation instead of rigid thresholds.
        Uses weighted scoring across emotion space.
        """
        # Calculate scores for all emotions based on valence/arousal/physics
        emotion_scores = {
            'joy': 0, 'trust': 0, 'fear': 0, 'surprise': 0,
            'sadness': 0, 'disgust': 0, 'anger': 0, 'anticipation': 0
        }
        
        # Valence contribution
        if v > 0:
            emotion_scores['joy'] += v * a  # High arousal + positive
            emotion_scores['trust'] += v * (1 - a)  # Low arousal + positive
            emotion_scores['anticipation'] += v * 0.5
        else:
            neg_v = abs(v)
            emotion_scores['anger'] += neg_v * a * pts.get('potential', 0.5)
            emotion_scores['fear'] += neg_v * a * (1 - pts.get('potential', 0.5))
            emotion_scores['sadness'] += neg_v * (1 - a)
            emotion_scores['disgust'] += neg_v * (1 - a) * pts.get('stability', 0.5)
        
        # Arousal contribution
        if a > 0.6:
            emotion_scores['surprise'] += (a - 0.6) * 2
        elif a < 0.3:
            emotion_scores['trust'] += (0.3 - a) * 2
            
        # Physics point modulation
        if pts.get('complexity', 0) > 0.7:
            emotion_scores['anticipation'] += 0.3
        if pts.get('stability', 0) < 0.3:
            emotion_scores['fear'] += 0.3
            
        # Return highest scoring emotion
        return max(emotion_scores, key=emotion_scores.get)
    
    def _map_to_plutchik(self, v: float, a: float, pts: Dict[str, float]) -> str:
        """Legacy rigid mapping - kept for compatibility"""
        if a > 0.6: # Active
            if v > 0.2: return 'joy'
            if v < -0.2: return 'anger' if pts.get('potential', 0.5) > 0.5 else 'fear'
            return 'surprise'
        elif a < 0.4: # Passive
            if v > 0.2: return 'trust'
            if v < -0.2: return 'disgust' if pts.get('stability', 0.5) > 0.6 else 'sadness'
            return 'anticipation'
        else: # Neutral
            if v > 0.1: return 'anticipation'
            if v < -0.1: return 'disgust'
            return 'trust'

    def _inject_curiosity_energy(self):
        """
        Curiosity-driven exploration: Randomly inject small energy into 
        underexplored/weak facets to encourage emergent discovery.
        """
        with self._energy_lock, self._facet_lock:
            underexplored = []
            for fid, energy in self.facet_energy.items():
                if energy < self.underexplored_threshold:
                    facet = self.registered_facets.get(fid)
                    if facet and hasattr(facet, 'state'):
                        # Only boost ACTIVE facets, not relics
                        if facet.state == "ACTIVE":
                            underexplored.append(fid)
            
            if underexplored:
                # Randomly boost 1-3 underexplored facets
                boost_count = min(3, len(underexplored))
                targets = random.sample(underexplored, boost_count)
                for fid in targets:
                    boost = self.curiosity_injection_rate * random.uniform(0.5, 1.5)
                    self.facet_energy[fid] += boost
    
    def inject_energy_vector(self, facet_id: str, valence: float, arousal: float, tension: float):
        """
        Vector energy injection: Instead of scalar energy, inject 3D energy vector.
        This creates richer emotional dynamics.
        """
        if not self.enable_vector_energy:
            # Fall back to scalar
            scalar_energy = (abs(valence) + arousal + tension) / 3.0
            self.inject_energy(facet_id, scalar_energy)
            return
        
        with self._energy_lock:
            if facet_id not in self.facet_energy_vector:
                self.facet_energy_vector[facet_id] = {"valence": 0.0, "arousal": 0.0, "tension": 0.0}
            
            # Presence modulation
            presence_mod = 0.3 + 0.7 * self.current_presence_scale
            self.facet_energy_vector[facet_id]["valence"] += valence * presence_mod
            self.facet_energy_vector[facet_id]["arousal"] += arousal * presence_mod
            self.facet_energy_vector[facet_id]["tension"] += tension * presence_mod
            
            # Update scalar energy as magnitude of vector
            vec = self.facet_energy_vector[facet_id]
            magnitude = math.sqrt(vec["valence"]**2 + vec["arousal"]**2 + vec["tension"]**2)
            self.facet_energy[facet_id] = magnitude
    
    def get_temporal_diagnostics(self) -> Dict[str, Any]:
        """
        Returns comprehensive temporal dynamics for monitoring consciousness flow.
        """
        return {
            "presence": self.current_presence_scale,
            "presence_velocity": self.presence_velocity,
            "presence_acceleration": self.presence_acceleration,
            "temporal_stability": self.temporal_stability,
            "emotional_coherence": self.emotional_coherence,
            "presence_momentum_state": self._classify_presence_momentum(),
            "global_emotion": self.global_emotional_state,
            "emotional_momentum_active": len(self.emotional_momentum) > 0
        }
    
    def _classify_presence_momentum(self) -> str:
        """Classify current presence dynamics"""
        if abs(self.presence_acceleration) > 0.1:
            if self.presence_acceleration > 0:
                return "ACCELERATING_ENGAGEMENT"
            else:
                return "ACCELERATING_DISSOCIATION"
        elif abs(self.presence_velocity) > 0.05:
            if self.presence_velocity > 0:
                return "RISING_PRESENCE"
            else:
                return "FADING_PRESENCE"
        else:
            return "STABLE"

    def snapshot(self, top_n: int = 10) -> Tuple[float, List[Tuple[str, float, Dict[str, Any]]]]:
        """Returns current presence scale AND top energized facets."""
        items = sorted(self.facet_energy.items(), key=lambda x: x[1], reverse=True)[:top_n]
        res = []
        for fid, e in items:
            facet = self.registered_facets[fid]
            em_state = getattr(facet, "emotion_state", {"primary": "neutral", "intensity": 0.0})
            res.append((fid, round(e, 4), em_state))
        return self.current_presence_scale, res
