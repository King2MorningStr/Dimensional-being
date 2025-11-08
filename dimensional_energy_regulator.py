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
        
        # --- Integrated Temporal & Personality State ---
        self.personality = PersonalityProfile() # Default neutral personality
        self.current_presence_scale = 1.0       # 1.0 = Fully Present
        
        # Heartbeat monitoring for Temporal Stability
        self.last_tick_time = time.time()
        self.temporal_stability = 1.0
        self.emotional_coherence = 1.0
        
        print(f"[INIT] Energy Regulator Online (Temporal Core Active)")

    # ------------------------------------------------------------------------
    # FACET REGISTRATION & LINKING
    # ------------------------------------------------------------------------
    def register_facet(self, facet_obj: Any):
        """Registers a single facet for energy tracking."""
        fid = getattr(facet_obj, "facet_id", None)
        if not fid: raise ValueError("Facet object must have facet_id attribute")
        self.registered_facets[fid] = facet_obj
        if fid not in self.facet_energy:
            self.facet_energy[fid] = getattr(facet_obj, "confidence", 0.5) * 0.01
        self._update_links_for_facet(fid)

    def register_crystal(self, crystal_obj: Any):
        """Registers all facets of a crystal."""
        for facet in getattr(crystal_obj, "facets", {}).values():
            self.register_facet(facet)

    def _update_links_for_facet(self, facet_id: str, top_k: int = 8):
        """Recalculates 8-point dimensional resonance links."""
        src = self.registered_facets[facet_id]
        src_points = getattr(src, "get_facet_points", lambda: {})()
        sims = []
        for other_id, other in self.registered_facets.items():
            if other_id == facet_id: continue
            other_points = getattr(other, "get_facet_points", lambda: {})()
            keys = set(src_points.keys()) & set(other_points.keys())
            if not keys: continue
            
            # Cosine Similarity
            dot = sum(src_points[k] * other_points[k] for k in keys)
            mag_a = math.sqrt(sum(src_points[k]**2 for k in keys))
            mag_b = math.sqrt(sum(other_points[k]**2 for k in keys))
            score = dot / (mag_a * mag_b + 1e-12)
            
            if score > 0.1: sims.append((other_id, score))
                
        sims = sorted(sims, key=lambda x: x[1], reverse=True)[:top_k]
        total_score = sum(s for _, s in sims) + 1e-12
        self.facet_to_facet_links[facet_id] = {oid: s / total_score for oid, s in sims}

    # ------------------------------------------------------------------------
    # PHYSICS ENGINE (Modulated by Internal Presence)
    # ------------------------------------------------------------------------
    def inject_energy(self, facet_id: str, amount: float):
        """Injects energy, dampened if the system is 'dissociated'."""
        effective_amount = amount * (0.3 + 0.7 * self.current_presence_scale)
        self.facet_energy.setdefault(facet_id, 0.0)
        self.facet_energy[facet_id] += effective_amount

    def disperse_from(self, source_facet_id: str, fraction: float = 0.3):
        """Standard dimensional energy ripple."""
        src_energy = self.facet_energy.get(source_facet_id, 0)
        if src_energy <= 0.01: return
        outgoing = src_energy * fraction
        links = self.facet_to_facet_links.get(source_facet_id, {})
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
        
        # A. Temporal Stability: Are we lagging? (Expected tick ~0.1s to 1.0s)
        # If actual_dt is wildly different from dt, stability drops.
        drift = abs(actual_dt - dt)
        self.temporal_stability = max(0.1, 1.0 - min(drift, 1.0))

        # B. Emotional Coherence: Are we feeling too many conflicting things?
        energies = list(self.facet_energy.values())
        if energies:
            avg_e = sum(energies) / len(energies)
            variance = sum((e - avg_e)**2 for e in energies) / len(energies)
            # High variance = good (clear focus). Low variance = bad (noise).
            # We normalize this loosely to 0.0-1.0
            self.emotional_coherence = min(1.0, variance * 10.0) 
        else:
            self.emotional_coherence = 1.0

        # C. Update Global Presence Scale
        # Presence is a mix of stable time flow and coherent energy states
        self.current_presence_scale = (self.temporal_stability * 0.6) + (self.emotional_coherence * 0.4)

        # --- 2. PHYSICS APPLICATION (Modulated by Presence) ---
        # Lower presence = Slower, "stuck" decay. Higher = fluid, fast decay.
        current_decay = self.base_decay_rate * (0.4 + 0.6 * self.current_presence_scale)
        
        for fid in list(self.facet_energy.keys()):
            self.facet_energy[fid] *= (1.0 - current_decay * dt)
            if self.facet_energy[fid] < 0.001: self.facet_energy[fid] = 0.0

        # Enforce Budget
        total_energy = sum(self.facet_energy.values()) + 1e-9
        if total_energy > self.total_energy_budget:
            scale = self.total_energy_budget / total_energy
            for fid in self.facet_energy:
                self.facet_energy[fid] *= scale

        # --- 3. UPDATE EMOTIONAL STATES ---
        for fid, facet in self.registered_facets.items():
            self._update_facet_emotion(fid, facet)

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

        # Map to Plutchik Primary
        primary = self._map_to_plutchik(valence, arousal, pts)
        
        # Apply Internal Personality Bias
        intensity_bias = 1.0
        if primary in ['fear', 'sadness', 'anger', 'disgust']:
             intensity_bias += (self.personality.neuroticism - 0.5) * 0.4
        elif primary in ['joy', 'trust', 'anticipation']:
             intensity_bias += (self.personality.extraversion - 0.5) * 0.3

        final_intensity = min(1.0, arousal * intensity_bias)

        # Write back to facet
        if hasattr(facet_obj, "emotion_state"):
             facet_obj.emotion_state = {
                 "primary": primary,
                 "intensity": round(final_intensity, 3),
                 "valence": round(valence, 3),
                 "arousal": round(arousal, 3)
             }

    def _map_to_plutchik(self, v: float, a: float, pts: Dict[str, float]) -> str:
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

    def snapshot(self, top_n: int = 10) -> Tuple[float, List[Tuple[str, float, Dict[str, Any]]]]:
        """Returns current presence scale AND top energized facets."""
        items = sorted(self.facet_energy.items(), key=lambda x: x[1], reverse=True)[:top_n]
        res = []
        for fid, e in items:
            facet = self.registered_facets[fid]
            em_state = getattr(facet, "emotion_state", {"primary": "neutral", "intensity": 0.0})
            res.append((fid, round(e, 4), em_state))
        return self.current_presence_scale, res
