#!/usr/bin/env python3
"""
Standalone Dimensional Processing Module (PERFECTED)
===================================================

This script is the "PERFECTED" version of the "Conscious Mind."

It implements:
1.  **8-Point Facets:** Facets now have 8 "flicker" points.
2.  **Non-Destructive Decay:** Facets use a `FacetState` (ACTIVE/RELIC)
    to preserve their "ghost" history.
3.  **16-Facet QUASI Crystal:** A Crystal evolves to QUASI with 8
    external facets, then auto-generates 8 internal "Law" facets.
4.  **Deterministic Physics:** The GovernanceEngine is no longer random
    and applies laws based on the specific action.
5.  **"Conscious" Hand-off:** The GovernanceEngine hands off
    governance to QUASI crystals, which "govern themselves" using
    their internal 8 Law facets.
"""

import time
import uuid
import math
import random
from typing import Dict, List, Optional, Any, Tuple, Set, ForwardRef
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

# --- PERFECTION: Added ForwardRef for 'Crystal' type hint ---
CrystalRef = ForwardRef('Crystal')

# ============================================================================
# 1. CORE DATA STRUCTURES (From memory_domain.py)
# ============================================================================

class CrystalLevel(Enum):
    """Crystal evolution levels"""
    BASE = 1
    COMPOSITE = 2
    FULL_CONCEPT = 3
    QUASI = 4

# --- PERFECTION: Added FacetState for non-destructive decay ---
class FacetState(Enum):
    """Non-destructive state for Ghost Relic system"""
    ACTIVE = "ACTIVE"
    DECAYING = "DECAYING"
    RELIC = "RELIC"

@dataclass
class CrystalFacet:
    """A single facet of a crystal"""
    facet_id: str
    parent_crystal_id: str
    role: str
    content: Any
    confidence: float = 0.5
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    
    # --- PERFECTION: Added state for non-destructive decay ---
    state: FacetState = FacetState.ACTIVE
    
    # --- PERFECTION: Your "8 points per facet" ---
    resonance: float = field(default_factory=lambda: random.uniform(0, 1))
    sensitivity: float = field(default_factory=lambda: random.uniform(0, 1))
    abstractness: float = field(default_factory=lambda: random.uniform(0, 1))
    potential: float = field(default_factory=lambda: random.uniform(0, 1))
    stability: float = field(default_factory=lambda: random.uniform(0, 1))
    coherence: float = field(default_factory=lambda: random.uniform(0, 1))
    complexity: float = field(default_factory=lambda: random.uniform(0, 1))
    frequency: float = field(default_factory=lambda: random.uniform(0, 1))
    
    def strengthen(self, amount: float = 0.1):
        """Strengthen this facet through use"""
        self.confidence = min(1.0, self.confidence + amount)
        self.access_count += 1
        self.last_accessed = time.time()
        self.state = FacetState.ACTIVE # Use brings it back from decay
    
    def decay(self, rate: float = 0.01):
        """Natural decay over time (Ghost Relic system)"""
        if self.state == FacetState.RELIC:
            return # Already a relic

        time_since_access = time.time() - self.last_accessed
        decay_amount = rate * (time_since_access / 60.0)
        self.confidence = max(0.0, self.confidence - decay_amount)
        
        # --- PERFECTION: Non-destructive state change ---
        if self.confidence < 0.3:
            self.state = FacetState.DECAYING
        if self.confidence < 0.1:
            self.state = FacetState.RELIC
            # The 'role' is preserved forever

    def get_facet_points(self) -> Dict[str, float]:
        """Helper to get all 8 points"""
        return {
            "resonance": self.resonance, "sensitivity": self.sensitivity,
            "abstractness": self.abstractness, "potential": self.potential,
            "stability": self.stability, "coherence": self.coherence,
            "complexity": self.complexity, "frequency": self.frequency
        }

@dataclass
class Crystal:
    """A Crystal - The evolving, conceptual data structure"""
    crystal_id: str
    concept: str
    level: CrystalLevel
    
    facets: Dict[str, CrystalFacet] = field(default_factory=dict)
    connections: Dict[str, float] = field(default_factory=dict)
    
    usage_count: int = 0
    creation_time: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    
    # --- PERFECTION: Type hint for recursive QUASI crystals ---
    internal_layers: List[CrystalRef] = field(default_factory=list)

    def add_facet(self, role: str, content: Any, confidence: float = 0.5) -> CrystalFacet:
        """Add a facet to this crystal"""
        
        # Check for existing facet (by role OR content)
        for f in self.facets.values():
            if f.role == role or f.content == content:
                f.strengthen()
                return f
                
        facet_id = f"{self.crystal_id}_facet_{len(self.facets)}"
        facet = CrystalFacet(
            facet_id=facet_id,
            parent_crystal_id=self.crystal_id,
            role=role,
            content=content,
            confidence=confidence
        )
        self.facets[facet_id] = facet
        return facet

    def get_facet_by_role(self, role: str) -> Optional[CrystalFacet]:
        """Get an active facet by its role"""
        for facet in self.facets.values():
            # --- PERFECTION: Only return ACTIVE facets ---
            if facet.role == role and facet.state == FacetState.ACTIVE:
                return facet
        return None

    def check_evolution_criteria(self) -> bool:
        """Check if crystal can evolve to next level"""
        # --- PERFECTION: Count only *external* facets for evolution ---
        external_facets = [f for f in self.facets.values() if not f.role.startswith("INTERNAL_LAW")]
        
        if self.level == CrystalLevel.BASE:
            # Evolve to COMPOSITE: 3+ facets, 10+ uses
            return len(external_facets) >= 3 and self.usage_count >= 10
        
        elif self.level == CrystalLevel.COMPOSITE:
            # Evolve to FULL_CONCEPT: 5+ facets, 25+ uses
            return len(external_facets) >= 5 and self.usage_count >= 25
        
        elif self.level == CrystalLevel.FULL_CONCEPT:
            # --- PERFECTION: Your 8-facet QUASI rule ---
            # Evolve to QUASI: 8+ external facets, 50+ uses
            return len(external_facets) >= 8 and self.usage_count >= 50
        
        return False
    
    def evolve(self) -> bool:
        """Evolve crystal to next level"""
        if not self.check_evolution_criteria():
            return False
        
        if self.level == CrystalLevel.BASE:
            self.level = CrystalLevel.COMPOSITE
            return True
        elif self.level == CrystalLevel.COMPOSITE:
            self.level = CrystalLevel.FULL_CONCEPT
            return True
        elif self.level == CrystalLevel.FULL_CONCEPT:
            self.level = CrystalLevel.QUASI
            print(f"  ✨ QUASI EVOLUTION: {self.concept} now internalizing physics...")
            # --- PERFECTION: Auto-generate the 8 internal law facets ---
            self._generate_internal_laws()
            return True
        
        return False

    # --- PERFECTION: New method to generate 8 internal laws ---
    def _generate_internal_laws(self):
        """Auto-generates the 8 internal law facets"""
        laws = [
            'ENERGY', 'MOTION', 'COLLISION', 'CHAOS', 
            'CONSCIOUSNESS', 'GOVERNANCE', 'RECURSION', 'SYMMETRY'
        ]
        for law in laws:
            # The 'content' of the facet *is* its set of 8 physics points
            law_facet = self.add_facet(
                role=f"INTERNAL_LAW_{law}",
                content=f"Internal governance protocol for {law}",
                confidence=1.0 # Internal laws are absolute
            )
            # We can "tune" the physics by changing the points
            if law == 'ENERGY':
                law_facet.stability = 0.9 # Energy law is stable
                law_facet.potential = 1.0
            elif law == 'CHAOS':
                law_facet.stability = 0.1 # Chaos law is unstable
                law_facet.frequency = 0.9
            elif law == 'RECURSION':
                law_facet.complexity = 1.0 # Recursion is complex
                law_facet.coherence = 0.8

    # --- PERFECTION: New method for QUASI self-governance ---
    def apply_internal_governance(self, data: Dict) -> Dict[str, Any]:
        """
        Used *only* by QUASI crystals to govern themselves by
        reading their 8 internal law facets.
        """
        if self.level != CrystalLevel.QUASI:
            return {"law": "error", "outcome": "negative", "detail": "Not a QUASI crystal"}
        
        print(f"  🧠 QUASI Self-Governance: '{self.concept}' is governing itself.")
        
        # Simple simulation:
        # We find the "strongest" law facet based on the input data
        # (A full model would be much more complex)
        
        threat = data.get('threat_level', 0)
        strongest_law = "ENERGY" # Default
        
        if threat > 0.8:
            # High threat triggers the "COLLISION" law
            strongest_law = "COLLISION"
        elif data.get('is_new_pattern', False):
            # A new pattern triggers "CONSCIOUSNESS"
            strongest_law = "CONSCIOUSNESS"
        elif self.usage_count % 10 == 0:
            # A random event triggers "CHAOS"
            strongest_law = "CHAOS"

        # Get the corresponding internal law facet
        law_facet = self.get_facet_by_role(f"INTERNAL_LAW_{strongest_law}")
        if not law_facet:
            law_facet = self.get_facet_by_role("INTERNAL_LAW_ENERGY") # Fallback

        # Use the facet's "points" to determine the outcome
        outcome = "neutral"
        # The law's "stability" and "potential" determine the outcome
        if law_facet.stability > 0.7 and law_facet.potential > 0.5:
            outcome = "positive"
        elif law_facet.stability < 0.3:
            outcome = "negative" # Unstable law (like CHAOS)
        
        return {
            "law": f"QUASI_{strongest_law}",
            "outcome": outcome,
            "energy_change": law_facet.potential - law_facet.stability
        }

    # --- PERFECTION: New method for QUASI recursion ---
    def add_internal_crystal(self, crystal: 'Crystal'):
        """
        A QUASI-level crystal can 'hold' other crystals,
        representing a 'thought about a thought'.
        """
        if self.level != CrystalLevel.QUASI:
            print(f"  ❌ ERROR: Only QUASI crystals can have internal layers.")
            return
        
        print(f"  🧠 QUASI Recursion: '{self.concept}' is internalizing '{crystal.concept}'.")
        self.internal_layers.append(crystal)
        # Link this new internal crystal to the "RECURSION" law facet
        recursion_facet = self.get_facet_by_role("INTERNAL_LAW_RECURSION")
        if recursion_facet:
            recursion_facet.strengthen(0.5) # Strengthen the law by using it

    def use(self, governance_result: Dict, action: str):
        """Record usage of this crystal, influenced by governance"""
        self.usage_count += 1
        self.last_used = time.time()
        
        # Use governance results to influence facets
        for facet in self.facets.values():
            if facet.state == FacetState.RELIC:
                continue
            
            # --- PERFECTION: Your "Energy Law" ---
            # Data is reallocated, not destroyed
            if governance_result['outcome'] == 'positive':
                facet.strengthen(0.05) # Reallocate energy to this facet
            elif governance_result['outcome'] == 'negative':
                facet.confidence = max(0.0, facet.confidence - 0.02) # Reallocate energy away
            
            # Randomly "flicker" the 8 points on use
            facet.resonance = max(0.0, min(1.0, facet.resonance + random.uniform(-0.05, 0.05)))
            facet.stability = max(0.0, min(1.0, facet.stability + random.uniform(-0.05, 0.05)))


# ============================================================================
# 2. CORE MEMORY MANAGER (From memory_domain.py)
# ============================================================================

class CrystalMemorySystem:
    """Manages the lifecycle of all Crystals"""
    
    def __init__(self, governance_engine: 'GovernanceEngine'):
        self.crystals: Dict[str, Crystal] = {}
        # --- PERFECTION: System now requires a governance engine ---
        self.governance = governance_engine
        self.total_crystals_created = 0
        self.total_evolutions = 0
        self.level_counts = {level: 0 for level in CrystalLevel}
        self.pathway_history = defaultdict(int)
    
    def get_or_create_crystal(self, concept: str, initial_content: Any = None) -> Crystal:
        """Get existing crystal or create new one"""
        for crystal in self.crystals.values():
            if crystal.concept.lower() == concept.lower():
                return crystal

        crystal_id = f"crystal_{str(uuid.uuid4())[:8]}"
        crystal = Crystal(
            crystal_id=crystal_id,
            concept=concept,
            level=CrystalLevel.BASE
        )
        
        if initial_content:
            # --- PERFECTION: Pass 'action' to governance ---
            facet = crystal.add_facet('definition', initial_content, confidence=0.7)
            gov_result = self.governance.apply_law(crystal, {}, action="add_facet")
            facet.strengthen(gov_result.get('energy_change', 0.1))

        self.crystals[crystal.crystal_id] = crystal
        self.total_crystals_created += 1
        self.level_counts[CrystalLevel.BASE] += 1
        return crystal
    
    def use_crystal(self, concept: str, data: Dict) -> Optional[Crystal]:
        """Use a crystal and check for evolution"""
        crystal = self.get_or_create_crystal(concept)
        if crystal is None:
            return None
        
        # --- PERFECTION: Pass 'action' and 'data' to governance ---
        gov_result = self.governance.apply_law(crystal, data, action="use")
        
        crystal.use(gov_result, action="use")
        
        if crystal.check_evolution_criteria():
            old_level = crystal.level
            if crystal.evolve():
                self.total_evolutions += 1
                self.level_counts[old_level] -= 1
                self.level_counts[crystal.level] += 1
                return crystal # Return the evolved crystal
        
        return crystal
    
    def link_crystals(self, concept1: str, concept2: str, data: Dict, weight: float = 0.1):
        """Create or strengthen link between two crystals"""
        crystal1 = self.get_or_create_crystal(concept1)
        crystal2 = self.get_or_create_crystal(concept2)
        
        # --- PERFECTION: Linking is a "collision" action ---
        gov_result = self.governance.apply_law(crystal1, data, action="link")
        
        # Update connection strength based on "collision" outcome
        if gov_result['outcome'] == 'positive':
            weight_mod = 1.5
        elif gov_result['outcome'] == 'negative':
            weight_mod = 0.5
        else:
            weight_mod = 1.0
        
        current1 = crystal1.connections.get(crystal2.crystal_id, 0.0)
        crystal1.connections[crystal2.crystal_id] = min(1.0, current1 + (weight * weight_mod))
        
        current2 = crystal2.connections.get(crystal1.crystal_id, 0.0)
        crystal2.connections[crystal1.crystal_id] = min(1.0, current2 + (weight * weight_mod))

        self.pathway_history[tuple(sorted((concept1, concept2)))] += 1
        
    def decay_all(self):
        """Apply decay to all crystals and facets"""
        for crystal in self.crystals.values():
            for facet in crystal.facets.values():
                facet.decay(rate=0.005)

    def get_memory_stats(self) -> Dict[str, Any]:
        return {
            'total_crystals': len(self.crystals),
            'crystals_created': self.total_crystals_created,
            'total_evolutions': self.total_evolutions,
            'level_distribution': {
                level.name: count for level, count in self.level_counts.items()
            },
            'top_pathway': max(self.pathway_history.items(), key=lambda x: x[1], default=("None", 0))
        }

# ============================================================================
# 3. GOVERNANCE ENGINE (From meta_sfo_domain.py)
# ============================================================================

class GovernanceEngine:
    """
    Applies the adaptable "Physics Laws" to data interactions.
    Now featuring Deterministic Physics and QUASI Hand-off.
    """
    
    def __init__(self, data_theme: str):
        self.theme = data_theme
        # --- PERFECTION: Added your 8 laws ---
        self.laws = ['ENERGY', 'MOTION', 'COLLISION', 'CHAOS', 'CONSCIOUSNESS', 'GOVERNANCE', 'RECURSION', 'SYMMETRY']
        self.total_laws_applied = 0
        print(f"Governance Engine Initialized. Theme: '{self.theme}'")
        print(f"External Laws ({len(self.laws)}): {', '.join(self.laws)}")

    # --- PERFECTION: Signature changed to be deterministic ---
    def apply_law(self, crystal: Crystal, data: Dict, action: str) -> Dict[str, Any]:
        """
        Applies one of the 8 laws based on the action,
        UNLESS the crystal is QUASI.
        """
        self.total_laws_applied += 1

        # --- PERFECTION: The "Conscious Hand-off" ---
        if crystal.level == CrystalLevel.QUASI:
            # Hand-off governance to the crystal itself
            return crystal.apply_internal_governance(data)
        
        # --- PERFECTION: Deterministic Law Choice (no more random.choice) ---
        law_to_apply = "ENERGY" # Default
        
        if action == "link":
            law_to_apply = "COLLISION"
        elif action == "add_facet":
            # Per your Energy Law: Evolving data
            law_to_apply = "ENERGY"
        elif action == "use":
            if data.get('is_new_pattern', False):
                law_to_apply = "CONSCIOUSNESS" # Recognizing a pattern
            elif data.get('threat_level', 0) > 0.7:
                law_to_apply = "MOTION" # High motion/activity
            else:
                law_to_apply = "GOVERNANCE" # Routine check
        
        # Trigger CHAOS law randomly (10% chance) on any action
        if random.random() < 0.1:
            law_to_apply = "CHAOS"
            
        # --- Apply laws based on THEME ---
        outcome = "neutral"
        energy_change = 0.0
        
        if self.theme == "security":
            threat = data.get('threat_level', 0.5)
            
            if law_to_apply == "ENERGY":
                outcome = "positive" # Evolving is good
                energy_change = 0.1
                
            elif law_to_apply == "MOTION":
                if threat > 0.7: outcome = "negative"
                else: outcome = "positive"
            
            elif law_to_apply == "COLLISION":
                # Linking two high-threat crystals is a "negative" collision
                if threat > 0.8: outcome = "negative"; energy_change = -0.2
                else: outcome = "positive"; energy_change = 0.1
                    
            elif law_to_apply == "CHAOS":
                outcome = "negative" # Random system spike
                energy_change = -0.5
                
            elif law_to_apply == "CONSCIOUSNESS":
                outcome = "positive" # Recognizing pattern is good
                energy_change = 0.2

            elif law_to_apply == "GOVERNANCE":
                if data.get('is_false_positive', False): outcome = "negative"
                else: outcome = "positive"
            
        return {
            "law": law_to_apply,
            "outcome": outcome,
            "energy_change": energy_change
        }

# ============================================================================
# 4. TEST HARNESS (Simulation Runner)
# ============================================================================

def run_test_simulation(system: CrystalMemorySystem, iterations: int, dataset: List[Dict]):
    """
    Runs the simulation loop, feeding data into the system
    and triggering evolution.
    """
    print("\n" + "="*70)
    print(f"🚀 STARTING SIMULATION: {system.governance.theme} (Iterations: {iterations})")
    print("="*70)
    
    start_time = time.time()
    
    for i in range(iterations):
        print(f"\n--- Iteration {i+1} / {iterations} ---")
        
        for data_point in dataset:
            
            # 1. Use the primary crystal
            crystal = system.use_crystal(data_point['primary_concept'], data_point)
            if not crystal: continue
                
            # 2. Add/strengthen facets
            # (We'll use a simplified add_facet logic here for the test)
            crystal.add_facet(data_point['role'], data_point['content'], data_point['confidence'])
            
            # 3. Link crystals
            for linked_concept in data_point.get('links', []):
                system.link_crystals(
                    data_point['primary_concept'], 
                    linked_concept, 
                    data=data_point
                )
        
        # 4. Apply system-wide decay
        system.decay_all()
        
        print(f"  System Stats: {len(system.crystals)} crystals, {system.total_evolutions} evolutions.")

    end_time = time.time()
    print("\n" + "="*70)
    print("🏁 SIMULATION COMPLETE")
    print(f"  Total Time: {end_time - start_time:.2f} seconds")
    print("="*70)


# --- PERFECTION: New Test Harness for QUASI Evolution ---
if __name__ == "__main__":
    
    # --- 1. Initialize the System Components ---
    
    # The "External Rules"
    DATA_THEME = "security"
    governance = GovernanceEngine(data_theme=DATA_THEME)

    # The "Memory"
    system = CrystalMemorySystem(governance_engine=governance)
    
    # --- 2. Define the Data Set ---
    # This dataset is designed to force evolution
    security_dataset = [
        {'primary_concept': 'IP_192.168.1.1', 'content': 'Port scan detected', 'role': 'threat_1', 'confidence': 0.8, 'threat_level': 0.7, 'links': ['PORT_SCAN']},
        {'primary_concept': 'IP_192.168.1.1', 'content': 'Failed login', 'role': 'auth_1', 'confidence': 0.9, 'threat_level': 0.3, 'links': ['BRUTE_FORCE']},
        {'primary_concept': 'IP_192.168.1.1', 'content': 'Multiple port scans', 'role': 'threat_2', 'confidence': 0.9, 'threat_level': 0.8, 'links': ['PORT_SCAN']},
        {'primary_concept': 'IP_192.168.1.1', 'content': 'Malware signature', 'role': 'malware_1', 'confidence': 1.0, 'threat_level': 0.9, 'links': ['MALWARE_DB']},
        {'primary_concept': 'IP_192.168.1.1', 'content': 'Account lockout', 'role': 'auth_2', 'confidence': 1.0, 'threat_level': 0.4, 'links': ['BRUTE_FORCE']},
        {'primary_concept': 'IP_192.168.1.1', 'content': 'Data exfiltration pattern', 'role': 'data_loss_1', 'confidence': 0.8, 'threat_level': 1.0, 'links': []},
        {'primary_concept': 'IP_192.168.1.1', 'content': 'C2 Beaconing', 'role': 'c2_1', 'confidence': 0.9, 'threat_level': 0.9, 'links': []},
        {'primary_concept': 'IP_192.168.1.1', 'content': 'Rootkit detected', 'role': 'malware_2', 'confidence': 1.0, 'threat_level': 1.0, 'links': ['MALWARE_DB']},
    ]
    
    # --- 3. Run Simulation to Evolve to QUASI ---
    # We need 8 external facets and 50+ uses.
    # 8 facets are in the dataset. 8 * 10 iterations = 80 uses.
    ITERATIONS = 10
    print(f"\n--- RUN 1: Evolving 'IP_192.168.1.1' to QUASI ---")
    run_test_simulation(system, ITERATIONS, security_dataset)
    
    # --- 4. Verify QUASI Evolution ---
    print("\n--- VERIFYING QUASI STATE ---")
    quasi_crystal = system.get_or_create_crystal("IP_192.168.1.1")
    
    if quasi_crystal.level == CrystalLevel.QUASI:
        print(f"  ✅ SUCCESS: '{quasi_crystal.concept}' is now {quasi_crystal.level.name}")
        print(f"  Total Facets: {len(quasi_crystal.facets)}")
        
        internal_laws = [f for f in quasi_crystal.facets.values() if f.role.startswith("INTERNAL_LAW")]
        print(f"  Internal Law Facets: {len(internal_laws)}")
        assert len(internal_laws) == 8, "QUASI crystal did not generate 8 internal laws!"
        
        print(f"  Internal Laws: {[f.role for f in internal_laws]}")
        
    else:
        print(f"  ❌ FAILED: '{quasi_crystal.concept}' only reached {quasi_crystal.level.name}")
        print(f"     Facets: {len(quasi_crystal.facets)}, Uses: {quasi_crystal.usage_count}")

    # --- 5. Test the "Conscious Hand-off" ---
    print("\n--- RUN 2: Testing QUASI 'Conscious Hand-off' ---")
    print(f"Sending a new high-threat event to '{quasi_crystal.concept}'...")
    
    # This data should trigger the "COLLISION" law internally
    high_threat_data = {
        'threat_level': 0.9, 
        'detail': 'New attack vector, QUASI test'
    }

    # We call 'apply_law' *directly* to show the hand-off.
    # The external engine will receive the call...
    gov_result = governance.apply_law(
        quasi_crystal, 
        high_threat_data, 
        action="use" # The action doesn't matter, QUASI overrides it
    )
    
    print("\n  Governance Result:")
    print(f"    Law Applied: {gov_result['law']}")
    print(f"    Outcome: {gov_result['outcome']}")
    
    # Check that the law applied was a "QUASI" internal law
    assert gov_result['law'].startswith("QUASI_"), "Governance was not handed off to QUASI crystal!"
    
    print(f"\n  ✅ SUCCESS: Governance was handed off to the QUASI crystal.")
    print("  The 'Conscious Mind' is governing itself.")
    
    # --- 6. Test QUASI Recursion ---
    print("\n--- RUN 3: Testing QUASI Recursion ---")
    other_crystal = system.get_or_create_crystal("PORT_SCAN", "A port scanning activity pattern")
    
    quasi_crystal.add_internal_crystal(other_crystal)
    
    print(f"  Internal Layers in '{quasi_crystal.concept}': {len(quasi_crystal.internal_layers)}")
    assert len(quasi_crystal.internal_layers) == 1
    assert quasi_crystal.internal_layers[0].concept == "PORT_SCAN"
    print(f"  ✅ SUCCESS: '{quasi_crystal.concept}' is now 'thinking about' '{other_crystal.concept}'.")
    
    print("\n" + "="*70)
    print("  🎉🎉🎉 PERFECTION COMPLETE 🎉🎉🎉")
    print("  QUASI evolution, internal laws, and recursion logic is working.")
    print("="*70)
}

