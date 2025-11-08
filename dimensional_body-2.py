#!/usr/bin/env python3
"""
Tactile Spatial Domain (DIMENSIONAL INTEGRATION)
================================================
Integrates the Virtual Body into the Dimensional System.
Body parts are Crystals, and physical sensations directly
modulate their dimensional physics vectors.
"""

import time
import uuid
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

# ============================================================================
# 1. DIMENSIONAL BODY STRUCTURES
# ============================================================================

class BodyPartType(Enum):
    HEAD = "CONCEPT_BODY_HEAD"
    TORSO = "CONCEPT_BODY_TORSO"
    LEFT_ARM = "CONCEPT_BODY_L_ARM"
    RIGHT_ARM = "CONCEPT_BODY_R_ARM"
    LEFT_LEG = "CONCEPT_BODY_L_LEG"
    RIGHT_LEG = "CONCEPT_BODY_R_LEG"

@dataclass
class DimensionalBodyPart:
    """A body part that maps to a Crystal."""
    part_type: BodyPartType
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Physics modifiers applied to the corresponding Crystal Facet
    tension: float = 0.0  # Maps to 'stability' (higher tension = lower stability)
    sensitivity: float = 0.5 # Maps to 'sensitivity'
    temperature: float = 36.5

class DimensionalBody:
    """The complete virtual body, acting as a dimensional anchor."""
    def __init__(self):
        self.parts: Dict[BodyPartType, DimensionalBodyPart] = {}
        self.balance: float = 1.0 # Global modifier for system stability
        self._init_parts()

    def _init_parts(self):
        for pt in BodyPartType:
            self.parts[pt] = DimensionalBodyPart(pt)
        # Set default positions (simplified T-pose)
        self.parts[BodyPartType.HEAD].position = (0, 1.7, 0)
        self.parts[BodyPartType.TORSO].position = (0, 1.2, 0)

# ============================================================================
# 2. THE DIMENSIONAL TACTILE SYSTEM
# ============================================================================

class TactileSpatialDomain:
    """
    Manages the Virtual Body and its connection to the Lattice.
    """
    def __init__(self):
        print("[INIT] Dimensional Tactile System Online.")
        self.body = DimensionalBody()
        self.processor = None
        self.regulator = None

    def set_dimensional_systems(self, processor, regulator):
        """Hooks into the core dimensional systems."""
        self.processor = processor
        self.regulator = regulator
        # Once connected, immediately reify the body in the Lattice
        if self.processor and self.regulator:
            self._reify_body_in_lattice()

    def _reify_body_in_lattice(self):
        """
        Ensures all body parts exist as Crystals in the main processor.
        """
        print("[TACTILE] Reifying virtual body in Crystal Lattice...")
        for part_type, part_data in self.body.parts.items():
            crystal = self.processor.get_or_create_crystal(part_type.value)
            if not crystal.facets:
                # Create the primary sensory facet for this body part
                f = crystal.add_facet("sensory", f"Virtual {part_type.name}", 0.8)
                self.regulator.register_facet(f)

    # ------------------------------------------------------------------------
    # CORE PROCESSING (Dimensionalized Sensation)
    # ------------------------------------------------------------------------
    def process_tactile_input(self, raw_data: Dict[str, Any]):
        """
        Converts raw touch/spatial data into Lattice modulations.
        """
        # 1. Map input to specific body part
        target_part = self._map_input_to_part(raw_data)
        if not target_part: return

        # 2. Update the virtual body state
        pressure = raw_data.get('pressure', 0.0)
        target_part.tension = min(1.0, target_part.tension + pressure * 0.2)
        target_part.sensitivity = min(1.0, pressure + 0.3)

        # 3. Modulate the Lattice immediately
        if self.processor and self.regulator:
            self._modulate_part_crystal(target_part, pressure)

        # 4. Update global balance if needed
        if raw_data.get('imbalance_detected'):
            self.body.balance = max(0.0, self.body.balance - 0.2)
            # Imbalance immediately affects global energy stability
            # (This would be a hook into the regulator's global state if it had one exposed,
            #  for now we can simulate it by injecting 'instability' into the torso)
            torso = self.body.parts[BodyPartType.TORSO]
            self._modulate_part_crystal(torso, instability_spike=0.5)

    def _map_input_to_part(self, data: Dict[str, Any]) -> Optional[DimensionalBodyPart]:
        """Simple mapping from input strings to BodyPartType."""
        pt_str = data.get('body_part', '').upper()
        for pt in BodyPartType:
            if pt.name in pt_str:
                return self.body.parts[pt]
        return self.body.parts[BodyPartType.TORSO] # Default to torso

    def _modulate_part_crystal(self, part: DimensionalBodyPart, intensity: float = 0.0, instability_spike: float = 0.0):
        """
        Directly alters the physics vector of the body part's Crystal Facet.
        """
        crystal = self.processor.crystals.get(part.part_type.value)
        if not crystal or not crystal.facets: return

        # Get the main sensory facet
        facet = list(crystal.facets.values())[0]
        
        # Apply physics modulations
        # High tension = Low stability
        facet.stability = max(0.0, 1.0 - part.tension - instability_spike)
        # High sensitivity = High sensitivity point
        facet.sensitivity = part.sensitivity
        # Touch increases resonance (it's active)
        facet.resonance = min(1.0, facet.resonance + intensity * 0.5)

        # Inject energy to bring this sensation to conscious awareness
        self.regulator.inject_energy(facet.facet_id, intensity * 2.0)
