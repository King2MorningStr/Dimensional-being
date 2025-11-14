"""
Ghost Relic System
==================
Stores templates and anchors (never concepts)
Accelerates re-formation via cached organization
Respects conservation: base sparks persist, relics are cached structure
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict
import time
import json

@dataclass
class GhostRelic:
    """
    Template of a dissolved crystal structure
    Stores organization, not concepts
    """
    relic_id: str
    original_level: int  # 2, 3, or 4
    original_id: str
    
    # Template data (structure only)
    facet_template: Dict[str, Dict[str, float]]  # facet_id -> scoring channels
    connection_template: Dict[str, float]  # connection patterns
    pathway_template: Dict[str, float]  # pathway involvement
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    dissolution_reason: str = ""
    reformation_count: int = 0
    last_reformed: Optional[float] = None
    
    # Performance cache
    symmetry_cache: float = 0.0
    bias_tallies: Dict[str, float] = field(default_factory=dict)
    
    def accelerate_reformation(self, new_crystal: Any) -> None:
        """Apply cached template to speed up new crystal formation"""
        # Warm symmetry checks
        if hasattr(new_crystal, '_symmetry_cache'):
            new_crystal._symmetry_cache = self.symmetry_cache
        
        # Apply bias tallies
        if hasattr(new_crystal, 'avg_bias'):
            new_crystal.avg_bias = self.bias_tallies.get('avg', 0.5)
        
        # Apply connection patterns
        if hasattr(new_crystal, 'connections'):
            for conn_id, weight in self.connection_template.items():
                new_crystal.connections[conn_id] = weight * 0.8  # Reduced weight for template
        
        self.reformation_count += 1
        self.last_reformed = time.time()

@dataclass
class RelicAnchor:
    """
    Anchor point for relic - references eternal concepts
    """
    anchor_id: str
    l1_concept_refs: List[str]  # References to eternal L1 concepts
    anchor_type: str  # 'cluster', 'quasicrystal', 'layer'
    strength: float = 1.0

class GhostRelicSystem:
    """Manages ghost relics for faster crystal re-formation"""
    
    def __init__(self, relic_storage_path: str = "ghost_relics.json"):
        self.relics: Dict[str, GhostRelic] = {}
        self.anchors: Dict[str, RelicAnchor] = {}
        self.storage_path = relic_storage_path
        
        # Performance tracking
        self.total_reformations = 0
        self.reformation_speedup: List[float] = []
        
        self._load_relics()
    
    def create_relic(self, dissolving_crystal: Any, reason: str = "low_activity") -> GhostRelic:
        """
        Create ghost relic from dissolving crystal
        Conservation: only structure is cached, concepts remain eternal
        """
        relic_id = f"relic_{len(self.relics)}_{int(time.time())}"
        
        # Extract template (structure only)
        facet_template = {}
        if hasattr(dissolving_crystal, 'facets'):
            for fid, facet in dissolving_crystal.facets.items():
                facet_template[fid] = {
                    'resonance': getattr(facet, 'resonance', 0.0),
                    'bias': getattr(facet, 'bias', 0.5),
                    'recency': getattr(facet, 'recency', 1.0),
                    'coherence': getattr(facet, 'coherence', 0.0),
                    'novelty': getattr(facet, 'novelty', 0.0)
                }
        
        # Extract connections
        connection_template = {}
        if hasattr(dissolving_crystal, 'connections'):
            connection_template = dissolving_crystal.connections.copy()
        
        # Determine original level
        original_level = 2  # Default
        if hasattr(dissolving_crystal, 'quasicrystal_id'):
            original_level = 3
        elif hasattr(dissolving_crystal, 'layer_id'):
            original_level = 4
        
        # Create relic
        relic = GhostRelic(
            relic_id=relic_id,
            original_level=original_level,
            original_id=getattr(dissolving_crystal, 'cluster_id', 
                              getattr(dissolving_crystal, 'quasicrystal_id',
                                    getattr(dissolving_crystal, 'layer_id', 'unknown'))),
            facet_template=facet_template,
            connection_template=connection_template,
            pathway_template={},
            dissolution_reason=reason
        )
        
        # Cache performance data
        if hasattr(dissolving_crystal, '_calculate_symmetry'):
            relic.symmetry_cache = dissolving_crystal._calculate_symmetry()
        
        if hasattr(dissolving_crystal, 'avg_bias'):
            relic.bias_tallies['avg'] = dissolving_crystal.avg_bias
        
        self.relics[relic_id] = relic
        
        # Create anchor
        self._create_anchor(relic, dissolving_crystal)
        
        return relic
    
    def _create_anchor(self, relic: GhostRelic, crystal: Any) -> None:
        """Create anchor linking relic to eternal L1 concepts"""
        anchor_id = f"anchor_{relic.relic_id}"
        
        # Extract L1 concept references
        l1_refs = []
        if hasattr(crystal, 'parent_l1_ids'):
            l1_refs = crystal.parent_l1_ids
        elif hasattr(crystal, 'parent_l2_ids'):
            # Trace back to L1 through L2
            l1_refs = getattr(crystal, '_traced_l1_refs', [])
        
        anchor = RelicAnchor(
            anchor_id=anchor_id,
            l1_concept_refs=l1_refs,
            anchor_type=f"level_{relic.original_level}",
            strength=1.0
        )
        
        self.anchors[anchor_id] = anchor
    
    def find_matching_relic(self, forming_crystal: Any, 
                          l1_concept_refs: List[str]) -> Optional[GhostRelic]:
        """Find matching relic to accelerate formation"""
        best_match = None
        best_score = 0.0
        
        for relic_id, relic in self.relics.items():
            # Find anchor
            anchor_id = f"anchor_{relic_id}"
            if anchor_id not in self.anchors:
                continue
            
            anchor = self.anchors[anchor_id]
            
            # Calculate match score based on L1 concept overlap
            overlap = len(set(anchor.l1_concept_refs) & set(l1_concept_refs))
            total = len(set(anchor.l1_concept_refs) | set(l1_concept_refs))
            
            if total == 0:
                continue
            
            score = (overlap / total) * anchor.strength
            
            if score > best_score:
                best_score = score
                best_match = relic
        
        # Require minimum match threshold
        if best_score > 0.6:
            return best_match
        
        return None
    
    def apply_relic(self, relic: GhostRelic, target_crystal: Any) -> float:
        """
        Apply relic template to accelerate crystal formation
        Returns speedup factor
        """
        start_time = time.time()
        
        relic.accelerate_reformation(target_crystal)
        
        elapsed = time.time() - start_time
        speedup = 1.0 - min(0.8, elapsed)  # Assume max 80% speedup
        
        self.reformation_speedup.append(speedup)
        self.total_reformations += 1
        
        return speedup
    
    def prune_old_relics(self, max_age_hours: float = 168.0) -> int:
        """Prune relics older than max_age (default 7 days)"""
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        pruned = []
        for relic_id, relic in list(self.relics.items()):
            age = current_time - relic.created_at
            
            # Keep if recently used or young
            if (relic.last_reformed and 
                current_time - relic.last_reformed < max_age_seconds):
                continue
            
            if age > max_age_seconds and relic.reformation_count == 0:
                pruned.append(relic_id)
                del self.relics[relic_id]
                
                # Remove anchor
                anchor_id = f"anchor_{relic_id}"
                if anchor_id in self.anchors:
                    del self.anchors[anchor_id]
        
        return len(pruned)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get relic system statistics"""
        return {
            'total_relics': len(self.relics),
            'total_anchors': len(self.anchors),
            'total_reformations': self.total_reformations,
            'avg_speedup': (sum(self.reformation_speedup) / len(self.reformation_speedup)
                          if self.reformation_speedup else 0.0),
            'relics_by_level': {
                level: sum(1 for r in self.relics.values() if r.original_level == level)
                for level in [2, 3, 4]
            }
        }
    
    def save_relics(self) -> None:
        """Persist relics to storage"""
        data = {
            'relics': {
                rid: {
                    'relic_id': r.relic_id,
                    'original_level': r.original_level,
                    'original_id': r.original_id,
                    'facet_template': r.facet_template,
                    'connection_template': r.connection_template,
                    'pathway_template': r.pathway_template,
                    'created_at': r.created_at,
                    'dissolution_reason': r.dissolution_reason,
                    'reformation_count': r.reformation_count,
                    'last_reformed': r.last_reformed,
                    'symmetry_cache': r.symmetry_cache,
                    'bias_tallies': r.bias_tallies
                }
                for rid, r in self.relics.items()
            },
            'anchors': {
                aid: {
                    'anchor_id': a.anchor_id,
                    'l1_concept_refs': a.l1_concept_refs,
                    'anchor_type': a.anchor_type,
                    'strength': a.strength
                }
                for aid, a in self.anchors.items()
            },
            'stats': self.get_statistics()
        }
        
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_relics(self) -> None:
        """Load relics from storage"""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            # Load relics
            for rid, rdata in data.get('relics', {}).items():
                self.relics[rid] = GhostRelic(
                    relic_id=rdata['relic_id'],
                    original_level=rdata['original_level'],
                    original_id=rdata['original_id'],
                    facet_template=rdata['facet_template'],
                    connection_template=rdata['connection_template'],
                    pathway_template=rdata['pathway_template'],
                    created_at=rdata['created_at'],
                    dissolution_reason=rdata['dissolution_reason'],
                    reformation_count=rdata['reformation_count'],
                    last_reformed=rdata.get('last_reformed'),
                    symmetry_cache=rdata.get('symmetry_cache', 0.0),
                    bias_tallies=rdata.get('bias_tallies', {})
                )
            
            # Load anchors
            for aid, adata in data.get('anchors', {}).items():
                self.anchors[aid] = RelicAnchor(
                    anchor_id=adata['anchor_id'],
                    l1_concept_refs=adata['l1_concept_refs'],
                    anchor_type=adata['anchor_type'],
                    strength=adata.get('strength', 1.0)
                )
        
        except FileNotFoundError:
            pass  # New system

if __name__ == "__main__":
    print("Ghost Relic System - Ready for integration")
