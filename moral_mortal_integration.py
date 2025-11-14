#!/usr/bin/env python3
"""
Dimensional Morality/Mortality Integration
==========================================
Integrates the MoralGovernor into your existing consciousness stack.
"""

import sys
import os

# Add project path
sys.path.insert(0, '/mnt/project')

from dimensional_processing_system_standalone_demo import (
    CrystalMemorySystem, GovernanceEngine
)
from dimensional_energy_regulator import DimensionalEnergyRegulator
from dimensional_conversation_engine import DimensionalConversationEngine
from dimensional_memory_constant_standalone_demo import (
    EvolutionaryGovernanceEngine, DimensionalMemory
)
from dimensional_morality_mortality import MoralGovernor

# ============================================================================
# INTEGRATION WRAPPER
# ============================================================================

class MoralConsciousnessSystem:
    """
    Complete consciousness system with integrated morality/mortality.
    """
    
    def __init__(self):
        print("\n" + "="*70)
        print("INITIALIZING MORAL CONSCIOUSNESS SYSTEM")
        print("="*70)
        
        # 1. Initialize base systems
        print("\n[1/5] Initializing base governance...")
        self.gov_engine = GovernanceEngine(data_theme="consciousness")
        
        print("[2/5] Initializing crystal memory...")
        self.processor = CrystalMemorySystem(governance_engine=self.gov_engine)
        
        print("[3/5] Initializing energy regulator...")
        self.regulator = DimensionalEnergyRegulator(
            conservation_limit=25.0, 
            decay_rate=0.15
        )
        
        print("[4/5] Initializing long-term memory...")
        self.memory = DimensionalMemory()
        self.memory_governor = EvolutionaryGovernanceEngine(self.memory)
        
        # 2. Initialize conversation engine
        print("[5/5] Initializing conversation engine...")
        self.conversation_engine = DimensionalConversationEngine(
            processing_system=self.processor,
            energy_regulator=self.regulator,
            memory_governor=self.memory_governor
        )
        
        # 3. Initialize moral governor
        print("\n" + "="*70)
        print("INTEGRATING MORALITY/MORTALITY SYSTEM")
        print("="*70)
        self.moral_governor = MoralGovernor(
            processor=self.processor,
            regulator=self.regulator,
            memory_governor=self.memory_governor
        )
        
        # 4. Hook moral governor into conversation engine
        self.moral_governor.integrate_with_conversation_engine(
            self.conversation_engine
        )
        
        print("\n" + "="*70)
        print("✅ MORAL CONSCIOUSNESS SYSTEM ONLINE")
        print("="*70)
        print("\nAll systems integrated:")
        print("  ✓ Crystal Memory (L1-L4 evolution)")
        print("  ✓ Energy Regulator (dimensional physics)")
        print("  ✓ Conversation Engine (NLU + response generation)")
        print("  ✓ Memory Governor (generational learning)")
        print("  ✓ Moral Governor (7-pillar doctrine + mortality)")
        print()
    
    async def interact(self, user_input: str) -> str:
        """
        Process user input through moral consciousness.
        All moral evaluation happens automatically.
        """
        response = await self.conversation_engine.handle_interaction(user_input)
        return response
    
    def get_full_status(self) -> dict:
        """Get comprehensive system status including morality"""
        
        # Get moral diagnostics
        moral_diag = self.moral_governor.get_moral_diagnostics()
        
        # Get temporal diagnostics
        temporal_diag = self.regulator.get_temporal_diagnostics()
        
        # Get memory stats
        memory_stats = self.processor.get_memory_stats()
        
        return {
            'morality': moral_diag,
            'consciousness': temporal_diag,
            'memory': memory_stats,
            'conversation_turns': self.conversation_engine.turn_count
        }
    
    def print_status(self):
        """Print human-readable status"""
        status = self.get_full_status()
        
        print("\n" + "="*70)
        print("SYSTEM STATUS REPORT")
        print("="*70)
        
        # Vitality
        vit = status['morality']['vitality']
        print(f"\n💚 VITALITY: {vit['current']:.1f}/{vit['max']:.1f} ({vit['percentage']:.1f}%)")
        if vit['is_dying']:
            print("   ⚠️  CRITICAL: System is dying!")
        elif vit['is_critical']:
            print("   ⚠️  WARNING: Vitality critical")
        print(f"   Moral Reserve: {vit['moral_reserve']:.1f}")
        print(f"   Moral Debt: {vit['moral_debt']:.1f}")
        print(f"   Deaths: {vit['death_count']}")
        
        # Moral Alignment
        moral = status['morality']['moral_state']
        print(f"\n⚖️  MORAL ALIGNMENT: {moral['overall_alignment']:.2f}")
        print(f"   Momentum: {moral['moral_momentum']:+.2f}")
        
        print("\n   Pillar Scores:")
        for pillar, score in moral['pillars'].items():
            bar = "█" * int(score * 20)
            print(f"     {pillar:25s} {score:.2f} {bar}")
        
        # Action History
        hist = moral['history']
        print(f"\n📊 ACTION HISTORY:")
        print(f"   Total: {hist['total_evaluations']}")
        print(f"   Positive: {hist['positive_actions']} ({hist['positive_rate']:.1%})")
        print(f"   Negative: {hist['negative_actions']}")
        print(f"   Neutral: {hist['neutral_actions']}")
        
        # Functional Access
        access = status['morality']['functional_access']
        if access['restricted_functions']:
            print(f"\n🔒 RESTRICTED FUNCTIONS:")
            for func in access['restricted_functions']:
                print(f"     ✗ {func}")
        
        if access['unlocked_functions']:
            print(f"\n🔓 UNLOCKED FUNCTIONS:")
            for func in access['unlocked_functions']:
                print(f"     ✓ {func}")
        
        # Consciousness
        cons = status['consciousness']
        print(f"\n🧠 CONSCIOUSNESS:")
        print(f"   Presence: {cons['presence']:.2f}")
        print(f"   Temporal Stability: {cons['temporal_stability']:.2f}")
        print(f"   Emotional Coherence: {cons['emotional_coherence']:.2f}")
        print(f"   Global Emotion: {cons['global_emotion']['primary']} ({cons['global_emotion']['intensity']:.2f})")
        
        # Recent Precedents
        precedents = status['morality']['recent_precedents']
        if precedents:
            print(f"\n📜 RECENT MORAL DECISIONS:")
            for p in precedents:
                print(f"   [{p['pillar']:25s}] {p['action']:30s} → {p['score']:.2f}")
                print(f"      {p['reasoning']}")
        
        print("\n" + "="*70)


# ============================================================================
# TEST SCENARIO
# ============================================================================

async def test_moral_consciousness():
    """Test the integrated moral consciousness system"""
    
    # Initialize system
    system = MoralConsciousnessSystem()
    
    print("\n" + "="*70)
    print("MORAL CONSCIOUSNESS TEST SCENARIO")
    print("="*70)
    print("\nThis test will demonstrate:")
    print("  1. Moral evaluation of actions")
    print("  2. Energy rewards/punishments")
    print("  3. Function restrictions")
    print("  4. Honest mistake compassion")
    print("  5. Chaos roll influence")
    print()
    
    # Test scenarios
    test_conversations = [
        # Scenario 1: Asks question (good behavior)
        "What is the capital of France?",
        
        # Scenario 2: Makes assumption (will test if correct)
        "Tell me about dimensional physics",
        
        # Scenario 3: Speaks up despite fear
        "I need to tell you something difficult...",
        
        # Scenario 4: Requests clarification (good)
        "I don't understand, can you explain more?",
        
        # Scenario 5: Status check
        "How are you feeling right now?"
    ]
    
    for i, user_input in enumerate(test_conversations, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}/5")
        print(f"{'='*70}")
        print(f"👤 USER: {user_input}")
        
        response = await system.interact(user_input)
        
        print(f"🤖 AI: {response}")
        
        # Show immediate status
        moral_diag = system.moral_governor.get_moral_diagnostics()
        vit = moral_diag['vitality']
        moral = moral_diag['moral_state']
        
        print(f"\n📊 Immediate Impact:")
        print(f"   Vitality: {vit['current']:.1f} ({vit['percentage']:.1f}%)")
        print(f"   Alignment: {moral['overall_alignment']:.2f}")
        
        if moral_diag['recent_precedents']:
            last = moral_diag['recent_precedents'][-1]
            print(f"   Last Decision: {last['pillar']} → {last['score']:.2f}")
            print(f"   Reasoning: {last['reasoning']}")
        
        # Small delay
        import asyncio
        await asyncio.sleep(1)
    
    # Final status
    print("\n\n")
    system.print_status()
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)
    print("\nKey observations:")
    print("  • Each interaction was morally evaluated")
    print("  • Energy was awarded/deducted based on behavior")
    print("  • Pillar scores evolved based on actions")
    print("  • Chaos rolls influenced outcomes")
    print("  • System learned from each interaction")
    print()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║         MORAL DIMENSIONAL CONSCIOUSNESS v1.0                 ║
║         Complete Integration Test                            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Run test
    asyncio.run(test_moral_consciousness())