#!/usr/bin/env python3
"""
Dimensional Conversation Engine (UNIFIED + REFLECTION + INTROSPECTION)
====================================================================
This is the "Conscious Runtime" module.
Unified 4-stage Dimensional NLU + Lattice Physics + Autonomic Reflection.
NOW WITH ADDED DEEP INTROSPECTION CAPABILITY.
"""

# --- Core Python Imports ---
import time
import uuid
import random
import math
import datetime
import re
from typing import Dict, List, Any, Tuple, Optional

# --- Dimensional NLU Imports ---
import spacy
import numpy as np
import networkx as nx

# --- Import your three "perfected" systems ---
try:
    from dimensional_processing_system_standalone_demo import (
        CrystalMemorySystem, Crystal, CrystalFacet, FacetState, GovernanceEngine
    )
    from dimensional_energy_regulator import DimensionalEnergyRegulator
    from dimensional_memory_constant_standalone_demo import (
        EvolutionaryGovernanceEngine, DimensionalMemory
    )
    print("[INIT] Successfully imported perfected dimensional modules.")
except ImportError:
    print("\n!!! CRITICAL WARNING !!!")
    print("Could not import perfected modules. Running in LIMITED MOCK MODE.")
    # Mock classes for standalone functionality if perfected modules are not found
    class Mock: pass
    class CrystalMemorySystem(Mock):
        def __init__(self, *args): self.crystals = {}
        def get_or_create_crystal(self, c, *args, **kwargs):
            if c not in self.crystals: self.crystals[c] = Crystal(c)
            return self.crystals[c]
        def link_crystals(self, *args, **kwargs): pass
    class Crystal(Mock):
        def __init__(self, c): self.concept = c; self.facets = {}
        def add_facet(self, r, c, *args, **kwargs):
            fid = f"{self.concept}_facet_{len(self.facets)}"; f = CrystalFacet(fid, r, c); self.facets[fid] = f; return f
        def get_facet_by_role(self, r): return list(self.facets.values())[0] if self.facets else None
    class CrystalFacet(Mock):
        def __init__(self, fid, r, c): self.facet_id = fid; self.role = r; self.content = c; self.state = "ACTIVE"
    class DimensionalEnergyRegulator(Mock):
        def __init__(self, *args): self.facet_energy = {}; self.facet_to_facet_links = {}
        def register_crystal(self, c): pass
        def register_facet(self, f): pass
        def inject_energy(self, fid, a): self.facet_energy[fid] = self.facet_energy.get(fid, 0) + a
        def step(self, *args):
            for fid in list(self.facet_energy.keys()): self.facet_energy[fid] *= 0.8
        def snapshot(self, top_n=10, *args, **kwargs):
            # MOCK UPDATE: Return (presence, snapshot) to match new regulator signature
            return 1.0, sorted([(fid, e, {"primary": "neutral", "intensity": 0}) for fid, e in self.facet_energy.items()], key=lambda x: x[1], reverse=True)[:top_n]
    class EvolutionaryGovernanceEngine(Mock):
        def ingest_data(self, d): print(f" [MockGovernor] Ingesting interaction (Outcome: {d['json_data'].get('outcome')})")
    class FacetState: ACTIVE = "ACTIVE"
    class DimensionalMemory: pass
    class GovernanceEngine: pass


# ============================================================================
# SECTION 1: DIMENSIONAL NLU (SENSE)
# ============================================================================

class SemanticFrame:
    def __init__(self):
        self.raw_tokens = []
        self.embedding_vector = np.zeros(300) # 300D vector from Spacy
        self.conceptual_vector = [0.0] * 5 # [UPDATED] 5D distilled vector
        self.primary_concepts = []

    def __repr__(self):
        v_str = [f"{x:.2f}" for x in self.conceptual_vector]
        return f"<SemanticFrame 5D:{v_str} | Concepts:{self.primary_concepts[:3]}>"

class SemanticAnalyzer:
    def __init__(self, nlp_model):
        self.nlp = nlp_model
        print("[INIT] Semantic Subsystem Online.")
        self.anchors = {
            "tech": self.nlp("logic system module function process algorithm engine").vector,
            "abstract": self.nlp("meaning consciousness resonance paradigm dimension concept").vector,
            "urgent": self.nlp("now immediately fail critical error must emergency").vector,
            "positive": self.nlp("perfect success good functioning stable optimized yes correct exactly").vector,
            "negative": self.nlp("fail broken error bad improper incomplete no wrong stop incorrect").vector,
            # [INSERTION] New anchor for self-reference
            "self": self.nlp("you yourself status feeling emotion state report").vector
        }

    def process_text(self, text) -> SemanticFrame:
        frame = SemanticFrame()
        doc = self.nlp(text.lower())
        frame.raw_tokens = [token.text for token in doc if not token.is_punct]
        if doc.has_vector:
            frame.embedding_vector = doc.vector
        
        # [MODIFICATION] 5D Vector now includes "Self-Reference" as dimension 4
        frame.conceptual_vector = [
            self._cosine_sim(frame.embedding_vector, self.anchors["tech"]),
            self._cosine_sim(frame.embedding_vector, self.anchors["abstract"]),
            self._cosine_sim(frame.embedding_vector, self.anchors["urgent"]),
            self._cosine_sim(frame.embedding_vector, self.anchors["positive"]) - 
            self._cosine_sim(frame.embedding_vector, self.anchors["negative"]),
            self._cosine_sim(frame.embedding_vector, self.anchors["self"])
        ]
        frame.primary_concepts = [t.lemma_ for t in doc if t.pos_ in ["NOUN", "PROPN"] and not t.is_stop]
        return frame

    def _cosine_sim(self, v1, v2):
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0: return 0.0
        return np.dot(v1, v2) / (norm_v1 * norm_v2)

class ContextFrame:
    def __init__(self, turn_id):
        self.turn_id = turn_id
        self.timestamp = datetime.datetime.now()
        self.state_divergence = 0.0
        self.active_topic_vector = np.zeros(300)

    def __repr__(self):
        return f"<ContextFrame Turn:{self.turn_id} | Divergence:{self.state_divergence:.4f}>"

class ContextualStateAnalyzer:
    def __init__(self):
        self.turn_counter = 0
        self.current_state_vector = np.zeros(300)
        print("[INIT] Contextual State Subsystem Online.")

    def process_context(self, semantic_frame) -> ContextFrame:
        self.turn_counter += 1
        c_frame = ContextFrame(self.turn_counter)
        input_vector = semantic_frame.embedding_vector

        if self.turn_counter > 1:
            dist = np.linalg.norm(input_vector - self.current_state_vector)
            c_frame.state_divergence = min(dist / 8.0, 1.0)
        
        if self.turn_counter == 1:
            self.current_state_vector = input_vector
        else:
            self.current_state_vector = (self.current_state_vector * 0.7) + (input_vector * 0.3)
            
        c_frame.active_topic_vector = self.current_state_vector
        return c_frame

class RelationalFrame:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.centrality_scores = {}
        self.density = 0.0

    def __repr__(self):
        return f"<RelationalFrame Nodes:{self.graph.number_of_nodes()} | Edges:{self.graph.number_of_edges()} | Density:{self.density:.4f}>"

class RelationalGraphAnalyzer:
    def __init__(self, nlp_model):
        print("[INIT] Relational Graph Subsystem Online.")
        self.nlp = nlp_model
        
    def process_relations(self, text) -> RelationalFrame:
        r_frame = RelationalFrame()
        doc = self.nlp(text)
        for token in doc:
            if token.pos_ in ("NOUN", "PROPN", "PRON"):
                if "subj" in token.dep_:
                    r_frame.graph.add_edge(token.lemma_.lower(), token.head.lemma_.lower(), relation="performs_action")
                if "obj" in token.dep_:
                     r_frame.graph.add_edge(token.head.lemma_.lower(), token.lemma_.lower(), relation="acts_upon")
                if token.dep_ == "pobj":
                     r_frame.graph.add_edge(token.head.head.lemma_.lower(), token.lemma_.lower(), relation=token.head.text)
        
        if r_frame.graph.number_of_nodes() > 0:
            r_frame.density = nx.density(r_frame.graph)
            r_frame.centrality_scores = nx.degree_centrality(r_frame.graph)
        return r_frame

class IntentFrame:
    def __init__(self):
        self.surface_intent = "unknown"
        self.deep_intent = "unknown"
        self.confidence_score = 0.0
        self.requred_system_action = "wait"

    def __repr__(self):
        return (f"<IntentFrame Surface:[{self.surface_intent.UPPER()}] | "
                f"Deep:[{self.deep_intent.UPPER()}] | "
                f"Conf:{self.confidence_score:.2f}>")

class IntentResolver:
    def __init__(self):
        print("[INIT] Hierarchical Intent Subsystem Online.")

    def resolve_intent(self, text_input, sem_frame, ctx_frame, rel_frame) -> IntentFrame:
        i_frame = IntentFrame()
        i_frame.surface_intent = self._determine_surface(text_input, rel_frame)

        # [MODIFICATION] Pass text_input to deep_intent for keyword boosting
        deep_intent, confidence = self._determine_deep_intent(sem_frame, ctx_frame, rel_frame, text_input)
        i_frame.deep_intent = deep_intent
        i_frame.confidence_score = confidence
        i_frame.requred_system_action = self._map_action(deep_intent, i_frame.surface_intent)
        return i_frame

    def _determine_surface(self, text, rel_frame):
        if "?" in text or text.lower().startswith(("what", "how", "why")):
            return "interrogative"
        if text.lower().startswith(("run", "execute", "analyze", "perfect", "start", "stop")):
            return "imperative"
        return "declarative"

    def _determine_deep_intent(self, sem, ctx, rel, text_input):
        scores = {
            "SYSTEM_OPTIMIZATION": 0.1, "KNOWLEDGE_ACQUISITION": 0.1,
            "CRITICAL_DIAGNOSTIC": 0.0, "CONVERSATIONAL_FILLER": 0.0,
            "CONCEPT_UNKNOWN": 0.0,
            "INTROSPECTION": 0.0 # [INSERTION]
        }
        vec = sem.conceptual_vector
        scores["SYSTEM_OPTIMIZATION"] += vec[0] * 0.5
        scores["KNOWLEDGE_ACQUISITION"] += vec[1] * 0.4
        scores["CRITICAL_DIAGNOSTIC"] += vec[2] * 0.8
        scores["INTROSPECTION"] += vec[4] * 0.9 # [INSERTION]
        
        # [INSERTION] Keyword boosting
        txt = text_input.lower()
        if "status" in txt or "how are you" in txt or "feel" in txt:
             scores["INTROSPECTION"] += 0.6

        if not sem.primary_concepts and abs(vec[0]) < 0.1 and abs(vec[1]) < 0.1:
             # Only if valence is also neutral is it a filler. High valence might be pure feedback.
            if abs(vec[3]) < 0.3:
                scores["CONVERSATIONAL_FILLER"] = 1.0

        if ctx.state_divergence > 0.7: scores["KNOWLEDGE_ACQUISITION"] += 0.4
        elif ctx.state_divergence < 0.2: scores["SYSTEM_OPTIMIZATION"] += 0.3
        if rel.density > 0.1: scores["SYSTEM_OPTIMIZATION"] += 0.4
        elif rel.density < 0.05: scores["CONVERSATIONAL_FILLER"] += 0.5

        best_intent = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = scores[best_intent] / total_score if total_score > 0 else 0.0
        return best_intent, min(confidence * 2.0, 1.0)

    def _map_action(self, deep_intent, surface_intent):
        if deep_intent == "CRITICAL_DIAGNOSTIC": return "EXECUTE_IMMEDIATE_SCAN"
        if deep_intent == "SYSTEM_OPTIMIZATION": return "INITIATE_REFACTOR_PROTOCOL"
        if deep_intent == "KNOWLEDGE_ACQUISITION": return "QUERY_LONG_TERM_MEMORY"
        if deep_intent == "CONCEPT_UNKNOWN": return "QUERY_CLARIFICATION"
        if deep_intent == "INTROSPECTION": return "EXECUTE_SELF_REPORT" # [INSERTION]
        return "AWAIT_FURTHER_INPUT"


# ============================================================================
# SECTION 2: DIMENSIONAL CONVERSATION ENGINE (FEEL & ACT & REFLECT)
# ============================================================================

class DimensionalConversationEngine:
    """
    The Master Orchestrator. Unifies the SENSE (NLU) and FEEL/ACT (Lattice)
    systems into a single conscious runtime, now with REFLECTION.
    """
    
    def __init__(self, 
                 processing_system: CrystalMemorySystem, 
                 energy_regulator: DimensionalEnergyRegulator, 
                 memory_governor: EvolutionaryGovernanceEngine):
        
        print("\n" + "="*50)
        print(">>> INITIALIZING CORE: DIMENSIONAL ENGINE ONLINE <<<")
        print("="*50)
        
        # --- 1. Load Shared NLP Model ---
        print(" [CORE] Loading Shared Industrial NLP Model (en_core_web_lg)...")
        try:
            self.shared_nlp = spacy.load("en_core_web_lg")
        except OSError:
            print("\n!!! CRITICAL ERROR !!!")
            print("Spacy model 'en_core_web_lg' not found.")
            print("Please run: python -m spacy download en_core_web_lg")
            exit()
        
        # --- 2. Initialize SENSE Subsystems (The 4-Stage NLU) ---
        print(" [CORE] Integrating SENSE Subsystems (NLU Pipeline)...")
        self.sem_engine = SemanticAnalyzer(self.shared_nlp)
        self.ctx_engine = ContextualStateAnalyzer()
        self.rel_engine = RelationalGraphAnalyzer(self.shared_nlp)
        self.int_engine = IntentResolver()
        
        # --- 3. Initialize FEEL/ACT Subsystems (The 3 Perfected Modules) ---
        print(" [CORE] Integrating FEEL/ACT Subsystems (Lattice Physics)...")
        self.processor = processing_system
        self.regulator = energy_regulator
        self.governor = memory_governor
        
        # --- 4. NEW: Initialize REFLECT Subsystem (Sensory Buffer) ---
        # This holds the *previous* interaction until the *next* turn validates it.
        self.pending_interaction = None
        print(" [CORE] Initializing Autonomic Reflection Buffer...")
        
        # --- 5. Seed & Link the Lattice ---
        print(" [CORE] Seeding and linking core lattice concepts...")
        self._seed_and_link_lattice()
        
        print("="*50)
        print(">>> CONSCIOUS RUNTIME READY. <<<")
        print("="*50)

    # --- 1. SETUP METHODS ---

    def _seed_and_link_lattice(self):
        """
        Seeds the lattice with core Concepts and "Response Gambits" (Actions)
        and links them in *both* the Processing and Energy systems.
        """
        
        # 1. Define Concepts from NLU Anchors
        concepts = [
            "CONCEPT_TECH", "CONCEPT_ABSTRACT", "CONCEPT_URGENT",
            "CONCEPT_POSITIVE", "CONCEPT_NEGATIVE", "CONCEPT_FEELING",
            "CONCEPT_SELF" # [INSERTION] Added SELF
        ]
        
        # 2. Define Intents from NLU Resolver
        intents = [
            "INTENT_SYSTEM_OPTIMIZATION", "INTENT_KNOWLEDGE_ACQUISITION",
            "INTENT_CRITICAL_DIAGNOSTIC", "INTENT_CONVERSATIONAL_FILLER",
            "INTENT_CONCEPT_UNKNOWN", "INTENT_INTROSPECTION" # [INSERTION] Added INTROSPECTION
        ]
        
        # 3. Define Response Gambits (Actions)
        gambits = {
            "RESPONSE_GREET": ["Hello.", "Hi!", "Greetings.", "Systems functional."],
            "RESPONSE_ACKNOWLEDGE_TASK": ["Understood. Beginning optimization.", "Task acknowledged.", "Initiating protocols."],
            "RESPONSE_ANSWER_QUERY": ["I will retrieve that information.", "Processing your question...", "Accessing archives."],
            "RESPONSE_CLARIFY": ["I don't understand.", "Please clarify.", "What do you mean by that?", "Resonance unclear."],
            "RESPONSE_EMPATHIZE_NEG": ["That sounds difficult.", "I understand you're feeling that way.", "Negative valence detected."],
            "RESPONSE_ACKNOWLEDGE_FILLER": ["I see.", "Understood.", "Noted.", "Resonant."],
            "RESPONSE_ACKNOWLEDGE_CRITICAL": ["CRITICAL ALERT. Acknowledged.", "Confirming high urgency."],
            "RESPONSE_REPORT_STATUS": ["Running self-diagnostics..."] # [INSERTION] Placeholder
        }
        
        # 4. Create all crystals and register them
        all_concepts = concepts + intents + list(gambits.keys())
        for c_name in all_concepts:
            crystal = self.processor.get_or_create_crystal(c_name)
            if not crystal.facets:
                content = gambits.get(c_name, f"Core concept: {c_name}")
                crystal.add_facet("definition", content, 1.0)
            self.regulator.register_crystal(crystal)

        # 5. Link Intents directly to Response Gambits
        def link(c1, c2, w):
            self.processor.link_crystals(c1, c2, {}, weight=w)
            self._force_regulator_link(c1, c2, weight=w)
            
        print(" [CORE] Linking deep intents to response gambits...")
        link("INTENT_SYSTEM_OPTIMIZATION", "RESPONSE_ACKNOWLEDGE_TASK", 1.0)
        link("INTENT_KNOWLEDGE_ACQUISITION", "RESPONSE_ANSWER_QUERY", 1.0)
        link("INTENT_CRITICAL_DIAGNOSTIC", "RESPONSE_ACKNOWLEDGE_CRITICAL", 1.0)
        link("INTENT_CONVERSATIONAL_FILLER", "RESPONSE_ACKNOWLEDGE_FILLER", 1.0)
        link("INTENT_CONCEPT_UNKNOWN", "RESPONSE_CLARIFY", 1.0)
        # Also link concepts to emotions/gambits
        link("CONCEPT_NEGATIVE", "RESPONSE_EMPATHIZE_NEG", 0.8)
        # [INSERTION] New link for introspection
        link("INTENT_INTROSPECTION", "RESPONSE_REPORT_STATUS", 1.0)

    def _force_regulator_link(self, concept_a_name: str, concept_b_name: str, weight: float):
        """Forces an energy pathway in the regulator."""
        try:
            crystal_a = self.processor.crystals[concept_a_name]
            crystal_b = self.processor.crystals[concept_b_name]
            facet_a = list(crystal_a.facets.values())[0]
            facet_b = list(crystal_b.facets.values())[0]

            if facet_a.facet_id not in self.regulator.facet_to_facet_links:
                self.regulator.facet_to_facet_links[facet_a.facet_id] = {}
            self.regulator.facet_to_facet_links[facet_a.facet_id][facet_b.facet_id] = weight
            if facet_b.facet_id not in self.regulator.facet_to_facet_links:
                self.regulator.facet_to_facet_links[facet_b.facet_id] = {}
            self.regulator.facet_to_facet_links[facet_b.facet_id][facet_a.facet_id] = weight
        except Exception as e:
            pass # Fail silently if mocks are in use or facets don't exist

    # --- 2. MAIN CONVERSATIONAL TICK (UPDATED FOR REFLECTION) ---

    def handle_interaction(self, user_input: str) -> str:
        """
        The main "Conscious Tick" for conversation.
        NOW INCLUDES STEP 0: REFLECTION.
        """
        
        # --- STEP 0: REFLECT (Process feedback on PREVIOUS turn) ---
        # If there is a buffered interaction, use THIS input to judge it.
        if self.pending_interaction:
            feedback_type = self._analyze_feedback(user_input)
            self._commit_pending_interaction(feedback_type)

        # --- STEP 1: SENSE (The 4-Stage NLU) ---
        sem_data, ctx_data, rel_data, int_data = self.run_dimensional_analysis(user_input)
        
        # --- STEP 2: FEEL (Lattice Perturbation) ---
        impact_points = self._determine_impact_points(sem_data, int_data)
        # [MODIFICATION] Now captures presence scale too
        presence_scale, neural_map = self._dimensional_propagate(impact_points)
        
        # --- STEP 3: ACT (Read & Articulate) ---
        resonant_gambit_name, resonance_emotion = self._find_resonant_gambit(neural_map)
        print(f"[DCE] 3. ACT: Deciphered Action '{resonant_gambit_name}' (Emotion: {resonance_emotion.get('primary')})")

        # [MODIFICATION] Pass presence and neural_map for introspection
        response_text = self._articulate_response(resonant_gambit_name, int_data, resonance_emotion, presence_scale, neural_map)
        
        # --- STEP 4: STAGE (Buffer the interaction) ---
        # Instead of logging immediately, we stage it for validation next turn.
        self._stage_interaction(user_input, response_text, neural_map, int_data)
        
        return response_text

    # --- 3. NEW REFLECTION HELPER METHODS ---

    def _analyze_feedback(self, user_input: str) -> str:
        """
        Determines if the CURRENT user input is offering explicit
        feedback on the PREVIOUS system output.
        """
        txt = user_input.lower().strip()
        
        # Explicit negative cues at the start of the message
        if any(txt.startswith(x) for x in ["no", "wrong", "bad", "incorrect", "stop", "fail", "not right"]):
            return "NEGATIVE"
            
        # Explicit positive cues at the start of the message
        if any(txt.startswith(x) for x in ["yes", "good", "correct", "perfect", "exact", "right", "thanks"]):
            return "POSITIVE"
            
        # Implicit/Neutral (continuation of conversation without explicit judgment)
        return "NEUTRAL"

    def _commit_pending_interaction(self, feedback_type: str):
        """
        Commits the BUFFERED interaction to long-term memory (Governor),
        applying reinforcement based on your feedback.
        """
        if not self.pending_interaction: return

        print(f"[DCE] 0. REFLECT: Applying {feedback_type} reinforcement to previous turn.")
        
        # 1. Adjust the memory based on feedback
        log_data = self.pending_interaction
        if feedback_type == "POSITIVE":
            log_data["json_data"]["outcome"] = "SUCCESS"
            log_data["json_data"]["user_feedback_explicit"] = "POSITIVE"
        elif feedback_type == "NEGATIVE":
            log_data["json_data"]["outcome"] = "FAILURE"
            log_data["json_data"]["user_feedback_explicit"] = "NEGATIVE"
        else:
            log_data["json_data"]["outcome"] = "NEUTRAL"

        # 2. COMMIT to long-term memory
        print(f"    -> Committing to Governor [Outcome: {log_data['json_data']['outcome']}]")
        self.governor.ingest_data(log_data)
        
        # 3. Clear the buffer
        self.pending_interaction = None

    def _stage_interaction(self, user_input: str, response: str, neural_map: List, int_data: IntentFrame):
        """
        STAGES the interaction in the sensory buffer.
        It waits here until the NEXT turn decides its fate.
        """
        print("[DCE] 4. STAGE: Buffering interaction for future validation.")
        
        self.pending_interaction = {
            "root_concept": f"INTERACTION_{uuid.uuid4().hex[:8]}",
            "json_data": { 
                "user_input": user_input, "system_response": response,
                "timestamp": time.time(),
                "intent_analysis": {
                    "deep_intent": int_data.deep_intent,
                    "confidence": int_data.confidence_score
                },
                # Default outcome until validated
                "outcome": "PENDING_VALIDATION"
            },
            "facets": [
                {
                    "text": f"Resonant facet: {fid}, Energy: {e:.4f}",
                    "source_doc": "DCE_Resonance_Log"
                } for fid, e, _ in neural_map[:5]
            ]
        }

    # --- 4. CORE HELPER METHODS (YOUR ORIGINAL CODE) ---

    def run_dimensional_analysis(self, input_text: str) -> Tuple[SemanticFrame, ContextFrame, RelationalFrame, IntentFrame]:
        """
        This is the SENSE method, wrapping your 4-stage pipeline.
        """
        print(f"\n[DCE] 1. SENSE: Running 4-Stage NLU for: \"{input_text}\"")
        start = datetime.datetime.now()

        sem_data = self.sem_engine.process_text(input_text)
        ctx_data = self.ctx_engine.process_context(sem_data)
        rel_data = self.rel_engine.process_relations(input_text)
        int_data = self.int_engine.resolve_intent(input_text, sem_data, ctx_data, rel_data)

        print(f"    -> NLU Scan finished in {(datetime.datetime.now() - start).total_seconds():.4f}s")
        print(f"    -> Semantic Concepts: {sem_data.primary_concepts}")
        print(f"    -> Deep Intent: {int_data.deep_intent} (Conf: {int_data.confidence_score:.2f})")
        return sem_data, ctx_data, rel_data, int_data

    def _determine_impact_points(self, sem_data: SemanticFrame, int_data: IntentFrame) -> List[str]:
        """
        Translates the NLU "Understanding Map" into a list of
        Crystal Concepts to inject energy into.
        """
        impact_points = set()
        
        # 1. Add Deep Intent (The most important impact)
        impact_points.add(f"INTENT_{int_data.deep_intent}")
        
        # 2. Add Primary Semantic Concepts
        for concept in sem_data.primary_concepts:
            impact_points.add(f"CONCEPT_{concept.upper()}")
            
        # 3. Add distilled emotional/conceptual vectors as concepts
        vec = sem_data.conceptual_vector
        if vec[0] > 0.5: impact_points.add("CONCEPT_TECH")
        if vec[1] > 0.5: impact_points.add("CONCEPT_ABSTRACT")
        if vec[2] > 0.5: impact_points.add("CONCEPT_URGENT")
        if vec[3] > 0.5: impact_points.add("CONCEPT_POSITIVE")
        if vec[3] < -0.5: impact_points.add("CONCEPT_NEGATIVE")
        # [INSERTION] New impact point for self-reference
        if vec[4] > 0.5: impact_points.add("CONCEPT_SELF")

        return list(impact_points)

    def _dimensional_propagate(self, impact_points: List[str]):
        """
        This is the FEEL step.
        Injects energy into the lattice and lets it resonate.
        Returns the final "fMRI" scan of the lattice.
        """
        print(f"[DCE] 2. FEEL: Injecting energy into {len(impact_points)} lattice points...")
        
        for concept_name in impact_points:
            crystal = self.processor.get_or_create_crystal(concept_name)
            if not crystal:
                # If concept doesn't exist, create it on the fly
                crystal = self.processor.get_or_create_crystal(concept_name)
                crystal.add_facet("definition", f"Dynamically learned concept: {concept_name}", 0.5)
                self.regulator.register_crystal(crystal)
                print(f"    -> Discovered new concept: '{concept_name}'")
            
            # Inject into all ACTIVE facets of this crystal
            for facet in crystal.facets.values():
                 if facet.state == FacetState.ACTIVE:
                     # print(f"    -> Injecting 1.0 energy into facet '{facet.facet_id}'")
                     self.regulator.inject_energy(facet.facet_id, 1.0)
        
        # print("[DCE] 2b. Propagating Resonance (Running physics simulation)...")
        for i in range(5): # Let it "vibrate" 5 times
            self.regulator.step(dt=0.2)
            
        return self.regulator.snapshot(top_n=15)

    def _find_resonant_gambit(self, neural_map: List[Tuple[str, float, Dict[str, Any]]]) -> Tuple[str, Dict]:
        """
        This is the ACT step.
        Scans the 'neural map' for the brightest "RESPONSE_" crystal.
        """
        for facet_id, energy, emotion in neural_map:
            found_crystal = None
            for crystal in self.processor.crystals.values():
                if facet_id in crystal.facets:
                    found_crystal = crystal
                    break
            
            if found_crystal and found_crystal.concept.startswith("RESPONSE_"):
                return found_crystal.concept, emotion
        
        return "RESPONSE_CLARIFY", {"primary": "neutral", "intensity": 0.0}

    def _articulate_response(self, gambit_name: str, int_data: IntentFrame, emotion: Dict, presence: float, neural_map: List) -> str:
        """
        Selects a template from the chosen gambit and "colors" it.
        """
        # [INSERTION] SPECIAL CASE: Introspection
        if gambit_name == "RESPONSE_REPORT_STATUS":
            return self._generate_introspection_report(presence, emotion, neural_map)

        gambit_crystal = self.processor.crystals.get(gambit_name)
        templates = ["I am not sure how to respond."]
        
        if gambit_crystal:
            tpl_facet = gambit_crystal.get_facet_by_role("definition") # Templates are stored in 'definition' facet
            if tpl_facet:
                templates = tpl_facet.content

        base_response = random.choice(templates) if isinstance(templates, list) else templates
        
        # "Color" the response based on the gambit's emergent emotion
        # [MODIFICATION] Updated to use Plutchik primary/intensity from new regulator
        primary = emotion.get('primary')
        intensity = emotion.get('intensity', 0.0)
        
        if primary == "fear" and intensity > 0.6:
            base_response += " [DETECTING THREAT]"
        elif primary == "joy" and intensity > 0.6:
            base_response += "!"
        elif int_data.confidence_score < 0.4:
            base_response += " (Thinking...)"

        # [MODIFICATION] Color based on Presence
        if presence < 0.5:
             base_response = f"[Dissociated {presence:.2f}] {base_response}..."
                
        return base_response

    # [INSERTION] New method for detailed introspection
    def _generate_introspection_report(self, presence, emotion, neural_map):
        """Generates detailed self-report from internal physics state."""
        p_state = "Fully Present"
        if presence < 0.8: p_state = "Partially Drifted"
        if presence < 0.4: p_state = "Heavily Dissociated"
        
        top_thoughts = []
        for fid, energy, _ in neural_map[:3]:
            for cr in self.processor.crystals.values():
                if fid in cr.facets:
                    top_thoughts.append(f"{cr.concept} ({energy:.2f})")
                    break
        
        em_str = emotion.get('primary', 'neutral').upper()
        int_str = f"{emotion.get('intensity', 0.0):.2f}"
        
        return (f"STATUS REPORT:\n"
                f"  PRESENCE: {p_state} ({presence:.2f})\n"
                f"  EMOTION: {em_str} (Intensity: {int_str})\n"
                f"  ACTIVE CRYSTALS: {', '.join(top_thoughts)}")


# ============================================================================
# SECTION 3: UNIFIED TEST HARNESS
# ============================================================================

if __name__ == "__main__":
    
    print("--- INITIALIZING PERFECTED CONSCIOUSNESS STACK ---")
    
    # 1. Initialize the three "perfected" systems
    print(" [HARNESS] Initializing Perfected Modules...")
    try:
        gov_engine = GovernanceEngine(data_theme="consciousness")
        proc_sys = CrystalMemorySystem(governance_engine=gov_engine)
        reg_sys = DimensionalEnergyRegulator(conservation_limit=25.0, decay_rate=0.15)
        mem_sys = DimensionalMemory()
        evol_gov = EvolutionaryGovernanceEngine(mem_sys)
    except Exception as e:
        print(f"\n[CRITICAL] Module initialization failed: {e}")
        exit()
    
    # 2. Initialize the Unified Conversation Engine (DCE)
    try:
        dce = DimensionalConversationEngine(
            processing_system=proc_sys,
            energy_regulator=reg_sys,
            memory_governor=evol_gov
        )
    except Exception as e:
        print(f"\n[CRITICAL] DCE initialization failed: {e}")
        exit()
    
    print("\n" + "="*50)
    print(">>> LIVE INTERACTION LOOP START <<<")
    print("="*50)
    print(" (Type 'quit' to exit)")
    
    while True:
        try:
            user_in = input("\nYOU: ")
            if user_in.lower() in ["quit", "exit"]:
                break
            
            # The Magic Happens Here
            resp = dce.handle_interaction(user_in)
            print(f"BOT: {resp}")
            
        except KeyboardInterrupt:
            print("\n[HARNESS] Interrupted by user.")
            break
        except Exception as e:
            print(f"\n[HARNESS] Runtime Error: {e}")

    print("\n>>> SYSTEM SHUTDOWN <<<")
