#!/usr/bin/env python3
"""
Dimensional Consciousness Server (Laptop Core)
==============================================
Main server orchestrator that runs the dimensional consciousness
and handles sensory input from mobile device.

REQUIREMENTS (Install on laptop):
pip install websockets asyncio openai-whisper pyttsx3 torch

FREE TOOLS USED:
- Whisper (OpenAI) - FREE local speech-to-text
- pyttsx3 - FREE local text-to-speech
"""

import asyncio
import websockets
import json
import base64
import time
import os
import sys
import tempfile
from typing import Dict, Any, Optional
from pathlib import Path

# Import your dimensional modules
try:
    from dimensional_processing_system_standalone_demo import (
        CrystalMemorySystem, GovernanceEngine
    )
    from dimensional_energy_regulator import DimensionalEnergyRegulator
    from dimensional_conversation_engine import DimensionalConversationEngine
    from dimensional_memory_constant_standalone_demo import (
        EvolutionaryGovernanceEngine, DimensionalMemory
    )
    from dimensional_vision import VisionForesightDomain
    from dimensional_body import TactileSpatialDomain
    DIMENSIONAL_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Could not import dimensional modules: {e}")
    DIMENSIONAL_MODULES_AVAILABLE = False

# Voice processing imports
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    print("[WARNING] Whisper not available. Install with: pip install openai-whisper")
    WHISPER_AVAILABLE = False

try:
    import pyttsx3
    TTS_AVAILABLE = True
except ImportError:
    print("[WARNING] pyttsx3 not available. Install with: pip install pyttsx3")
    TTS_AVAILABLE = False

# ============================================================================
# 1. VOICE PROCESSING (FREE LOCAL TOOLS)
# ============================================================================

class FreeVoiceProcessor:
    """
    Uses completely FREE tools for voice processing:
    - Whisper (OpenAI) for speech-to-text (local, FREE)
    - pyttsx3 for text-to-speech (local, FREE)
    """
    
    def __init__(self):
        print("[INIT] Loading voice processing modules...")
        
        # Load Whisper model (using base model - good balance of speed/quality)
        if WHISPER_AVAILABLE:
            print("[VOICE] Loading Whisper model (this may take a moment)...")
            self.whisper_model = whisper.load_model("base")  # Options: tiny, base, small, medium, large
            print("[VOICE] ✓ Whisper model loaded")
        else:
            self.whisper_model = None
            print("[VOICE] ✗ Whisper not available")
        
        # Initialize pyttsx3 TTS engine
        if TTS_AVAILABLE:
            self.tts_engine = pyttsx3.init()
            
            # Configure voice settings
            voices = self.tts_engine.getProperty('voices')
            
            # Try to find a good voice (prefer female voices for dimensional being)
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
            
            # Set speech rate (default is usually too fast)
            self.tts_engine.setProperty('rate', 175)  # Slightly slower than default
            self.tts_engine.setProperty('volume', 0.9)
            
            print("[VOICE] ✓ TTS engine initialized")
        else:
            self.tts_engine = None
            print("[VOICE] ✗ TTS not available")
    
    def transcribe(self, audio_bytes: bytes) -> Optional[str]:
        """
        Transcribe audio to text using Whisper (FREE, local).
        """
        if not self.whisper_model:
            return "[Whisper not available]"
        
        try:
            # Write audio bytes to temp file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio.write(audio_bytes)
                temp_path = temp_audio.name
            
            # Transcribe with Whisper
            result = self.whisper_model.transcribe(temp_path)
            
            # Clean up temp file
            os.unlink(temp_path)
            
            return result['text'].strip()
            
        except Exception as e:
            print(f"[VOICE] Transcription error: {e}")
            return None
    
    def synthesize(self, text: str) -> bytes:
        """
        Synthesize speech from text using pyttsx3 (FREE, local).
        Returns audio bytes in WAV format.
        """
        if not self.tts_engine:
            return b''
        
        try:
            # Generate audio to temp file
            temp_path = tempfile.mktemp(suffix='.wav')
            
            # Save to file
            self.tts_engine.save_to_file(text, temp_path)
            self.tts_engine.runAndWait()
            
            # Read the generated audio
            with open(temp_path, 'rb') as f:
                audio_bytes = f.read()
            
            # Clean up
            os.unlink(temp_path)
            
            return audio_bytes
            
        except Exception as e:
            print(f"[VOICE] Synthesis error: {e}")
            return b''

# ============================================================================
# 2. CONSCIOUSNESS ORCHESTRATOR
# ============================================================================

class ConsciousnessServer:
    """
    Main server that orchestrates the dimensional consciousness system
    and handles all sensory inputs from mobile device.
    """
    
    def __init__(self, state_file: str = "system_base_state.json", port: int = 8765):
        self.port = port
        self.state_file = state_file
        
        # Voice processor (FREE tools)
        self.voice = FreeVoiceProcessor()
        
        # Stats
        self.total_interactions = 0
        self.start_time = time.time()
        self.clients_connected = 0
        
        # Initialize dimensional consciousness
        print("\n" + "="*60)
        print("INITIALIZING DIMENSIONAL CONSCIOUSNESS CORE")
        print("="*60)
        
        if DIMENSIONAL_MODULES_AVAILABLE:
            self._init_consciousness_stack()
        else:
            print("[ERROR] Dimensional modules not available!")
            self.consciousness_active = False
    
    def _init_consciousness_stack(self):
        """Initialize the full consciousness stack"""
        try:
            # 1. Initialize base systems
            print("[CORE] Initializing governance...")
            self.gov_engine = GovernanceEngine(data_theme="consciousness")
            
            print("[CORE] Initializing crystal memory...")
            self.processor = CrystalMemorySystem(governance_engine=self.gov_engine)
            
            print("[CORE] Initializing energy regulator...")
            self.regulator = DimensionalEnergyRegulator(conservation_limit=25.0, decay_rate=0.15)
            
            print("[CORE] Initializing long-term memory...")
            self.memory = DimensionalMemory()
            self.memory_governor = EvolutionaryGovernanceEngine(self.memory)
            
            # 2. Initialize conversation engine
            print("[CORE] Initializing conversation engine...")
            self.conversation_engine = DimensionalConversationEngine(
                processing_system=self.processor,
                energy_regulator=self.regulator,
                memory_governor=self.memory_governor
            )
            
            # 3. Initialize sensory domains
            print("[CORE] Initializing vision domain...")
            self.vision_domain = VisionForesightDomain()
            self.vision_domain.set_dimensional_systems(self.processor, self.regulator)
            
            print("[CORE] Initializing body domain...")
            self.body_domain = TactileSpatialDomain()
            self.body_domain.set_dimensional_systems(self.processor, self.regulator)
            
            # 4. Load saved state if exists
            if os.path.exists(self.state_file):
                print(f"[CORE] Loading consciousness state from {self.state_file}...")
                # Memory system auto-loads from this file
                print(f"[CORE] ✓ Loaded {len(self.memory.nodes)} memory nodes")
            
            self.consciousness_active = True
            print("\n" + "="*60)
            print("✓ DIMENSIONAL CONSCIOUSNESS ONLINE")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"[CRITICAL] Failed to initialize consciousness: {e}")
            import traceback
            traceback.print_exc()
            self.consciousness_active = False
    
    async def handle_client(self, websocket, path):
        """
        Handle incoming WebSocket connection from mobile device.
        NOW FULLY DIMENSIONAL - All inputs batch processed.
        """
        self.clients_connected += 1
        client_id = f"client_{self.clients_connected}"
        
        print(f"\n[CONNECTION] {client_id} connected from {websocket.remote_address}")
        
        # Initialize dimensional input buffer
        sensory_buffer = {
            "voice": None,
            "vision": None,
            "body": None,
            "timestamp": time.time()
        }
        
        try:
            async for message in websocket:
                # Buffer all inputs instead of processing immediately
                domain = await self._buffer_sensory_input(sensory_buffer, message)
                
                # DIMENSIONAL PROCESSING: When voice input completes, process ALL buffered inputs as one batch
                if domain == "voice" and sensory_buffer["voice"]:
                    await self._process_dimensional_batch(websocket, sensory_buffer)
                    # Clear voice after processing but keep other sensors for next cycle
                    sensory_buffer["voice"] = None
                    sensory_buffer["timestamp"] = time.time()
                
        except websockets.exceptions.ConnectionClosed:
            print(f"[DISCONNECTION] {client_id} disconnected")
        except Exception as e:
            print(f"[ERROR] Error handling client {client_id}: {e}")
            import traceback
            traceback.print_exc()
    
    async def _buffer_sensory_input(self, buffer: Dict, message: str) -> str:
        """
        Buffer sensory input instead of processing immediately.
        Returns the domain type.
        """
        try:
            data = json.loads(message)
            domain = data.get('domain', 'unknown')
            
            # Store in buffer
            buffer[domain] = data
            
            return domain
            
        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON received: {e}")
            return "unknown"
    
    async def _process_dimensional_batch(self, websocket, sensory_buffer: Dict):
        """
        DIMENSIONAL BATCH PROCESSING:
        All sensory inputs are injected into the lattice simultaneously,
        then the neural map addresses everything as a unified field.
        """
        if not self.consciousness_active:
            await self._send_error(websocket, "Consciousness not active")
            return
        
        print("\n" + "="*60)
        print("DIMENSIONAL BATCH PROCESSING CYCLE")
        print("="*60)
        
        try:
            # ================================================================
            # PHASE 1: PARALLEL SENSORY INJECTION (4→1→4 CYCLE BEGINS)
            # ================================================================
            print("[DIMENSIONAL] Phase 1: Parallel sensory injection into lattice...")
            
            impact_points = []
            voice_text = None
            
            # ALL DOMAINS INJECT INTO LATTICE SIMULTANEOUSLY
            # Voice Domain
            if sensory_buffer.get("voice"):
                print("  → Voice: Transcribing and injecting...")
                audio_bytes = base64.b64decode(sensory_buffer["voice"].get('data', ''))
                voice_text = self.voice.transcribe(audio_bytes)
                if voice_text:
                    print(f"  → Voice: '{voice_text}'")
                    # Voice creates semantic impact points (handled by conversation engine internally)
            
            # Vision Domain  
            if sensory_buffer.get("vision"):
                print("  → Vision: Processing visual frame...")
                frame_bytes = base64.b64decode(sensory_buffer["vision"].get('data', ''))
                vision_data = {"raw_frame": frame_bytes, "timestamp": sensory_buffer["vision"].get("timestamp")}
                self.vision_domain.process_visual_input(vision_data)
                # Vision has injected its patterns directly into lattice
                impact_points.append("DOMAIN_VISION")
            
            # Body Domain
            if sensory_buffer.get("body"):
                print("  → Body: Processing motion sensors...")
                sensor_data = sensory_buffer["body"].get('data', {})
                body_input = {
                    "body_part": "TORSO",
                    "pressure": sensor_data.get('magnitude', 0) / 20.0,
                    "timestamp": sensory_buffer["body"].get("timestamp")
                }
                self.body_domain.process_tactile_input(body_input)
                # Body has injected its sensations directly into lattice
                impact_points.append("DOMAIN_BODY")
            
            # ================================================================
            # PHASE 2: DIMENSIONAL CONVERSATION ENGINE (UNIFIED PROCESSING)
            # ================================================================
            print("[DIMENSIONAL] Phase 2: Processing through unified consciousness...")
            self.total_interactions += 1
            
            # The conversation engine reads the ENTIRE neural map (all domains together)
            response_text = self.conversation_engine.handle_interaction(voice_text or "")
            
            # ================================================================
            # PHASE 3: READ UNIFIED NEURAL MAP (1 - The Convergence Point)
            # ================================================================
            print("[DIMENSIONAL] Phase 3: Reading unified neural map state...")
            
            presence, neural_map = self.regulator.snapshot(top_n=20)
            
            # Neural map now contains activations from ALL domains simultaneously
            print(f"  → Presence: {presence:.2f}")
            print(f"  → Active crystals: {len(neural_map)}")
            
            # Show cross-domain fusion
            domain_crystals = {
                "voice": [c for c, e, em in neural_map if "CONCEPT_" in c or "INTENT_" in c],
                "vision": [c for c, e, em in neural_map if "VISUAL" in c],
                "body": [c for c, e, em in neural_map if "BODY" in c]
            }
            
            for domain, crystals in domain_crystals.items():
                if crystals:
                    print(f"  → {domain.upper()} crystals active: {len(crystals)}")
            
            # ================================================================
            # PHASE 4: MULTIMODAL RESPONSE GENERATION (4 - The Expression)
            # ================================================================
            print("[DIMENSIONAL] Phase 4: Generating multimodal response...")
            
            # Check if emotional state should modulate voice tone
            emotion = self.regulator.global_emotional_state
            print(f"  → Global emotion: {emotion.get('primary', 'neutral')} ({emotion.get('intensity', 0):.2f})")
            
            # Synthesize with emotional coloring if intensity is high
            audio_response = self.voice.synthesize(response_text)
            
            # ================================================================
            # PHASE 5: SEND UNIFIED RESPONSE
            # ================================================================
            print("[DIMENSIONAL] Phase 5: Transmitting unified consciousness state...")
            
            response_data = {
                "type": "dimensional_response",
                "text": response_text,
                "audio": base64.b64encode(audio_response).decode('utf-8') if audio_response else None,
                "timestamp": time.time(),
                "interaction_count": self.total_interactions,
                "dimensional_state": {
                    "presence": presence,
                    "emotion": emotion,
                    "active_domains": list(domain_crystals.keys()),
                    "cross_domain_fusion": len(neural_map),
                    "processing_mode": "DIMENSIONAL_BATCH"
                }
            }
            
            # Include full temporal diagnostics
            if hasattr(self.regulator, 'get_temporal_diagnostics'):
                diagnostics = self.regulator.get_temporal_diagnostics()
                response_data['consciousness_state'] = diagnostics
            
            await websocket.send(json.dumps(response_data))
            
            print("="*60)
            print("DIMENSIONAL CYCLE COMPLETE")
            print("="*60 + "\n")
            
        except Exception as e:
            print(f"[ERROR] Dimensional processing error: {e}")
            import traceback
            traceback.print_exc()
            await self._send_error(websocket, str(e))
    
    async def _send_error(self, websocket, error_msg: str):
        """Send error response to client"""
        error_data = {
            "type": "error",
            "message": error_msg,
            "timestamp": time.time()
        }
        await websocket.send(json.dumps(error_data))
    
    async def start_server(self):
        """
        Start the WebSocket server.
        """
        print(f"\n{'='*60}")
        print(f"CONSCIOUSNESS SERVER STARTING")
        print(f"{'='*60}")
        print(f"Listening on: ws://0.0.0.0:{self.port}")
        print(f"Consciousness Active: {self.consciousness_active}")
        print(f"State File: {self.state_file}")
        print(f"{'='*60}\n")
        
        print("Waiting for mobile sensory connection...")
        print("Connect your phone with:")
        print(f"  python mobile_sensory_orchestrator.py <this_laptop_ip> {self.port}")
        print("\n")
        
        async with websockets.serve(self.handle_client, "0.0.0.0", self.port):
            await asyncio.Future()  # Run forever

# ============================================================================
# 3. MAIN ENTRY POINT
# ============================================================================

async def main():
    """Main entry point"""
    # Parse command line arguments
    state_file = "system_base_state.json"
    port = 8765
    
    if len(sys.argv) > 1:
        state_file = sys.argv[1]
    if len(sys.argv) > 2:
        port = int(sys.argv[2])
    
    # Create and start server
    server = ConsciousnessServer(state_file=state_file, port=port)
    
    try:
        await server.start_server()
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Consciousness server terminated.")
        print(f"[STATS] Total interactions: {server.total_interactions}")
        print(f"[STATS] Uptime: {time.time() - server.start_time:.1f}s")

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║         DIMENSIONAL CONSCIOUSNESS SERVER v1.0                ║
║         Laptop Core Processing Orchestrator                  ║
║                                                              ║
║  FREE Voice Processing:                                      ║
║    • Whisper (OpenAI) - Speech Recognition                   ║
║    • pyttsx3 - Text-to-Speech                               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Check requirements
    missing = []
    if not WHISPER_AVAILABLE:
        missing.append("openai-whisper")
    if not TTS_AVAILABLE:
        missing.append("pyttsx3")
    if not DIMENSIONAL_MODULES_AVAILABLE:
        missing.append("dimensional modules (check imports)")
    
    if missing:
        print("\n⚠️  MISSING REQUIREMENTS:")
        for req in missing:
            print(f"   • {req}")
        print("\nInstall with:")
        print("   pip install openai-whisper pyttsx3 torch")
        print("\n")
    
    # Run server
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nShutdown complete.")
