#!/usr/bin/env python3
"""
Mobile Sensory Orchestrator (Android Termux)
============================================
Captures phone sensors (camera, mic, accelerometer) and streams
to laptop consciousness core via WebSocket.

REQUIREMENTS (Install in Termux):
pip install websockets asyncio Pillow
pkg install python ffmpeg
"""

import asyncio
import websockets
import json
import base64
import time
import os
import sys
from typing import Optional, Dict, Any
from io import BytesIO

# Try to import Android-specific modules
try:
    from android.permissions import request_permissions, Permission
    from android.storage import app_storage_path
    ANDROID_AVAILABLE = True
except ImportError:
    print("[WARNING] Android modules not found. Running in mock mode.")
    ANDROID_AVAILABLE = False

# ============================================================================
# 1. SENSORY CAPTURE MODULES
# ============================================================================

class VoiceCapture:
    """Captures audio from microphone using ffmpeg"""
    
    def __init__(self):
        self.recording = False
        self.sample_rate = 16000  # Whisper prefers 16kHz
        
    def record_audio(self, duration: float = 3.0) -> Optional[bytes]:
        """
        Records audio for specified duration.
        Returns raw audio bytes in WAV format.
        """
        try:
            output_file = "/sdcard/temp_audio.wav"
            
            # Use ffmpeg to record from microphone
            cmd = f"ffmpeg -f android_camera -i 0:a -t {duration} -ar {self.sample_rate} -y {output_file} -loglevel quiet"
            
            # Fallback: Use termux-microphone-record if ffmpeg fails
            if os.system(cmd) != 0:
                print("[VOICE] Falling back to termux-microphone-record...")
                cmd = f"termux-microphone-record -d -l 1 -f {output_file}"
                os.system(cmd)
                time.sleep(duration)
                os.system("termux-microphone-record -q")
            
            # Read recorded file
            if os.path.exists(output_file):
                with open(output_file, 'rb') as f:
                    audio_data = f.read()
                os.remove(output_file)
                return audio_data
            else:
                print("[VOICE] Recording failed - file not created")
                return None
                
        except Exception as e:
            print(f"[VOICE] Error recording audio: {e}")
            return None

class VisionCapture:
    """Captures camera frames"""
    
    def __init__(self):
        self.enabled = False
        
    def capture_frame(self) -> Optional[bytes]:
        """
        Captures a single frame from camera.
        Returns JPEG bytes.
        """
        try:
            output_file = "/sdcard/temp_frame.jpg"
            
            # Use termux-camera-photo
            cmd = f"termux-camera-photo {output_file}"
            result = os.system(cmd)
            
            if result == 0 and os.path.exists(output_file):
                with open(output_file, 'rb') as f:
                    frame_data = f.read()
                os.remove(output_file)
                return frame_data
            else:
                print("[VISION] Camera capture failed")
                return None
                
        except Exception as e:
            print(f"[VISION] Error capturing frame: {e}")
            return None

class BodySensors:
    """Reads phone sensors (accelerometer, etc.)"""
    
    def __init__(self):
        self.last_accel = {"x": 0, "y": 0, "z": 0}
        
    def read_accelerometer(self) -> Dict[str, float]:
        """
        Reads accelerometer data.
        Returns dict with x, y, z values.
        """
        try:
            # Use termux-sensor to read accelerometer
            import subprocess
            result = subprocess.run(
                ["termux-sensor", "-s", "accelerometer", "-n", "1"],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                if "accelerometer" in data:
                    values = data["accelerometer"]["values"]
                    self.last_accel = {
                        "x": values[0],
                        "y": values[1],
                        "z": values[2]
                    }
            
            return self.last_accel
            
        except Exception as e:
            print(f"[BODY] Error reading accelerometer: {e}")
            return self.last_accel

# ============================================================================
# 2. AUDIO PLAYBACK
# ============================================================================

class AudioPlayer:
    """Plays audio responses from consciousness core"""
    
    def play_audio(self, audio_bytes: bytes):
        """
        Plays audio through phone speaker.
        """
        try:
            output_file = "/sdcard/response_audio.wav"
            
            # Write audio to temp file
            with open(output_file, 'wb') as f:
                f.write(audio_bytes)
            
            # Play using termux-media-player
            os.system(f"termux-media-player play {output_file}")
            
            # Wait for playback to finish (estimate based on file size)
            # Rough estimate: 16kHz mono = 32KB/sec
            duration = len(audio_bytes) / 32000
            time.sleep(duration + 0.5)
            
            # Cleanup
            os.remove(output_file)
            
        except Exception as e:
            print(f"[AUDIO] Error playing audio: {e}")

# ============================================================================
# 3. MAIN ORCHESTRATOR
# ============================================================================

class MobileSensoryOrchestrator:
    """
    Main coordinator for all phone sensors and consciousness connection.
    """
    
    def __init__(self, server_ip: str, server_port: int = 8765):
        self.server_uri = f"ws://{server_ip}:{server_port}"
        
        # Sensory modules
        self.voice = VoiceCapture()
        self.vision = VisionCapture()
        self.body = BodySensors()
        self.audio_player = AudioPlayer()
        
        # State
        self.connected = False
        self.voice_active = True
        self.vision_active = False  # Start with vision off to save bandwidth
        self.body_active = True
        
        # Stats
        self.messages_sent = 0
        self.messages_received = 0
        
    async def connect_to_consciousness(self):
        """
        Main connection loop - maintains WebSocket to consciousness core.
        """
        print(f"[INIT] Mobile Sensory Orchestrator Starting...")
        print(f"[INIT] Connecting to consciousness core: {self.server_uri}")
        
        while True:
            try:
                async with websockets.connect(self.server_uri) as websocket:
                    self.connected = True
                    print("[CONNECTED] ✓ Link established with dimensional core")
                    
                    # Start sensory streaming
                    await self._sensory_loop(websocket)
                    
            except Exception as e:
                self.connected = False
                print(f"[ERROR] Connection lost: {e}")
                print("[RETRY] Reconnecting in 5 seconds...")
                await asyncio.sleep(5)
    
    async def _sensory_loop(self, websocket):
        """
        Main sensory streaming loop.
        """
        print("\n" + "="*50)
        print("SENSORY INTERFACE ACTIVE")
        print("="*50)
        print("Commands:")
        print("  [SPACE] - Start voice input")
        print("  'v' - Toggle vision")
        print("  'q' - Quit")
        print("="*50 + "\n")
        
        last_body_update = time.time()
        
        while True:
            try:
                # --- VOICE INPUT (Primary interaction) ---
                if self.voice_active:
                    print("\n🎤 [Listening... Press ENTER when done speaking]")
                    input()  # Wait for user to indicate they're done speaking
                    
                    print("🎤 [Recording...]")
                    audio_data = self.voice.record_audio(duration=3.0)
                    
                    if audio_data:
                        print(f"🎤 [Sending {len(audio_data)} bytes to consciousness core]")
                        
                        # Send voice data
                        message = {
                            "domain": "voice",
                            "type": "audio",
                            "data": base64.b64encode(audio_data).decode('utf-8'),
                            "timestamp": time.time()
                        }
                        
                        await websocket.send(json.dumps(message))
                        self.messages_sent += 1
                        
                        # Wait for response
                        print("💭 [Processing through dimensional consciousness...]")
                        response = await websocket.recv()
                        self.messages_received += 1
                        
                        response_data = json.loads(response)
                        
                        if response_data.get("type") == "voice_response":
                            print(f"🔊 [Response received: {len(response_data.get('text', ''))} chars]")
                            print(f"\n💬 AI: {response_data.get('text', '')}\n")
                            
                            # Play audio if available
                            if "audio" in response_data:
                                audio_bytes = base64.b64decode(response_data["audio"])
                                self.audio_player.play_audio(audio_bytes)
                        
                        elif response_data.get("type") == "status":
                            print(f"📊 Status: {response_data.get('message', '')}")
                
                # --- VISION INPUT (Periodic snapshots) ---
                if self.vision_active:
                    print("📷 [Capturing visual frame...]")
                    frame_data = self.vision.capture_frame()
                    
                    if frame_data:
                        message = {
                            "domain": "vision",
                            "type": "frame",
                            "data": base64.b64encode(frame_data).decode('utf-8'),
                            "timestamp": time.time()
                        }
                        await websocket.send(json.dumps(message))
                        self.messages_sent += 1
                
                # --- BODY SENSORS (Periodic updates) ---
                if self.body_active and (time.time() - last_body_update) > 2.0:
                    accel = self.body.read_accelerometer()
                    
                    # Only send if significant movement
                    magnitude = (accel['x']**2 + accel['y']**2 + accel['z']**2)**0.5
                    if magnitude > 12.0:  # Threshold for "movement detected"
                        message = {
                            "domain": "body",
                            "type": "motion",
                            "data": {
                                "accelerometer": accel,
                                "magnitude": magnitude
                            },
                            "timestamp": time.time()
                        }
                        await websocket.send(json.dumps(message))
                        self.messages_sent += 1
                        print(f"🏃 [Movement detected: {magnitude:.1f}]")
                    
                    last_body_update = time.time()
                
                # Small delay to prevent overwhelming the connection
                await asyncio.sleep(0.1)
                
            except KeyboardInterrupt:
                print("\n[SHUTDOWN] Disconnecting from consciousness core...")
                break
            except Exception as e:
                print(f"[ERROR] Sensory loop error: {e}")
                raise

# ============================================================================
# 4. MAIN ENTRY POINT
# ============================================================================

def request_android_permissions():
    """Request necessary Android permissions"""
    if ANDROID_AVAILABLE:
        print("[INIT] Requesting Android permissions...")
        permissions = [
            Permission.RECORD_AUDIO,
            Permission.CAMERA,
            Permission.WRITE_EXTERNAL_STORAGE,
            Permission.READ_EXTERNAL_STORAGE
        ]
        request_permissions(permissions)
        time.sleep(2)  # Give time for permission dialogs

async def main():
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python mobile_sensory_orchestrator.py <laptop_ip> [port]")
        print("Example: python mobile_sensory_orchestrator.py 192.168.1.100 8765")
        sys.exit(1)
    
    server_ip = sys.argv[1].rstrip(':') # Defensively remove trailing colon
    server_port = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
    
    # Request permissions if on Android
    if ANDROID_AVAILABLE:
        request_android_permissions()
    
    # Create and run orchestrator
    orchestrator = MobileSensoryOrchestrator(server_ip, server_port)
    
    try:
        await orchestrator.connect_to_consciousness()
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Mobile interface terminated.")
        print(f"[STATS] Messages sent: {orchestrator.messages_sent}")
        print(f"[STATS] Messages received: {orchestrator.messages_received}")

if __name__ == "__main__":
    # Set up event loop
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Run the orchestrator
    asyncio.run(main())
