#!/usr/bin/env python3
"""
BRIDGE V5 (EXPLICIT DATA + DNA SYSTEM SCAN)
===========================================
- Data Source: EXPLICIT absolute path (configured below).
- System Source: AUTO-DNA scan in current folder.
"""

import json
import time
import os
import uuid
import sys
import importlib.util

# === USER CONFIGURATION ===
# Your exact data location:
EXPLICIT_DATA_PATH = "/storage/emulated/0/true/Dip1/11_03_2025/chatgpt_backup/conversations.json"
# ==========================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"[SETUP] Bridge running in: {CURRENT_DIR}\n")

# --- 1. LOCATE DATA (EXPLICIT) ---
if os.path.exists(EXPLICIT_DATA_PATH):
    EXPORT_FILE = EXPLICIT_DATA_PATH
    print(f"[SETUP] DATA: Found at explicit path:\n --> {EXPORT_FILE}")
else:
    # Fallback to local folder if explicit path fails
    print(f"[SETUP] WARNING: Explicit path not found: {EXPLICIT_DATA_PATH}")
    print("[SETUP] Attempting fallback to local folder...")
    candidates = [f for f in os.listdir(CURRENT_DIR) if "conversations" in f and f.endswith(".json")]
    if candidates:
        EXPORT_FILE = os.path.join(CURRENT_DIR, candidates[0])
        print(f"[SETUP] DATA: Found local fallback: {candidates[0]}")
    else:
        print("\n[ERROR] Could not find 'conversations.json' anywhere.")
        sys.exit()

# --- 2. LOCATE SYSTEM (DNA SCAN) ---
SYSTEM_PATH = None
print("\n[SETUP] SYSTEM: Scanning local folder for DNA match ('continuous_save_thread')...")
for f in os.listdir(CURRENT_DIR):
    if f.endswith(".py") and f != os.path.basename(__file__):
        try:
            with open(os.path.join(CURRENT_DIR, f), 'r', encoding='utf-8') as pyfile:
                if "def continuous_save_thread" in pyfile.read():
                        SYSTEM_PATH = os.path.join(CURRENT_DIR, f)
                        print(f"[SETUP] DNA MATCH: {f}")
                        break
        except: continue

if not SYSTEM_PATH:
    print("\n[ERROR] Could not find the Memory System file in this folder.")
    sys.exit()

# --- 3. LINK & RUN ---
try:
    spec = importlib.util.spec_from_file_location("sys", SYSTEM_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    start_sys, stop_sys = mod.start_memory_system, mod.stop_memory_system
    print("[SETUP] LINK: Success.\n")
except Exception as e:
    print(f"[ERROR] Link failed: {e}"); sys.exit()

def run():
    gov, mem, t1, t2 = start_sys()
    print(">>> FEED STARTED <<<\n")
    try:
        with open(EXPORT_FILE, 'r', encoding='utf-8') as f: data = json.load(f)
    except Exception as e:
        print(f"[ERROR] Data read failed: {e}"); stop_sys(t1, t2); return

    total = len(data)
    for i, conv in enumerate(data):
        msgs = []
        for n in conv.get("mapping", {}).values():
            if n and n.get("message") and n["message"].get("content", {}).get("content_type") == "text":
                txt = "".join(str(p) for p in n["message"]["content"].get("parts", []) if p).strip()
                if txt:
                     msgs.append({
                        "concept": f"MSG_{uuid.uuid4().hex[:8]}", "text": txt,
                        "source_doc": "chatgpt_history", "role": n["message"]["author"]["role"],
                        "timestamp": n["message"].get("create_time", time.time())
                    })
        if msgs:
            print(f"[{i+1}/{total}] Absorbing: {conv.get('title')[:30]}...")
            gov.ingest_data({
                "root_concept": f"CONV_{conv.get('id', str(i))[:8]}",
                "json_data": {"title": conv.get("title"), "imported": time.time()},
                "conversation_messages": msgs
            })
            time.sleep(0.01)
            
    print("\n>>> COMPLETE <<<"); stop_sys(t1, t2)

if __name__ == "__main__":
    run()
