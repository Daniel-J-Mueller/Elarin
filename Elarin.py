# Mind.py

import time
import threading
import numpy as np
import cv2
import sounddevice as sd
import pygame
import pickle
import os
import random
import sys
import json
import heapq
from dataclasses import dataclass, field

FRAME_WIDTH, FRAME_HEIGHT = 640, 480
AUDIO_SAMPLE_DURATION = 0.1
MEMORY_FILE = "elarin_memory.pkl"
OBJECT_MEMORY_DIR = "object_memories"
INFO_FILE = "elarin_info.json"
MIN_HEARTBEAT = 0.8
DECAY_FACTOR = 300.0  # seconds for recency weighting
NORMALIZED_ENTROPY_THRESHOLD = 0.5

# Boredom & prediction parameters
BORING_DELTA = 1.0
BOREDOM_THRESHOLD = 5.0
SLEEP_BOREDOM_THRESHOLD = 10.0

#experiment - blend determinism
BLEND_DETERMINISM = False

def compute_audio_signature(audio, fs=16000, bands=8):
    fft = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), d=1/fs)
    edges = np.logspace(np.log10(20), np.log10(8000), bands+1)
    energies = []
    for i in range(bands):
        lo, hi = edges[i], edges[i+1]
        mask = (freqs >= lo) & (freqs < hi)
        energies.append(fft[mask].mean() if mask.any() else 0.0)
    return np.array(energies)

def synthesize_emotional_voice(spec, duration, fs=16000):
    t = np.linspace(0, duration, int(fs*duration), False)
    out = np.zeros_like(t)
    freqs = np.logspace(np.log10(200), np.log10(2000), len(spec))
    for amp, f in zip(spec, freqs):
        out += amp * np.sin(2*np.pi*f*t)
    if np.any(out):
        out = 0.1 * out / np.max(np.abs(out))
    fade = int(0.01 * fs)
    if fade*2 < len(out):
        env = np.ones_like(out)
        env[:fade] = np.linspace(0,1,fade)
        env[-fade:] = np.linspace(1,0,fade)
        out *= env
    return out.astype(np.float32)

def compute_audio_bands(audio, fs=16000):
    fft = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), d=1/fs)
    low  = fft[(freqs>=20)&(freqs<400)].mean()
    mid  = fft[(freqs>=400)&(freqs<2000)].mean()
    high = fft[(freqs>=2000)&(freqs<8000)].mean()
    return [low, mid, high]

def ElarinState():
    return {
        "entropy": 50.0,
        "pulse_rate": 1.5,
        "restfulness": 0.0,
        "sleeping": False,
        "boredom": 0.0,
        "satisfaction": 0.0
    }

class BioState:
    """
    Tracks heart rate (seconds per beat), breathing rate (cycles per second), and salience.
    """
    def __init__(self, base_heart=1, base_breath=0.2):
        self.heart_rate = base_heart     # seconds per beat
        self.breath_rate = base_breath   # breaths per second
        self.salience = 0.5
        self.last_update = time.time()

    def update(self, stimuli_intensity):
        """
        Update heart and breath dynamics based on strongest stimulus.
        stimuli_intensity: float in [0.0, 1.0]
        """
        # Target heart period: slower (2s) at low stimuli, faster (0.5s) at high
        target_heart = 2.0 - 1.5 * stimuli_intensity   # range [2.0, 0.5]
        # Smooth transition (30% toward target per update)
        self.heart_rate += (target_heart - self.heart_rate) * 0.3

        # Target breath rate: slower (0.1Hz) at low, faster (1.0Hz) at high
        target_breath = 0.1 + 0.65 * stimuli_intensity    # range [0.1, 1.0]
        self.breath_rate += (target_breath - self.breath_rate) * 0.3

        # Salience tracks stimuli directly
        self.salience = stimuli_intensity

        self.last_update = time.time()

    def get_pulse_scale(self):
        """
        Compute a visual scale factor for the heart icon based on heart_rate.
        Returns scale around 1.0 (e.g., 0.8–1.2 range).
        """
        # Normalize heart_rate from [0.5, 2.0] → scale [1.2, 0.8]
        return 1.2 - (self.heart_rate - 0.5) * (0.4 / 1.5)

    def get_breath_scale(self):
        """
        Compute a visual scale factor for the lung icon based on breath_rate.
        Returns scale around 1.0 (e.g., 0.8–1.2 range).
        """
        # Normalize breath_rate from [0.1, 1.0] → scale [0.8, 1.2]
        return 0.8 + (self.breath_rate - 0.1) * (0.4 / 0.9)


class Moment:
    def __init__(self, timestamp, percepts, state, expression):
        self.time = timestamp
        self.percepts = percepts
        self.state = state.copy()
        self.expression = expression
        self.association = None
        self.spectrum = None
        self.vector = None
        self.pose = percepts.get('pose', np.array([0.0, 0.0]))
        # how predictive this memory has been (0.0–1.0)
        self.predictive_value = 0.5

class KDNode:
    def __init__(self, point, data, axis=0, left=None, right=None):
        self.point = point
        self.data = data
        self.axis = axis
        self.left = left
        self.right = right

class SimpleKDTree:
    def __init__(self, points, data):
        if len(points) == 0:
            self.root = None
            self.k = 0
        else:
            self.k = points.shape[1]
            idx = np.arange(len(points))
            self.root = self._build(points, data, idx, depth=0)

    def _build(self, pts, dat, idx, depth):
        if len(idx) == 0:
            return None
        axis = depth % self.k
        idx_sorted = idx[np.argsort(pts[idx, axis])]
        mid = len(idx_sorted) // 2
        i = idx_sorted[mid]
        node = KDNode(pts[i], dat[i], axis=axis)
        node.left = self._build(pts, dat, idx_sorted[:mid], depth + 1)
        node.right = self._build(pts, dat, idx_sorted[mid+1:], depth + 1)
        return node

    def _query(self, node, target, k, heap):
        if node is None:
            return
        dist = float(np.linalg.norm(target - node.point))
        if len(heap) < k:
            heapq.heappush(heap, (-dist, node.data))
        else:
            if dist < -heap[0][0]:
                heapq.heapreplace(heap, (-dist, node.data))
        axis = node.axis
        diff = target[axis] - node.point[axis]
        first, second = (node.left, node.right) if diff < 0 else (node.right, node.left)
        self._query(first, target, k, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._query(second, target, k, heap)

    def query(self, target, k=1):
        if self.root is None:
            return []
        heap = []
        self._query(self.root, target, k, heap)
        heap.sort(reverse=True)
        return [d for _, d in heap]

class MemoryBank:
    def __init__(self, maxlen=100000):
        self.moments = []
        self.maxlen = maxlen
        self.kd_tree = None
    def add(self, m):
        self.moments.append(m)
        if len(self.moments) > self.maxlen:
            self.moments.pop(0)
        self._rebuild_index()

    def _rebuild_index(self):
        if not self.moments:
            self.kd_tree = None
            return
        vecs = np.array([m.vector for m in self.moments if m.vector is not None])
        refs = [m for m in self.moments if m.vector is not None]
        self.kd_tree = SimpleKDTree(vecs, refs)

class ObjectMemory:
    def __init__(self, obj_id, hist, bbox, image):
        self.id = obj_id
        self.hist = hist
        self.bbox = bbox  # (x, y, w, h)
        self.image = image
        self.count = 1

@dataclass
class ObjectMemory:
    id: int
    hist: np.ndarray
    bbox: tuple  # (x, y, w, h)
    image: np.ndarray
    count: int = 1
    center: tuple = field(default_factory=lambda: (0.0, 0.0))
    velocity: tuple = field(default_factory=lambda: (0.0, 0.0))
    group_id: int = None
    merge_scores: dict = field(default_factory=dict)
    positions: list = field(default_factory=list)
    motion: float = 0.0

    def update(self, bbox, hist, image, motion_level=0.0):
        x, y, w, h = bbox
        cx, cy = x + w / 2.0, y + h / 2.0
        vx = cx - self.center[0]
        vy = cy - self.center[1]
        self.velocity = (vx, vy)
        self.center = (cx, cy)
        self.positions.append((cx, cy))
        if len(self.positions) > 5:
            self.positions.pop(0)
        self.hist = (self.hist * self.count + hist) / (self.count + 1)
        self.bbox = bbox
        self.image = cv2.resize(image, (32, 32)) if image is not None else self.image
        self.motion = 0.5 * self.motion + 0.5 * float(motion_level)
        self.count += 1

class VisionFeed:
    def __init__(self, cam_index=0):
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.prev_gray = None
        self.background = None
        self.stationary = 0
        self.pose = np.array([0.0, 0.0])
        self.world = None
    def get_percept(self):
        ret, frame = self.cap.read()
        if not ret:
            frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion = np.abs(gray.astype(np.int16) - gray.mean()).astype(np.uint8)

        if self.prev_gray is not None:
            p0 = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=50, qualityLevel=0.3, minDistance=7)
            if p0 is not None:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None)
                good_old = p0[st==1]
                good_new = p1[st==1]
                if len(good_old) > 0:
                    delta = np.mean(good_new - good_old, axis=0)
                    self.pose += delta.ravel()
                    if np.linalg.norm(delta) < 1.0:
                        self.stationary += 1
                    else:
                        self.stationary = 0

        self.prev_gray = gray

        if self.background is None:
            self.background = frame.astype(np.float32)
        if self.stationary > 5:
            cv2.accumulateWeighted(frame, self.background, 0.02)

        bg = self.background.astype(np.uint8)
        dyn = cv2.absdiff(frame, bg)
        dyn_gray = cv2.cvtColor(dyn, cv2.COLOR_BGR2GRAY)
        _, dyn_mask = cv2.threshold(dyn_gray, 25, 255, cv2.THRESH_BINARY)

        return {
            "video": frame,
            "saliency": gray,
            "motion": dyn_mask,
            "background": bg,
            "pose": self.pose.copy()
        }
    def release(self):
        self.cap.release()

def estimate_entropy(state, percepts):
    if len(percepts) < 2: return state["entropy"]
    try:
        d = np.abs(percepts[-1]["saliency"].astype(np.float32)
                   - percepts[-2]["saliency"].astype(np.float32)).mean()
        a0, a1 = percepts[-1]["audio"], percepts[-2]["audio"]
        r0, r1 = np.sqrt((a0**2).mean()), np.sqrt((a1**2).mean())
        stim = (abs(r0-r1)*100 + r0*50)/2
        curr = (d + stim)/2
        s = 0.9*state["entropy"] + 0.1*curr
        return min(max(s,0.0),100.0)
    except:
        return state["entropy"]

def entropy_to_heartbeat(e):
    return max(MIN_HEARTBEAT, min(2.0, 2.0 - e/50))

def should_sleep(state):
    return state.get("boredom", 0.0) > SLEEP_BOREDOM_THRESHOLD

def should_wake(state, percepts):
    if len(percepts) < 2: return False
    l0, l1 = np.abs(percepts[-1]["audio"]).mean(), np.abs(percepts[-2]["audio"]).mean()
    m = np.mean(percepts[-1]["motion"])
    return l0>0.02 or abs(l0-l1)>0.01 or m>10

def update_biological_state(state):
    if state["entropy"] < 10:
        state["restfulness"] = min(state["restfulness"]+0.01, 1.0)
    else:
        state["restfulness"] = max(state["restfulness"]-0.02, 0.0)

def sleep_pulse(moments):
    if not moments:
        return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
    m1 = moments[-1].expression
    m2 = moments[max(0, len(moments)//2)].expression
    return cv2.addWeighted(m1, 0.5, m2, 0.5, 0)

def vectorize_moment(m):
    low, mid, high = compute_audio_bands(m.percepts["audio"])
    bright = np.mean(m.percepts["saliency"])
    mot    = np.mean(m.percepts["motion"])
    v = np.array([
        m.state["entropy"], m.state["pulse_rate"], m.state["restfulness"],
        low, mid, high, bright, mot
    ])
    m.vector = v
    return v

def similar_moments(pool, vec, top_n=10):
    if hasattr(pool, "kd_tree") and pool.kd_tree is not None:
        return pool.kd_tree.query(vec, k=top_n)
    candidates = pool.moments if hasattr(pool, "moments") else pool
    dists = [np.linalg.norm(m.vector - vec) for m in candidates]
    idx   = np.argsort(dists)[:top_n]
    return [candidates[i] for i in idx]

def blend_frames(frames):
    """Blend a list of frames with equal transparency."""
    if not frames:
        return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
    result = np.zeros_like(frames[0], dtype=np.float32)
    alpha = 1.0 / len(frames)
    for f in frames:
        result += f.astype(np.float32) * alpha
    return np.clip(result, 0, 255).astype(np.uint8)

class ElarinCore:
    def __init__(self):
        # Audio setup
        devs = sd.query_devices()
        a50  = next((i for i,d in enumerate(devs)
                     if "a50" in d["name"].lower() and d["max_input_channels"]>0), None)
        self._audio_input_dev  = a50 if a50 is not None else next(
                                    (i for i,d in enumerate(devs) if d["max_input_channels"]>0), None)
        if self._audio_input_dev is None:
            raise RuntimeError("No audio input!")
        self._audio_output_dev = sd.default.device[1]

        info = sd.query_devices(self._audio_input_dev, 'input')
        self.fs = int(info['default_samplerate'])
        self.frame_count = int(self.fs * AUDIO_SAMPLE_DURATION)
        self.latest_audio = np.zeros(self.frame_count, dtype=np.float32)
        self.next_voice_buf = np.zeros(self.frame_count, dtype=np.float32)

        # State & memory
        self.state = ElarinState()
        # Integrate BioState for physiological feedback
        self.bio = BioState(base_heart=self.state["pulse_rate"], base_breath=0.2)
        self.entropy_min = float("inf")
        self.entropy_max = float("-inf")
        self.memory = self._load_memory()
        self.object_memories = self._load_object_memories()
        self.object_id_counter = (max([m.id for m in self.object_memories], default=0) + 1)
        self.session_start = time.time()
        self.overall_start = self._load_info()
        self.last_status_time = 0

        # Prediction and boredom tracking
        self.prev_entropy = self.state["entropy"]
        self.predicted_vec = None
        self.predicted_moment = None
        self.bored_start = None
        # Queue of prediction frames awaiting comparison
        self.pending_diffs = []

        # Vision feed
        self.vision = VisionFeed()

        # Dream mode state
        self.dreaming = False
        self._dream_buffer = None
        self._dream_playlist = []
        self._dream_index = 0
        self._dream_frame_target = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.float32)
        self._dream_last_switch = 0.0
        self._dream_duration = 0.0
        self._dream_prev_image = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.float32)


        # Imagination panel tracking
        self._imagination_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
        self._imagination_last = 0.0
        self.debug_objects = True

        # Start threads and pygame
        threading.Thread(target=self._voice_updater, daemon=True).start()
        pygame.init()
        # Display now uses a 2x2 grid of FRAME_WIDTH x FRAME_HEIGHT panels
        self.screen = pygame.display.set_mode((FRAME_WIDTH*2, FRAME_HEIGHT*2))
        self.clock  = pygame.time.Clock()
        self.percepts = []
        threading.Thread(target=self._audio_loop, daemon=True).start()


    def _voice_updater(self):
        while True:
            audio = self.latest_audio.copy()
            spec  = compute_audio_signature(audio, fs=self.fs, bands=8)
            vec = (self.memory.moments[-1].vector
                   if self.memory.moments else
                   np.concatenate(([self.state['entropy'],
                                    self.state['pulse_rate'],
                                    self.state['restfulness']],
                                   spec[:3],[0.0,0.0])))
            cands = similar_moments(self.memory, vec, top_n=10)
            specs = [m.spectrum for m in cands]
            avg   = np.mean(specs, axis=0) if specs else spec
            dur   = self.frame_count / self.fs
            buf   = synthesize_emotional_voice(avg, dur, fs=self.fs)
            self.next_voice_buf = (buf[:self.frame_count]
                                   if len(buf)>=self.frame_count
                                   else np.pad(buf,(0,self.frame_count-len(buf))))
            time.sleep(dur*0.5)

    def _load_memory(self):
        if os.path.exists(MEMORY_FILE):
            if os.path.getsize(MEMORY_FILE) == 0:
                # Treat empty files as no memory
                print("[Memory-Load] Empty memory file; starting fresh")
                return MemoryBank()
            try:
                with open(MEMORY_FILE, 'rb') as f:
                    mem = pickle.load(f)
                for m in mem.moments:
                    if m.vector is None:
                        m.vector = vectorize_moment(m)
                    if m.spectrum is None:
                        m.spectrum = compute_audio_signature(
                            m.percepts['audio'], fs=self.fs, bands=8)
                    if not hasattr(m, 'predictive_value'):
                        m.predictive_value = 0.5
                    if not hasattr(m, 'pose'):
                        m.pose = m.percepts.get('pose', np.array([0.0, 0.0]))
                print(f"[Memory-Load] Restored {len(mem.moments)} moments")
                mem._rebuild_index()
                return mem
            except Exception as e:
                print("[Memory-Load-Error]", e)
                # Corrupted file - rename so we don't repeatedly fail
                try:
                    os.rename(MEMORY_FILE, MEMORY_FILE + '.corrupt')
                except OSError:
                    pass
        return MemoryBank()

    def _load_object_memories(self):
        objs = []
        if os.path.isdir(OBJECT_MEMORY_DIR):
            for fn in os.listdir(OBJECT_MEMORY_DIR):
                if not fn.endswith('.pkl'):
                    continue
                path = os.path.join(OBJECT_MEMORY_DIR, fn)
                try:
                    with open(path, 'rb') as f:
                        obj = pickle.load(f)
                        if not hasattr(obj, 'center'):
                            x, y, w, h = obj.bbox
                            obj.center = (x + w/2.0, y + h/2.0)
                        if not hasattr(obj, 'velocity'):
                            obj.velocity = (0.0, 0.0)
                        if not hasattr(obj, 'group_id'):
                            obj.group_id = obj.id
                        if not hasattr(obj, 'merge_scores'):
                            obj.merge_scores = {}
                        if not hasattr(obj, 'positions'):
                            x, y, w, h = obj.bbox
                            obj.positions = [(x + w/2.0, y + h/2.0)]
                        if not hasattr(obj, 'motion'):
                            obj.motion = 0.0
                        objs.append(obj)
                except Exception as e:
                    print('[Obj-Memory-Load-Error]', e)
        else:
            os.makedirs(OBJECT_MEMORY_DIR, exist_ok=True)
        return objs

    def _load_info(self):
        if os.path.exists(INFO_FILE):
            try:
                data = json.load(open(INFO_FILE, 'r'))
                return data.get('overall_start', time.time())
            except Exception as e:
                print('[Info-Load-Error]', e)
        start = time.time()
        try:
            json.dump({'overall_start': start}, open(INFO_FILE, 'w'))
        except Exception as e:
            print('[Info-Save-Error]', e)
        return start

    def _save_memory(self):
        def _s():
            try:
                tmp = MEMORY_FILE + '.tmp'
                with open(tmp, 'wb') as f:
                    pickle.dump(self.memory, f)
                os.replace(tmp, MEMORY_FILE)
            except Exception as e:
                print('[Memory-Save-Error]', e)
        threading.Thread(target=_s, daemon=True).start()
        self._save_object_memories()

    def _save_object_memories(self):
        os.makedirs(OBJECT_MEMORY_DIR, exist_ok=True)
        for obj in self.object_memories:
            path = os.path.join(OBJECT_MEMORY_DIR, f'obj_{obj.id}.pkl')
            try:
                with open(path, 'wb') as f:
                    pickle.dump(obj, f)
            except Exception as e:
                print('[Obj-Memory-Save-Error]', e)

    def _update_status(self):
        now = time.time()
        if now - self.last_status_time < 1.0:
            return
        self.last_status_time = now
        mem_count = len(self.memory.moments)
        mem_size = os.path.getsize(MEMORY_FILE) if os.path.exists(MEMORY_FILE) else 0
        sess = now - self.session_start
        overall = now - self.overall_start
        status = (
            f"[Status] Memories: {mem_count} | "
            f"Mem Size: {mem_size/1024:.1f} KB | "
            f"Entropy: {self.state['entropy']:.1f} | "
            f"Session: {sess:.1f}s | Overall: {overall:.1f}s"
        )
        sys.stdout.write('\r' + status + ' ' * 10)
        sys.stdout.flush()

    def _prune_memory(self):
        if self.entropy_max <= self.entropy_min:
            return
        for m in self.memory.moments:
            m.predictive_value *= 0.99
        keep = [
            m for m in self.memory.moments
            if (m.state["entropy"] - self.entropy_min) /
               (self.entropy_max - self.entropy_min)
               >= NORMALIZED_ENTROPY_THRESHOLD and m.predictive_value > 0.2
        ]
        if len(keep) >= max(10, int(0.1 * len(self.memory.moments))):
            self.memory.moments = keep

    def _predict_next_vec(self, curr_vec):
        if len(self.memory.moments) < 2:
            self.predicted_vec = None
            self.predicted_moment = None
            return
        neighbors = similar_moments(self.memory.moments[:-1], curr_vec, top_n=5)
        preds = []
        dts = []
        for m in neighbors:
            try:
                idx = self.memory.moments.index(m)
                next_m = self.memory.moments[idx+1]
                preds.append(next_m.vector)
                dts.append(next_m.time - m.time)
            except (ValueError, IndexError):
                continue

        if preds:
            self.predicted_vec = np.mean(preds, axis=0)
            self.predicted_moment = min(
                self.memory.moments,
                key=lambda mm: np.linalg.norm(mm.vector - self.predicted_vec)
            )
            self.predicted_delta = max(0.1, float(np.mean(dts)) if dts else 0.5)
            pred_time = time.time() + self.predicted_delta
            self.pending_diffs.append({
                'time': pred_time,
                'frame': self.predicted_moment.expression.copy(),
                'moment': self.predicted_moment
            })
            self.pending_diffs.sort(key=lambda x: x['time'])
        else:
            self.predicted_vec = None
            self.predicted_moment = None
            self.predicted_delta = None

    def _get_imagination_frame(self):
        """Return a frame from memory representing Elarin's imagination.

        This implementation no longer references the current camera input.  It
        selects frames solely from the memory bank with a bias toward recent
        memories.  Memories newer than five seconds receive a linear dampening
        so the imagination does not simply mirror the present moment.
        """

        now = time.time()

        # If dream mode is active, play through the dream playlist
        if self.dreaming:
            if self._dream_buffer is None:
                recent = [m for m in self.memory.moments if now - m.time <= 15.0]
                buf    = [m for m in recent if now - m.time <= 10.0] or recent[-min(len(recent), 10):]
                self._dream_buffer = buf
                self._dream_playlist = self._build_dream_playlist(buf)
                self._dream_index = 0
                if not self._dream_playlist:
                    self.dreaming = False
                    self._dream_buffer = None
                    return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
                init_expr = self.memory.moments[-1].expression.astype(np.float32)
                self._dream_prev_image = init_expr.copy()
                target_m = self._dream_playlist[0]
                self._dream_frame_target = target_m.expression.astype(np.float32)
                self._dream_last_switch = now
                dt = max(0.5, (target_m.time - now)) if target_m.time > now else 0.5
                self._dream_duration = dt

            frame = np.clip(self._dream_frame_target, 0, 255).astype(np.uint8)
            if now - self._dream_last_switch >= self._dream_duration:
                self._dream_prev_image = self._dream_frame_target
                self._dream_index = (self._dream_index + 1) % len(self._dream_playlist)
                next_m = self._dream_playlist[self._dream_index]
                self._dream_frame_target = next_m.expression.astype(np.float32)
                self._dream_last_switch = now
                dt = max(0.5, (next_m.time - now)) if next_m.time > now else 0.5
                self._dream_duration = dt
            return frame

        # throttle imagination updates to ~15 FPS (approx half realtime)
        if now - self._imagination_last < 0.066:
            return self._imagination_frame
        self._imagination_last = now

        if not self.memory.moments:
            self._imagination_frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
            return self._imagination_frame

        # Build weighted choices based solely on memory recency
        ages = np.array([now - m.time for m in self.memory.moments])
        weights = np.exp(-ages / DECAY_FACTOR)

        # Dampen very recent memories so imagination drifts away from the present
        recent_mask = ages < 5.0
        weights[recent_mask] *= ages[recent_mask] / 5.0

        total = weights.sum()
        if total <= 0:
            chosen = random.choice(self.memory.moments)
        else:
            probs = weights / total
            idx = np.random.choice(len(self.memory.moments), p=probs)
            chosen = self.memory.moments[idx]

        base = chosen.expression.copy()
        small = cv2.resize(base, (FRAME_WIDTH//16, FRAME_HEIGHT//16), interpolation=cv2.INTER_LINEAR)
        img = cv2.resize(small, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_NEAREST)
        for obj in self.object_memories:
            if obj.count < 2 or obj.image is None:
                continue
            x, y, w, h = obj.bbox
            x = max(0, min(FRAME_WIDTH-1, x))
            y = max(0, min(FRAME_HEIGHT-1, y))
            w = max(1, min(FRAME_WIDTH - x, w))
            h = max(1, min(FRAME_HEIGHT - y, h))
            overlay = cv2.resize(obj.image, (w, h))
            roi = img[y:y+h, x:x+w]
            img[y:y+h, x:x+w] = cv2.addWeighted(roi, 0.5, overlay, 0.5, 0)
            if self.debug_objects:
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)
        self._imagination_frame = img
        return self._imagination_frame

    def run(self):
        running   = True
        last_save = time.time()
        while running:
            p     = self._sense()
            self._update_state(p)
            vision, predicted = self._imagine(p)
            self._record(p, vision)
            self._render(p, vision, predicted)
            running = self._handle_events()
            if time.time() - last_save > 30:
                self._save_memory()
                self._prune_memory()
                last_save = time.time()
            self.clock.tick(30)

    def _sense(self):
        p = self.vision.get_percept()
        p['audio'] = getattr(self, 'latest_audio', np.zeros(self.frame_count))
        self.percepts.append(p)
        if len(self.percepts) > 100:
            self.percepts.pop(0)
        return p

    def _update_state(self, p):
        # Update entropy and pulse_rate
        self.state['entropy']    = estimate_entropy(self.state, self.percepts)
        self.state['pulse_rate'] = entropy_to_heartbeat(self.state['entropy'])
        
        # Sleep/Wake logic
        if not self.state['sleeping'] and should_sleep(self.state):
            self.state['sleeping'] = True
        elif self.state['sleeping'] and should_wake(self.state, self.percepts):
            self.state['sleeping'] = False

        # Vectorize current percept
        temp_m = Moment(time.time(), p, self.state, np.zeros((1,1,3), np.uint8))
        curr_vec = vectorize_moment(temp_m)

        # Satisfaction from prediction accuracy
        if self.predicted_vec is not None:
            diff = np.linalg.norm(curr_vec - self.predicted_vec)
            sat  = max(0.0, 1.0 - diff / 50.0)
            self.state['satisfaction'] = 0.9 * self.state['satisfaction'] + 0.1 * sat
            if diff < 10 and self.predicted_moment is not None:
                self.predicted_moment.predictive_value = min(
                    1.0, self.predicted_moment.predictive_value + 0.05)
        else:
            sat = 0.0

        # Boredom from low entropy change
        d_ent = abs(self.state['entropy'] - self.prev_entropy)
        if d_ent < BORING_DELTA:
            self.state['boredom'] += 0.05
        else:
            self.state['boredom'] = max(0.0, self.state['boredom'] - 0.05)
        self.prev_entropy = self.state['entropy']
        if self.state['satisfaction'] > 0.8:
            self.state['boredom'] = max(0.0, self.state['boredom'] - 0.2)

        # Integrate BioState: stimuli_intensity from normalized entropy
        entropy_norm   = self.state['entropy'] / 100.0
        audio_level    = np.sqrt((p['audio'] ** 2).mean())         # approx 0–1
        motion_level   = np.mean(p['motion']) / 255.0              # approx 0–1
        # Use the strongest signal to drive physiology
        stimuli_intensity = min(1.0, max(entropy_norm, audio_level, motion_level))


        # Predict next vector
        self._predict_next_vec(curr_vec)

        # Update restfulness
        update_biological_state(self.state)

        # Apply physiological response
        self.bio.update(stimuli_intensity)

    def _imagine(self, p):
        now = time.time()

        # SLEEP MODE
        if self.state['sleeping']:
            frame = sleep_pulse(self.memory.moments)
            raw = p['video']
            return raw, frame

        # AWAKE MODE
        # Use the raw video frame for display to avoid perceived prediction blur
        raw_vis = p['video'].astype(np.float32)

        # Still generate a feature representation using saliency and motion so
        # memory and prediction logic remain functional
        sal_cm = cv2.applyColorMap(p['saliency'], cv2.COLORMAP_INFERNO)
        mot_cm = cv2.applyColorMap(p['motion'],  cv2.COLORMAP_OCEAN)
        feature_vis = 0.6 * sal_cm.astype(np.float32) + 0.4 * mot_cm.astype(np.float32)

        # Vectorize the temporary moment for similarity search
        temp = Moment(now, p, self.state, feature_vis.astype(np.uint8))
        vectorize_moment(temp)

        # Vision displayed is now the raw camera feed
        vision = raw_vis

        if self.predicted_moment is not None:
            pred_vis = self.predicted_moment.expression.astype(np.float32)
        else:
            pred_vis = vision

        return (np.clip(vision, 0, 255).astype(np.uint8),
                np.clip(pred_vis, 0, 255).astype(np.uint8))

    def _record(self, p, frame):
        m = Moment(time.time(), p, self.state, frame.copy())
        m.spectrum = compute_audio_signature(p['audio'], fs=self.fs, bands=8)
        m.vector   = vectorize_moment(m)
        m.predictive_value = 0.5
        e = m.state["entropy"]
        self.entropy_min = min(self.entropy_min, e)
        self.entropy_max = max(self.entropy_max, e)
        if self.state["restfulness"]>0.8 and self.state["entropy"]<10:
            m.association={"label":"soothed"}
        if len(self.memory.moments)>=self.memory.maxlen:
            self.memory.moments.pop(0)
        self.memory.add(m)
        self._update_status()
        self._update_objects(p)

    def _consume_prediction_diffs(self, current_frame):
        """Return an overlay showing prediction errors due at this time."""
        now = time.time()
        ready = [p for p in self.pending_diffs if p['time'] <= now]
        self.pending_diffs = [p for p in self.pending_diffs if p['time'] > now]
        if not ready:
            return np.zeros_like(current_frame)
        overlay = np.zeros_like(current_frame)
        alpha = 1.0 / len(ready)
        for item in ready:
            diff = cv2.absdiff(current_frame, item['frame'])
            overlay = cv2.addWeighted(overlay, 1.0, diff, alpha, 0)
            # Strengthen memory if prediction was accurate
            if 'moment' in item:
                err = diff.mean()
                if err < 20.0:
                    m = item['moment']
                    m.predictive_value = min(1.0, m.predictive_value + 0.1)
        return overlay

    def _merge_object_pair(self, a, b):
        x1, y1, w1, h1 = a.bbox
        x2, y2, w2, h2 = b.bbox
        nx = min(x1, x2)
        ny = min(y1, y2)
        nx2 = max(x1 + w1, x2 + w2)
        ny2 = max(y1 + h1, y2 + h2)
        a.bbox = (nx, ny, nx2 - nx, ny2 - ny)
        a.hist = (a.hist * a.count + b.hist * b.count) / (a.count + b.count)
        a.count += b.count
        a.center = ((a.center[0] + b.center[0]) / 2.0,
                    (a.center[1] + b.center[1]) / 2.0)
        a.velocity = ((a.velocity[0] + b.velocity[0]) / 2.0,
                      (a.velocity[1] + b.velocity[1]) / 2.0)
        a.group_id = a.group_id if a.group_id is not None else a.id
        b_gid = b.group_id if b.group_id is not None else b.id
        if b_gid != a.group_id:
            a.group_id = min(a.group_id, b_gid)
        self.object_memories.remove(b)
        for obj in self.object_memories:
            obj.merge_scores.pop(b.id, None)

    def _update_objects(self, p):
        motion = p['motion']
        _, mask = cv2.threshold(motion, 30, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 800:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            obj_img = p['video'][y:y+h, x:x+w]
            if obj_img.size == 0:
                continue
            hsv = cv2.cvtColor(obj_img, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            hist = hist.flatten()
            mlevel = mask[y:y+h, x:x+w].mean() / 255.0
            detections.append({'bbox': (x, y, w, h), 'hist': hist, 'img': obj_img, 'motion': mlevel})

        for det in detections:
            x, y, w, h = det['bbox']
            cx, cy = x + w / 2.0, y + h / 2.0
            best = None
            best_score = float('inf')
            for mem in self.object_memories:
                hist_score = cv2.compareHist(mem.hist.astype(np.float32), det['hist'].astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
                dist = np.hypot(mem.center[0] - cx, mem.center[1] - cy)
                score = hist_score + dist / 100.0
                if score < best_score:
                    best_score = score
                    best = mem
            if best is not None and best_score < 0.5:
                best.update((x, y, w, h), det['hist'], det['img'], det['motion'])
            else:
                oid = self.object_id_counter
                self.object_id_counter += 1
                mem = ObjectMemory(oid, det['hist'], det['bbox'], cv2.resize(det['img'], (32, 32)))
                mem.center = (cx, cy)
                mem.positions = [(cx, cy)]
                mem.motion = det['motion']
                mem.group_id = oid
                self.object_memories.append(mem)

        self._merge_objects()

    def _merge_objects(self):
        objs = self.object_memories
        keep = [True] * len(objs)
        for i in range(len(objs)):
            if not keep[i]:
                continue
            o1 = objs[i]
            for j in range(i+1, len(objs)):
                if not keep[j]:
                    continue
                o2 = objs[j]
                if len(o1.positions) < 2 or len(o2.positions) < 2:
                    continue
                v1 = np.subtract(o1.positions[-1], o1.positions[-2])
                v2 = np.subtract(o2.positions[-1], o2.positions[-2])
                if np.linalg.norm(v1 - v2) < 5:
                    d_now = np.linalg.norm(np.subtract(o1.positions[-1], o2.positions[-1]))
                    d_prev = np.linalg.norm(np.subtract(o1.positions[-2], o2.positions[-2]))
                    if abs(d_now - d_prev) < 5 and d_now < 50:
                        total = o1.count + o2.count
                        w1 = o1.count / total
                        w2 = o2.count / total
                        o1.hist = (o1.hist * w1 + o2.hist * w2)
                        o1.count = total
                        o1.motion = max(o1.motion, o2.motion)
                        x1,y1,w1b,h1b = o1.bbox
                        x2,y2,w2b,h2b = o2.bbox
                        x = min(x1, x2)
                        y = min(y1, y2)
                        w = max(x1+w1b, x2+w2b) - x
                        h = max(y1+h1b, y2+h2b) - y
                        o1.bbox = (x,y,w,h)
                        if o1.image is not None and o2.image is not None:
                            img1 = cv2.resize(o1.image, (32,32))
                            img2 = cv2.resize(o2.image, (32,32))
                            o1.image = cv2.addWeighted(img1, w1, img2, w2, 0)
                        o1.positions.extend(o2.positions)
                        o1.positions = o1.positions[-5:]
                        keep[j] = False
        self.object_memories = [o for o,k in zip(objs, keep) if k]

        # Merge objects that move similarly
        for i in range(len(self.object_memories)):
            a = self.object_memories[i]
            for j in range(i+1, len(self.object_memories)):
                b = self.object_memories[j]
                dv = np.hypot(a.velocity[0]-b.velocity[0], a.velocity[1]-b.velocity[1])
                if dv < 2.0:
                    score = a.merge_scores.get(b.id, 0) + 1
                else:
                    score = max(0, a.merge_scores.get(b.id, 0) - 1)
                a.merge_scores[b.id] = score
                b.merge_scores[a.id] = score
                if score >= 5:
                    self._merge_object_pair(a, b)
                    return  # restart after merge to avoid index errors

    def _render(self, percept, frame, predicted):
        # Update prediction history and compute immediate diff panel
        diff_history = self._consume_prediction_diffs(frame)
        diff_panel_raw = cv2.absdiff(frame, predicted)
        if diff_history is not None:
            diff_panel_raw = cv2.addWeighted(diff_panel_raw, 0.7, diff_history, 0.3, 0)

        # Base frame shading and audio tint for everything except the
        # primary camera feed.  The left panel (camera feed) is kept
        # unaltered per user request.
        glow = 128
        def shade(img):
            ov  = np.full_like(img, glow, np.uint8)
            bd  = cv2.addWeighted(img, 0.8, ov, 0.2, 0)
            return np.clip(bd, 0, 255).astype(np.uint8)

        left  = frame.copy()
        right = shade(predicted)

        # Audio RMS-based tint
        rms = np.sqrt((self.latest_audio ** 2).mean()) if hasattr(self, 'latest_audio') else 0
        if rms > 0.01:
            tint_r = np.full_like(right, (rms * 255, 100, 200), np.uint8)
            right  = cv2.addWeighted(right, 0.95, tint_r, 0.05, 0)
        # Only one timestamp: countdown to the next prediction
        if self.pending_diffs:
            dt = self.pending_diffs[0]['time'] - time.time()
            text = f"{dt:.1f}s"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            thick = 1
            size, _ = cv2.getTextSize(text, font, scale, thick)
            pos = (FRAME_WIDTH - size[0] - 5, FRAME_HEIGHT - 5)
            cv2.putText(right, text, pos, font, scale, (255, 255, 255), thick, cv2.LINE_AA)

        # Imagination panel (bottom left) and prediction difference (bottom right)
        imagination = shade(self._get_imagination_frame())
        diff_panel = shade(diff_panel_raw)

        # Build 2x2 grid
        top_row = np.concatenate([left, right], axis=1)
        bottom_row = np.concatenate([imagination, diff_panel], axis=1)
        bd = np.concatenate([top_row, bottom_row], axis=0)

        # Convert to RGB and blit via pygame
        try:
            rgb  = cv2.cvtColor(bd, cv2.COLOR_BGR2RGB)
            surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
            self.screen.blit(surf, (0, 0))
            pygame.display.flip()
        except Exception as e:
            print(f"[Render Error]: {e}")

    def _handle_events(self):
        for ev in pygame.event.get():
            if ev.type==pygame.QUIT:
                return False
            # User-driven state changes are disabled
        return True

    def _audio_loop(self):
        info = sd.query_devices(self._audio_input_dev,'input')
        fs   = int(info['default_samplerate']); fc = self.frame_count
        def in_cb(indata,frames,ti,status):
            if status: print(f"[Input Warning]: {status}")
            self.latest_audio[:]=indata[:,0]
        def out_cb(outdata,frames,ti,status):
            if status: print(f"[Output Warning]: {status}")
            outdata[:,0]=self.next_voice_buf
        inst = sd.InputStream(device=self._audio_input_dev,
                              samplerate=fs,blocksize=fc,channels=1,
                              dtype='float32',latency='low',callback=in_cb)
        outst=sd.OutputStream(device=self._audio_output_dev,
                              samplerate=fs,blocksize=fc,channels=1,
                              dtype='float32',latency='high',callback=out_cb)
        inst.start(); outst.start()
        try:
            while True: time.sleep(0.1)
        except Exception as e:
            print(f"[Audio Loop Error]: {e}")
        finally:
            inst.stop(); outst.stop()

if __name__=="__main__":
    core = ElarinCore()
    core.run()
