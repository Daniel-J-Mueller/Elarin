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

FRAME_WIDTH, FRAME_HEIGHT = 640, 480
AUDIO_SAMPLE_DURATION = 0.1
MEMORY_FILE = "elarin_memory.pkl"
INFO_FILE = "elarin_info.json"
MIN_HEARTBEAT = 0.8
DECAY_FACTOR = 300.0  # seconds for recency weighting
NORMALIZED_ENTROPY_THRESHOLD = 0.5
REPULSION_FACTOR = 0.5  # for dream-mode deviation

# Boredom & prediction parameters
BORING_DELTA = 1.0
BOREDOM_THRESHOLD = 5.0
SLEEP_BOREDOM_THRESHOLD = 10.0

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
        # how predictive this memory has been (0.0–1.0)
        self.predictive_value = 0.5

class MemoryBank:
    def __init__(self, maxlen=100000):
        self.moments = []
        self.maxlen = maxlen
    def add(self, m):
        self.moments.append(m)
        if len(self.moments) > self.maxlen:
            self.moments.pop(0)

class VisionFeed:
    def __init__(self, cam_index=0):
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    def get_percept(self):
        ret, frame = self.cap.read()
        if not ret:
            frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion = np.abs(gray.astype(np.int16) - gray.mean()).astype(np.uint8)
        return {"video": frame, "saliency": gray, "motion": motion}
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
    candidates = pool.moments if hasattr(pool, "moments") else pool
    dists = [np.linalg.norm(m.vector - vec) for m in candidates]
    idx   = np.argsort(dists)[:top_n]
    return [candidates[i] for i in idx]

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
        self.session_start = time.time()
        self.overall_start = self._load_info()
        self.last_status_time = 0

        # Prediction and boredom tracking
        self.prev_entropy = self.state["entropy"]
        self.predicted_vec = None
        self.predicted_moment = None
        self.predicted_delta = None
        self.bored_start = None

        # Vision feed
        self.vision = VisionFeed()

        # Dream variables
        self.dreaming = False
        self._dream_buffer        = None
        self._dream_playlist      = None
        self._dream_index         = 0
        self._dream_last_switch   = 0.0
        self._dream_duration      = 0.5

        # Start threads and pygame
        threading.Thread(target=self._voice_updater, daemon=True).start()
        pygame.init()
        # Display now uses a 2x2 grid of FRAME_WIDTH x FRAME_HEIGHT panels
        self.screen = pygame.display.set_mode((FRAME_WIDTH*2, FRAME_HEIGHT*2))
        self.clock  = pygame.time.Clock()
        self.percepts = []
        threading.Thread(target=self._audio_loop, daemon=True).start()

    def _build_dream_playlist(self, buffer):
        # iterative repulsion-based sampling
        playlist = []
        if not buffer:
            return playlist
        # seed: highest entropy
        seed = max(buffer, key=lambda m: m.state["entropy"])
        playlist.append(seed)
        attempts = 0
        while len(playlist) < min(len(buffer), 10) and attempts < 20:
            prev = playlist[-1]
            weights = []
            cands   = [m for m in buffer if m not in playlist]
            for m in cands:
                d   = np.linalg.norm(m.vector - prev.vector)
                rec = np.exp(-(time.time()-m.time)/DECAY_FACTOR)
                emo = 1.5 if m.association else 1.0
                base_w = (1.0/(1.0+d)) * rec * emo
                # repulsion from recent in playlist
                recent = playlist[-4:]
                if recent:
                    sims = [1.0/(1.0+np.linalg.norm(m.vector - h.vector)) for h in recent]
                    rep  = max(0.0, 1.0 - REPULSION_FACTOR * max(sims))
                else:
                    rep = 1.0
                weights.append(base_w * rep)
            total = sum(weights)
            if total <= 0:
                break
            probs = [w/total for w in weights]
            choice = np.random.choice(len(cands), p=probs)
            playlist.append(cands[choice])
            attempts += 1
        return playlist

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
            try:
                mem = pickle.load(open(MEMORY_FILE, 'rb'))
                for m in mem.moments:
                    if m.vector  is None:
                        m.vector  = vectorize_moment(m)
                    if m.spectrum is None:
                        m.spectrum = compute_audio_signature(m.percepts['audio'], fs=self.fs, bands=8)
                    if not hasattr(m, 'predictive_value'):
                        m.predictive_value = 0.5
                print(f"[Memory-Load] Restored {len(mem.moments)} moments")
                return mem
            except Exception as e:
                print("[Memory-Load-Error]", e)
        return MemoryBank()

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
                pickle.dump(self.memory, open(MEMORY_FILE, 'wb'))
            except:
                pass
        threading.Thread(target=_s, daemon=True).start()

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
            self.predicted_delta = None
            return
        neighbors = similar_moments(self.memory.moments[:-1], curr_vec, top_n=5)
        preds = []
        deltas = []
        for m in neighbors:
            try:
                idx = self.memory.moments.index(m)
                next_m = self.memory.moments[idx+1]
                preds.append(next_m.vector)
                deltas.append(next_m.time - m.time)
            except (ValueError, IndexError):
                continue
        if preds:
            self.predicted_vec = np.mean(preds, axis=0)
            self.predicted_moment = min(
                self.memory.moments,
                key=lambda mm: np.linalg.norm(mm.vector - self.predicted_vec)
            )
            self.predicted_delta = float(np.mean(deltas)) if deltas else None
        else:
            self.predicted_vec = None
            self.predicted_moment = None
            self.predicted_delta = None

    def _get_imagination_frame(self):
        """Return a frame from memory representing Elarin's imagination.

        Instead of always pulling the highest-entropy frames (which tended to
        lock onto the initial moments of a run), try to imagine what might come
        next based on similar past experiences.  We look for moments in memory
        that resemble the most recent perception and display the frame that
        followed each similar moment.  This links memories together and keeps
        the imagination panel fresh.
        """

        if len(self.memory.moments) < 2:
            return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)

        # Use the latest moment as the reference point
        last_m = self.memory.moments[-1]
        vec = last_m.vector
        neighbors = similar_moments(self.memory.moments[:-1], vec, top_n=5)

        # Gather the moments that occurred right after each neighbor
        next_frames = []
        for n in neighbors:
            try:
                idx = self.memory.moments.index(n)
                if idx + 1 < len(self.memory.moments):
                    next_frames.append(self.memory.moments[idx + 1])
            except ValueError:
                continue

        if next_frames:
            chosen = random.choice(next_frames)
            return chosen.expression.copy()

        # Fallback to high-entropy sampling if no neighbors found
        threshold = 50.0
        interesting = [m for m in self.memory.moments
                       if m.state.get("entropy", 0.0) > threshold]
        candidates = interesting if interesting else self.memory.moments
        chosen = random.choice(candidates)
        return chosen.expression.copy()

    def run(self):
        running   = True
        last_save = time.time()
        while running:
            p     = self._sense()
            self._update_state(p)
            vision, predicted = self._imagine(p)
            self._record(p, vision)
            self._render(vision, predicted)
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
            self.state['boredom'] += 0.1
        else:
            self.state['boredom'] = max(0.0, self.state['boredom'] - 0.05)
        self.prev_entropy = self.state['entropy']
        if self.state['satisfaction'] > 0.8:
            self.state['boredom'] = max(0.0, self.state['boredom'] - 0.2)

        # Dreaming triggered by boredom
        if self.state['boredom'] > BOREDOM_THRESHOLD and not self.dreaming and not self.state['sleeping']:
            self.dreaming = True
            self._dream_buffer = None
            self._dream_playlist = None
        elif self.dreaming and self.state['boredom'] <= BOREDOM_THRESHOLD:
            self.dreaming = False
            self._dream_current = None

        # Predict next vector
        self._predict_next_vec(curr_vec)

        # Update restfulness
        update_biological_state(self.state)
        
        # Integrate BioState: stimuli_intensity from normalized entropy
        # Dynamic stimuli intensity based on entropy, audio, and motion
        entropy_norm   = self.state['entropy'] / 100.0
        audio_level    = np.sqrt((p['audio'] ** 2).mean())         # approx 0–1
        motion_level   = np.mean(p['motion']) / 255.0              # approx 0–1
        # Use the strongest signal to drive physiology
        stimuli_intensity = min(1.0, max(entropy_norm, audio_level, motion_level))
        self.bio.update(stimuli_intensity)

    def _imagine(self, p):
        now = time.time()

        # DREAM MODE
        if self.dreaming:
            if self._dream_buffer is None:
                # build buffer of recent frames
                recent = [m for m in self.memory.moments if now - m.time <= 15.0]
                buf    = [m for m in recent if now - m.time <= 10.0] or recent[-min(len(recent),10):]
                self._dream_buffer = buf
                # build playlist
                self._dream_playlist = self._build_dream_playlist(buf)
                self._dream_index    = 0
                if not self._dream_playlist:
                    self.dreaming = False
                    self._dream_buffer = None
                    return p['video']
                # initialize prev and target images
                init_expr = p['video'].astype(np.float32)
                self._dream_prev_image   = init_expr.copy()
                target_m = self._dream_playlist[0]
                self._dream_frame_target = target_m.expression.astype(np.float32)
                self._dream_last_switch   = now
                dt = max(0.5, (target_m.time - now)) if target_m.time > now else 0.5
                self._dream_duration      = dt

            # Display the target frame directly (no cross-fade) to avoid motion blur
            blended = self._dream_frame_target

            # switch if time
            if now - self._dream_last_switch >= self._dream_duration:
                self._dream_prev_image = self._dream_frame_target
                self._dream_index = (self._dream_index + 1) % len(self._dream_playlist)
                next_m = self._dream_playlist[self._dream_index]
                self._dream_frame_target = next_m.expression.astype(np.float32)
                self._dream_last_switch  = now
                dt = max(0.5, (next_m.time - now)) if next_m.time > now else 0.5
                self._dream_duration     = dt

            frame = np.clip(blended, 0, 255).astype(np.uint8)
            return frame, frame

        # SLEEP MODE
        if self.state['sleeping']:
            frame = sleep_pulse(self.memory.moments)
            return frame, frame

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

    def _render(self, frame, predicted):
        # Base frame shading and audio tint
        # Use a fixed overlay brightness to avoid pulsing/flashing
        glow = 128
        def shade(img):
            ov  = np.full_like(img, glow, np.uint8)
            bd  = cv2.addWeighted(img, 0.8, ov, 0.2, 0)
            return np.clip(bd, 0, 255).astype(np.uint8)
        left  = shade(frame)
        right = shade(predicted)

        # Audio RMS-based tint
        rms = np.sqrt((self.latest_audio ** 2).mean()) if hasattr(self, 'latest_audio') else 0
        if rms > 0.01:
            tint_l = np.full_like(left, (rms * 255, 100, 200), np.uint8)
            left   = cv2.addWeighted(left, 0.95, tint_l, 0.05, 0)
            tint_r = np.full_like(right, (rms * 255, 100, 200), np.uint8)
            right  = cv2.addWeighted(right, 0.95, tint_r, 0.05, 0)

        # Draw audio activity bar
        bar_h = int(min(1.0, rms * 50) * FRAME_HEIGHT)
        cv2.rectangle(left, (10, FRAME_HEIGHT - bar_h), (30, FRAME_HEIGHT), (255, 200, 100), -1)

        # --- NEW: Overlay physiological indicators ---
        # Heart indicator (red circle)
        # Compute a phase [0,1) based on the heart_rate period
        heart_period = self.bio.heart_rate  # seconds per beat
        heart_phase  = (time.time() % heart_period) / heart_period
        heart_pulse  = 1.0 + 0.2 * (1.0 - abs(heart_phase * 2.0 - 1.0))
        base_radius  = 15
        radius_h     = int(base_radius * heart_pulse)
        cv2.circle(left, (30, 30), radius_h, (0, 0, 255), -1)

        # Lung indicator (blue circle)
        # Compute breath cycle period and phase
        # --- With this clamped period for realistic baseline and range ---
        raw_period     = 1.0 / self.bio.breath_rate   # ideal cycle from bio state
        min_period     = 2.0    # fastest realistic breath: 1 breath per 2s (30 bpm)
        max_period     = 5.0    # slowest realistic breath: 1 breath per 5s (12 bpm)
        breath_period  = min(max_period, max(min_period, raw_period))
        breath_phase  = (time.time() % breath_period) / breath_period
        breath_pulse  = 1.0 + 0.2 * (1.0 - abs(breath_phase * 2.0 - 1.0))
        base_radius   = 15
        radius_l      = int(base_radius * breath_pulse)
        cv2.circle(left, (70, 30), radius_l, (255, 0, 0), -1)
        # -------------------------------------------------

        # Overlay prediction lead time on the predicted frame
        if self.predicted_delta is not None:
            text = f"{self.predicted_delta:.2f}s"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            thick = 1
            size, _ = cv2.getTextSize(text, font, scale, thick)
            pos = (FRAME_WIDTH - size[0] - 5, FRAME_HEIGHT - 5)
            cv2.putText(right, text, pos, font, scale, (255, 255, 255), thick, cv2.LINE_AA)

        # Imagination panel (bottom left) and empty panel (bottom right)
        imagination = shade(self._get_imagination_frame())
        empty = np.zeros_like(imagination)

        # Build 2x2 grid
        top_row = np.concatenate([left, right], axis=1)
        bottom_row = np.concatenate([imagination, empty], axis=1)
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
