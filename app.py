import sys
import os
import time
import json
import wave
import struct
import zlib
import hashlib
import threading
import asyncio
import queue
import subprocess
import uuid

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

def ensure_deps():
    needed = {
        "numpy": "numpy",
        "scipy": "scipy",
        "Crypto": "pycryptodome",
        "flask": "flask",
        "flask_cors": "flask-cors",
        "flask_socketio": "flask-socketio",
    }
    missing = []
    for mod, pkg in needed.items():
        try:
            __import__(mod)
        except (ImportError, OSError):
            missing.append(pkg)
    if missing:
        print(f"Installing: {', '.join(missing)}")
        for pkg in missing:
            install(pkg)
        print("Done. Restarting...\n")
        os.execv(sys.executable, [sys.executable] + sys.argv)

ensure_deps()

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import butter, lfilter
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room

try:
    from bleak import BleakScanner
except ImportError:
    BleakScanner = None

try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None

SAMPLE_RATE = 44100
CHANNELS = 1
DTYPE = 'float32'

BASE_FREQ = 1875
HIGH_FREQ = 6328
FREQ_SPACING = 46.875
LOW_CUTOFF = 1500
HIGH_CUTOFF = 7000

BITS_PER_TONE = 4
TONES_PER_SYMBOL = 6
SYMBOL_DURATION = 0.05

ECC_LIGHT = 4
ECC_MEDIUM = 8
ECC_HEAVY = 16
ECC_MAXIMUM = 32

NOISE_GATE_DB = -40
FILTER_ORDER = 4

PREAMBLE_PATTERN = bytes([0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55])
POSTAMBLE_PATTERN = bytes([0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA, 0x55, 0xAA])

HEADER_SIZE = 4
NONCE_SIZE = 12
LENGTH_SIZE = 2
MAX_PAYLOAD_SIZE = 200
ACK_TIMEOUT = 3.0
MAX_RETRIES = 3

VERSION = 1
FLAG_ENCRYPTED = 0x01
FLAG_COMPRESSED = 0x02
FLAG_FRAGMENTED = 0x04
FLAG_ACK_REQUIRED = 0x08

AUDIO_BUFFER_SIZE = 4096
RECORD_SECONDS = 0.1

FREQ_0 = 800
FREQ_1 = 1600
BIT_DURATION = 0.02
SAMPLES_PER_BIT = int(SAMPLE_RATE * BIT_DURATION)
PREAMBLE_FREQ = 2400
POSTAMBLE_FREQ = 400
MARKER_DURATION = 0.1


def bytes_to_hex(data):
    return data.hex()

def hex_to_bytes(hex_str):
    return bytes.fromhex(hex_str)

def calculate_checksum(data):
    return hashlib.sha256(data).digest()[:4]

def verify_checksum(data, checksum):
    return calculate_checksum(data) == checksum

def generate_device_id():
    return hashlib.md5(str(time.time()).encode()).hexdigest()[:8]

def timestamp_ms():
    return int(time.time() * 1000)

def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"

def format_bytes(data):
    if len(data) < 1024:
        return f"{len(data)} B"
    elif len(data) < 1024 * 1024:
        return f"{len(data) / 1024:.1f} KB"
    else:
        return f"{len(data) / (1024 * 1024):.1f} MB"

def signal_strength_indicator(snr_db):
    if snr_db > 20:
        return "Excellent"
    elif snr_db > 10:
        return "Good"
    elif snr_db > 0:
        return "Fair"
    else:
        return "Poor"

def signal_bar(snr_db):
    level = max(0, min(10, int(snr_db / 4)))
    filled = "\u2588" * level
    empty = "\u2591" * (10 - level)
    return f"[{filled}{empty}]"


class AudioIO:
    def __init__(self, sample_rate=SAMPLE_RATE, channels=CHANNELS):
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_recording = False
        self.recorded_frames = []
        self.audio_queue = queue.Queue()

    def play(self, waveform, blocking=True):
        if sd is None:
            return
        waveform = np.clip(waveform, -1.0, 1.0)
        waveform = waveform.astype(np.float32)
        if blocking:
            sd.play(waveform, self.sample_rate)
            sd.wait()
        else:
            sd.play(waveform, self.sample_rate)

    def save_wav(self, waveform, filename):
        waveform = np.clip(waveform, -1.0, 1.0)
        waveform = (waveform * 32767).astype(np.int16)
        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(waveform.tobytes())

    def load_wav(self, filename):
        with wave.open(filename, 'r') as wav_file:
            frames = wav_file.readframes(wav_file.getnframes())
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sampwidth = wav_file.getsampwidth()

        if sampwidth == 2:
            data = np.frombuffer(frames, dtype=np.int16)
            data = data.astype(np.float32) / 32767.0
        elif sampwidth == 4:
            data = np.frombuffer(frames, dtype=np.int32)
            data = data.astype(np.float32) / 2147483647.0
        else:
            data = np.frombuffer(frames, dtype=np.uint8)
            data = (data.astype(np.float32) - 128) / 128.0

        if channels > 1:
            data = data.reshape(-1, channels)[:, 0]

        return data, sample_rate


class ReedSolomon:
    def __init__(self, ecc_bytes=8):
        self.ecc_bytes = ecc_bytes
        self.msg_len = 255 - ecc_bytes
        self.gf_exp = [0] * 512
        self.gf_log = [0] * 256
        self._init_galois()

    def _init_galois(self):
        x = 1
        for i in range(255):
            self.gf_exp[i] = x
            self.gf_log[x] = i
            x <<= 1
            if x & 256:
                x ^= 0x11d
        for i in range(255, 512):
            self.gf_exp[i] = self.gf_exp[i - 255]

    def _gf_mul(self, a, b):
        if a == 0 or b == 0:
            return 0
        return self.gf_exp[self.gf_log[a] + self.gf_log[b]]

    def _gf_pow(self, x, power):
        return self.gf_exp[(self.gf_log[x] * power) % 255]

    def _poly_eval(self, poly, x):
        y = poly[0]
        for i in range(1, len(poly)):
            y = self._gf_mul(y, x) ^ poly[i]
        return y

    def _poly_mul(self, p, q):
        result = [0] * (len(p) + len(q) - 1)
        for i in range(len(p)):
            for j in range(len(q)):
                result[i + j] ^= self._gf_mul(p[i], q[j])
        return result

    def _generator_poly(self):
        g = [1]
        for i in range(self.ecc_bytes):
            g = self._poly_mul(g, [1, self.gf_pow(2, i)])
        return g

    def encode(self, data):
        if len(data) > self.msg_len:
            raise ValueError(f"Data too long: {len(data)} > {self.msg_len}")
        padded = data + b'\x00' * (self.msg_len - len(data))
        gen = self._generator_poly()
        msg_out = list(padded) + [0] * self.ecc_bytes
        for i in range(len(padded)):
            coef = msg_out[i]
            if coef != 0:
                for j in range(1, len(gen)):
                    msg_out[i + j] ^= self._gf_mul(gen[j], coef)
        msg_out[:len(padded)] = list(padded)
        return bytes(msg_out)

    def decode(self, data):
        if len(data) != 255:
            data = data.ljust(255, b'\x00')
        msg_in = list(data[:255])
        syndromes = [0] * self.ecc_bytes
        for i in range(self.ecc_bytes):
            syndromes[i] = self._poly_eval(msg_in, self.gf_pow(2, i))
        if max(syndromes) == 0:
            return bytes(msg_in[:self.msg_len])
        return bytes(msg_in[:self.msg_len])


class SimpleParity:
    def __init__(self, block_size=32):
        self.block_size = block_size

    def encode(self, data):
        result = bytearray()
        for i in range(0, len(data), self.block_size):
            block = data[i:i + self.block_size]
            parity = 0
            for byte in block:
                parity ^= byte
            result.extend(block)
            result.append(parity)
        return bytes(result)

    def decode(self, data):
        result = bytearray()
        errors = 0
        for i in range(0, len(data), self.block_size + 1):
            block = data[i:i + self.block_size]
            if len(block) < self.block_size:
                result.extend(block)
                break
            parity_byte = data[i + self.block_size] if i + self.block_size < len(data) else 0
            calculated_parity = 0
            for byte in block:
                calculated_parity ^= byte
            if calculated_parity == parity_byte:
                result.extend(block)
            else:
                errors += 1
                result.extend(block)
        return bytes(result), errors


class CRC32:
    def __init__(self):
        self.crc_table = self._make_table()

    def _make_table(self):
        table = []
        for i in range(256):
            crc = i
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ 0xEDB88320
                else:
                    crc >>= 1
            table.append(crc)
        return table

    def calculate(self, data):
        crc = 0xFFFFFFFF
        for byte in data:
            crc = self.crc_table[(crc ^ byte) & 0xFF] ^ (crc >> 8)
        return crc ^ 0xFFFFFFFF

    def append(self, data):
        crc = self.calculate(data)
        return data + crc.to_bytes(4, 'big')

    def verify(self, data):
        if len(data) < 4:
            return False, data
        payload = data[:-4]
        received_crc = int.from_bytes(data[-4:], 'big')
        calculated_crc = self.calculate(payload)
        return received_crc == calculated_crc, payload


class ErrorCorrection:
    def __init__(self, ecc_level='medium'):
        self.ecc_level = ecc_level
        self.crc = CRC32()

    def encode(self, data):
        crc_bytes = self.crc.calculate(data).to_bytes(4, 'big')
        return data + crc_bytes

    def decode(self, data):
        if len(data) < 4:
            return None, -1
        payload = data[:-4]
        received_crc = int.from_bytes(data[-4:], 'big')
        calculated_crc = self.crc.calculate(payload)
        if received_crc != calculated_crc:
            return None, -1
        return payload, 0


class SecureChannel:
    def __init__(self, key=None):
        if key is None:
            self.key = get_random_bytes(32)
        elif isinstance(key, str):
            self.key = hashlib.sha256(key.encode()).digest()
        else:
            self.key = key

    def encrypt(self, plaintext):
        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')
        nonce = get_random_bytes(12)
        cipher = AES.new(self.key, AES.MODE_GCM, nonce=nonce)
        ciphertext, tag = cipher.encrypt_and_digest(pad(plaintext, AES.block_size))
        return nonce + tag + ciphertext

    def decrypt(self, data):
        if len(data) < 28:
            raise ValueError("Data too short")
        nonce = data[:12]
        tag = data[12:28]
        ciphertext = data[28:]
        cipher = AES.new(self.key, AES.MODE_GCM, nonce=nonce)
        plaintext = unpad(cipher.decrypt_and_verify(ciphertext, tag), AES.block_size)
        return plaintext.decode('utf-8')

    def get_key_hex(self):
        return self.key.hex()

    @classmethod
    def from_hex(cls, hex_key):
        return cls(bytes.fromhex(hex_key))


def bytes_to_bits(data):
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits

def bits_to_bytes(bits):
    result = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte <<= 1
            if i + j < len(bits):
                byte |= bits[i + j]
        result.append(byte)
    return bytes(result)


class FSKModulator:
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.freq_0 = FREQ_0
        self.freq_1 = FREQ_1
        self.samples_per_bit = SAMPLES_PER_BIT

    def _generate_tone(self, freq, num_samples):
        t = np.arange(num_samples) / self.sample_rate
        tone = np.sin(2 * np.pi * freq * t)
        fade = min(num_samples // 5, 200)
        if fade > 0:
            tone[:fade] *= np.linspace(0, 1, fade)
            tone[-fade:] *= np.linspace(1, 0, fade)
        return tone

    def modulate(self, data):
        bits = bytes_to_bits(data)
        waveform = np.array([], dtype=np.float64)
        for bit in bits:
            freq = self.freq_1 if bit else self.freq_0
            tone = self._generate_tone(freq, self.samples_per_bit)
            waveform = np.concatenate([waveform, tone])
        return waveform


class FSKDemodulator:
    def __init__(self):
        self.sample_rate = SAMPLE_RATE
        self.freq_0 = FREQ_0
        self.freq_1 = FREQ_1
        self.samples_per_bit = SAMPLES_PER_BIT

    def demodulate(self, waveform):
        n_bits = len(waveform) // self.samples_per_bit
        bits = []
        for i in range(n_bits):
            start = i * self.samples_per_bit
            end = start + self.samples_per_bit
            segment = waveform[start:end]
            if len(segment) < self.samples_per_bit:
                break
            windowed = segment * np.hanning(len(segment))
            spectrum = np.abs(fft(windowed))
            freqs = fftfreq(len(segment), 1.0 / self.sample_rate)
            idx_0 = np.argmin(np.abs(freqs - self.freq_0))
            idx_1 = np.argmin(np.abs(freqs - self.freq_1))
            energy_0 = spectrum[idx_0]
            energy_1 = spectrum[idx_1]
            bits.append(1 if energy_1 > energy_0 else 0)
        return bits_to_bytes(bits)


class AcousticModem:
    def __init__(self):
        self.modulator = FSKModulator()
        self.demodulator = FSKDemodulator()

    def _generate_marker(self, freq, duration=MARKER_DURATION):
        n = int(SAMPLE_RATE * duration)
        t = np.arange(n) / SAMPLE_RATE
        return np.sin(2 * np.pi * freq * t) * 0.9

    def encode(self, data):
        preamble = self._generate_marker(PREAMBLE_FREQ, MARKER_DURATION)
        postamble = self._generate_marker(POSTAMBLE_FREQ, MARKER_DURATION)
        data_wave = self.modulator.modulate(data)
        return np.concatenate([preamble, data_wave, postamble])

    def decode(self, waveform):
        clean = self._find_data_region(waveform)
        if clean is None or len(clean) < self.modulator.samples_per_bit:
            return None
        return self.demodulator.demodulate(clean)

    def _find_data_region(self, waveform):
        if len(waveform) == 0:
            return None
        envelope = np.abs(waveform)
        kernel_size = min(len(waveform) // 10, 4000)
        if kernel_size < 1:
            kernel_size = 1
        envelope = np.convolve(envelope, np.ones(kernel_size) / kernel_size, mode='same')
        max_env = np.max(envelope)
        if max_env == 0:
            return None
        threshold = max_env * 0.1
        above = envelope > threshold
        if not np.any(above):
            return None
        indices = np.where(above)[0]
        if len(indices) == 0:
            return None
        start = indices[0]
        end = indices[-1] + 1
        preamble_samples = int(SAMPLE_RATE * MARKER_DURATION)
        postamble_samples = int(SAMPLE_RATE * MARKER_DURATION)
        start = min(start + preamble_samples, len(waveform))
        end = max(end - postamble_samples, start + self.modulator.samples_per_bit)
        if end > len(waveform):
            end = len(waveform)
        return waveform[start:end]

    def measure_snr(self, waveform):
        if len(waveform) == 0:
            return 0
        n = len(waveform)
        half = n // 2
        signal_power = np.mean(waveform[:half] ** 2)
        noise_power = np.mean(np.diff(waveform[half:]) ** 2)
        if noise_power == 0:
            return 40.0
        snr = 10 * np.log10(signal_power / noise_power)
        return max(0, min(snr, 40))


class BandpassFilter:
    def __init__(self, low_freq=LOW_CUTOFF, high_freq=HIGH_CUTOFF,
                 sample_rate=SAMPLE_RATE, order=FILTER_ORDER):
        self.sample_rate = sample_rate
        nyq = sample_rate / 2
        low = max(low_freq / nyq, 0.001)
        high = min(high_freq / nyq, 0.999)
        self.b, self.a = butter(order, [low, high], btype='band')

    def apply(self, signal):
        return lfilter(self.b, self.a, signal)


class NoiseGate:
    def __init__(self, threshold_db=NOISE_GATE_DB, attack_ms=10, release_ms=50,
                 sample_rate=SAMPLE_RATE):
        self.threshold = 10 ** (threshold_db / 20)
        self.attack_samples = int(sample_rate * attack_ms / 1000)
        self.release_samples = int(sample_rate * release_ms / 1000)
        self.sample_rate = sample_rate

    def apply(self, signal):
        envelope = np.abs(signal)
        kernel_size = int(self.sample_rate * 0.005)
        if kernel_size % 2 == 0:
            kernel_size += 1
        envelope = np.convolve(envelope, np.ones(kernel_size) / kernel_size, mode='same')
        gate = np.zeros(len(signal))
        in_signal = False
        counter = 0
        for i in range(len(signal)):
            if envelope[i] > self.threshold:
                in_signal = True
                counter = self.release_samples
                gate[i] = 1.0
            elif counter > 0:
                counter -= 1
                gate[i] = counter / self.release_samples
            else:
                gate[i] = 0.0
        return signal * gate


class AGC:
    def __init__(self, target_level=0.5, gain_min=0.1, gain_max=10.0,
                 adaptation_rate=0.01, sample_rate=SAMPLE_RATE):
        self.target_level = target_level
        self.gain_min = gain_min
        self.gain_max = gain_max
        self.adaptation_rate = adaptation_rate
        self.sample_rate = sample_rate
        self.current_gain = 1.0

    def apply(self, signal):
        output = np.zeros_like(signal)
        block_size = int(self.sample_rate * 0.05)
        gain = self.current_gain
        for i in range(0, len(signal), block_size):
            block = signal[i:i + block_size]
            if len(block) == 0:
                continue
            rms = np.sqrt(np.mean(block ** 2))
            if rms > 0:
                desired_gain = self.target_level / rms
                desired_gain = np.clip(desired_gain, self.gain_min, self.gain_max)
                gain = gain + self.adaptation_rate * (desired_gain - gain)
            output[i:i + len(block)] = block * gain
        self.current_gain = gain
        return output


class SignalProcessor:
    def __init__(self):
        self.bandpass = BandpassFilter()
        self.noise_gate = NoiseGate()
        self.agc = AGC()

    def process(self, signal, noise_reduction=True, agc=True):
        processed = signal.copy()
        if noise_reduction:
            processed = self.noise_gate.apply(processed)
        processed = self.bandpass.apply(processed)
        if agc:
            processed = self.agc.apply(processed)
        return processed


class Frame:
    def __init__(self, msg_id=0, frag_id=0, total_frags=1, flags=0, payload=b''):
        self.version = VERSION
        self.msg_id = msg_id
        self.frag_id = frag_id
        self.total_frags = total_frags
        self.flags = flags
        self.payload = payload

    def serialize(self):
        header = struct.pack('!BBB B H',
                           self.version,
                           self.msg_id & 0xFF,
                           self.frag_id & 0xFF,
                           self.total_frags & 0xFF,
                           self.flags & 0xFFFF)
        length = struct.pack('!H', len(self.payload))
        return header + length + self.payload

    @classmethod
    def deserialize(cls, data):
        if len(data) < HEADER_SIZE + LENGTH_SIZE:
            return None
        version, msg_id, frag_id, total_frags = struct.unpack('!BBBB', data[0:4])
        flags = struct.unpack('!H', data[4:6])[0]
        length = struct.unpack('!H', data[6:8])[0]
        if len(data) < HEADER_SIZE + LENGTH_SIZE + length:
            return None
        payload = data[HEADER_SIZE + LENGTH_SIZE:HEADER_SIZE + LENGTH_SIZE + length]
        frame = cls(msg_id, frag_id, total_frags, flags, payload)
        frame.version = version
        return frame

    @property
    def is_fragmented(self):
        return bool(self.flags & FLAG_FRAGMENTED)

    @property
    def is_encrypted(self):
        return bool(self.flags & FLAG_ENCRYPTED)

    @property
    def is_compressed(self):
        return bool(self.flags & FLAG_COMPRESSED)


class MessageFramer:
    def __init__(self, max_payload=MAX_PAYLOAD_SIZE):
        self.max_payload = max_payload
        self.msg_counter = 0

    def frame_message(self, data, compress=True):
        if compress:
            compressed = zlib.compress(data, level=6)
            if len(compressed) < len(data):
                data = compressed
                compressed_flag = FLAG_COMPRESSED
            else:
                compressed_flag = 0
        else:
            compressed_flag = 0
        fragments = []
        for i in range(0, len(data), self.max_payload):
            chunk = data[i:i + self.max_payload]
            fragments.append(chunk)
        frames = []
        for i, chunk in enumerate(fragments):
            flags = compressed_flag
            if len(fragments) > 1:
                flags |= FLAG_FRAGMENTED
            frame = Frame(
                msg_id=self.msg_counter & 0xFF,
                frag_id=i,
                total_frags=len(fragments),
                flags=flags,
                payload=chunk
            )
            frames.append(frame)
        self.msg_counter += 1
        return frames

    def reassemble(self, frames):
        frames.sort(key=lambda f: f.frag_id)
        if len(frames) == 1:
            data = frames[0].payload
        else:
            data = b''.join(f.payload for f in frames)
        if frames[0].is_compressed:
            try:
                data = zlib.decompress(data)
            except zlib.error:
                pass
        return data


app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=30000000)

modem = AcousticModem()
ecc = ErrorCorrection('medium')
audio = AudioIO()
framer = MessageFramer()
signal_proc = SignalProcessor()

needs_list = []
needs_lock = threading.Lock()

private_chats = {}


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/api/send', methods=['POST'])
def api_send():
    try:
        data = request.json
        message = data.get('message', '')
        password = data.get('password', '')
        sender = data.get('sender', 'anonymous')

        if not message or not password:
            return jsonify({'success': False, 'error': 'Missing message or password'})

        payload = f"{sender}:{message}"
        crypto = SecureChannel(password)
        encrypted = crypto.encrypt(payload)
        protected = ecc.encode(encrypted)
        waveform = modem.encode(protected)

        duration = round(len(waveform) / SAMPLE_RATE, 2)

        wav_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sounds')
        os.makedirs(wav_dir, exist_ok=True)
        wav_file = os.path.join(wav_dir, f"msg_{int(time.time())}.wav")
        audio.save_wav(waveform, wav_file)

        waveform_data = waveform[::max(1, len(waveform) // 500)].tolist()

        return jsonify({
            'success': True,
            'duration': duration,
            'size': len(protected),
            'file': os.path.basename(wav_file),
            'waveform': waveform_data
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/receive', methods=['POST'])
def api_receive():
    try:
        data = request.json
        password = data.get('password', '')
        duration = data.get('duration', 5)

        if not password:
            return jsonify({'success': False, 'error': 'Missing password'})

        if sd is None:
            return jsonify({'success': False, 'error': 'Audio hardware not available on server. Use mic recording on client.'})

        recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                          channels=1, dtype='float32')
        sd.wait()
        audio_data = recording.flatten()

        processed = signal_proc.process(audio_data)
        raw = modem.decode(processed)
        if raw is None:
            return jsonify({'success': True, 'message': None})

        text, errors = ecc.decode(raw)
        if text is None:
            return jsonify({'success': False, 'error': 'CRC check failed'})

        crypto = SecureChannel(password)
        try:
            decrypted = crypto.decrypt(text)
        except Exception:
            return jsonify({'success': False, 'error': 'Decryption failed (wrong password?)'})

        waveform_data = audio_data[::max(1, len(audio_data) // 500)].tolist()

        return jsonify({
            'success': True,
            'message': decrypted,
            'waveform': waveform_data
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/decode_pcm', methods=['POST'])
def api_decode_pcm():
    try:
        password = request.form.get('password', '')
        pcm_file = request.files.get('pcm')
        if not pcm_file or not password:
            return jsonify({'success': False, 'error': 'Missing audio or password'})

        pcm_bytes = pcm_file.read()
        audio_data = np.frombuffer(pcm_bytes, dtype=np.float32)

        processed = signal_proc.process(audio_data)
        raw = modem.decode(processed)
        if raw is None:
            return jsonify({'success': True, 'message': None})

        text, errors = ecc.decode(raw)
        if text is None:
            return jsonify({'success': False, 'error': 'CRC check failed'})

        crypto = SecureChannel(password)
        try:
            decrypted = crypto.decrypt(text)
        except Exception:
            return jsonify({'success': False, 'error': 'Decryption failed (wrong password?)'})

        waveform_data = audio_data[::max(1, len(audio_data) // 500)].tolist()

        return jsonify({
            'success': True,
            'message': decrypted,
            'waveform': waveform_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/send-file', methods=['POST'])
def api_send_file():
    try:
        password = request.form.get('password', '')
        file_obj = request.files.get('file')
        if not file_obj or not password:
            return jsonify({'success': False, 'error': 'Missing file or password'})

        file_bytes = file_obj.read()
        if len(file_bytes) > 200:
            return jsonify({'success': False, 'error': 'File > 200 bytes will take too long on acoustic. Use Digital Mode!'})

        import base64
        b64_data = base64.b64encode(file_bytes).decode('utf-8')
        payload = f"FILE:{file_obj.filename}:{b64_data}"

        crypto = SecureChannel(password)
        encrypted = crypto.encrypt(payload)

        raw = ecc.encode(encrypted)
        waveform = modem.encode(raw)

        wav_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sounds')
        os.makedirs(wav_dir, exist_ok=True)
        wav_file = f"file_{int(time.time())}.wav"
        wav_path = os.path.join(wav_dir, wav_file)
        audio.save_wav(waveform, wav_path)

        waveform_data = waveform[::max(1, len(waveform) // 500)].tolist()
        duration = len(waveform) / 44100.0

        return jsonify({
            'success': True,
            'size': len(file_bytes),
            'duration': round(duration, 1),
            'waveform': waveform_data,
            'file_url': f'/api/files/{wav_file}'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/decode', methods=['POST'])
def api_decode():
    try:
        data = request.json
        password = data.get('password', '')
        filename = data.get('filename', '')

        if not password or not filename:
            return jsonify({'success': False, 'error': 'Missing password or filename'})

        wav_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sounds')
        wav_path = os.path.join(wav_dir, filename)

        if not os.path.exists(wav_path):
            return jsonify({'success': False, 'error': f'File not found: {filename}'})

        waveform, sr = audio.load_wav(wav_path)
        processed = signal_proc.process(waveform)
        raw = modem.decode(processed)

        if raw is None:
            return jsonify({'success': False, 'error': 'No signal found in file'})

        text, errors = ecc.decode(raw)
        if text is None:
            return jsonify({'success': False, 'error': 'CRC check failed'})

        crypto = SecureChannel(password)
        try:
            decrypted = crypto.decrypt(text)
        except Exception:
            return jsonify({'success': False, 'error': 'Decryption failed'})

        waveform_data = waveform[::max(1, len(waveform) // 500)].tolist()

        return jsonify({
            'success': True,
            'message': decrypted,
            'duration': round(len(waveform) / sr, 2),
            'waveform': waveform_data
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/files', methods=['GET'])
def api_files():
    wav_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sounds')
    if not os.path.exists(wav_dir):
        return jsonify({'files': []})
    files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
    files.sort(reverse=True)
    return jsonify({'files': files})

@app.route('/api/files/<filename>', methods=['GET'])
def serve_wav(filename):
    wav_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sounds')
    return send_from_directory(wav_dir, filename)


@app.route('/api/needs', methods=['GET'])
def api_get_needs():
    with needs_lock:
        return jsonify({'needs': needs_list})

@app.route('/api/needs', methods=['POST'])
def api_post_need():
    try:
        data = request.json
        entry = {
            'id': str(uuid.uuid4())[:8],
            'user': data.get('user', 'anonymous'),
            'type': data.get('type', 'need'),
            'title': data.get('title', ''),
            'description': data.get('description', ''),
            'timestamp': int(time.time() * 1000)
        }
        with needs_lock:
            needs_list.append(entry)
        socketio.emit('new_need', entry, broadcast=True)
        return jsonify({'success': True, 'need': entry})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/needs/<need_id>', methods=['DELETE'])
def api_delete_need(need_id):
    with needs_lock:
        global needs_list
        needs_list = [n for n in needs_list if n['id'] != need_id]
    return jsonify({'success': True})


connected_clients = {}
taken_names = set()
taken_names_lock = threading.Lock()
bt_devices_count = 0

@socketio.on('connect')
def handle_connect():
    connected_clients[request.sid] = {'name': 'unknown'}
    emit('device_update', {'count': len(connected_clients)}, broadcast=True)
    emit('bt_update', {'count': bt_devices_count})

@socketio.on('disconnect')
def handle_disconnect():
    sid = request.sid
    old = connected_clients.pop(sid, {})
    name = old.get('name', '')
    if name and name != 'unknown':
        with taken_names_lock:
            taken_names.discard(name.lower())
    for room_id, chat in list(private_chats.items()):
        if sid in chat.get('members', []):
            chat['members'].discard(sid)
            leave_room(room_id)
            if not chat['members']:
                del private_chats[room_id]
            else:
                emit('chat_partner_left', {'room': room_id}, room=room_id)
    emit('device_update', {'count': len(connected_clients)}, broadcast=True)

@socketio.on('set_name')
def handle_set_name(data):
    name = data.get('name', '').strip()
    if not name:
        emit('name_result', {'ok': False, 'error': 'Name cannot be empty'})
        return
    lower = name.lower()
    with taken_names_lock:
        if lower in taken_names:
            emit('name_result', {'ok': False, 'error': f'"{name}" is already taken'})
            return
        old_name = connected_clients.get(request.sid, {}).get('name', '')
        if old_name and old_name != 'unknown':
            taken_names.discard(old_name.lower())
        taken_names.add(lower)
    connected_clients[request.sid] = {'name': name}
    emit('name_result', {'ok': True, 'name': name})
    emit('device_update', {'count': len(connected_clients), 'names': [c['name'] for c in connected_clients.values() if c['name'] != 'unknown']}, broadcast=True)

@socketio.on('digital_message')
def handle_digital_message(data):
    payload = data.get('payload', '')
    if payload:
        sender_name = connected_clients.get(request.sid, {}).get('name', 'unknown')
        emit('new_message', {'payload': payload, 'sender': sender_name}, broadcast=True, include_self=False)

@socketio.on('claim_need')
def handle_claim_need(data):
    need_id = data.get('need_id', '')
    claimer = data.get('claimer', 'anonymous')
    owner = data.get('owner', 'anonymous')

    room_id = f"chat_{need_id}"
    private_chats[room_id] = {
        'need_id': need_id,
        'members': set(),
        'created': int(time.time() * 1000)
    }

    join_room(room_id, request.sid)
    private_chats[room_id]['members'].add(request.sid)
    emit('private_joined', {'room': room_id, 'partner': owner}, room=request.sid)

    for sid, info in connected_clients.items():
        if info.get('name', '').lower() == owner.lower() and sid != request.sid:
            join_room(room_id, sid)
            private_chats[room_id]['members'].add(sid)
            emit('private_joined', {'room': room_id, 'partner': claimer, 'need_id': need_id}, room=sid)
            break

    with needs_lock:
        global needs_list
        needs_list = [n for n in needs_list if n['id'] != need_id]
    socketio.emit('need_claimed', {'need_id': need_id, 'claimer': claimer, 'owner': owner}, broadcast=True)

@socketio.on('join_private')
def handle_join_private(data):
    room_id = data.get('room', '')
    if room_id in private_chats:
        join_room(room_id, request.sid)
        private_chats[room_id]['members'].add(request.sid)
        emit('private_joined', {'room': room_id}, room=request.sid)

@socketio.on('private_message')
def handle_private_message(data):
    room = data.get('room', '')
    payload = data.get('payload', '')
    sender = data.get('sender', 'anonymous')
    if room and payload:
        emit('new_private_message', {
            'room': room,
            'payload': payload,
            'sender': sender
        }, room=room, include_self=False)


def bt_scan_thread():
    if BleakScanner is None:
        return
    global bt_devices_count
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def scan():
        global bt_devices_count
        while True:
            try:
                devices = await BleakScanner.discover(timeout=3.0, return_adv=True)
                nearby = [{'mac': addr, 'rssi': adv.rssi} for addr, (dev, adv) in devices.items() if adv.rssi >= -70]
                bt_devices_count = len(nearby)
                socketio.emit('bt_update', {'count': bt_devices_count, 'devices': nearby})
            except Exception:
                pass
            await asyncio.sleep(1)

    loop.run_until_complete(scan())

if __name__ == '__main__':
    print("\n  WHISPER - Acoustic + Digital Messenger")
    print("  Open http://localhost:8080 in your browser\n")
    print("  Starting background Bluetooth radar...")
    threading.Thread(target=bt_scan_thread, daemon=True).start()
    port = int(os.environ.get("PORT", 8080))
    socketio.run(app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True)
