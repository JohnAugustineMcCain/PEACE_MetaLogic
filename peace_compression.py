import enum
import json
import numpy as np
from typing import List, Tuple, Callable, Dict, Union
import zlib
import nltk  # For text; install with `pip install nltk`
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import difflib  # For text similarity
from skimage import io, color  # For images; install with `pip install scikit-image`
from sklearn.decomposition import PCA  # For image compression
import matplotlib.pyplot as plt  # For visualization (optional)
from scipy.io import wavfile  # For audio; install with `pip install scipy`
from scipy.fft import fft, ifft  # For FFT-based compression

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TruthValue(enum.Enum):
    T = 'T'
    F = 'F'
    B = 'B'

def negation(val: TruthValue) -> TruthValue:
    if val == TruthValue.T:
        return TruthValue.F
    elif val == TruthValue.F:
        return TruthValue.T
    else:
        return TruthValue.B

def conjunction(val1: TruthValue, val2: TruthValue) -> TruthValue:
    if val1 == TruthValue.F or val2 == TruthValue.F:
        return TruthValue.F
    if val1 == TruthValue.T and val2 == TruthValue.T:
        return TruthValue.T
    return TruthValue.B

def disjunction(val1: TruthValue, val2: TruthValue) -> TruthValue:
    if val1 == TruthValue.T or val2 == TruthValue.T:
        return TruthValue.T
    if val1 == TruthValue.F and val2 == TruthValue.F:
        return TruthValue.F
    return TruthValue.B

def implication(val1: TruthValue, val2: TruthValue) -> TruthValue:
    if val1 == TruthValue.T and val2 == TruthValue.F:
        return TruthValue.F
    if val1 == TruthValue.F or val2 == TruthValue.T:
        return TruthValue.T
    return TruthValue.B

class Perspective:
    def __init__(self, name: str, func: Callable[[any, Dict], Tuple[TruthValue, float]]):
        self.name = name
        self.func = func
        self.weight = 0.5  # Initial confidence, can be updated

def compute_cc(available: set, required: set) -> float:
    if not required:
        return 1.0
    intersection = available.intersection(required)
    return len(intersection) / len(required)

class PEACE:
    def __init__(self, cc_threshold_low=0.3, cc_threshold_high=0.8, confidence_threshold=0.7, q_max=3):
        self.cc_threshold_low = cc_threshold_low
        self.cc_threshold_high = cc_threshold_high
        self.confidence_threshold = confidence_threshold
        self.q_max = q_max
        self.perspectives: List[Perspective] = []

    def add_perspective(self, perspective: Perspective):
        self.perspectives.append(perspective)

    def evaluate(self, phi: any, context: Dict, required_context: set) -> TruthValue:
        available_context = set(context.keys())
        cc = compute_cc(available_context, required_context)
        if cc < self.cc_threshold_low:
            return TruthValue.B
        if cc < self.cc_threshold_high:
            return TruthValue.B  # Could trigger questions, but proceed for simplicity

        valuations = []
        for p in self.perspectives:
            val, conf = p.func(phi, context)
            valuations.append((val, p.weight * conf))

        t_score = sum(w for v, w in valuations if v == TruthValue.T)
        f_score = sum(w for v, w in valuations if v == TruthValue.F)
        b_score = sum(w for v, w in valuations if v == TruthValue.B)

        total = t_score + f_score + b_score
        if total == 0:
            return TruthValue.B

        max_score = max(t_score, f_score, b_score) / total
        if max_score >= self.confidence_threshold:
            if t_score == max(t_score, f_score, b_score):
                return TruthValue.T
            if f_score == max(t_score, f_score, b_score):
                return TruthValue.F
        return TruthValue.B

    def update_weights(self, perspective_name: str, reward: float):
        for p in self.perspectives:
            if p.name == perspective_name:
                p.weight = min(1.0, max(0.0, p.weight + reward))
                break

class PeaceCompressor:
    def __init__(self, error_threshold=0.05, max_degree=10):
        self.peace = PEACE()
        self.error_threshold = error_threshold
        self.max_degree = max_degree

        def utility_perspective(phi, context):
            original = context.get('original')
            reconstructed = context.get('reconstructed')
            if original is None or reconstructed is None:
                return TruthValue.B, 0.5
            error = np.mean(np.abs(original - reconstructed)) / (np.mean(np.abs(original)) + 1e-10)
            if error < self.error_threshold:
                return TruthValue.T, 1 - error
            return TruthValue.F, error

        self.peace.add_perspective(Perspective("utility", utility_perspective))

        def smoothness_perspective(phi, context):
            reconstructed = context.get('reconstructed')
            if reconstructed is None:
                return TruthValue.B, 0.5
            diffs = np.diff(reconstructed)
            var = np.var(diffs)
            if var < 0.1:
                return TruthValue.T, 0.8
            return TruthValue.B, 0.5

        self.peace.add_perspective(Perspective("smoothness", smoothness_perspective))

    def compress(self, data: np.ndarray) -> bytes:
        x = np.arange(len(data))
        best_meta = None
        best_recon = None
        for deg in range(1, self.max_degree + 1):
            coeffs = np.polyfit(x, data, deg)
            recon = np.polyval(coeffs, x)
            context = {'original': data, 'reconstructed': recon}
            val = self.peace.evaluate("reconstruction", context, {'original', 'reconstructed'})
            if val == TruthValue.T:
                best_meta = {
                    'type': 'poly',
                    'degree': deg,
                    'coeffs': coeffs.tolist(),
                    'length': len(data)
                }
                best_recon = recon
                break

        if best_meta is None:
            deg = self.max_degree
            coeffs = np.polyfit(x, data, deg)
            best_meta = {
                'type': 'poly',
                'degree': deg,
                'coeffs': coeffs.tolist(),
                'length': len(data)
            }

        meta_json = json.dumps(best_meta).encode('utf-8')
        compressed = zlib.compress(meta_json)
        return compressed

    def decompress(self, compressed: bytes) -> np.ndarray:
        meta_json = zlib.decompress(compressed)
        meta = json.loads(meta_json.decode('utf-8'))

        if meta['type'] == 'poly':
            x = np.arange(meta['length'])
            coeffs = np.array(meta['coeffs'])
            recon = np.polyval(coeffs, x)
            context = {'reconstructed': recon}
            val = self.peace.evaluate("reconstruction", context, {'reconstructed'})
            if val == TruthValue.F:
                print("Warning: Reconstruction may have issues based on perspectives.")
            return recon
        raise ValueError("Unknown compression type")

class TextCompressor:
    def __init__(self, similarity_threshold=0.85, max_ngram=3):
        self.peace = PEACE()
        self.similarity_threshold = similarity_threshold
        self.max_ngram = max_ngram

        def coherence_perspective(phi, context):
            original = context.get('original')
            reconstructed = context.get('reconstructed')
            if original is None or reconstructed is None:
                return TruthValue.B, 0.5
            matcher = difflib.SequenceMatcher(None, original.split(), reconstructed.split())
            similarity = matcher.ratio()
            if similarity >= self.similarity_threshold:
                return TruthValue.T, similarity
            return TruthValue.F, similarity

        self.peace.add_perspective(Perspective("coherence", coherence_perspective))

        def readability_perspective(phi, context):
            reconstructed = context.get('reconstructed')
            if reconstructed is None:
                return TruthValue.B, 0.5
            words = len(word_tokenize(reconstructed))
            if 0.8 <= words / 10 <= 1.2:  # Arbitrary range for small text
                return TruthValue.T, 0.7
            return TruthValue.B, 0.5

        self.peace.add_perspective(Perspective("readability", readability_perspective))

    def compress(self, text: str) -> bytes:
        tokens = word_tokenize(text.lower())
        best_meta = None
        best_recon = None

        for n in range(1, self.max_ngram + 1):
            n_grams = list(ngrams(tokens, n))
            freq_dict = {}
            for ng in n_grams:
                freq_dict[" ".join(ng)] = freq_dict.get(" ".join(ng), 0) + 1

            threshold = int(len(freq_dict) * 0.2)
            sorted_ngrams = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:threshold]
            dictionary = {k: i for i, (k, _) in enumerate(sorted_ngrams)}

            recon_tokens = []
            for i in range(0, len(tokens), n):
                window = " ".join(tokens[i:i + n])
                if window in dictionary:
                    recon_tokens.append(f"{dictionary[window]}")
                else:
                    recon_tokens.append(tokens[i])
            reconstructed = " ".join(recon_tokens)

            context = {'original': text, 'reconstructed': reconstructed}
            val = self.peace.evaluate("reconstruction", context, {'original', 'reconstructed'})
            if val == TruthValue.T:
                best_meta = {
                    'type': 'ngram',
                    'n': n,
                    'dictionary': dictionary,
                    'length': len(tokens)
                }
                best_recon = reconstructed
                break

        if best_meta is None:
            n = self.max_ngram
            n_grams = list(ngrams(tokens, n))
            freq_dict = {}
            for ng in n_grams:
                freq_dict[" ".join(ng)] = freq_dict.get(" ".join(ng), 0) + 1
            threshold = int(len(freq_dict) * 0.2)
            sorted_ngrams = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)[:threshold]
            best_meta = {
                'type': 'ngram',
                'n': n,
                'dictionary': {k: i for i, (k, _) in enumerate(sorted_ngrams)},
                'length': len(tokens)
            }

        meta_json = json.dumps(best_meta).encode('utf-8')
        compressed = zlib.compress(meta_json)
        return compressed

    def decompress(self, compressed: bytes) -> str:
        meta_json = zlib.decompress(compressed)
        meta = json.loads(meta_json.decode('utf-8'))

        if meta['type'] == 'ngram':
            rev_dict = {v: k for k, v in meta['dictionary'].items()}
            recon_tokens = []
            for i in range(meta['length']):
                if str(i) in rev_dict:
                    recon_tokens.append(rev_dict[str(i)].split())
                else:
                    recon_tokens.append(['<unk>'])
            reconstructed = " ".join(word for sublist in recon_tokens for word in sublist)

            context = {'reconstructed': reconstructed}
            val = self.peace.evaluate("reconstruction", context, {'reconstructed'})
            if val == TruthValue.F:
                print("Warning: Reconstruction may have issues based on perspectives.")
            return reconstructed
        raise ValueError("Unknown compression type")

class ImageCompressor:
    def __init__(self, psnr_threshold=30.0, max_components=50):
        self.peace = PEACE()
        self.psnr_threshold = psnr_threshold
        self.max_components = max_components

        def quality_perspective(phi, context):
            original = context.get('original')
            reconstructed = context.get('reconstructed')
            if original is None or reconstructed is None:
                return TruthValue.B, 0.5
            mse = np.mean((original - reconstructed) ** 2)
            if mse == 0:
                return TruthValue.T, 1.0
            max_pixel = np.max(original)
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse)) if mse > 0 else 100
            if psnr >= self.psnr_threshold:
                return TruthValue.T, psnr / 100
            return TruthValue.F, psnr / 100

        self.peace.add_perspective(Perspective("quality", quality_perspective))

        def structure_perspective(phi, context):
            reconstructed = context.get('reconstructed')
            if reconstructed is None:
                return TruthValue.B, 0.5
            grad_x = np.abs(np.gradient(reconstructed, axis=1))
            grad_y = np.abs(np.gradient(reconstructed, axis=0))
            grad_var = np.var(grad_x) + np.var(grad_y)
            if grad_var > 0.01:
                return TruthValue.T, 0.7
            return TruthValue.B, 0.5

        self.peace.add_perspective(Perspective("structure", structure_perspective))

    def compress(self, image_path: str) -> bytes:
        img = io.imread(image_path, as_gray=True)
        img = img.astype(np.float32) / 255.0

        best_meta = None
        best_recon = None

        for factor in [2, 4, 8]:
            downsampled = img[::factor, ::factor]
            flat_data = downsampled.reshape(-1, downsampled.shape[0] * downsampled.shape[1]).T
            for n_components in range(1, self.max_components + 1, 5):
                pca = PCA(n_components=n_components)
                compressed_data = pca.fit_transform(flat_data)
                recon_flat = pca.inverse_transform(compressed_data)
                recon_downsampled = recon_flat.T.reshape(downsampled.shape)
                recon = np.zeros_like(img)
                for i in range(0, img.shape[0], factor):
                    for j in range(0, img.shape[1], factor):
                        recon[i:i+factor, j:j+factor] = recon_downsampled[i//factor, j//factor]

                context = {'original': img, 'reconstructed': recon}
                val = self.peace.evaluate("reconstruction", context, {'original', 'reconstructed'})
                if val == TruthValue.T:
                    best_meta = {
                        'type': 'pca',
                        'downsample_factor': factor,
                        'n_components': n_components,
                        'mean': pca.mean_.tolist(),
                        'components': pca.components_.tolist(),
                        'shape': img.shape
                    }
                    best_recon = recon
                    break
            if best_meta:
                break

        if best_meta is None:
            factor = 8
            downsampled = img[::factor, ::factor]
            flat_data = downsampled.reshape(-1, downsampled.shape[0] * downsampled.shape[1]).T
            pca = PCA(n_components=self.max_components)
            compressed_data = pca.fit_transform(flat_data)
            best_meta = {
                'type': 'pca',
                'downsample_factor': factor,
                'n_components': self.max_components,
                'mean': pca.mean_.tolist(),
                'components': pca.components_.tolist(),
                'shape': img.shape
            }

        meta_json = json.dumps(best_meta).encode('utf-8')
        compressed = zlib.compress(meta_json)
        return compressed

    def decompress(self, compressed: bytes) -> np.ndarray:
        meta_json = zlib.decompress(compressed)
        meta = json.loads(meta_json.decode('utf-8'))

        if meta['type'] == 'pca':
            pca = PCA(n_components=meta['n_components'])
            pca.mean_ = np.array(meta['mean'])
            pca.components_ = np.array(meta['components'])
            flat_data = np.random.rand(meta['n_components'])  # Placeholder
            recon_flat = pca.inverse_transform(flat_data)
            recon_downsampled = recon_flat.T.reshape(-1, int(np.sqrt(recon_flat.shape[0])))
            factor = meta['downsample_factor']
            shape = meta['shape']
            recon = np.zeros(shape)
            for i in range(0, shape[0], factor):
                for j in range(0, shape[1], factor):
                    recon[i:i+factor, j:j+factor] = recon_downsampled[i//factor, j//factor]

            context = {'reconstructed': recon}
            val = self.peace.evaluate("reconstruction", context, {'reconstructed'})
            if val == TruthValue.F:
                print("Warning: Reconstruction may have issues based on perspectives.")
            return recon
        raise ValueError("Unknown compression type")

class AudioCompressor:
    def __init__(self, snr_threshold=30.0, max_freq_bands=50):
        self.peace = PEACE()
        self.snr_threshold = snr_threshold  # Signal-to-Noise Ratio threshold (dB)
        self.max_freq_bands = max_freq_bands

        # Perspective: Perceptual Quality (SNR-based)
        def quality_perspective(phi, context):
            original = context.get('original')
            reconstructed = context.get('reconstructed')
            if original is None or reconstructed is None:
                return TruthValue.B, 0.5
            noise = original - reconstructed
            signal_power = np.mean(original ** 2)
            noise_power = np.mean(noise ** 2)
            snr = 10 * np.log10(signal_power / (noise_power + 1e-10)) if noise_power > 0 else 100
            if snr >= self.snr_threshold:
                return TruthValue.T, snr / 100  # Normalize confidence
            return TruthValue.F, snr / 100

        self.peace.add_perspective(Perspective("quality", quality_perspective))

        # Perspective: Temporal Consistency (variance of differences)
        def consistency_perspective(phi, context):
            reconstructed = context.get('reconstructed')
            if reconstructed is None:
                return TruthValue.B, 0.5
            diffs = np.diff(reconstructed)
            var = np.var(diffs)
            if var < 0.1:  # Low variance indicates smooth transitions
                return TruthValue.T, 0.7
            return TruthValue.B, 0.5

        self.peace.add_perspective(Perspective("consistency", consistency_perspective))

    def compress(self, audio_path: str) -> bytes:
        """
        Compress audio using downsampling + FFT-based frequency selection as meta-rule.
        Stores downsample factor and significant frequency components.
        """
        sample_rate, audio = wavfile.read(audio_path)
        audio = audio.astype(np.float32) / np.max(np.abs(audio))  # Normalize to [-1, 1]

        best_meta = None
        best_recon = None

        # Try different downsample factors and frequency band counts
        for factor in [2, 4, 8]:  # Downsample by 2x, 4x, 8x
            downsampled = audio[::factor]
            n_samples = len(downsampled)
            fft_data = np.abs(fft(downsampled))
            freq_bands = np.linspace(0, sample_rate // 2, len(fft_data) // 2 + 1)
            for n_bands in range(1, self.max_freq_bands + 1, 5):
                # Select top n_bands frequencies
                indices = np.argsort(fft_data[:len(fft_data) // 2])[-n_bands:]
                selected_mags = fft_data[indices]
                selected_phases = np.angle(fft(downsampled))[indices]

                # Reconstruct in frequency domain
                recon_fft = np.zeros(len(fft_data), dtype=complex)
                recon_fft[indices] = selected_mags * np.exp(1j * selected_phases)
                recon_fft[len(fft_data) // 2 + 1:] = np.conj(recon_fft[1:len(fft_data) // 2][::-1])  # Symmetry
                recon = np.real(ifft(recon_fft))[:n_samples]  # Inverse FFT
                recon_upsampled = np.interp(np.arange(len(audio)), np.arange(0, len(audio), factor), recon)

                context = {'original': audio, 'reconstructed': recon_upsampled}
                val = self.peace.evaluate("reconstruction", context, {'original', 'reconstructed'})
                if val == TruthValue.T:
                    best_meta = {
                        'type': 'fft',
                        'downsample_factor': factor,
                        'n_bands': n_bands,
                        'freq_indices': indices.tolist(),
                        'magnitudes': selected_mags.tolist(),
                        'phases': selected_phases.tolist(),
                        'sample_rate': sample_rate,
                        'length': len(audio)
                    }
                    best_recon = recon_upsampled
                    break
            if best_meta:
                break

        if best_meta is None:
            factor = 8
            downsampled = audio[::factor]
            n_samples = len(downsampled)
            fft_data = np.abs(fft(downsampled))
            indices = np.argsort(fft_data[:len(fft_data) // 2])[-self.max_freq_bands:]
            selected_mags = fft_data[indices]
            selected_phases = np.angle(fft(downsampled))[indices]
            recon_fft = np.zeros(len(fft_data), dtype=complex)
            recon_fft[indices] = selected_mags * np.exp(1j * selected_phases)
            recon_fft[len(fft_data) // 2 + 1:] = np.conj(recon_fft[1:len(fft_data) // 2][::-1])
            recon = np.real(ifft(recon_fft))[:n_samples]
            recon_upsampled = np.interp(np.arange(len(audio)), np.arange(0, len(audio), factor), recon)
            best_meta = {
                'type': 'fft',
                'downsample_factor': factor,
                'n_bands': self.max_freq_bands,
                'freq_indices': indices.tolist(),
                'magnitudes': selected_mags.tolist(),
                'phases': selected_phases.tolist(),
                'sample_rate': sample_rate,
                'length': len(audio)
            }

        meta_json = json.dumps(best_meta).encode('utf-8')
        compressed = zlib.compress(meta_json)
        return compressed

    def decompress(self, compressed: bytes) -> np.ndarray:
        """
        Reconstruct audio using meta-rule (FFT + upsampling).
        Tolerates errors if PEACE evaluates as T.
        """
        meta_json = zlib.decompress(compressed)
        meta = json.loads(meta_json.decode('utf-8'))

        if meta['type'] == 'fft':
            n_samples = meta['length']
            recon_fft = np.zeros(n_samples // 2 + 1, dtype=complex)
            indices = meta['freq_indices']
            magnitudes = np.array(meta['magnitudes'])
            phases = np.array(meta['phases'])
            recon_fft[indices] = magnitudes * np.exp(1j * phases)
            recon_fft[1:n_samples // 2] = np.conj(recon_fft[n_samples // 2:0:-1])  # Symmetry
            recon = np.real(ifft(np.concatenate([recon_fft, np.conj(recon_fft[-2:0:-1])])))[:n_samples]
            recon_upsampled = np.interp(np.arange(n_samples * meta['downsample_factor']),
                                      np.arange(0, n_samples, meta['downsample_factor']), recon)

            context = {'reconstructed': recon_upsampled}
            val = self.peace.evaluate("reconstruction", context, {'reconstructed'})
            if val == TruthValue.F:
                print("Warning: Reconstruction may have issues based on perspectives.")
            return recon_upsampled.astype(np.int16)  # Convert back to integer for WAV
        raise ValueError("Unknown compression type")

# Example usage
if __name__ == "__main__":
    # Test numerical data
    x = np.linspace(0, 10, 200)
    data = x**2 + np.random.normal(0, 1, 200)
    num_compressor = PeaceCompressor(error_threshold=0.01)
    compressed_num = num_compressor.compress(data)
    print(f"Compressed numerical size: {len(compressed_num)} bytes")
    decompressed_num = num_compressor.decompress(compressed_num)
    error = np.mean(np.abs(data - decompressed_num))
    print(f"Mean absolute error (numerical): {error}")

    # Test text data
    text = "The quick brown fox jumps over the lazy dog repeatedly and quickly again."
    text_compressor = TextCompressor(similarity_threshold=0.85)
    compressed_text = text_compressor.compress(text)
    print(f"Compressed text size: {len(compressed_text)} bytes")
    decompressed_text = text_compressor.decompress(compressed_text)
    print(f"Decompressed text: {decompressed_text}")

    # Test image data
    image_path = "sample_image.jpg"  # Replace with a real image path
    img_compressor = ImageCompressor(psnr_threshold=30.0)
    compressed_img = img_compressor.compress(image_path)
    print(f"Compressed image size: {len(compressed_img)} bytes")
    decompressed_img = img_compressor.decompress(compressed_img)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(io.imread(image_path, as_gray=True), cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("Decompressed Image")
    plt.imshow(decompressed_img, cmap='gray')
    plt.show()

    # Test audio data
    audio_path = "sample_audio.wav"  # Replace with a real audio path (e.g., .wav file)
    audio_compressor = AudioCompressor(snr_threshold=30.0)
    compressed_audio = audio_compressor.compress(audio_path)
    print(f"Compressed audio size: {len(compressed_audio)} bytes")
    decompressed_audio = audio_compressor.decompress(compressed_audio)
    wavfile.write("decompressed_audio.wav", 44100, decompressed_audio)  # Save for listening
    print("Decompressed audio saved as 'decompressed_audio.wav'")
