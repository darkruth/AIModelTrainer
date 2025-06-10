import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import librosa
import io
import wave
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq

from utils.logger import Logger

class AudioRNN(nn.Module):
    """
    Red neuronal recurrente para procesamiento de audio
    Implementa características cuánticas en el procesamiento secuencial
    """
    
    def __init__(self, input_size=128, hidden_size=256, num_layers=2, output_size=64):
        super(AudioRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN bidireccional para captar patrones temporales
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        
        # Atención temporal
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # Bidireccional
            num_heads=8,
            batch_first=True
        )
        
        # Capas de procesamiento cuántico-inspiradas
        self.quantum_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),  # Función de activación cuántica-like
            nn.Dropout(0.1)
        )
        
        # Capa de salida
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_size, output_size)
        )
        
        # Normalización
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
    
    def forward(self, x, hidden_state=None):
        """
        Forward pass del RNN de audio
        
        Args:
            x: Input tensor [batch, seq_len, input_size]
            hidden_state: Estado oculto previo
        
        Returns:
            output: Características procesadas
            hidden_state: Nuevo estado oculto
            attention_weights: Pesos de atención
        """
        
        # Procesar con RNN
        rnn_out, hidden_state = self.rnn(x, hidden_state)
        
        # Normalizar
        rnn_out = self.layer_norm(rnn_out)
        
        # Aplicar atención temporal
        attended_out, attention_weights = self.attention(rnn_out, rnn_out, rnn_out)
        
        # Procesamiento cuántico
        quantum_out = self.quantum_layer(attended_out)
        
        # Pooling temporal (promedio)
        pooled = torch.mean(quantum_out, dim=1)
        
        # Capa de salida
        output = self.output_layer(pooled)
        
        return output, hidden_state, attention_weights

class EmotionAudioClassifier(nn.Module):
    """
    Clasificador de emociones en audio
    """
    
    def __init__(self, input_size=64, num_emotions=8):
        super(EmotionAudioClassifier, self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_emotions),
            nn.Softmax(dim=1)
        )
        
        self.emotion_labels = [
            'felicidad', 'tristeza', 'miedo', 'ira',
            'sorpresa', 'disgusto', 'neutral', 'amor'
        ]
    
    def forward(self, x):
        return self.classifier(x)

class AudioProcessor:
    """
    Procesador de audio principal con capacidades de análisis consciente
    Implementa procesamiento de audio usando redes recurrentes cuánticas
    """
    
    def __init__(self):
        self.logger = Logger()
        
        # Modelos de audio
        self.audio_rnn = AudioRNN()
        self.emotion_classifier = EmotionAudioClassifier()
        
        # Configuración de audio
        self.sample_rate = 22050  # Hz estándar para análisis
        self.hop_length = 512
        self.n_mels = 128  # Número de filtros mel
        self.n_fft = 2048
        
        # Configuración de análisis
        self.window_size = 2048
        self.overlap = 0.5
        
        # Rangos de frecuencia para análisis
        self.freq_ranges = {
            'sub_bass': (20, 60),
            'bass': (60, 250),
            'low_midrange': (250, 500),
            'midrange': (500, 2000),
            'upper_midrange': (2000, 4000),
            'presence': (4000, 6000),
            'brilliance': (6000, 20000)
        }
        
        # Estadísticas de procesamiento
        self.processing_stats = {
            'audios_processed': 0,
            'avg_duration': 0.0,
            'emotion_distribution': {},
            'audio_quality_scores': []
        }
        
        self.logger.log("INFO", "AudioProcessor initialized successfully")
    
    def analyze_audio(self, audio_file) -> Dict:
        """
        Análisis completo de audio usando conciencia artificial
        
        Args:
            audio_file: Archivo de audio (Streamlit UploadedFile o similar)
        
        Returns:
            Diccionario con análisis completo
        """
        
        try:
            # Cargar audio
            audio_data, sr = self._load_audio(audio_file)
            
            # Análisis básico
            basic_analysis = self._analyze_basic_properties(audio_data, sr)
            
            # Análisis espectral
            spectral_analysis = self._analyze_spectral_features(audio_data, sr)
            
            # Análisis temporal
            temporal_analysis = self._analyze_temporal_features(audio_data, sr)
            
            # Análisis de frecuencias
            frequency_analysis = self._analyze_frequency_content(audio_data, sr)
            
            # Análisis de calidad
            quality_analysis = self._analyze_audio_quality(audio_data, sr)
            
            # Análisis de contenido (voz vs música vs ruido)
            content_analysis = self._analyze_content_type(audio_data, sr)
            
            # Análisis de patrones rítmicos
            rhythm_analysis = self._analyze_rhythm_patterns(audio_data, sr)
            
            analysis = {
                'basic_properties': basic_analysis,
                'spectral_features': spectral_analysis,
                'temporal_features': temporal_analysis,
                'frequency_content': frequency_analysis,
                'quality_metrics': quality_analysis,
                'content_type': content_analysis,
                'rhythm_patterns': rhythm_analysis,
                'processing_timestamp': self._get_timestamp()
            }
            
            # Actualizar estadísticas
            self._update_processing_stats(analysis)
            
            self.logger.log("DEBUG", f"Audio analysis completed: {basic_analysis.get('duration', 0):.2f}s")
            
            return analysis
            
        except Exception as e:
            self.logger.log("ERROR", f"Audio analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def analyze_emotions(self, audio_file) -> Dict[str, float]:
        """
        Analiza emociones en el audio
        
        Args:
            audio_file: Archivo de audio
        
        Returns:
            Diccionario con puntuaciones emocionales
        """
        
        try:
            # Cargar y procesar audio
            audio_data, sr = self._load_audio(audio_file)
            
            # Extraer características para análisis emocional
            emotion_features = self._extract_emotion_features(audio_data, sr)
            
            # Análisis con red neuronal
            with torch.no_grad():
                features_tensor = torch.tensor(emotion_features, dtype=torch.float32).unsqueeze(0)
                emotion_scores = self.emotion_classifier(features_tensor)
                emotion_scores = emotion_scores.squeeze().tolist()
            
            # Crear diccionario de emociones
            neural_emotions = {
                emotion: score for emotion, score in 
                zip(self.emotion_classifier.emotion_labels, emotion_scores)
            }
            
            # Análisis adicional basado en características acústicas
            acoustic_emotions = self._analyze_acoustic_emotions(audio_data, sr)
            
            # Combinar análisis
            combined_emotions = {}
            for emotion in neural_emotions:
                neural_score = neural_emotions[emotion]
                acoustic_score = acoustic_emotions.get(emotion, 0.0)
                combined_emotions[emotion] = (neural_score * 0.6 + acoustic_score * 0.4)
            
            # Métricas adicionales
            dominant_emotion = max(combined_emotions.keys(), key=lambda k: combined_emotions[k])
            emotional_intensity = max(combined_emotions.values())
            emotional_stability = 1.0 - np.std(list(combined_emotions.values()))
            
            return {
                'neural_emotions': neural_emotions,
                'acoustic_emotions': acoustic_emotions,
                'combined_emotions': combined_emotions,
                'dominant_emotion': dominant_emotion,
                'emotional_intensity': emotional_intensity,
                'emotional_stability': emotional_stability
            }
            
        except Exception as e:
            self.logger.log("ERROR", f"Emotion analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def transcribe_audio(self, audio_file) -> str:
        """
        Transcribe audio a texto (implementación simplificada)
        
        Args:
            audio_file: Archivo de audio
        
        Returns:
            Transcripción del audio
        """
        
        try:
            # Cargar audio
            audio_data, sr = self._load_audio(audio_file)
            
            # Análisis de contenido de voz
            voice_analysis = self._analyze_voice_content(audio_data, sr)
            
            # En una implementación real, aquí se usaría un modelo de ASR
            # Por ahora, generamos una transcripción basada en el análisis
            
            if voice_analysis['voice_probability'] > 0.7:
                # Estimación de palabras basada en patrones de voz
                estimated_words = max(1, int(voice_analysis['speech_segments'] * 2.5))
                
                # Transcripción simulada basada en características del audio
                if voice_analysis['emotional_tone'] > 0.6:
                    transcription = f"[Audio detectado con {estimated_words} palabras aproximadamente. Tono emocional positivo detectado.]"
                elif voice_analysis['emotional_tone'] < 0.4:
                    transcription = f"[Audio detectado con {estimated_words} palabras aproximadamente. Tono emocional bajo detectado.]"
                else:
                    transcription = f"[Audio detectado con {estimated_words} palabras aproximadamente. Tono neutral.]"
                
                # Añadir información sobre calidad de voz
                if voice_analysis['clarity_score'] > 0.8:
                    transcription += " Voz clara y bien definida."
                elif voice_analysis['clarity_score'] < 0.4:
                    transcription += " Calidad de voz limitada o distorsionada."
                
            else:
                # No es principalmente voz
                content_type = voice_analysis.get('primary_content', 'desconocido')
                transcription = f"[Audio no vocal detectado. Tipo de contenido: {content_type}]"
            
            return transcription
            
        except Exception as e:
            self.logger.log("ERROR", f"Audio transcription failed: {str(e)}")
            return f"[Error en transcripción: {str(e)}]"
    
    def extract_features(self, audio_file) -> torch.Tensor:
        """
        Extrae características vectoriales del audio para fusión multimodal
        
        Args:
            audio_file: Archivo de audio
        
        Returns:
            Tensor de características [256] dimensional
        """
        
        try:
            # Cargar audio
            audio_data, sr = self._load_audio(audio_file)
            
            # Extraer características mel-spectrogram
            mel_features = self._extract_mel_features(audio_data, sr)
            
            # Procesar con RNN
            with torch.no_grad():
                features_tensor = torch.tensor(mel_features, dtype=torch.float32).unsqueeze(0)
                rnn_output, _, _ = self.audio_rnn(features_tensor)
                features = rnn_output.squeeze()
            
            # Asegurar dimensión correcta
            if features.dim() == 0:
                features = features.unsqueeze(0)
            
            # Padding o truncate a 256 dimensiones
            target_size = 256
            current_size = features.size(0)
            
            if current_size < target_size:
                padding = torch.zeros(target_size - current_size)
                features = torch.cat([features, padding])
            elif current_size > target_size:
                features = features[:target_size]
            
            return features
            
        except Exception as e:
            self.logger.log("ERROR", f"Audio feature extraction failed: {str(e)}")
            # Retornar vector de ceros en caso de error
            return torch.zeros(256, dtype=torch.float32)
    
    def _load_audio(self, audio_file) -> Tuple[np.ndarray, int]:
        """Carga archivo de audio"""
        
        if hasattr(audio_file, 'read'):
            # Streamlit UploadedFile
            audio_bytes = audio_file.read()
            
            # Intentar cargar con librosa
            try:
                audio_data, sr = librosa.load(io.BytesIO(audio_bytes), sr=self.sample_rate)
            except:
                # Fallback: intentar como WAV
                with wave.open(io.BytesIO(audio_bytes), 'rb') as wav_file:
                    frames = wav_file.readframes(-1)
                    audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    sr = wav_file.getframerate()
                    
                    # Resample si es necesario
                    if sr != self.sample_rate:
                        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=self.sample_rate)
                        sr = self.sample_rate
        else:
            # Ruta de archivo
            audio_data, sr = librosa.load(audio_file, sr=self.sample_rate)
        
        return audio_data, sr
    
    def _analyze_basic_properties(self, audio_data: np.ndarray, sr: int) -> Dict:
        """Analiza propiedades básicas del audio"""
        
        duration = len(audio_data) / sr
        
        # Análisis de amplitud
        rms_energy = np.sqrt(np.mean(audio_data**2))
        peak_amplitude = np.max(np.abs(audio_data))
        dynamic_range = 20 * np.log10(peak_amplitude / (rms_energy + 1e-10))
        
        # Análisis de silencios
        silence_threshold = 0.01
        silence_mask = np.abs(audio_data) < silence_threshold
        silence_ratio = np.sum(silence_mask) / len(audio_data)
        
        return {
            'duration': duration,
            'sample_rate': sr,
            'num_samples': len(audio_data),
            'rms_energy': float(rms_energy),
            'peak_amplitude': float(peak_amplitude),
            'dynamic_range': float(dynamic_range),
            'silence_ratio': float(silence_ratio)
        }
    
    def _analyze_spectral_features(self, audio_data: np.ndarray, sr: int) -> Dict:
        """Analiza características espectrales"""
        
        # Calcular espectrograma
        stft = librosa.stft(audio_data, hop_length=self.hop_length, n_fft=self.n_fft)
        magnitude = np.abs(stft)
        
        # Centroide espectral
        spectral_centroids = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
        
        # Ancho de banda espectral
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sr)[0]
        
        # Rolloff espectral
        spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sr)[0]
        
        # Flujo espectral
        spectral_flux = np.sum(np.diff(magnitude, axis=1)**2, axis=0)
        
        # Contraste espectral
        spectral_contrast = librosa.feature.spectral_contrast(S=magnitude, sr=sr)
        
        return {
            'spectral_centroid_mean': float(np.mean(spectral_centroids)),
            'spectral_centroid_std': float(np.std(spectral_centroids)),
            'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
            'spectral_flux_mean': float(np.mean(spectral_flux)),
            'spectral_contrast_mean': float(np.mean(spectral_contrast))
        }
    
    def _analyze_temporal_features(self, audio_data: np.ndarray, sr: int) -> Dict:
        """Analiza características temporales"""
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        
        # Tasa de ataque temporal
        onset_frames = librosa.onset.onset_detect(y=audio_data, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        # Tempo
        tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sr)
        
        # Análisis de envolvente
        envelope = np.abs(librosa.stft(audio_data))
        envelope_mean = np.mean(envelope)
        envelope_std = np.std(envelope)
        
        return {
            'zero_crossing_rate_mean': float(np.mean(zcr)),
            'zero_crossing_rate_std': float(np.std(zcr)),
            'onset_density': len(onset_times) / (len(audio_data) / sr),
            'tempo': float(tempo),
            'envelope_mean': float(envelope_mean),
            'envelope_std': float(envelope_std)
        }
    
    def _analyze_frequency_content(self, audio_data: np.ndarray, sr: int) -> Dict:
        """Analiza contenido de frecuencias"""
        
        # FFT para análisis de frecuencias
        fft_data = fft(audio_data)
        freqs = fftfreq(len(fft_data), 1/sr)
        magnitude = np.abs(fft_data)
        
        # Análisis por rangos de frecuencia
        freq_content = {}
        for range_name, (low_freq, high_freq) in self.freq_ranges.items():
            mask = (freqs >= low_freq) & (freqs <= high_freq)
            power = np.sum(magnitude[mask]**2)
            freq_content[range_name] = float(power)
        
        # Normalizar por potencia total
        total_power = sum(freq_content.values())
        if total_power > 0:
            freq_content = {k: v/total_power for k, v in freq_content.items()}
        
        # Frecuencia fundamental
        fundamental_freq = self._estimate_fundamental_frequency(audio_data, sr)
        
        return {
            'frequency_distribution': freq_content,
            'fundamental_frequency': fundamental_freq,
            'frequency_range': (float(np.min(freqs[freqs > 0])), float(np.max(freqs)))
        }
    
    def _estimate_fundamental_frequency(self, audio_data: np.ndarray, sr: int) -> float:
        """Estima la frecuencia fundamental"""
        
        # Usar autocorrelación para estimar pitch
        autocorr = np.correlate(audio_data, audio_data, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Encontrar picos en autocorrelación
        min_period = int(sr / 800)  # 800 Hz máximo
        max_period = int(sr / 50)   # 50 Hz mínimo
        
        if max_period < len(autocorr):
            autocorr_section = autocorr[min_period:max_period]
            if len(autocorr_section) > 0:
                peak_idx = np.argmax(autocorr_section) + min_period
                fundamental_freq = sr / peak_idx
                return float(fundamental_freq)
        
        return 0.0
    
    def _analyze_audio_quality(self, audio_data: np.ndarray, sr: int) -> Dict:
        """Analiza calidad del audio"""
        
        # SNR estimado
        signal_power = np.mean(audio_data**2)
        noise_estimate = np.std(np.diff(audio_data))  # Aproximación simple
        snr = 10 * np.log10(signal_power / (noise_estimate**2 + 1e-10))
        
        # THD estimado (distorsión armónica total)
        # Simplificado: ratio de energía en armónicos vs fundamental
        fundamental = self._estimate_fundamental_frequency(audio_data, sr)
        if fundamental > 0:
            harmonics_power = 0
            for i in range(2, 6):  # 2do a 5to armónico
                harmonic_freq = fundamental * i
                if harmonic_freq < sr/2:
                    # Buscar energía cerca del armónico
                    fft_data = fft(audio_data)
                    freqs = fftfreq(len(fft_data), 1/sr)
                    harmonic_idx = np.argmin(np.abs(freqs - harmonic_freq))
                    window = 5  # Ventana alrededor del armónico
                    start_idx = max(0, harmonic_idx - window)
                    end_idx = min(len(fft_data), harmonic_idx + window)
                    harmonics_power += np.sum(np.abs(fft_data[start_idx:end_idx])**2)
            
            # THD como ratio de armónicos a fundamental
            fundamental_idx = np.argmin(np.abs(freqs - fundamental))
            fundamental_power = np.sum(np.abs(fft_data[fundamental_idx-5:fundamental_idx+5])**2)
            thd = harmonics_power / (fundamental_power + 1e-10)
        else:
            thd = 0.0
        
        # Clipping detection
        clipping_threshold = 0.95
        clipped_samples = np.sum(np.abs(audio_data) > clipping_threshold)
        clipping_ratio = clipped_samples / len(audio_data)
        
        return {
            'estimated_snr': float(snr),
            'estimated_thd': float(thd),
            'clipping_ratio': float(clipping_ratio),
            'overall_quality_score': float(np.clip((snr + 60) / 80, 0, 1))  # Normalizar SNR
        }
    
    def _analyze_content_type(self, audio_data: np.ndarray, sr: int) -> Dict:
        """Analiza tipo de contenido (voz, música, ruido)"""
        
        # Características para clasificación de contenido
        
        # 1. Regularidad espectral (música tiende a ser más regular)
        stft = librosa.stft(audio_data, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        spectral_regularity = 1.0 - np.std(magnitude, axis=1).mean()
        
        # 2. Armonicidad (voz y música tienen más estructura armónica)
        harmonic, percussive = librosa.effects.hpss(audio_data)
        harmonicity = np.sum(harmonic**2) / (np.sum(audio_data**2) + 1e-10)
        
        # 3. Periodicidad (voz tiende a ser periódica)
        autocorr = np.correlate(audio_data, audio_data, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        periodicity = np.max(autocorr[100:1000]) / (np.max(autocorr) + 1e-10)
        
        # 4. Variabilidad temporal (música tiende a variar más)
        rms_frames = librosa.feature.rms(y=audio_data, hop_length=self.hop_length)[0]
        temporal_variability = np.std(rms_frames)
        
        # Clasificación heurística
        content_scores = {
            'voice': harmonicity * periodicity * (1 - temporal_variability),
            'music': spectral_regularity * harmonicity * temporal_variability,
            'noise': (1 - harmonicity) * (1 - periodicity),
            'silence': 1.0 if np.mean(np.abs(audio_data)) < 0.01 else 0.0
        }
        
        # Normalizar
        total_score = sum(content_scores.values())
        if total_score > 0:
            content_scores = {k: v/total_score for k, v in content_scores.items()}
        
        primary_content = max(content_scores.keys(), key=lambda k: content_scores[k])
        
        return {
            'content_scores': content_scores,
            'primary_content': primary_content,
            'confidence': content_scores[primary_content],
            'spectral_regularity': float(spectral_regularity),
            'harmonicity': float(harmonicity),
            'periodicity': float(periodicity)
        }
    
    def _analyze_rhythm_patterns(self, audio_data: np.ndarray, sr: int) -> Dict:
        """Analiza patrones rítmicos"""
        
        # Detección de tempo y beats
        tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sr)
        
        # Intervalos entre beats
        beat_times = librosa.frames_to_time(beats, sr=sr)
        if len(beat_times) > 1:
            beat_intervals = np.diff(beat_times)
            rhythm_regularity = 1.0 - (np.std(beat_intervals) / (np.mean(beat_intervals) + 1e-10))
        else:
            rhythm_regularity = 0.0
        
        # Análisis de onset
        onset_envelope = librosa.onset.onset_strength(y=audio_data, sr=sr)
        onset_times = librosa.onset.onset_detect(y=audio_data, sr=sr, units='time')
        
        # Densidad rítmica
        duration = len(audio_data) / sr
        rhythmic_density = len(onset_times) / duration if duration > 0 else 0
        
        return {
            'tempo': float(tempo),
            'num_beats': len(beats),
            'rhythm_regularity': float(rhythm_regularity),
            'onset_density': float(rhythmic_density),
            'beat_strength': float(np.mean(onset_envelope))
        }
    
    def _extract_emotion_features(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """Extrae características para análisis emocional"""
        
        features = []
        
        # Características espectrales
        stft = librosa.stft(audio_data, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        features.extend([np.mean(mfccs[i]) for i in range(13)])
        features.extend([np.std(mfccs[i]) for i in range(13)])
        
        # Características de pitch
        pitch, _ = librosa.piptrack(y=audio_data, sr=sr)
        pitch_values = pitch[pitch > 0]
        if len(pitch_values) > 0:
            features.extend([
                np.mean(pitch_values),
                np.std(pitch_values),
                np.max(pitch_values),
                np.min(pitch_values)
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Características de energía
        rms = librosa.feature.rms(y=audio_data)[0]
        features.extend([np.mean(rms), np.std(rms)])
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        features.extend([np.mean(zcr), np.std(zcr)])
        
        # Centroide espectral
        spectral_centroids = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
        features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])
        
        # Asegurar 64 características
        while len(features) < 64:
            features.append(0.0)
        
        return np.array(features[:64], dtype=np.float32)
    
    def _analyze_acoustic_emotions(self, audio_data: np.ndarray, sr: int) -> Dict[str, float]:
        """Analiza emociones basado en características acústicas"""
        
        # Análisis de pitch para emociones
        pitch, _ = librosa.piptrack(y=audio_data, sr=sr)
        pitch_values = pitch[pitch > 0]
        
        # Análisis de energía
        rms = librosa.feature.rms(y=audio_data)[0]
        energy_mean = np.mean(rms)
        energy_std = np.std(rms)
        
        # Análisis de tempo
        tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sr)
        
        # Mapeo heurístico a emociones
        emotions = {}
        
        if len(pitch_values) > 0:
            pitch_mean = np.mean(pitch_values)
            pitch_std = np.std(pitch_values)
            
            # Felicidad: pitch alto, energía alta, tempo rápido
            emotions['felicidad'] = min(1.0, (pitch_mean / 500.0) * energy_mean * (tempo / 120.0))
            
            # Tristeza: pitch bajo, energía baja, tempo lento
            emotions['tristeza'] = min(1.0, (1.0 - pitch_mean / 500.0) * (1.0 - energy_mean) * (1.0 - tempo / 120.0))
            
            # Ira: pitch variable, energía alta, tempo rápido
            emotions['ira'] = min(1.0, (pitch_std / 100.0) * energy_mean * (tempo / 120.0))
            
            # Miedo: pitch alto variable, energía media
            emotions['miedo'] = min(1.0, (pitch_std / 100.0) * (pitch_mean / 500.0) * energy_mean)
            
        else:
            # Sin pitch detectado
            emotions['felicidad'] = energy_mean * (tempo / 120.0)
            emotions['tristeza'] = (1.0 - energy_mean) * (1.0 - tempo / 120.0)
            emotions['ira'] = energy_std * (tempo / 120.0)
            emotions['miedo'] = energy_std * energy_mean
        
        # Emociones adicionales
        emotions['sorpresa'] = min(1.0, energy_std * 2.0)
        emotions['disgusto'] = min(1.0, (1.0 - energy_mean) * energy_std)
        emotions['neutral'] = min(1.0, 1.0 - max(emotions.values()) if emotions else 1.0)
        emotions['amor'] = min(1.0, energy_mean * (1.0 - energy_std))
        
        # Normalizar
        max_emotion = max(emotions.values()) if emotions else 1.0
        if max_emotion > 0:
            emotions = {k: v / max_emotion for k, v in emotions.items()}
        
        return emotions
    
    def _analyze_voice_content(self, audio_data: np.ndarray, sr: int) -> Dict:
        """Analiza contenido de voz específicamente"""
        
        # Características específicas de voz
        
        # 1. Formantes (resonancias vocales)
        # Simplificado: buscar picos en espectro
        fft_data = fft(audio_data)
        freqs = fftfreq(len(fft_data), 1/sr)
        magnitude = np.abs(fft_data)
        
        # Buscar formantes en rangos típicos
        formant_ranges = [(300, 800), (800, 2500), (2500, 4000)]  # F1, F2, F3
        formants_detected = 0
        
        for low, high in formant_ranges:
            mask = (freqs >= low) & (freqs <= high)
            if np.any(mask):
                range_magnitude = magnitude[mask]
                if np.max(range_magnitude) > np.mean(magnitude) * 2:
                    formants_detected += 1
        
        voice_probability = formants_detected / 3.0
        
        # 2. Análisis de segmentos de habla
        # Detectar segmentos con energía consistente (característica de habla)
        rms = librosa.feature.rms(y=audio_data, hop_length=self.hop_length)[0]
        speech_threshold = np.mean(rms) * 0.5
        speech_frames = rms > speech_threshold
        
        # Contar segmentos de habla
        speech_segments = 0
        in_speech = False
        for frame in speech_frames:
            if frame and not in_speech:
                speech_segments += 1
                in_speech = True
            elif not frame:
                in_speech = False
        
        # 3. Claridad de voz
        # Basado en regularidad espectral y ausencia de ruido
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
        clarity_score = 1.0 - (np.std(spectral_centroid) / (np.mean(spectral_centroid) + 1e-10))
        
        # 4. Tono emocional
        # Basado en variaciones de pitch y energía
        pitch, _ = librosa.piptrack(y=audio_data, sr=sr)
        pitch_values = pitch[pitch > 0]
        
        if len(pitch_values) > 0:
            pitch_variation = np.std(pitch_values) / (np.mean(pitch_values) + 1e-10)
            emotional_tone = min(1.0, pitch_variation * 2)  # Mayor variación = más emocional
        else:
            emotional_tone = 0.5
        
        return {
            'voice_probability': float(voice_probability),
            'speech_segments': speech_segments,
            'clarity_score': float(np.clip(clarity_score, 0, 1)),
            'emotional_tone': float(emotional_tone),
            'formants_detected': formants_detected
        }
    
    def _extract_mel_features(self, audio_data: np.ndarray, sr: int) -> np.ndarray:
        """Extrae características mel-spectrogram"""
        
        # Calcular mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data, 
            sr=sr, 
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        
        # Convertir a dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Transponer para tener tiempo en primera dimensión
        mel_features = mel_spec_db.T
        
        return mel_features
    
    def _get_timestamp(self) -> str:
        """Obtiene timestamp actual"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _update_processing_stats(self, analysis: Dict):
        """Actualiza estadísticas de procesamiento"""
        
        self.processing_stats['audios_processed'] += 1
        
        # Duración promedio
        duration = analysis.get('basic_properties', {}).get('duration', 0)
        total_processed = self.processing_stats['audios_processed']
        
        if total_processed == 1:
            self.processing_stats['avg_duration'] = duration
        else:
            self.processing_stats['avg_duration'] = (
                (self.processing_stats['avg_duration'] * (total_processed - 1) + duration) / total_processed
            )
        
        # Calidad promedio
        quality_score = analysis.get('quality_metrics', {}).get('overall_quality_score', 0.5)
        self.processing_stats['audio_quality_scores'].append(quality_score)
        
        # Mantener solo las últimas 100 puntuaciones
        if len(self.processing_stats['audio_quality_scores']) > 100:
            self.processing_stats['audio_quality_scores'] = self.processing_stats['audio_quality_scores'][-100:]
    
    def get_processing_statistics(self) -> Dict:
        """Obtiene estadísticas de procesamiento"""
        
        stats = self.processing_stats.copy()
        
        # Calcular calidad promedio
        if stats['audio_quality_scores']:
            stats['avg_quality_score'] = np.mean(stats['audio_quality_scores'])
        else:
            stats['avg_quality_score'] = 0.0
        
        return stats
    
    def check_health(self) -> bool:
        """Verifica el estado de salud del procesador de audio"""
        
        try:
            # Crear audio de prueba (1 segundo de tono)
            duration = 1.0  # segundos
            freq = 440  # Hz (La4)
            t = np.linspace(0, duration, int(self.sample_rate * duration))
            test_audio = np.sin(2 * np.pi * freq * t) * 0.5
            
            # Test de análisis básico
            basic_analysis = self._analyze_basic_properties(test_audio, self.sample_rate)
            if not basic_analysis or 'duration' not in basic_analysis:
                return False
            
            # Test de extracción de características
            mel_features = self._extract_mel_features(test_audio, self.sample_rate)
            if mel_features.size == 0:
                return False
            
            # Test de RNN
            with torch.no_grad():
                features_tensor = torch.tensor(mel_features, dtype=torch.float32).unsqueeze(0)
                output, _, _ = self.audio_rnn(features_tensor)
                if output.numel() == 0:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.log("ERROR", f"Audio processor health check failed: {str(e)}")
            return False
