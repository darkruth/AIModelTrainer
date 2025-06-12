"""
Moduladores de Sistema - Modulación dinámica de procesos cognitivos
Sistema de modulación adaptativa para control de intensidad y flujo neural
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
import math
from collections import defaultdict, deque

class ModulationType(Enum):
    """Tipos de modulación disponibles"""
    AMPLITUDE = "amplitude"          # Modulación de amplitud
    FREQUENCY = "frequency"          # Modulación de frecuencia  
    PHASE = "phase"                 # Modulación de fase
    ATTENTION = "attention"         # Modulación atencional
    EMOTIONAL = "emotional"         # Modulación emocional
    TEMPORAL = "temporal"           # Modulación temporal
    CONTEXTUAL = "contextual"       # Modulación contextual
    ADAPTIVE = "adaptive"           # Modulación adaptativa

class ModulationIntensity(Enum):
    """Intensidades de modulación"""
    MINIMAL = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    MAXIMUM = 0.9

@dataclass
class ModulationConfig:
    """Configuración de un modulador"""
    modulation_type: ModulationType
    intensity: float = 0.5
    frequency: float = 1.0  # Hz
    phase_offset: float = 0.0
    decay_rate: float = 0.95
    adaptation_rate: float = 0.01
    target_modules: List[str] = field(default_factory=list)
    activation_threshold: float = 0.1
    max_modulation_depth: float = 0.8

class BaseModulator:
    """Modulador base para todos los tipos de modulación"""
    
    def __init__(self, config: ModulationConfig, name: str):
        self.config = config
        self.name = name
        self.is_active = False
        self.current_intensity = 0.0
        self.phase = 0.0
        self.adaptation_history = deque(maxlen=100)
        self.modulation_history = deque(maxlen=1000)
        self.lock = threading.Lock()
        
    def activate(self):
        """Activa el modulador"""
        with self.lock:
            self.is_active = True
            
    def deactivate(self):
        """Desactiva el modulador"""
        with self.lock:
            self.is_active = False
            
    def modulate(self, input_tensor: torch.Tensor, context: Dict[str, Any] = None) -> torch.Tensor:
        """Aplica modulación al tensor de entrada"""
        if not self.is_active:
            return input_tensor
            
        with self.lock:
            # Calcular intensidad actual
            current_time = time.time()
            self._update_phase(current_time)
            self._update_intensity(context)
            
            # Aplicar modulación específica del tipo
            modulated_tensor = self._apply_modulation(input_tensor, context)
            
            # Registrar modulación
            self._record_modulation(input_tensor, modulated_tensor, current_time)
            
            return modulated_tensor
    
    def _update_phase(self, current_time: float):
        """Actualiza la fase de modulación"""
        self.phase = (current_time * self.config.frequency * 2 * math.pi + 
                     self.config.phase_offset) % (2 * math.pi)
    
    def _update_intensity(self, context: Dict[str, Any] = None):
        """Actualiza la intensidad de modulación"""
        base_intensity = self.config.intensity
        
        # Modulación sinusoidal basada en fase
        phase_modulation = math.sin(self.phase) * 0.1
        
        # Adaptación basada en contexto
        context_adaptation = self._calculate_context_adaptation(context)
        
        # Aplicar decay si no hay activación reciente
        if len(self.adaptation_history) > 0:
            last_adaptation = self.adaptation_history[-1]['timestamp']
            time_since_last = time.time() - last_adaptation
            decay_factor = self.config.decay_rate ** time_since_last
        else:
            decay_factor = 1.0
            
        self.current_intensity = min(
            base_intensity + phase_modulation + context_adaptation,
            self.config.max_modulation_depth
        ) * decay_factor
        
    def _calculate_context_adaptation(self, context: Dict[str, Any] = None) -> float:
        """Calcula adaptación basada en contexto"""
        if not context:
            return 0.0
            
        adaptation = 0.0
        
        # Adaptación por nivel de consciencia
        if 'consciousness_level' in context:
            consciousness_factor = context['consciousness_level'] * 0.2
            adaptation += consciousness_factor
            
        # Adaptación por nivel emocional
        if 'emotion_level' in context:
            emotion_factor = context['emotion_level'] * 0.1
            adaptation += emotion_factor
            
        # Adaptación por actividad neural
        if 'neural_activity' in context:
            activity_factor = context['neural_activity'] * 0.15
            adaptation += activity_factor
            
        return adaptation * self.config.adaptation_rate
    
    def _apply_modulation(self, input_tensor: torch.Tensor, context: Dict[str, Any] = None) -> torch.Tensor:
        """Método abstracto para aplicar modulación específica"""
        raise NotImplementedError("Subclases deben implementar _apply_modulation")
    
    def _record_modulation(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor, timestamp: float):
        """Registra información de la modulación"""
        modulation_effect = torch.norm(output_tensor - input_tensor).item()
        
        record = {
            'timestamp': timestamp,
            'intensity': self.current_intensity,
            'phase': self.phase,
            'effect_magnitude': modulation_effect,
            'input_norm': torch.norm(input_tensor).item(),
            'output_norm': torch.norm(output_tensor).item()
        }
        
        self.modulation_history.append(record)
        
        # Registrar adaptación si el efecto es significativo
        if modulation_effect > self.config.activation_threshold:
            self.adaptation_history.append({
                'timestamp': timestamp,
                'effect': modulation_effect,
                'intensity': self.current_intensity
            })

class AmplitudeModulator(BaseModulator):
    """Modulador de amplitud - ajusta la magnitud de activaciones"""
    
    def _apply_modulation(self, input_tensor: torch.Tensor, context: Dict[str, Any] = None) -> torch.Tensor:
        # Modulación de amplitud multiplicativa
        amplitude_factor = 1.0 + (self.current_intensity * math.sin(self.phase))
        return input_tensor * amplitude_factor

class FrequencyModulator(BaseModulator):
    """Modulador de frecuencia - ajusta patrones temporales"""
    
    def __init__(self, config: ModulationConfig, name: str):
        super().__init__(config, name)
        self.temporal_buffer = deque(maxlen=50)
        
    def _apply_modulation(self, input_tensor: torch.Tensor, context: Dict[str, Any] = None) -> torch.Tensor:
        # Almacenar tensor en buffer temporal
        self.temporal_buffer.append(input_tensor.clone())
        
        if len(self.temporal_buffer) < 2:
            return input_tensor
            
        # Crear modulación basada en frecuencia temporal
        freq_factor = 1.0 + self.current_intensity * math.cos(self.phase * self.config.frequency)
        
        # Interpolar entre estados temporales
        if len(self.temporal_buffer) >= 3:
            prev_tensor = self.temporal_buffer[-2]
            interpolation_weight = (freq_factor + 1.0) / 2.0
            modulated = input_tensor * interpolation_weight + prev_tensor * (1.0 - interpolation_weight)
        else:
            modulated = input_tensor * freq_factor
            
        return modulated

class PhaseModulator(BaseModulator):
    """Modulador de fase - ajusta relaciones temporales"""
    
    def _apply_modulation(self, input_tensor: torch.Tensor, context: Dict[str, Any] = None) -> torch.Tensor:
        # Aplicar desplazamiento de fase a través de rotación de características
        phase_shift = self.current_intensity * self.phase
        
        # Crear matriz de rotación
        dim = input_tensor.shape[-1]
        if dim % 2 == 0:
            # Rotación par de características
            rotated = input_tensor.clone()
            for i in range(0, dim, 2):
                if i + 1 < dim:
                    cos_shift = math.cos(phase_shift)
                    sin_shift = math.sin(phase_shift)
                    
                    x = input_tensor[..., i]
                    y = input_tensor[..., i + 1]
                    
                    rotated[..., i] = x * cos_shift - y * sin_shift
                    rotated[..., i + 1] = x * sin_shift + y * cos_shift
            
            return rotated
        else:
            # Para dimensiones impares, aplicar modulación simple
            return input_tensor * (1.0 + self.current_intensity * math.sin(phase_shift))

class AttentionModulator(BaseModulator):
    """Modulador atencional - ajusta patrones de atención"""
    
    def __init__(self, config: ModulationConfig, name: str):
        super().__init__(config, name)
        self.attention_weights = None
        
    def _apply_modulation(self, input_tensor: torch.Tensor, context: Dict[str, Any] = None) -> torch.Tensor:
        # Generar pesos de atención dinámicos
        if self.attention_weights is None or self.attention_weights.shape != input_tensor.shape:
            self.attention_weights = torch.randn_like(input_tensor)
            
        # Normalizar pesos de atención
        attention_normalized = F.softmax(self.attention_weights.flatten(), dim=0)
        attention_reshaped = attention_normalized.view_as(input_tensor)
        
        # Aplicar modulación atencional
        attention_factor = 1.0 + self.current_intensity * math.sin(self.phase)
        modulated_attention = attention_reshaped * attention_factor
        
        return input_tensor * modulated_attention

class EmotionalModulator(BaseModulator):
    """Modulador emocional - ajusta procesamiento según estado emocional"""
    
    def _apply_modulation(self, input_tensor: torch.Tensor, context: Dict[str, Any] = None) -> torch.Tensor:
        # Obtener estado emocional del contexto
        emotion_valence = context.get('emotion_valence', 0.0) if context else 0.0
        emotion_arousal = context.get('emotion_arousal', 0.5) if context else 0.5
        
        # Modulación basada en valencia emocional
        valence_factor = 1.0 + (emotion_valence * self.current_intensity * 0.5)
        
        # Modulación basada en arousal emocional
        arousal_factor = 1.0 + (emotion_arousal * self.current_intensity * 0.3)
        
        # Combinar factores emocionales
        emotional_factor = valence_factor * arousal_factor
        
        return input_tensor * emotional_factor

class TemporalModulator(BaseModulator):
    """Modulador temporal - ajusta dinámicas temporales"""
    
    def __init__(self, config: ModulationConfig, name: str):
        super().__init__(config, name)
        self.temporal_memory = deque(maxlen=20)
        
    def _apply_modulation(self, input_tensor: torch.Tensor, context: Dict[str, Any] = None) -> torch.Tensor:
        current_time = time.time()
        
        # Almacenar en memoria temporal
        self.temporal_memory.append({
            'tensor': input_tensor.clone(),
            'timestamp': current_time
        })
        
        if len(self.temporal_memory) < 3:
            return input_tensor
            
        # Calcular derivada temporal
        recent_tensors = [mem['tensor'] for mem in list(self.temporal_memory)[-3:]]
        temporal_gradient = recent_tensors[-1] - recent_tensors[0]
        
        # Aplicar modulación temporal
        temporal_factor = self.current_intensity * math.tanh(torch.norm(temporal_gradient).item())
        modulation = temporal_gradient * temporal_factor
        
        return input_tensor + modulation

class ContextualModulator(BaseModulator):
    """Modulador contextual - ajusta procesamiento según contexto global"""
    
    def _apply_modulation(self, input_tensor: torch.Tensor, context: Dict[str, Any] = None) -> torch.Tensor:
        if not context:
            return input_tensor
            
        # Calcular factor contextual basado en múltiples variables
        contextual_factors = []
        
        # Factor de actividad neural
        if 'neural_activity' in context:
            neural_factor = context['neural_activity'] * 0.3
            contextual_factors.append(neural_factor)
            
        # Factor de coherencia
        if 'coherence_level' in context:
            coherence_factor = context['coherence_level'] * 0.4
            contextual_factors.append(coherence_factor)
            
        # Factor de complejidad de tarea
        if 'task_complexity' in context:
            complexity_factor = context['task_complexity'] * 0.2
            contextual_factors.append(complexity_factor)
            
        # Factor de urgencia
        if 'urgency_level' in context:
            urgency_factor = context['urgency_level'] * 0.1
            contextual_factors.append(urgency_factor)
            
        # Combinar factores contextuales
        if contextual_factors:
            combined_factor = sum(contextual_factors) / len(contextual_factors)
            modulation_strength = self.current_intensity * combined_factor
            return input_tensor * (1.0 + modulation_strength)
        
        return input_tensor

class AdaptiveModulator(BaseModulator):
    """Modulador adaptativo - aprende patrones óptimos de modulación"""
    
    def __init__(self, config: ModulationConfig, name: str):
        super().__init__(config, name)
        self.performance_history = deque(maxlen=500)
        self.adaptation_weights = torch.randn(10) * 0.1
        self.learning_rate = 0.001
        
    def _apply_modulation(self, input_tensor: torch.Tensor, context: Dict[str, Any] = None) -> torch.Tensor:
        # Extraer características del contexto y tensor
        features = self._extract_features(input_tensor, context)
        
        # Calcular modulación adaptativa
        adaptive_signal = torch.dot(features, self.adaptation_weights).item()
        adaptive_factor = 1.0 + self.current_intensity * math.tanh(adaptive_signal)
        
        modulated_tensor = input_tensor * adaptive_factor
        
        # Aprender de la modulación si hay información de performance
        if context and 'performance_feedback' in context:
            self._update_adaptation_weights(features, context['performance_feedback'])
            
        return modulated_tensor
    
    def _extract_features(self, input_tensor: torch.Tensor, context: Dict[str, Any] = None) -> torch.Tensor:
        """Extrae características para adaptación"""
        features = torch.zeros(10)
        
        # Características del tensor
        features[0] = torch.norm(input_tensor).item()
        features[1] = torch.mean(input_tensor).item()
        features[2] = torch.std(input_tensor).item()
        features[3] = torch.max(input_tensor).item()
        features[4] = torch.min(input_tensor).item()
        
        # Características del contexto
        if context:
            features[5] = context.get('consciousness_level', 0.0)
            features[6] = context.get('emotion_level', 0.0)
            features[7] = context.get('neural_activity', 0.0)
            features[8] = context.get('coherence_level', 0.0)
            features[9] = context.get('task_complexity', 0.0)
            
        return features
    
    def _update_adaptation_weights(self, features: torch.Tensor, performance_feedback: float):
        """Actualiza pesos adaptativos basado en feedback"""
        # Gradiente simple para aprendizaje
        prediction = torch.dot(features, self.adaptation_weights).item()
        error = performance_feedback - prediction
        
        # Actualización de pesos
        gradient = features * error * self.learning_rate
        self.adaptation_weights += gradient
        
        # Registrar performance
        self.performance_history.append({
            'timestamp': time.time(),
            'features': features.clone(),
            'feedback': performance_feedback,
            'error': error
        })

class ModulationManager:
    """Gestor central de todos los moduladores del sistema"""
    
    def __init__(self):
        self.modulators: Dict[str, BaseModulator] = {}
        self.active_modulators: List[str] = []
        self.global_context = {}
        self.modulation_statistics = {
            'total_modulations': 0,
            'average_effect': 0.0,
            'most_active_modulator': None
        }
        self.lock = threading.Lock()
        
    def register_modulator(self, name: str, modulator: BaseModulator):
        """Registra un nuevo modulador"""
        with self.lock:
            self.modulators[name] = modulator
            
    def activate_modulator(self, name: str):
        """Activa un modulador específico"""
        with self.lock:
            if name in self.modulators:
                self.modulators[name].activate()
                if name not in self.active_modulators:
                    self.active_modulators.append(name)
                    
    def deactivate_modulator(self, name: str):
        """Desactiva un modulador específico"""
        with self.lock:
            if name in self.modulators:
                self.modulators[name].deactivate()
                if name in self.active_modulators:
                    self.active_modulators.remove(name)
                    
    def update_global_context(self, context: Dict[str, Any]):
        """Actualiza el contexto global para todos los moduladores"""
        with self.lock:
            self.global_context.update(context)
            
    def apply_modulation(self, 
                        input_tensor: torch.Tensor, 
                        target_modules: Optional[List[str]] = None,
                        additional_context: Dict[str, Any] = None) -> torch.Tensor:
        """Aplica modulación de todos los moduladores activos relevantes"""
        
        # Combinar contexto global y adicional
        combined_context = self.global_context.copy()
        if additional_context:
            combined_context.update(additional_context)
            
        # Determinar moduladores a aplicar
        modulators_to_apply = target_modules or self.active_modulators
        
        result_tensor = input_tensor.clone()
        applied_modulators = []
        
        for modulator_name in modulators_to_apply:
            if modulator_name in self.modulators and modulator_name in self.active_modulators:
                modulator = self.modulators[modulator_name]
                result_tensor = modulator.modulate(result_tensor, combined_context)
                applied_modulators.append(modulator_name)
                
        # Actualizar estadísticas
        with self.lock:
            self.modulation_statistics['total_modulations'] += len(applied_modulators)
            if applied_modulators:
                effect_magnitude = torch.norm(result_tensor - input_tensor).item()
                current_avg = self.modulation_statistics['average_effect']
                total_count = self.modulation_statistics['total_modulations']
                self.modulation_statistics['average_effect'] = (
                    (current_avg * (total_count - len(applied_modulators)) + effect_magnitude) / total_count
                )
                
        return result_tensor
    
    def get_modulator_status(self) -> Dict[str, Any]:
        """Obtiene estado de todos los moduladores"""
        status = {
            'total_modulators': len(self.modulators),
            'active_modulators': len(self.active_modulators),
            'modulators': {},
            'statistics': self.modulation_statistics.copy()
        }
        
        for name, modulator in self.modulators.items():
            status['modulators'][name] = {
                'is_active': modulator.is_active,
                'current_intensity': modulator.current_intensity,
                'phase': modulator.phase,
                'type': modulator.config.modulation_type.value,
                'modulation_count': len(modulator.modulation_history),
                'adaptation_count': len(modulator.adaptation_history)
            }
            
        return status
    
    def optimize_modulators(self):
        """Optimiza parámetros de moduladores basado en historial"""
        for modulator in self.modulators.values():
            if len(modulator.modulation_history) > 50:
                # Análisis de efectividad
                recent_effects = [
                    m['effect_magnitude'] 
                    for m in list(modulator.modulation_history)[-50:]
                ]
                
                avg_effect = np.mean(recent_effects)
                
                # Ajustar intensidad basado en efectividad
                if avg_effect < 0.1:  # Poco efecto
                    modulator.config.intensity = min(modulator.config.intensity * 1.1, 0.9)
                elif avg_effect > 0.5:  # Mucho efecto
                    modulator.config.intensity = max(modulator.config.intensity * 0.9, 0.1)

# Instancia global del gestor de modulación
modulation_manager = ModulationManager()

def initialize_modulation_system():
    """Inicializa el sistema completo de modulación"""
    
    # Crear configuraciones por defecto
    configs = {
        'amplitude_mod': ModulationConfig(
            ModulationType.AMPLITUDE,
            intensity=0.3,
            frequency=1.0
        ),
        'frequency_mod': ModulationConfig(
            ModulationType.FREQUENCY,
            intensity=0.4,
            frequency=0.5
        ),
        'attention_mod': ModulationConfig(
            ModulationType.ATTENTION,
            intensity=0.6,
            frequency=2.0
        ),
        'emotional_mod': ModulationConfig(
            ModulationType.EMOTIONAL,
            intensity=0.5,
            frequency=0.3
        ),
        'temporal_mod': ModulationConfig(
            ModulationType.TEMPORAL,
            intensity=0.4,
            frequency=1.5
        ),
        'contextual_mod': ModulationConfig(
            ModulationType.CONTEXTUAL,
            intensity=0.7,
            frequency=0.8
        ),
        'adaptive_mod': ModulationConfig(
            ModulationType.ADAPTIVE,
            intensity=0.5,
            frequency=1.0
        )
    }
    
    # Crear y registrar moduladores
    modulators = {
        'amplitude_mod': AmplitudeModulator(configs['amplitude_mod'], 'amplitude_mod'),
        'frequency_mod': FrequencyModulator(configs['frequency_mod'], 'frequency_mod'),
        'attention_mod': AttentionModulator(configs['attention_mod'], 'attention_mod'),
        'emotional_mod': EmotionalModulator(configs['emotional_mod'], 'emotional_mod'),
        'temporal_mod': TemporalModulator(configs['temporal_mod'], 'temporal_mod'),
        'contextual_mod': ContextualModulator(configs['contextual_mod'], 'contextual_mod'),
        'adaptive_mod': AdaptiveModulator(configs['adaptive_mod'], 'adaptive_mod')
    }
    
    for name, modulator in modulators.items():
        modulation_manager.register_modulator(name, modulator)
        
    # Activar moduladores esenciales
    essential_modulators = ['attention_mod', 'emotional_mod', 'contextual_mod', 'adaptive_mod']
    for mod_name in essential_modulators:
        modulation_manager.activate_modulator(mod_name)
        
    return modulation_manager

def get_modulation_manager() -> ModulationManager:
    """Obtiene la instancia global del gestor de modulación"""
    return modulation_manager

def apply_system_modulation(tensor: torch.Tensor, context: Dict[str, Any] = None) -> torch.Tensor:
    """Función de conveniencia para aplicar modulación del sistema"""
    return modulation_manager.apply_modulation(tensor, additional_context=context)