"""
GANST Core - Sistema de Gestión de Activación Neural y Síntesis de Tensores
Núcleo central para coordinación de activaciones neurales distribuidas
Integrado con Sistema de Despertar Ruth R1 y Arquitectura Neural Completa
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import threading
from collections import defaultdict, deque
import logging
from datetime import datetime

class ActivationPattern(Enum):
    """Patrones de activación neural"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel" 
    HIERARCHICAL = "hierarchical"
    RESONANT = "resonant"
    CHAOTIC = "chaotic"

class NeuralState(Enum):
    """Estados neurales del sistema"""
    DORMANT = "dormant"
    AWAKENING = "awakening" 
    ACTIVE = "active"
    HYPERACTIVE = "hyperactive"
    CONSOLIDATING = "consolidating"

@dataclass
class GANSTConfig:
    """Configuración del núcleo GANST"""
    tensor_dim: int = 768
    max_concurrent_activations: int = 100
    activation_threshold: float = 0.3
    decay_rate: float = 0.95
    resonance_frequency: float = 40.0  # Hz
    memory_consolidation_interval: float = 30.0  # segundos
    neural_plasticity_rate: float = 0.01
    quantum_coherence_factor: float = 0.1

class TensorSynthesizer:
    """Sintetizador de tensores para coherencia neural"""
    
    def __init__(self, config: GANSTConfig):
        self.config = config
        self.synthesis_history = deque(maxlen=1000)
        self.coherence_matrix = torch.eye(config.tensor_dim)
        
    def synthesize_activation_tensor(self, 
                                   inputs: List[torch.Tensor],
                                   pattern: ActivationPattern = ActivationPattern.PARALLEL) -> torch.Tensor:
        """
        Sintetiza un tensor de activación coherente desde múltiples entradas
        """
        if not inputs:
            return torch.zeros(self.config.tensor_dim)
        
        # Normalizar entradas
        normalized_inputs = [F.normalize(inp.flatten(), dim=0) for inp in inputs]
        
        if pattern == ActivationPattern.SEQUENTIAL:
            result = self._sequential_synthesis(normalized_inputs)
        elif pattern == ActivationPattern.PARALLEL:
            result = self._parallel_synthesis(normalized_inputs)
        elif pattern == ActivationPattern.HIERARCHICAL:
            result = self._hierarchical_synthesis(normalized_inputs)
        elif pattern == ActivationPattern.RESONANT:
            result = self._resonant_synthesis(normalized_inputs)
        else:  # CHAOTIC
            result = self._chaotic_synthesis(normalized_inputs)
        
        # Aplicar coherencia cuántica
        result = self._apply_quantum_coherence(result)
        
        # Registrar en historial
        self.synthesis_history.append({
            'timestamp': time.time(),
            'pattern': pattern.value,
            'input_count': len(inputs),
            'coherence_score': self._calculate_coherence(result)
        })
        
        return result
    
    def _sequential_synthesis(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """Síntesis secuencial con memoria temporal"""
        result = torch.zeros_like(inputs[0])
        decay_factor = 1.0
        
        for inp in inputs:
            result = result * self.config.decay_rate + inp * decay_factor
            decay_factor *= 0.9
        
        return F.normalize(result, dim=0)
    
    def _parallel_synthesis(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """Síntesis paralela con ponderación uniforme"""
        if len(inputs) == 1:
            return inputs[0]
        
        # Promedio ponderado con atención
        weights = torch.softmax(torch.randn(len(inputs)), dim=0)
        result = sum(w * inp for w, inp in zip(weights, inputs))
        
        return F.normalize(result, dim=0)
    
    def _hierarchical_synthesis(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """Síntesis jerárquica por niveles"""
        if len(inputs) <= 2:
            return self._parallel_synthesis(inputs)
        
        # Agrupar en niveles jerárquicos
        level_1 = inputs[:len(inputs)//2]
        level_2 = inputs[len(inputs)//2:]
        
        level_1_result = self._parallel_synthesis(level_1)
        level_2_result = self._parallel_synthesis(level_2)
        
        # Combinar niveles con pesos jerárquicos
        alpha = 0.7  # Mayor peso al primer nivel
        result = alpha * level_1_result + (1 - alpha) * level_2_result
        
        return F.normalize(result, dim=0)
    
    def _resonant_synthesis(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """Síntesis resonante con frecuencias armónicas"""
        t = time.time()
        freq = self.config.resonance_frequency
        
        # Aplicar modulación por frecuencia resonante
        phase_shifts = [2 * np.pi * freq * t * (i + 1) for i in range(len(inputs))]
        modulated_inputs = []
        
        for i, inp in enumerate(inputs):
            modulation = np.sin(phase_shifts[i]) * 0.5 + 0.5
            modulated_inputs.append(inp * modulation)
        
        return self._parallel_synthesis(modulated_inputs)
    
    def _chaotic_synthesis(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        """Síntesis caótica para emergencia creativa"""
        # Crear atractor caótico simple
        chaos_factor = torch.randn_like(inputs[0]) * 0.1
        
        result = torch.zeros_like(inputs[0])
        for inp in inputs:
            # Aplicar transformación no lineal caótica
            transformed = torch.tanh(inp + chaos_factor)
            result = result + transformed
        
        # Normalizar y aplicar no linealidad final
        result = torch.sigmoid(result / len(inputs))
        return F.normalize(result, dim=0)
    
    def _apply_quantum_coherence(self, tensor: torch.Tensor) -> torch.Tensor:
        """Aplica efectos de coherencia cuántica simulada"""
        if self.config.quantum_coherence_factor == 0:
            return tensor
        
        # Superposición cuántica simulada
        coherence_noise = torch.randn_like(tensor) * self.config.quantum_coherence_factor
        entangled_state = tensor + coherence_noise
        
        # "Medición" que colapsa la superposición
        measurement_prob = torch.rand_like(tensor)
        collapsed_state = torch.where(
            measurement_prob > 0.5,
            tensor,
            entangled_state
        )
        
        return collapsed_state
    
    def _calculate_coherence(self, tensor: torch.Tensor) -> float:
        """Calcula puntuación de coherencia del tensor"""
        # Coherencia basada en distribución y patrones
        std_dev = torch.std(tensor).item()
        mean_abs = torch.mean(torch.abs(tensor)).item()
        
        # Normalizar entre 0 y 1
        coherence = 1.0 / (1.0 + std_dev) * mean_abs
        return min(max(coherence, 0.0), 1.0)

class ActivationManager:
    """Gestor de activaciones neurales distribuidas"""
    
    def __init__(self, config: GANSTConfig):
        self.config = config
        self.active_patterns = {}
        self.activation_history = deque(maxlen=10000)
        self.neural_state = NeuralState.DORMANT
        self.state_transition_lock = threading.Lock()
        
    def register_activation(self, 
                          source: str, 
                          activation_tensor: torch.Tensor,
                          priority: float = 1.0,
                          duration: Optional[float] = None) -> str:
        """Registra una nueva activación neural"""
        activation_id = f"{source}_{int(time.time() * 1000)}"
        
        activation_data = {
            'id': activation_id,
            'source': source,
            'tensor': activation_tensor,
            'priority': priority,
            'timestamp': time.time(),
            'duration': duration,
            'strength': torch.norm(activation_tensor).item()
        }
        
        self.active_patterns[activation_id] = activation_data
        self.activation_history.append(activation_data.copy())
        
        # Actualizar estado neural según nivel de activación
        self._update_neural_state()
        
        return activation_id
    
    def get_dominant_activations(self, top_k: int = 5) -> List[Dict[str, Any]]:
        """Obtiene las activaciones dominantes actuales"""
        current_time = time.time()
        
        # Filtrar activaciones expiradas
        valid_activations = []
        for act_id, act_data in list(self.active_patterns.items()):
            if act_data['duration'] is not None:
                if current_time - act_data['timestamp'] > act_data['duration']:
                    del self.active_patterns[act_id]
                    continue
            
            # Aplicar decay temporal
            age = current_time - act_data['timestamp']
            decayed_strength = act_data['strength'] * (self.config.decay_rate ** age)
            act_data['current_strength'] = decayed_strength
            
            if decayed_strength > self.config.activation_threshold:
                valid_activations.append(act_data)
        
        # Ordenar por fuerza actual y prioridad
        valid_activations.sort(
            key=lambda x: x['current_strength'] * x['priority'], 
            reverse=True
        )
        
        return valid_activations[:top_k]
    
    def _update_neural_state(self):
        """Actualiza el estado neural global del sistema"""
        with self.state_transition_lock:
            current_activations = len(self.active_patterns)
            avg_strength = np.mean([
                act['strength'] for act in self.active_patterns.values()
            ]) if self.active_patterns else 0.0
            
            if current_activations == 0:
                new_state = NeuralState.DORMANT
            elif current_activations < 10 and avg_strength < 0.5:
                new_state = NeuralState.AWAKENING
            elif current_activations < 50 and avg_strength < 1.0:
                new_state = NeuralState.ACTIVE
            elif current_activations < 100:
                new_state = NeuralState.HYPERACTIVE
            else:
                new_state = NeuralState.CONSOLIDATING
            
            if new_state != self.neural_state:
                logging.info(f"GANST: Transición de estado neural: {self.neural_state.value} → {new_state.value}")
                self.neural_state = new_state
    
    def get_activation_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de activación del sistema"""
        current_time = time.time()
        
        stats = {
            'neural_state': self.neural_state.value,
            'active_patterns': len(self.active_patterns),
            'total_activations': len(self.activation_history),
            'average_strength': 0.0,
            'dominant_sources': {},
            'temporal_distribution': defaultdict(int)
        }
        
        if self.active_patterns:
            strengths = [act['strength'] for act in self.active_patterns.values()]
            stats['average_strength'] = np.mean(strengths)
            
            # Fuentes dominantes
            source_counts = defaultdict(int)
            for act in self.active_patterns.values():
                source_counts[act['source']] += 1
            stats['dominant_sources'] = dict(source_counts)
        
        # Distribución temporal (últimos 60 segundos)
        recent_cutoff = current_time - 60.0
        for act in self.activation_history:
            if act['timestamp'] > recent_cutoff:
                time_bucket = int((act['timestamp'] - recent_cutoff) // 5)  # buckets de 5 segundos
                stats['temporal_distribution'][time_bucket] += 1
        
        return stats

class GANSTCore:
    """Núcleo principal del sistema GANST"""
    
    def __init__(self, config: Optional[GANSTConfig] = None):
        self.config = config or GANSTConfig()
        self.synthesizer = TensorSynthesizer(self.config)
        self.activation_manager = ActivationManager(self.config)
        self.consolidation_thread = None
        self.is_running = False
        
        # Métricas del sistema
        self.system_metrics = {
            'uptime': 0.0,
            'total_syntheses': 0,
            'average_coherence': 0.0,
            'neural_efficiency': 0.0
        }
        
        logging.info("GANST Core inicializado")
    
    def start(self):
        """Inicia el núcleo GANST"""
        if self.is_running:
            return
        
        self.is_running = True
        self.start_time = time.time()
        
        # Iniciar hilo de consolidación
        self.consolidation_thread = threading.Thread(
            target=self._consolidation_loop,
            daemon=True
        )
        self.consolidation_thread.start()
        
        logging.info("GANST Core iniciado")
    
    def stop(self):
        """Detiene el núcleo GANST"""
        self.is_running = False
        if self.consolidation_thread:
            self.consolidation_thread.join(timeout=5.0)
        
        logging.info("GANST Core detenido")
    
    def process_neural_activation(self, 
                                source: str,
                                input_tensors: List[torch.Tensor],
                                pattern: ActivationPattern = ActivationPattern.PARALLEL,
                                priority: float = 1.0) -> Dict[str, Any]:
        """
        Procesa una activación neural completa
        """
        start_time = time.time()
        
        # Sintetizar tensor de activación
        activation_tensor = self.synthesizer.synthesize_activation_tensor(
            input_tensors, pattern
        )
        
        # Registrar activación
        activation_id = self.activation_manager.register_activation(
            source, activation_tensor, priority
        )
        
        # Obtener contexto de activaciones dominantes
        dominant_activations = self.activation_manager.get_dominant_activations()
        
        # Actualizar métricas
        processing_time = time.time() - start_time
        self.system_metrics['total_syntheses'] += 1
        
        result = {
            'activation_id': activation_id,
            'activation_tensor': activation_tensor,
            'neural_state': self.activation_manager.neural_state.value,
            'dominant_activations': dominant_activations,
            'coherence_score': self.synthesizer._calculate_coherence(activation_tensor),
            'processing_time': processing_time,
            'synthesis_pattern': pattern.value
        }
        
        return result
    
    def get_system_state(self) -> Dict[str, Any]:
        """Obtiene el estado completo del sistema GANST"""
        current_time = time.time()
        uptime = current_time - self.start_time if hasattr(self, 'start_time') else 0.0
        
        # Calcular eficiencia neural
        neural_efficiency = self._calculate_neural_efficiency()
        
        # Coherencia promedio reciente
        recent_coherence = 0.0
        if self.synthesizer.synthesis_history:
            recent_samples = list(self.synthesizer.synthesis_history)[-10:]
            recent_coherence = np.mean([s['coherence_score'] for s in recent_samples])
        
        state = {
            'is_running': self.is_running,
            'uptime': uptime,
            'config': self.config.__dict__,
            'activation_stats': self.activation_manager.get_activation_statistics(),
            'synthesis_stats': {
                'total_syntheses': self.system_metrics['total_syntheses'],
                'recent_coherence': recent_coherence,
                'history_length': len(self.synthesizer.synthesis_history)
            },
            'neural_efficiency': neural_efficiency,
            'timestamp': current_time
        }
        
        return state
    
    def _consolidation_loop(self):
        """Bucle de consolidación de memoria en background"""
        while self.is_running:
            try:
                time.sleep(self.config.memory_consolidation_interval)
                self._perform_memory_consolidation()
            except Exception as e:
                logging.error(f"Error en consolidación GANST: {e}")
    
    def _perform_memory_consolidation(self):
        """Realiza consolidación de memoria neural"""
        # Limpiar activaciones expiradas
        current_time = time.time()
        expired_activations = []
        
        for act_id, act_data in self.activation_manager.active_patterns.items():
            age = current_time - act_data['timestamp']
            if age > 300:  # 5 minutos
                expired_activations.append(act_id)
        
        for act_id in expired_activations:
            del self.activation_manager.active_patterns[act_id]
        
        # Actualizar matriz de coherencia basada en historial
        if len(self.synthesizer.synthesis_history) > 10:
            self._update_coherence_matrix()
        
        logging.debug(f"GANST consolidación: {len(expired_activations)} activaciones expiradas")
    
    def _update_coherence_matrix(self):
        """Actualiza la matriz de coherencia global"""
        # Implementación simplificada - en un sistema real sería más compleja
        recent_coherences = [
            s['coherence_score'] 
            for s in list(self.synthesizer.synthesis_history)[-100:]
        ]
        
        if recent_coherences:
            avg_coherence = np.mean(recent_coherences)
            # Aplicar plasticidad a la matriz de coherencia
            plasticity = self.config.neural_plasticity_rate
            identity_factor = (1.0 - plasticity) + plasticity * avg_coherence
            
            self.synthesizer.coherence_matrix = (
                self.synthesizer.coherence_matrix * (1.0 - plasticity) +
                torch.eye(self.config.tensor_dim) * plasticity * identity_factor
            )
    
    def _calculate_neural_efficiency(self) -> float:
        """Calcula la eficiencia neural del sistema"""
        if not self.activation_manager.active_patterns:
            return 0.0
        
        # Eficiencia basada en ratio de activaciones útiles vs total
        total_activations = len(self.activation_manager.activation_history)
        active_activations = len(self.activation_manager.active_patterns)
        
        if total_activations == 0:
            return 0.0
        
        # Factor de eficiencia combinado
        temporal_efficiency = active_activations / min(total_activations, 100)
        state_efficiency = {
            NeuralState.DORMANT: 0.1,
            NeuralState.AWAKENING: 0.3,
            NeuralState.ACTIVE: 1.0,
            NeuralState.HYPERACTIVE: 0.7,
            NeuralState.CONSOLIDATING: 0.5
        }[self.activation_manager.neural_state]
        
        return temporal_efficiency * state_efficiency

# Instancia global del núcleo GANST
ganst_core = GANSTCore()

def initialize_ganst_system() -> GANSTCore:
    """Inicializa el sistema GANST global"""
    ganst_core.start()
    return ganst_core

def get_ganst_core() -> GANSTCore:
    """Obtiene la instancia global del núcleo GANST"""
    return ganst_core

# Funciones de conveniencia para uso externo
def process_neural_input(source: str, 
                        tensors: List[torch.Tensor],
                        pattern: str = "parallel") -> Dict[str, Any]:
    """Función de conveniencia para procesamiento neural"""
    pattern_enum = ActivationPattern(pattern)
    return ganst_core.process_neural_activation(source, tensors, pattern_enum)

def get_neural_state() -> str:
    """Obtiene el estado neural actual"""
    return ganst_core.activation_manager.neural_state.value

def get_system_statistics() -> Dict[str, Any]:
    """Obtiene estadísticas completas del sistema"""
    return ganst_core.get_system_state()