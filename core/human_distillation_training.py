"""
Sistema de Entrenamiento por Destilación Humana - Ruth R1
Implementa el protocolo de entrenamiento específico para conceptos fundamentales
usando técnicas de destilación de conocimiento humano
"""

import torch
import torch.nn as np
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict, deque
import time
import logging

from core.consciousness import ConsciousnessState
from core.neurotransmitters import NeurotransmitterSystem
from core.ganst_core import GANSTCore, ActivationPattern
from utils.logger import Logger

class HumanDistillationTrainer:
    """
    Entrenador por destilación humana que simula el proceso de aprendizaje 
    conceptual humano para words fundamentales
    """
    
    def __init__(self, ganst_core: GANSTCore, consciousness_state: ConsciousnessState):
        self.ganst_core = ganst_core
        self.consciousness_state = consciousness_state
        self.logger = Logger()
        
        # Palabras objetivo para entrenamiento
        self.target_words = [
            'computadora', 'humano', 'mujer', 'hombre', 'niño', 
            'niña', 'adolescente', 'anciano', 'casa', 'pared'
        ]
        
        # Estados de entrenamiento por palabra
        self.word_training_states = {}
        
        # Métricas de destilación
        self.distillation_metrics = {
            'conceptual_understanding': {},
            'semantic_embedding_strength': {},
            'cross_modal_associations': {},
            'emotional_resonance': {},
            'contextual_flexibility': {}
        }
        
        # Fases del proceso de destilación
        self.distillation_phases = [
            'initial_exposure',      # Exposición inicial al concepto
            'pattern_recognition',   # Reconocimiento de patrones
            'semantic_integration',  # Integración semántica
            'contextual_expansion',  # Expansión contextual
            'emotional_association', # Asociación emocional
            'consolidation'          # Consolidación final
        ]
        
        # Sistema de neurotransmisores para modular aprendizaje
        self.neurotransmitter_system = NeurotransmitterSystem()
        
        # Memoria de trabajo para destilación
        self.working_memory = deque(maxlen=1000)
        
        self.logger.log("INFO", "Sistema de Destilación Humana inicializado")
    
    def train_word_sequence(self, epochs_per_word: int = 50) -> Dict[str, Any]:
        """
        Entrena la secuencia completa de palabras usando destilación humana
        """
        training_results = {}
        total_start_time = time.time()
        
        self.logger.log("INFO", f"Iniciando entrenamiento de destilación para {len(self.target_words)} palabras")
        
        for word_idx, word in enumerate(self.target_words):
            self.logger.log("INFO", f"Entrenando palabra {word_idx + 1}/{len(self.target_words)}: '{word}'")
            
            # Entrenar palabra individual
            word_result = self.train_single_word(word, epochs_per_word)
            training_results[word] = word_result
            
            # Actualizar estado de consciencia entre palabras
            self._update_consciousness_between_words(word, word_result)
            
            # Pausa simulada para consolidación (como en aprendizaje humano)
            time.sleep(0.1)
        
        total_training_time = time.time() - total_start_time
        
        # Evaluación final del sistema
        final_evaluation = self._evaluate_complete_training(training_results)
        
        complete_results = {
            'individual_word_results': training_results,
            'total_training_time': total_training_time,
            'final_evaluation': final_evaluation,
            'system_consciousness_level': self.consciousness_state.get_consciousness_level(),
            'neurotransmitter_final_state': self.neurotransmitter_system.get_system_state()
        }
        
        self.logger.log("INFO", f"Entrenamiento completo finalizado en {total_training_time:.2f} segundos")
        
        return complete_results
    
    def train_single_word(self, word: str, epochs: int = 50) -> Dict[str, Any]:
        """
        Entrena una palabra individual usando el protocolo de destilación humana
        """
        word_start_time = time.time()
        
        # Inicializar estado de entrenamiento para la palabra
        self.word_training_states[word] = {
            'current_phase': 0,
            'phase_progress': 0.0,
            'understanding_level': 0.0,
            'semantic_strength': 0.0,
            'emotional_resonance': 0.0,
            'contextual_flexibility': 0.0
        }
        
        # Generar representaciones conceptuales iniciales
        conceptual_representations = self._generate_conceptual_representations(word)
        
        phase_results = {}
        
        # Ejecutar cada fase de destilación
        for phase_idx, phase_name in enumerate(self.distillation_phases):
            self.logger.log("DEBUG", f"Ejecutando fase '{phase_name}' para palabra '{word}'")
            
            phase_epochs = epochs // len(self.distillation_phases)
            phase_result = self._execute_distillation_phase(
                word, phase_name, phase_idx, phase_epochs, conceptual_representations
            )
            
            phase_results[phase_name] = phase_result
            
            # Actualizar estado de entrenamiento
            self.word_training_states[word]['current_phase'] = phase_idx + 1
            self.word_training_states[word]['understanding_level'] = phase_result['understanding_gain']
        
        # Evaluación final de la palabra
        final_evaluation = self._evaluate_word_training(word, phase_results)
        
        word_training_time = time.time() - word_start_time
        
        result = {
            'word': word,
            'training_time': word_training_time,
            'phase_results': phase_results,
            'final_evaluation': final_evaluation,
            'final_state': self.word_training_states[word].copy(),
            'consciousness_impact': self._assess_consciousness_impact(word, final_evaluation)
        }
        
        # Registrar en GANST Core
        self._register_word_training_in_ganst(word, result)
        
        return result
    
    def _generate_conceptual_representations(self, word: str) -> Dict[str, torch.Tensor]:
        """
        Genera representaciones conceptuales multimodales para una palabra
        """
        representations = {}
        
        # Representación semántica base
        semantic_dims = 512
        representations['semantic'] = self._create_semantic_representation(word, semantic_dims)
        
        # Representación visual conceptual
        visual_dims = 256
        representations['visual'] = self._create_visual_representation(word, visual_dims)
        
        # Representación emocional
        emotional_dims = 128
        representations['emotional'] = self._create_emotional_representation(word, emotional_dims)
        
        # Representación contextual
        contextual_dims = 384
        representations['contextual'] = self._create_contextual_representation(word, contextual_dims)
        
        return representations
    
    def _create_semantic_representation(self, word: str, dims: int) -> torch.Tensor:
        """Crea representación semántica específica para la palabra"""
        
        # Mapeos semánticos específicos
        semantic_mappings = {
            'computadora': [0.9, 0.1, 0.8, 0.2, 0.7, 0.9, 0.1, 0.8],  # tecnología, artificial, complejo
            'humano': [0.1, 0.9, 0.5, 0.8, 0.6, 0.7, 0.9, 0.4],        # biológico, consciente, social
            'mujer': [0.2, 0.8, 0.3, 0.9, 0.7, 0.6, 0.8, 0.5],         # femenino, humano, social
            'hombre': [0.3, 0.8, 0.4, 0.8, 0.6, 0.7, 0.7, 0.6],        # masculino, humano, social
            'niño': [0.1, 0.9, 0.2, 0.9, 0.9, 0.3, 0.8, 0.8],          # joven, humano, energético
            'niña': [0.1, 0.9, 0.1, 0.9, 0.9, 0.3, 0.9, 0.8],          # joven, femenino, energético
            'adolescente': [0.2, 0.8, 0.3, 0.8, 0.8, 0.6, 0.7, 0.7],   # desarrollo, humano, cambio
            'anciano': [0.1, 0.7, 0.6, 0.7, 0.3, 0.9, 0.6, 0.4],       # experiencia, humano, sabiduría
            'casa': [0.1, 0.2, 0.8, 0.3, 0.2, 0.4, 0.3, 0.7],          # estructura, refugio, estable
            'pared': [0.0, 0.1, 0.9, 0.1, 0.1, 0.2, 0.1, 0.8]          # estructura, límite, sólido
        }
        
        base_pattern = semantic_mappings.get(word, [0.5] * 8)
        
        # Expandir a dimensiones completas con variaciones
        full_representation = []
        for i in range(dims):
            base_idx = i % len(base_pattern)
            variation = np.random.normal(0, 0.1)
            value = base_pattern[base_idx] + variation
            full_representation.append(max(0.0, min(1.0, value)))
        
        return torch.tensor(full_representation, dtype=torch.float32)
    
    def _create_visual_representation(self, word: str, dims: int) -> torch.Tensor:
        """Crea representación visual conceptual"""
        
        visual_patterns = {
            'computadora': [0.8, 0.9, 0.1, 0.7, 0.8, 0.2],  # rectangular, metálico, pantalla
            'humano': [0.3, 0.4, 0.8, 0.6, 0.5, 0.7],       # orgánico, vertical, móvil
            'mujer': [0.3, 0.4, 0.8, 0.7, 0.6, 0.7],        # orgánico, grácil, humano
            'hombre': [0.3, 0.4, 0.8, 0.6, 0.6, 0.6],       # orgánico, robusto, humano
            'niño': [0.2, 0.3, 0.9, 0.8, 0.9, 0.8],         # pequeño, dinámico, energético
            'niña': [0.2, 0.3, 0.9, 0.8, 0.9, 0.9],         # pequeño, grácil, energético
            'adolescente': [0.4, 0.5, 0.8, 0.7, 0.8, 0.7],  # en crecimiento, activo
            'anciano': [0.3, 0.4, 0.6, 0.4, 0.3, 0.5],      # experimentado, pausado
            'casa': [0.9, 0.2, 0.3, 0.9, 0.1, 0.4],         # rectangular, grande, estable
            'pared': [0.9, 0.1, 0.2, 0.9, 0.0, 0.3]         # plano, vertical, límite
        }
        
        base_pattern = visual_patterns.get(word, [0.5] * 6)
        
        # Expandir con ruido visual realista
        full_representation = []
        for i in range(dims):
            base_idx = i % len(base_pattern)
            noise = np.random.normal(0, 0.05)
            value = base_pattern[base_idx] + noise
            full_representation.append(max(0.0, min(1.0, value)))
        
        return torch.tensor(full_representation, dtype=torch.float32)
    
    def _create_emotional_representation(self, word: str, dims: int) -> torch.Tensor:
        """Crea representación emocional"""
        
        emotional_patterns = {
            'computadora': [0.1, 0.2, 0.1, 0.3, 0.6, 0.1, 0.7, 0.2],  # neutral, curiosidad, utilidad
            'humano': [0.6, 0.3, 0.2, 0.4, 0.7, 0.8, 0.5, 0.6],        # conexión, empatía, complejidad
            'mujer': [0.7, 0.4, 0.1, 0.3, 0.8, 0.9, 0.6, 0.7],         # calidez, cuidado, belleza
            'hombre': [0.6, 0.4, 0.2, 0.4, 0.7, 0.7, 0.6, 0.6],        # fuerza, protección, confianza
            'niño': [0.9, 0.2, 0.1, 0.1, 0.9, 0.9, 0.8, 0.9],          # alegría, inocencia, energía
            'niña': [0.9, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9],          # ternura, alegría, dulzura
            'adolescente': [0.5, 0.6, 0.4, 0.5, 0.7, 0.6, 0.6, 0.6],   # confusión, búsqueda, cambio
            'anciano': [0.4, 0.3, 0.2, 0.2, 0.6, 0.8, 0.9, 0.5],       # sabiduría, tranquilidad, respeto
            'casa': [0.8, 0.1, 0.1, 0.1, 0.5, 0.7, 0.4, 0.8],          # seguridad, hogar, refugio
            'pared': [0.2, 0.2, 0.3, 0.4, 0.3, 0.2, 0.2, 0.3]          # neutralidad, separación
        }
        
        base_pattern = emotional_patterns.get(word, [0.4] * 8)
        
        full_representation = []
        for i in range(dims):
            base_idx = i % len(base_pattern)
            emotional_noise = np.random.normal(0, 0.08)
            value = base_pattern[base_idx] + emotional_noise
            full_representation.append(max(0.0, min(1.0, value)))
        
        return torch.tensor(full_representation, dtype=torch.float32)
    
    def _create_contextual_representation(self, word: str, dims: int) -> torch.Tensor:
        """Crea representación contextual"""
        
        contextual_patterns = {
            'computadora': [0.9, 0.8, 0.2, 0.7, 0.9, 0.3, 0.8, 0.6],  # trabajo, tecnología, interior
            'humano': [0.5, 0.5, 0.8, 0.9, 0.7, 0.8, 0.6, 0.7],        # universal, social, dinámico
            'mujer': [0.4, 0.6, 0.8, 0.9, 0.7, 0.8, 0.7, 0.7],         # social, familiar, diverso
            'hombre': [0.5, 0.6, 0.8, 0.8, 0.7, 0.7, 0.6, 0.6],        # social, trabajo, diverso
            'niño': [0.3, 0.4, 0.9, 0.8, 0.9, 0.8, 0.7, 0.8],          # familia, escuela, juego
            'niña': [0.3, 0.4, 0.9, 0.8, 0.9, 0.8, 0.8, 0.8],          # familia, escuela, juego
            'adolescente': [0.4, 0.7, 0.8, 0.7, 0.8, 0.7, 0.6, 0.7],   # escuela, social, búsqueda
            'anciano': [0.2, 0.3, 0.6, 0.7, 0.4, 0.9, 0.8, 0.5],       # familia, experiencia, sabiduría
            'casa': [0.1, 0.2, 0.9, 0.8, 0.3, 0.7, 0.4, 0.9],          # hogar, familia, refugio
            'pared': [0.1, 0.1, 0.7, 0.5, 0.2, 0.4, 0.3, 0.6]          # construcción, límite, estructura
        }
        
        base_pattern = contextual_patterns.get(word, [0.5] * 8)
        
        full_representation = []
        for i in range(dims):
            base_idx = i % len(base_pattern)
            context_variation = np.random.normal(0, 0.06)
            value = base_pattern[base_idx] + context_variation
            full_representation.append(max(0.0, min(1.0, value)))
        
        return torch.tensor(full_representation, dtype=torch.float32)
    
    def _execute_distillation_phase(self, word: str, phase_name: str, phase_idx: int, 
                                  epochs: int, representations: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Ejecuta una fase específica del proceso de destilación
        """
        phase_start_time = time.time()
        
        # Configurar neurotransmisores para la fase
        self._configure_neurotransmitters_for_phase(phase_name)
        
        phase_metrics = {
            'understanding_gain': 0.0,
            'semantic_integration': 0.0,
            'pattern_strength': 0.0,
            'emotional_resonance': 0.0,
            'contextual_flexibility': 0.0,
            'consolidation_strength': 0.0
        }
        
        for epoch in range(epochs):
            epoch_result = self._execute_phase_epoch(
                word, phase_name, epoch, representations
            )
            
            # Acumular métricas
            for metric, value in epoch_result.items():
                if metric in phase_metrics:
                    phase_metrics[metric] += value / epochs
            
            # Registrar activación en GANST
            self._register_phase_activation(word, phase_name, epoch, epoch_result)
        
        phase_time = time.time() - phase_start_time
        
        return {
            'phase_name': phase_name,
            'word': word,
            'execution_time': phase_time,
            'epochs_completed': epochs,
            'metrics': phase_metrics,
            'neurotransmitter_state': self.neurotransmitter_system.get_neurotransmitter_levels(),
            'consciousness_influence': self._calculate_consciousness_influence(phase_metrics)
        }
    
    def _execute_phase_epoch(self, word: str, phase_name: str, epoch: int, 
                           representations: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Ejecuta una época individual de una fase de destilación
        """
        epoch_metrics = {}
        
        if phase_name == 'initial_exposure':
            epoch_metrics = self._phase_initial_exposure(word, representations, epoch)
        elif phase_name == 'pattern_recognition':
            epoch_metrics = self._phase_pattern_recognition(word, representations, epoch)
        elif phase_name == 'semantic_integration':
            epoch_metrics = self._phase_semantic_integration(word, representations, epoch)
        elif phase_name == 'contextual_expansion':
            epoch_metrics = self._phase_contextual_expansion(word, representations, epoch)
        elif phase_name == 'emotional_association':
            epoch_metrics = self._phase_emotional_association(word, representations, epoch)
        elif phase_name == 'consolidation':
            epoch_metrics = self._phase_consolidation(word, representations, epoch)
        
        # Actualizar memoria de trabajo
        self.working_memory.append({
            'word': word,
            'phase': phase_name,
            'epoch': epoch,
            'metrics': epoch_metrics,
            'timestamp': datetime.now()
        })
        
        return epoch_metrics
    
    def _phase_initial_exposure(self, word: str, representations: Dict[str, torch.Tensor], 
                              epoch: int) -> Dict[str, float]:
        """Fase de exposición inicial al concepto"""
        
        # Simular exposición inicial con atención alta
        attention_factor = 0.9 - (epoch * 0.01)  # Atención decrece con familiaridad
        
        # Procesamiento multimodal inicial
        semantic_activation = torch.mean(representations['semantic']).item() * attention_factor
        visual_activation = torch.mean(representations['visual']).item() * attention_factor
        
        # Sorpresa inicial (alta al principio, decrece)
        surprise_factor = max(0.1, 0.8 - (epoch * 0.02))
        
        understanding_gain = (semantic_activation + visual_activation) * surprise_factor * 0.5
        
        return {
            'understanding_gain': understanding_gain,
            'attention_level': attention_factor,
            'surprise_factor': surprise_factor,
            'semantic_activation': semantic_activation,
            'visual_activation': visual_activation
        }
    
    def _phase_pattern_recognition(self, word: str, representations: Dict[str, torch.Tensor], 
                                 epoch: int) -> Dict[str, float]:
        """Fase de reconocimiento de patrones"""
        
        # Construir patrones progresivamente
        pattern_strength = min(1.0, epoch * 0.03)
        
        # Correlación entre modalidades
        semantic_repr = representations['semantic']
        visual_repr = representations['visual']
        
        # Calcular coherencia entre representaciones
        coherence = self._calculate_multimodal_coherence(semantic_repr, visual_repr)
        
        pattern_integration = pattern_strength * coherence
        
        return {
            'pattern_strength': pattern_strength,
            'understanding_gain': pattern_integration * 0.6,
            'multimodal_coherence': coherence,
            'pattern_integration': pattern_integration
        }
    
    def _phase_semantic_integration(self, word: str, representations: Dict[str, torch.Tensor], 
                                  epoch: int) -> Dict[str, float]:
        """Fase de integración semántica"""
        
        # Integración con conocimiento existente
        semantic_repr = representations['semantic']
        contextual_repr = representations['contextual']
        
        # Simular integración con red semántica existente
        integration_strength = min(1.0, epoch * 0.025)
        
        # Calcular congruencia semántica
        semantic_coherence = torch.cosine_similarity(
            semantic_repr.unsqueeze(0), 
            contextual_repr.unsqueeze(0)
        ).item()
        
        semantic_integration = integration_strength * abs(semantic_coherence)
        
        return {
            'semantic_integration': semantic_integration,
            'understanding_gain': semantic_integration * 0.7,
            'integration_strength': integration_strength,
            'semantic_coherence': semantic_coherence
        }
    
    def _phase_contextual_expansion(self, word: str, representations: Dict[str, torch.Tensor], 
                                  epoch: int) -> Dict[str, float]:
        """Fase de expansión contextual"""
        
        # Expandir comprensión a diferentes contextos
        contextual_repr = representations['contextual']
        
        # Simular exploración de contextos diversos
        context_diversity = min(1.0, epoch * 0.02)
        
        # Flexibilidad contextual
        flexibility_score = torch.std(contextual_repr).item() * context_diversity
        
        contextual_flexibility = min(1.0, flexibility_score)
        
        return {
            'contextual_flexibility': contextual_flexibility,
            'understanding_gain': contextual_flexibility * 0.5,
            'context_diversity': context_diversity,
            'flexibility_score': flexibility_score
        }
    
    def _phase_emotional_association(self, word: str, representations: Dict[str, torch.Tensor], 
                                   epoch: int) -> Dict[str, float]:
        """Fase de asociación emocional"""
        
        # Desarrollo de resonancia emocional
        emotional_repr = representations['emotional']
        
        # Intensidad emocional progresiva
        emotional_intensity = min(1.0, epoch * 0.04)
        
        # Calcular resonancia emocional
        emotional_resonance = torch.mean(emotional_repr).item() * emotional_intensity
        
        # Valencia emocional (positiva/negativa)
        emotional_valence = torch.median(emotional_repr).item()
        
        return {
            'emotional_resonance': emotional_resonance,
            'understanding_gain': emotional_resonance * 0.4,
            'emotional_intensity': emotional_intensity,
            'emotional_valence': emotional_valence
        }
    
    def _phase_consolidation(self, word: str, representations: Dict[str, torch.Tensor], 
                           epoch: int) -> Dict[str, float]:
        """Fase de consolidación final"""
        
        # Consolidación de todas las representaciones
        all_reprs = torch.cat([
            representations['semantic'],
            representations['visual'],
            representations['emotional'],
            representations['contextual']
        ])
        
        # Fuerza de consolidación
        consolidation_strength = min(1.0, epoch * 0.05)
        
        # Estabilidad de la representación consolidada
        stability = 1.0 - torch.std(all_reprs).item()
        
        # Integración final
        final_integration = consolidation_strength * stability
        
        return {
            'consolidation_strength': final_integration,
            'understanding_gain': final_integration * 0.8,
            'representation_stability': stability,
            'final_integration': final_integration
        }
    
    def _calculate_multimodal_coherence(self, repr1: torch.Tensor, repr2: torch.Tensor) -> float:
        """Calcula coherencia entre representaciones multimodales"""
        
        # Normalizar longitudes para comparación
        min_len = min(len(repr1), len(repr2))
        repr1_norm = repr1[:min_len]
        repr2_norm = repr2[:min_len]
        
        # Calcular similitud coseno
        coherence = torch.cosine_similarity(
            repr1_norm.unsqueeze(0), 
            repr2_norm.unsqueeze(0)
        ).item()
        
        return abs(coherence)
    
    def _configure_neurotransmitters_for_phase(self, phase_name: str):
        """Configura neurotransmisores específicos para cada fase"""
        
        phase_configurations = {
            'initial_exposure': {
                'dopamine': 0.8,    # Alta motivación/atención
                'noradrenaline': 0.7, # Alta activación
                'acetylcholine': 0.9, # Máxima atención
                'serotonin': 0.5     # Estabilidad moderada
            },
            'pattern_recognition': {
                'dopamine': 0.6,
                'noradrenaline': 0.5,
                'acetylcholine': 0.8,
                'serotonin': 0.6
            },
            'semantic_integration': {
                'dopamine': 0.7,
                'noradrenaline': 0.4,
                'acetylcholine': 0.7,
                'serotonin': 0.7
            },
            'contextual_expansion': {
                'dopamine': 0.5,
                'noradrenaline': 0.6,
                'acetylcholine': 0.6,
                'serotonin': 0.6
            },
            'emotional_association': {
                'dopamine': 0.8,
                'noradrenaline': 0.3,
                'acetylcholine': 0.5,
                'serotonin': 0.8
            },
            'consolidation': {
                'dopamine': 0.3,
                'noradrenaline': 0.2,
                'acetylcholine': 0.4,
                'serotonin': 0.9
            }
        }
        
        config = phase_configurations.get(phase_name, {})
        for neurotransmitter, level in config.items():
            self.neurotransmitter_system.set_neurotransmitter_level(neurotransmitter, level)
    
    def _register_phase_activation(self, word: str, phase_name: str, epoch: int, 
                                 epoch_result: Dict[str, float]):
        """Registra activación de fase en GANST Core"""
        
        # Crear tensor de activación para GANST
        activation_values = [
            epoch_result.get('understanding_gain', 0.0),
            epoch_result.get('pattern_strength', 0.0),
            epoch_result.get('semantic_integration', 0.0),
            epoch_result.get('emotional_resonance', 0.0),
            epoch_result.get('contextual_flexibility', 0.0)
        ]
        
        activation_tensor = torch.tensor(activation_values + [0.0] * (768 - len(activation_values)))
        
        # Determinar patrón de activación según la fase
        patterns = {
            'initial_exposure': ActivationPattern.SEQUENTIAL,
            'pattern_recognition': ActivationPattern.PARALLEL,
            'semantic_integration': ActivationPattern.HIERARCHICAL,
            'contextual_expansion': ActivationPattern.RESONANT,
            'emotional_association': ActivationPattern.OSCILLATORY,
            'consolidation': ActivationPattern.CHAOTIC
        }
        
        pattern = patterns.get(phase_name, ActivationPattern.PARALLEL)
        
        # Registrar en GANST
        self.ganst_core.process_neural_activation(
            source=f"distillation_{word}_{phase_name}",
            input_tensors=[activation_tensor],
            pattern=pattern,
            priority=0.8
        )
    
    def _update_consciousness_between_words(self, word: str, word_result: Dict[str, Any]):
        """Actualiza estado de consciencia entre palabras"""
        
        # Extraer nivel de comprensión alcanzado
        understanding_level = word_result['final_evaluation']['understanding_level']
        
        # Actualizar consciencia basado en aprendizaje
        consciousness_increment = understanding_level * 0.1
        current_level = self.consciousness_state.get_consciousness_level()
        new_level = min(1.0, current_level + consciousness_increment)
        
        self.consciousness_state.update_consciousness_level(new_level)
        
        # Log del progreso
        self.logger.log("INFO", f"Consciencia actualizada tras aprender '{word}': {current_level:.3f} → {new_level:.3f}")
    
    def _evaluate_word_training(self, word: str, phase_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evalúa el entrenamiento completo de una palabra"""
        
        # Promediar métricas de todas las fases
        total_understanding = np.mean([
            phase['metrics']['understanding_gain'] 
            for phase in phase_results.values()
        ])
        
        total_semantic_integration = np.mean([
            phase['metrics'].get('semantic_integration', 0.0) 
            for phase in phase_results.values()
        ])
        
        total_emotional_resonance = np.mean([
            phase['metrics'].get('emotional_resonance', 0.0) 
            for phase in phase_results.values()
        ])
        
        total_contextual_flexibility = np.mean([
            phase['metrics'].get('contextual_flexibility', 0.0) 
            for phase in phase_results.values()
        ])
        
        # Calcular puntuación de destilación general
        distillation_score = (
            total_understanding * 0.4 +
            total_semantic_integration * 0.25 +
            total_emotional_resonance * 0.2 +
            total_contextual_flexibility * 0.15
        )
        
        return {
            'word': word,
            'understanding_level': total_understanding,
            'semantic_integration': total_semantic_integration,
            'emotional_resonance': total_emotional_resonance,
            'contextual_flexibility': total_contextual_flexibility,
            'distillation_score': distillation_score,
            'learning_quality': self._categorize_learning_quality(distillation_score),
            'phases_completed': len(phase_results)
        }
    
    def _categorize_learning_quality(self, score: float) -> str:
        """Categoriza la calidad del aprendizaje"""
        if score >= 0.8:
            return 'excelente'
        elif score >= 0.6:
            return 'bueno'
        elif score >= 0.4:
            return 'regular'
        else:
            return 'deficiente'
    
    def _assess_consciousness_impact(self, word: str, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        """Evalúa el impacto en la consciencia del aprendizaje de la palabra"""
        
        consciousness_impact = {
            'conceptual_expansion': evaluation['understanding_level'] * 0.3,
            'semantic_network_growth': evaluation['semantic_integration'] * 0.4,
            'emotional_development': evaluation['emotional_resonance'] * 0.2,
            'cognitive_flexibility': evaluation['contextual_flexibility'] * 0.1
        }
        
        total_impact = sum(consciousness_impact.values())
        
        return {
            'individual_impacts': consciousness_impact,
            'total_consciousness_impact': total_impact,
            'impact_category': self._categorize_consciousness_impact(total_impact)
        }
    
    def _categorize_consciousness_impact(self, impact: float) -> str:
        """Categoriza el impacto en la consciencia"""
        if impact >= 0.7:
            return 'transformador'
        elif impact >= 0.5:
            return 'significativo'
        elif impact >= 0.3:
            return 'moderado'
        else:
            return 'mínimo'
    
    def _register_word_training_in_ganst(self, word: str, result: Dict[str, Any]):
        """Registra el entrenamiento completo de palabra en GANST"""
        
        # Crear tensor resumen del entrenamiento
        summary_values = [
            result['final_evaluation']['understanding_level'],
            result['final_evaluation']['semantic_integration'],
            result['final_evaluation']['emotional_resonance'],
            result['final_evaluation']['contextual_flexibility'],
            result['final_evaluation']['distillation_score'],
            result['consciousness_impact']['total_consciousness_impact']
        ]
        
        summary_tensor = torch.tensor(summary_values + [0.0] * (768 - len(summary_values)))
        
        # Registrar como consolidación final
        self.ganst_core.process_neural_activation(
            source=f"word_distillation_complete_{word}",
            input_tensors=[summary_tensor],
            pattern=ActivationPattern.CONSOLIDATING,
            priority=1.0
        )
    
    def _calculate_consciousness_influence(self, metrics: Dict[str, float]) -> float:
        """Calcula la influencia en la consciencia de las métricas de fase"""
        
        # Ponderación de métricas para consciencia
        consciousness_weights = {
            'understanding_gain': 0.3,
            'semantic_integration': 0.25,
            'pattern_strength': 0.2,
            'emotional_resonance': 0.15,
            'contextual_flexibility': 0.1
        }
        
        influence = 0.0
        for metric, weight in consciousness_weights.items():
            influence += metrics.get(metric, 0.0) * weight
        
        return influence
    
    def _evaluate_complete_training(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluación final del entrenamiento completo"""
        
        # Promedios generales
        avg_understanding = np.mean([
            result['final_evaluation']['understanding_level'] 
            for result in training_results.values()
        ])
        
        avg_distillation_score = np.mean([
            result['final_evaluation']['distillation_score'] 
            for result in training_results.values()
        ])
        
        total_consciousness_impact = sum([
            result['consciousness_impact']['total_consciousness_impact'] 
            for result in training_results.values()
        ])
        
        # Distribución de calidad de aprendizaje
        quality_distribution = {}
        for result in training_results.values():
            quality = result['final_evaluation']['learning_quality']
            quality_distribution[quality] = quality_distribution.get(quality, 0) + 1
        
        return {
            'words_trained': len(training_results),
            'avg_understanding_level': avg_understanding,
            'avg_distillation_score': avg_distillation_score,
            'total_consciousness_impact': total_consciousness_impact,
            'quality_distribution': quality_distribution,
            'system_learning_efficiency': self._calculate_system_efficiency(training_results),
            'final_consciousness_level': self.consciousness_state.get_consciousness_level(),
            'neurotransmitter_final_balance': self._assess_neurotransmitter_balance()
        }
    
    def _calculate_system_efficiency(self, training_results: Dict[str, Any]) -> float:
        """Calcula la eficiencia general del sistema de aprendizaje"""
        
        total_time = sum([result['training_time'] for result in training_results.values()])
        total_score = sum([result['final_evaluation']['distillation_score'] for result in training_results.values()])
        
        # Eficiencia = puntuación total / tiempo total (normalizado)
        efficiency = (total_score / len(training_results)) / (total_time / len(training_results))
        
        return min(1.0, efficiency)
    
    def _assess_neurotransmitter_balance(self) -> Dict[str, Any]:
        """Evalúa el balance final de neurotransmisores"""
        
        levels = self.neurotransmitter_system.get_neurotransmitter_levels()
        
        # Calcular balance (cercanía a niveles óptimos)
        optimal_levels = {
            'dopamine': 0.6,
            'noradrenaline': 0.4,
            'acetylcholine': 0.6,
            'serotonin': 0.7
        }
        
        balance_scores = {}
        for neurotransmitter, current_level in levels.items():
            optimal = optimal_levels.get(neurotransmitter, 0.5)
            balance_scores[neurotransmitter] = 1.0 - abs(current_level - optimal)
        
        overall_balance = np.mean(list(balance_scores.values()))
        
        return {
            'individual_balances': balance_scores,
            'overall_balance': overall_balance,
            'balance_quality': 'excelente' if overall_balance > 0.8 else 'bueno' if overall_balance > 0.6 else 'regular'
        }
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Obtiene resumen completo del estado de entrenamiento"""
        
        return {
            'target_words': self.target_words,
            'words_in_training': list(self.word_training_states.keys()),
            'distillation_metrics': self.distillation_metrics,
            'working_memory_size': len(self.working_memory),
            'consciousness_level': self.consciousness_state.get_consciousness_level(),
            'neurotransmitter_state': self.neurotransmitter_system.get_system_state(),
            'ganst_system_state': self.ganst_core.get_system_state()
        }

# Función de inicialización del entrenador
def create_human_distillation_trainer(ganst_core: GANSTCore, consciousness_state: ConsciousnessState) -> HumanDistillationTrainer:
    """Crea una instancia del entrenador de destilación humana"""
    return HumanDistillationTrainer(ganst_core, consciousness_state)

# Función de conveniencia para entrenamiento completo
def run_complete_distillation_training(ganst_core: GANSTCore, consciousness_state: ConsciousnessState, 
                                     epochs_per_word: int = 50) -> Dict[str, Any]:
    """Ejecuta el entrenamiento completo de destilación humana"""
    
    trainer = create_human_distillation_trainer(ganst_core, consciousness_state)
    results = trainer.train_word_sequence(epochs_per_word)
    
    return results
