import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from scipy.linalg import expm
from scipy.stats import entropy
from collections import defaultdict, deque
from datetime import datetime
import json

from utils.logger import Logger

class QuantumDensityMatrix:
    """
    Representa una matriz de densidad cuántica para el sistema Bayesiano
    """
    
    def __init__(self, matrix: np.ndarray):
        """
        Inicializa matriz de densidad
        
        Args:
            matrix: Matriz de densidad (debe ser hermítica y traza = 1)
        """
        self.matrix = matrix.astype(complex)
        self._normalize()
        
    def _normalize(self):
        """Normaliza la matriz para que tenga traza 1"""
        trace = np.trace(self.matrix)
        if abs(trace) > 1e-10:
            self.matrix = self.matrix / trace
    
    @classmethod
    def from_state_vector(cls, state_vector: np.ndarray) -> 'QuantumDensityMatrix':
        """Crea matriz de densidad desde vector de estado puro"""
        state_vector = state_vector / np.linalg.norm(state_vector)
        matrix = np.outer(state_vector, np.conj(state_vector))
        return cls(matrix)
    
    @classmethod
    def mixed_state(cls, states: List[np.ndarray], probabilities: List[float]) -> 'QuantumDensityMatrix':
        """Crea estado mixto desde múltiples estados puros"""
        assert len(states) == len(probabilities)
        assert abs(sum(probabilities) - 1.0) < 1e-10
        
        dim = len(states[0])
        matrix = np.zeros((dim, dim), dtype=complex)
        
        for state, prob in zip(states, probabilities):
            state = state / np.linalg.norm(state)
            matrix += prob * np.outer(state, np.conj(state))
        
        return cls(matrix)
    
    def purity(self) -> float:
        """Calcula la pureza del estado cuántico"""
        return np.real(np.trace(self.matrix @ self.matrix))
    
    def von_neumann_entropy(self) -> float:
        """Calcula la entropía de von Neumann"""
        eigenvals = np.linalg.eigvals(self.matrix)
        eigenvals = eigenvals[eigenvals > 1e-12]  # Evitar log(0)
        return -np.sum(eigenvals * np.log2(eigenvals + 1e-12))
    
    def trace_distance(self, other: 'QuantumDensityMatrix') -> float:
        """Calcula la distancia de traza con otra matriz de densidad"""
        diff = self.matrix - other.matrix
        eigenvals = np.linalg.eigvals(diff @ np.conj(diff.T))
        return 0.5 * np.sum(np.sqrt(np.abs(eigenvals)))
    
    def fidelity(self, other: 'QuantumDensityMatrix') -> float:
        """Calcula la fidelidad cuántica con otra matriz de densidad"""
        # F = Tr(√(√ρ σ √ρ))
        sqrt_self = self._matrix_sqrt()
        sqrt_other = other._matrix_sqrt()
        
        product = sqrt_self @ other.matrix @ sqrt_self
        sqrt_product = self._matrix_sqrt_of_matrix(product)
        
        return np.real(np.trace(sqrt_product))
    
    def _matrix_sqrt(self) -> np.ndarray:
        """Calcula la raíz cuadrada de la matriz de densidad"""
        eigenvals, eigenvecs = np.linalg.eigh(self.matrix)
        sqrt_eigenvals = np.sqrt(np.maximum(eigenvals, 0))
        return eigenvecs @ np.diag(sqrt_eigenvals) @ np.conj(eigenvecs.T)
    
    def _matrix_sqrt_of_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Calcula la raíz cuadrada de una matriz arbitraria"""
        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        sqrt_eigenvals = np.sqrt(np.maximum(eigenvals, 0))
        return eigenvecs @ np.diag(sqrt_eigenvals) @ np.conj(eigenvecs.T)
    
    def partial_trace(self, subsystem_dims: List[int], traced_systems: List[int]) -> 'QuantumDensityMatrix':
        """
        Calcula la traza parcial sobre subsistemas específicos
        
        Args:
            subsystem_dims: Dimensiones de cada subsistema
            traced_systems: Índices de los subsistemas a trazar
        
        Returns:
            Matriz de densidad reducida
        """
        # Implementación simplificada para sistemas pequeños
        total_dim = np.prod(subsystem_dims)
        remaining_dim = np.prod([subsystem_dims[i] for i in range(len(subsystem_dims)) if i not in traced_systems])
        
        if remaining_dim == 0:
            remaining_dim = 1
        
        # Para simplificar, retornamos una aproximación
        # En implementación completa se haría la traza parcial exacta
        reduced_matrix = np.eye(remaining_dim, dtype=complex) / remaining_dim
        
        return QuantumDensityMatrix(reduced_matrix)
    
    def apply_unitary(self, unitary: np.ndarray):
        """Aplica una transformación unitaria a la matriz de densidad"""
        self.matrix = unitary @ self.matrix @ np.conj(unitary.T)
    
    def apply_channel(self, kraus_operators: List[np.ndarray]):
        """
        Aplica un canal cuántico usando operadores de Kraus
        
        Args:
            kraus_operators: Lista de operadores de Kraus del canal
        """
        new_matrix = np.zeros_like(self.matrix)
        
        for kraus_op in kraus_operators:
            new_matrix += kraus_op @ self.matrix @ np.conj(kraus_op.T)
        
        self.matrix = new_matrix
        self._normalize()
    
    def measure_observable(self, observable: np.ndarray) -> float:
        """
        Calcula el valor esperado de un observable
        
        Args:
            observable: Matriz hermítica representando el observable
        
        Returns:
            Valor esperado del observable
        """
        return np.real(np.trace(self.matrix @ observable))
    
    def __repr__(self):
        return f"QuantumDensityMatrix(shape={self.matrix.shape}, purity={self.purity():.3f})"

class QuantumBayesianEvent:
    """
    Representa un evento en el marco Bayesiano cuántico
    """
    
    def __init__(self, name: str, initial_density: QuantumDensityMatrix, 
                 prior_probability: float = 1.0):
        self.name = name
        self.density_matrix = initial_density
        self.prior_probability = prior_probability
        self.posterior_probability = prior_probability
        
        # Historia de actualizaciones
        self.update_history = []
        
        # Evidencias acumuladas
        self.evidences = []
        
    def update_posterior(self, likelihood_density: QuantumDensityMatrix, 
                        evidence_strength: float = 1.0):
        """
        Actualiza la probabilidad posterior usando evidencia cuántica
        
        Args:
            likelihood_density: Matriz de densidad de la verosimilitud
            evidence_strength: Fuerza de la evidencia (0-1)
        """
        
        # Guardar estado anterior
        old_density = QuantumDensityMatrix(self.density_matrix.matrix.copy())
        old_probability = self.posterior_probability
        
        # Actualización cuántica de la matriz de densidad
        # Simular canal cuántico que incorpora la evidencia
        
        # Operador de evolución basado en evidencia
        evolution_strength = evidence_strength * 0.5  # Factor de modulación
        
        # Crear Hamiltoniano de evolución
        dim = self.density_matrix.matrix.shape[0]
        hamiltonian = np.random.hermitian(dim) * evolution_strength
        
        # Evolución unitaria
        unitary = expm(-1j * hamiltonian)
        
        # Aplicar evolución
        self.density_matrix.apply_unitary(unitary)
        
        # Mezclar con evidencia
        mixing_parameter = evidence_strength * 0.3
        mixed_matrix = ((1 - mixing_parameter) * self.density_matrix.matrix + 
                       mixing_parameter * likelihood_density.matrix)
        
        self.density_matrix = QuantumDensityMatrix(mixed_matrix)
        
        # Actualizar probabilidad clásica
        fidelity_with_evidence = self.density_matrix.fidelity(likelihood_density)
        self.posterior_probability *= (1 + evidence_strength * fidelity_with_evidence)
        
        # Normalizar probabilidad
        self.posterior_probability = min(1.0, max(0.0, self.posterior_probability))
        
        # Registrar actualización
        update_record = {
            'timestamp': datetime.now().isoformat(),
            'evidence_strength': evidence_strength,
            'fidelity_change': fidelity_with_evidence,
            'probability_change': self.posterior_probability - old_probability,
            'entropy_change': (self.density_matrix.von_neumann_entropy() - 
                             old_density.von_neumann_entropy())
        }
        
        self.update_history.append(update_record)
        self.evidences.append({
            'likelihood_density': likelihood_density,
            'strength': evidence_strength,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_quantum_information(self) -> Dict[str, float]:
        """Obtiene información cuántica del evento"""
        return {
            'purity': self.density_matrix.purity(),
            'von_neumann_entropy': self.density_matrix.von_neumann_entropy(),
            'prior_probability': self.prior_probability,
            'posterior_probability': self.posterior_probability,
            'total_updates': len(self.update_history),
            'total_evidences': len(self.evidences)
        }

class QuantumBayesianInference:
    """
    Motor de inferencia Bayesiana cuántica
    
    Implementa la ecuación: ρ(A|B) = ρ(B|A) · ρ(A) / ρ(B)
    """
    
    def __init__(self, default_dimension: int = 4):
        self.default_dimension = default_dimension
        
        # Registro de eventos cuánticos
        self.events = {}
        
        # Registro de relaciones causales
        self.causal_relationships = defaultdict(list)
        
        # Historia de inferencias
        self.inference_history = deque(maxlen=1000)
        
        # Cache de cálculos
        self.calculation_cache = {}
        
        self.logger = Logger()
        self.logger.log("INFO", "QuantumBayesianInference initialized")
    
    def register_event(self, event_name: str, initial_state: np.ndarray = None, 
                      prior_probability: float = 0.5) -> QuantumBayesianEvent:
        """
        Registra un nuevo evento en el sistema Bayesiano cuántico
        
        Args:
            event_name: Nombre del evento
            initial_state: Estado cuántico inicial (opcional)
            prior_probability: Probabilidad a priori
        
        Returns:
            Evento cuántico creado
        """
        
        if initial_state is None:
            # Estado inicial mixto uniforme
            initial_state = np.ones(self.default_dimension) / np.sqrt(self.default_dimension)
        
        initial_density = QuantumDensityMatrix.from_state_vector(initial_state)
        
        event = QuantumBayesianEvent(event_name, initial_density, prior_probability)
        self.events[event_name] = event
        
        self.logger.log("DEBUG", f"Registered quantum Bayesian event: {event_name}")
        
        return event
    
    def add_causal_relationship(self, cause: str, effect: str, strength: float = 1.0):
        """
        Añade una relación causal entre eventos
        
        Args:
            cause: Evento causa
            effect: Evento efecto
            strength: Fuerza de la relación causal
        """
        
        self.causal_relationships[cause].append({
            'effect': effect,
            'strength': strength,
            'established': datetime.now().isoformat()
        })
        
        self.logger.log("DEBUG", f"Added causal relationship: {cause} -> {effect} (strength: {strength})")
    
    def quantum_bayes_update(self, hypothesis: str, evidence: str, 
                           evidence_data: Any = None, confidence: float = 1.0) -> Dict[str, Any]:
        """
        Realiza actualización Bayesiana cuántica
        
        Implementa: ρ(H|E) = ρ(E|H) · ρ(H) / ρ(E)
        
        Args:
            hypothesis: Nombre del evento hipótesis
            evidence: Nombre del evento evidencia
            evidence_data: Datos de evidencia (opcional)
            confidence: Confianza en la evidencia
        
        Returns:
            Resultado de la actualización
        """
        
        # Verificar que los eventos existen
        if hypothesis not in self.events:
            self.register_event(hypothesis)
        
        if evidence not in self.events:
            self.register_event(evidence)
        
        hypothesis_event = self.events[hypothesis]
        evidence_event = self.events[evidence]
        
        # Calcular verosimilitud cuántica P(E|H)
        likelihood_density = self._calculate_quantum_likelihood(
            hypothesis_event, evidence_event, evidence_data, confidence
        )
        
        # Guardar estado anterior para comparación
        prior_density = QuantumDensityMatrix(hypothesis_event.density_matrix.matrix.copy())
        prior_probability = hypothesis_event.posterior_probability
        
        # Actualizar hipótesis con evidencia
        hypothesis_event.update_posterior(likelihood_density, confidence)
        
        # Calcular métricas de la actualización
        posterior_density = hypothesis_event.density_matrix
        
        update_metrics = {
            'hypothesis': hypothesis,
            'evidence': evidence,
            'prior_probability': prior_probability,
            'posterior_probability': hypothesis_event.posterior_probability,
            'probability_change': hypothesis_event.posterior_probability - prior_probability,
            'fidelity_change': prior_density.fidelity(posterior_density),
            'entropy_change': (posterior_density.von_neumann_entropy() - 
                             prior_density.von_neumann_entropy()),
            'purity_change': posterior_density.purity() - prior_density.purity(),
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }
        
        # Propagar efectos causales
        self._propagate_causal_effects(hypothesis, confidence * 0.5)
        
        # Registrar en historial
        self.inference_history.append(update_metrics)
        
        self.logger.log("INFO", 
            f"Quantum Bayes update: {hypothesis} | {evidence} -> "
            f"P: {prior_probability:.3f} -> {hypothesis_event.posterior_probability:.3f}"
        )
        
        return update_metrics
    
    def _calculate_quantum_likelihood(self, hypothesis_event: QuantumBayesianEvent, 
                                    evidence_event: QuantumBayesianEvent,
                                    evidence_data: Any, confidence: float) -> QuantumDensityMatrix:
        """
        Calcula la verosimilitud cuántica P(E|H)
        
        Args:
            hypothesis_event: Evento hipótesis
            evidence_event: Evento evidencia
            evidence_data: Datos de evidencia
            confidence: Confianza en la evidencia
        
        Returns:
            Matriz de densidad de la verosimilitud
        """
        
        # Base: matriz de densidad de la evidencia
        base_likelihood = evidence_event.density_matrix
        
        # Modificar según la relación con la hipótesis
        
        # 1. Buscar relaciones causales existentes
        causal_strength = 0.0
        if hypothesis_event.name in self.causal_relationships:
            for relation in self.causal_relationships[hypothesis_event.name]:
                if relation['effect'] == evidence_event.name:
                    causal_strength = max(causal_strength, relation['strength'])
        
        # 2. Calcular similitud cuántica entre eventos
        quantum_similarity = hypothesis_event.density_matrix.fidelity(evidence_event.density_matrix)
        
        # 3. Incorporar datos de evidencia si están disponibles
        data_factor = 1.0
        if evidence_data is not None:
            data_factor = self._analyze_evidence_data(evidence_data)
        
        # 4. Combinar factores para crear verosimilitud
        combined_strength = (causal_strength * 0.4 + 
                           quantum_similarity * 0.4 + 
                           data_factor * 0.2) * confidence
        
        # 5. Crear matriz de densidad de verosimilitud
        if combined_strength > 0.5:
            # Alta verosimilitud: estado más puro hacia la evidencia
            likelihood_matrix = (combined_strength * evidence_event.density_matrix.matrix + 
                               (1 - combined_strength) * hypothesis_event.density_matrix.matrix)
        else:
            # Baja verosimilitud: estado más mixto
            dim = base_likelihood.matrix.shape[0]
            mixed_state = np.eye(dim) / dim
            likelihood_matrix = (combined_strength * evidence_event.density_matrix.matrix + 
                               (1 - combined_strength) * mixed_state)
        
        return QuantumDensityMatrix(likelihood_matrix)
    
    def _analyze_evidence_data(self, evidence_data: Any) -> float:
        """
        Analiza datos de evidencia para extraer factor de relevancia
        
        Args:
            evidence_data: Datos de evidencia a analizar
        
        Returns:
            Factor de relevancia (0-1)
        """
        
        if evidence_data is None:
            return 0.5
        
        try:
            if isinstance(evidence_data, (int, float)):
                # Datos numéricos: normalizar
                return min(1.0, max(0.0, abs(evidence_data) / 10.0))
            
            elif isinstance(evidence_data, str):
                # Datos textuales: analizar contenido
                relevance_keywords = ['importante', 'crucial', 'significativo', 'clave', 'crítico']
                uncertainty_keywords = ['quizás', 'posiblemente', 'tal vez', 'incierto']
                
                relevance_score = sum(1 for keyword in relevance_keywords 
                                    if keyword in evidence_data.lower()) * 0.2
                uncertainty_penalty = sum(1 for keyword in uncertainty_keywords 
                                        if keyword in evidence_data.lower()) * 0.1
                
                base_score = min(len(evidence_data) / 100.0, 1.0)  # Longitud como indicador
                
                return max(0.0, min(1.0, base_score + relevance_score - uncertainty_penalty))
            
            elif isinstance(evidence_data, dict):
                # Datos estructurados: analizar completitud
                completeness = len([v for v in evidence_data.values() if v is not None]) / len(evidence_data)
                return completeness
            
            elif isinstance(evidence_data, (list, tuple)):
                # Datos secuenciales: analizar variabilidad
                if len(evidence_data) > 0:
                    try:
                        numeric_data = [float(x) for x in evidence_data if isinstance(x, (int, float))]
                        if numeric_data:
                            variability = np.std(numeric_data) / (np.mean(numeric_data) + 1e-10)
                            return min(1.0, variability)
                    except:
                        pass
                
                return len(evidence_data) / 20.0  # Longitud normalizada
            
            else:
                return 0.5  # Valor neutro para tipos desconocidos
        
        except Exception as e:
            self.logger.log("WARNING", f"Error analyzing evidence data: {str(e)}")
            return 0.5
    
    def _propagate_causal_effects(self, source_event: str, effect_strength: float):
        """
        Propaga efectos causales a eventos relacionados
        
        Args:
            source_event: Evento fuente del efecto
            effect_strength: Fuerza del efecto a propagar
        """
        
        if source_event not in self.causal_relationships:
            return
        
        for relation in self.causal_relationships[source_event]:
            effect_event = relation['effect']
            causal_strength = relation['strength']
            
            if effect_event in self.events:
                # Calcular fuerza del efecto propagado
                propagated_strength = effect_strength * causal_strength * 0.5
                
                if propagated_strength > 0.1:  # Umbral mínimo
                    # Crear evidencia sintética para el efecto
                    source_density = self.events[source_event].density_matrix
                    
                    # Actualizar evento efecto
                    self.events[effect_event].update_posterior(source_density, propagated_strength)
                    
                    self.logger.log("DEBUG", 
                        f"Propagated causal effect: {source_event} -> {effect_event} "
                        f"(strength: {propagated_strength:.3f})"
                    )
    
    def calculate_mutual_information(self, event_a: str, event_b: str) -> float:
        """
        Calcula información mutua cuántica entre dos eventos
        
        Args:
            event_a: Primer evento
            event_b: Segundo evento
        
        Returns:
            Información mutua cuántica
        """
        
        if event_a not in self.events or event_b not in self.events:
            return 0.0
        
        density_a = self.events[event_a].density_matrix
        density_b = self.events[event_b].density_matrix
        
        # Calcular entropías
        entropy_a = density_a.von_neumann_entropy()
        entropy_b = density_b.von_neumann_entropy()
        
        # Simular entropía conjunta (en implementación real se calcularía el producto tensorial)
        # Por simplicidad, estimamos basado en fidelidad
        fidelity = density_a.fidelity(density_b)
        joint_entropy = entropy_a + entropy_b - fidelity * min(entropy_a, entropy_b)
        
        # Información mutua: I(A;B) = H(A) + H(B) - H(A,B)
        mutual_info = entropy_a + entropy_b - joint_entropy
        
        return max(0.0, mutual_info)
    
    def compute_quantum_correlation(self, event_a: str, event_b: str) -> Dict[str, float]:
        """
        Calcula correlaciones cuánticas entre eventos
        
        Args:
            event_a: Primer evento
            event_b: Segundo evento
        
        Returns:
            Diccionario con métricas de correlación
        """
        
        if event_a not in self.events or event_b not in self.events:
            return {'error': 'Events not found'}
        
        density_a = self.events[event_a].density_matrix
        density_b = self.events[event_b].density_matrix
        
        # Métricas de correlación cuántica
        correlations = {
            'fidelity': density_a.fidelity(density_b),
            'trace_distance': density_a.trace_distance(density_b),
            'mutual_information': self.calculate_mutual_information(event_a, event_b),
            'entropy_difference': abs(density_a.von_neumann_entropy() - density_b.von_neumann_entropy()),
            'purity_correlation': abs(density_a.purity() - density_b.purity())
        }
        
        # Correlación clásica de probabilidades
        prob_a = self.events[event_a].posterior_probability
        prob_b = self.events[event_b].posterior_probability
        correlations['probability_correlation'] = abs(prob_a - prob_b)
        
        return correlations
    
    def perform_quantum_hypothesis_testing(self, hypothesis: str, 
                                         evidence_list: List[Tuple[str, Any, float]],
                                         significance_level: float = 0.05) -> Dict[str, Any]:
        """
        Realiza prueba de hipótesis cuántica
        
        Args:
            hypothesis: Hipótesis a probar
            evidence_list: Lista de (evidencia, datos, confianza)
            significance_level: Nivel de significancia
        
        Returns:
            Resultado de la prueba de hipótesis
        """
        
        if hypothesis not in self.events:
            self.register_event(hypothesis)
        
        # Estado inicial de la hipótesis
        initial_probability = self.events[hypothesis].posterior_probability
        initial_entropy = self.events[hypothesis].density_matrix.von_neumann_entropy()
        
        # Aplicar evidencias secuencialmente
        evidence_effects = []
        
        for evidence_name, evidence_data, confidence in evidence_list:
            update_result = self.quantum_bayes_update(
                hypothesis, evidence_name, evidence_data, confidence
            )
            evidence_effects.append(update_result)
        
        # Estado final de la hipótesis
        final_probability = self.events[hypothesis].posterior_probability
        final_entropy = self.events[hypothesis].density_matrix.von_neumann_entropy()
        
        # Calcular estadísticas de la prueba
        probability_change = final_probability - initial_probability
        entropy_change = final_entropy - initial_entropy
        
        # Decisión basada en cambio de probabilidad y significancia
        is_significant = abs(probability_change) > significance_level
        accept_hypothesis = final_probability > 0.5 and is_significant
        
        # Calcular p-valor simulado basado en cambios cuánticos
        p_value = max(0.001, 1.0 - abs(probability_change))
        
        test_result = {
            'hypothesis': hypothesis,
            'initial_probability': initial_probability,
            'final_probability': final_probability,
            'probability_change': probability_change,
            'entropy_change': entropy_change,
            'is_significant': is_significant,
            'accept_hypothesis': accept_hypothesis,
            'p_value': p_value,
            'significance_level': significance_level,
            'evidence_count': len(evidence_list),
            'evidence_effects': evidence_effects,
            'quantum_coherence': 1.0 - final_entropy,  # Coherencia como 1 - entropía
            'test_timestamp': datetime.now().isoformat()
        }
        
        self.logger.log("INFO", 
            f"Quantum hypothesis test: {hypothesis} -> "
            f"Accept: {accept_hypothesis}, P-value: {p_value:.4f}"
        )
        
        return test_result
    
    def get_system_state(self) -> Dict[str, Any]:
        """Obtiene estado completo del sistema Bayesiano cuántico"""
        
        # Estadísticas por evento
        event_stats = {}
        for name, event in self.events.items():
            event_stats[name] = event.get_quantum_information()
        
        # Estadísticas globales
        total_updates = sum(len(event.update_history) for event in self.events.values())
        avg_entropy = np.mean([event.density_matrix.von_neumann_entropy() 
                              for event in self.events.values()]) if self.events else 0.0
        avg_purity = np.mean([event.density_matrix.purity() 
                             for event in self.events.values()]) if self.events else 0.0
        
        # Relaciones causales
        causal_stats = {
            'total_relationships': sum(len(relations) for relations in self.causal_relationships.values()),
            'relationships_by_cause': {cause: len(relations) 
                                     for cause, relations in self.causal_relationships.items()}
        }
        
        return {
            'events': event_stats,
            'global_statistics': {
                'total_events': len(self.events),
                'total_updates': total_updates,
                'average_entropy': avg_entropy,
                'average_purity': avg_purity,
                'total_inferences': len(self.inference_history)
            },
            'causal_relationships': causal_stats,
            'recent_inferences': list(self.inference_history)[-10:],  # Últimas 10
            'system_coherence': 1.0 - avg_entropy,
            'timestamp': datetime.now().isoformat()
        }
    
    def export_quantum_beliefs(self) -> Dict[str, Any]:
        """Exporta el estado de creencias cuánticas"""
        
        beliefs = {}
        
        for name, event in self.events.items():
            beliefs[name] = {
                'posterior_probability': event.posterior_probability,
                'quantum_purity': event.density_matrix.purity(),
                'quantum_entropy': event.density_matrix.von_neumann_entropy(),
                'update_count': len(event.update_history),
                'last_update': event.update_history[-1]['timestamp'] if event.update_history else None
            }
        
        return {
            'beliefs': beliefs,
            'export_timestamp': datetime.now().isoformat(),
            'system_version': '1.0'
        }
    
    def reset_system(self):
        """Reinicia completamente el sistema Bayesiano cuántico"""
        
        self.events.clear()
        self.causal_relationships.clear()
        self.inference_history.clear()
        self.calculation_cache.clear()
        
        self.logger.log("INFO", "Quantum Bayesian system reset completed")

class BayesianQuantumSystem:
    """
    Sistema principal que integra toda la funcionalidad Bayesiana cuántica
    """
    
    def __init__(self):
        self.inference_engine = QuantumBayesianInference()
        self.decision_threshold = 0.7  # Umbral para toma de decisiones
        self.uncertainty_tolerance = 0.3  # Tolerancia a la incertidumbre
        
        # Contexto del sistema de conciencia
        self.consciousness_context = {}
        
        # Integración con GANST
        self.ganst_integration = {
            'activation_buffer': deque(maxlen=100),
            'coherence_sync': 0.0,
            'last_ganst_sync': None
        }
        
        self.logger = Logger()
        self.logger.log("INFO", "BayesianQuantumSystem initialized")
    
    def process_consciousness_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa entrada desde el sistema de conciencia
        
        Args:
            input_data: Datos de entrada con contexto de conciencia
        
        Returns:
            Resultado del procesamiento Bayesiano
        """
        
        # Actualizar contexto
        self.consciousness_context.update(input_data)
        
        # Extraer eventos y evidencias
        events_to_process = []
        
        if 'user_input' in input_data:
            events_to_process.append(('user_query', input_data['user_input'], 0.8))
        
        if 'neurotransmitter_levels' in input_data:
            nt_levels = input_data['neurotransmitter_levels']
            
            # Crear eventos basados en neurotransmisores
            for nt_name, level in nt_levels.items():
                event_name = f"neurotransmitter_{nt_name}"
                confidence = min(1.0, level / 50.0)  # Normalizar
                events_to_process.append((event_name, level, confidence))
        
        if 'emotional_state' in input_data:
            events_to_process.append(('emotional_context', input_data['emotional_state'], 0.6))
        
        # Procesar eventos con inferencia Bayesiana cuántica
        processing_results = []
        
        for event_name, event_data, confidence in events_to_process:
            # Registrar evento si no existe
            if event_name not in self.inference_engine.events:
                self.inference_engine.register_event(event_name)
            
            # Crear hipótesis relacionadas
            hypothesis = f"response_to_{event_name}"
            
            update_result = self.inference_engine.quantum_bayes_update(
                hypothesis, event_name, event_data, confidence
            )
            
            processing_results.append(update_result)
        
        # Generar decisión basada en estados cuánticos
        decision = self._make_quantum_decision(processing_results)
        
        return {
            'processing_results': processing_results,
            'quantum_decision': decision,
            'system_state': self.inference_engine.get_system_state(),
            'consciousness_integration': self._assess_consciousness_integration()
        }
    
    def _make_quantum_decision(self, processing_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Toma decisión basada en resultados de procesamiento cuántico
        
        Args:
            processing_results: Resultados del procesamiento Bayesiano
        
        Returns:
            Decisión cuántica
        """
        
        if not processing_results:
            return {'decision': 'no_action', 'confidence': 0.0}
        
        # Analizar cambios de probabilidad
        probability_changes = [result.get('probability_change', 0) for result in processing_results]
        avg_change = np.mean(probability_changes)
        max_change = max(probability_changes, key=abs) if probability_changes else 0
        
        # Analizar coherencia cuántica
        entropy_changes = [result.get('entropy_change', 0) for result in processing_results]
        avg_entropy_change = np.mean(entropy_changes)
        
        # Tomar decisión basada en umbrales
        decision_confidence = abs(max_change)
        
        if decision_confidence > self.decision_threshold:
            if max_change > 0:
                decision_type = 'positive_action'
                action_strength = min(1.0, decision_confidence)
            else:
                decision_type = 'negative_action'
                action_strength = min(1.0, decision_confidence)
        else:
            decision_type = 'uncertain'
            action_strength = decision_confidence
        
        # Calcular métricas de decisión
        quantum_coherence = max(0.0, 1.0 - abs(avg_entropy_change))
        uncertainty_level = 1.0 - decision_confidence
        
        decision = {
            'decision': decision_type,
            'confidence': decision_confidence,
            'action_strength': action_strength,
            'quantum_coherence': quantum_coherence,
            'uncertainty_level': uncertainty_level,
            'requires_more_evidence': uncertainty_level > self.uncertainty_tolerance,
            'evidence_quality': np.mean([result.get('confidence', 0) for result in processing_results]),
            'reasoning': self._generate_decision_reasoning(processing_results, decision_type)
        }
        
        return decision
    
    def _generate_decision_reasoning(self, processing_results: List[Dict[str, Any]], 
                                   decision_type: str) -> str:
        """Genera explicación del razonamiento de la decisión"""
        
        if not processing_results:
            return "No hay suficiente información para generar razonamiento."
        
        # Analizar evidencias más influyentes
        significant_results = [r for r in processing_results 
                             if abs(r.get('probability_change', 0)) > 0.1]
        
        if not significant_results:
            return "Los cambios de probabilidad son mínimos, manteniendo estado neutro."
        
        reasoning_parts = []
        
        # Describir evidencias principales
        for result in significant_results[:3]:  # Top 3
            hypothesis = result.get('hypothesis', 'unknown')
            evidence = result.get('evidence', 'unknown')
            change = result.get('probability_change', 0)
            
            if change > 0:
                reasoning_parts.append(f"La evidencia '{evidence}' apoya fuertemente '{hypothesis}' (+{change:.2f})")
            else:
                reasoning_parts.append(f"La evidencia '{evidence}' contradice '{hypothesis}' ({change:.2f})")
        
        # Describir decisión
        if decision_type == 'positive_action':
            reasoning_parts.append("El análisis cuántico sugiere una respuesta positiva y proactiva.")
        elif decision_type == 'negative_action':
            reasoning_parts.append("El análisis cuántico indica precaución o acción correctiva.")
        else:
            reasoning_parts.append("El estado cuántico permanece en superposición, requiriendo más información.")
        
        return " ".join(reasoning_parts)
    
    def _assess_consciousness_integration(self) -> Dict[str, float]:
        """Evalúa la integración con el sistema de conciencia"""
        
        integration_metrics = {
            'neurotransmitter_influence': 0.0,
            'emotional_coherence': 0.0,
            'cognitive_alignment': 0.0,
            'quantum_consciousness_sync': 0.0
        }
        
        # Analizar influencia de neurotransmisores
        nt_events = [name for name in self.inference_engine.events.keys() 
                    if name.startswith('neurotransmitter_')]
        
        if nt_events:
            nt_entropies = [self.inference_engine.events[name].density_matrix.von_neumann_entropy() 
                           for name in nt_events]
            integration_metrics['neurotransmitter_influence'] = 1.0 - np.mean(nt_entropies)
        
        # Analizar coherencia emocional
        emotional_events = [name for name in self.inference_engine.events.keys() 
                           if 'emotional' in name]
        
        if emotional_events:
            emotional_purities = [self.inference_engine.events[name].density_matrix.purity() 
                                 for name in emotional_events]
            integration_metrics['emotional_coherence'] = np.mean(emotional_purities)
        
        # Analizar alineación cognitiva
        total_correlations = 0
        correlation_count = 0
        
        event_names = list(self.inference_engine.events.keys())
        for i, event_a in enumerate(event_names):
            for event_b in event_names[i+1:]:
                correlation = self.inference_engine.compute_quantum_correlation(event_a, event_b)
                if 'fidelity' in correlation:
                    total_correlations += correlation['fidelity']
                    correlation_count += 1
        
        if correlation_count > 0:
            integration_metrics['cognitive_alignment'] = total_correlations / correlation_count
        
        # Sincronización de conciencia cuántica
        system_state = self.inference_engine.get_system_state()
        integration_metrics['quantum_consciousness_sync'] = system_state.get('system_coherence', 0.0)
        
        return integration_metrics
    
    def get_quantum_insights(self) -> Dict[str, Any]:
        """Obtiene insights cuánticos del sistema"""
        
        insights = {
            'quantum_state_summary': self.inference_engine.get_system_state(),
            'consciousness_integration': self._assess_consciousness_integration(),
            'decision_patterns': self._analyze_decision_patterns(),
            'uncertainty_analysis': self._analyze_system_uncertainty(),
            'causal_network': self._analyze_causal_network()
        }
        
        return insights
    
    def _analyze_decision_patterns(self) -> Dict[str, Any]:
        """Analiza patrones en las decisiones tomadas"""
        
        recent_inferences = list(self.inference_engine.inference_history)[-20:]
        
        if not recent_inferences:
            return {'no_data': True}
        
        # Analizar tipos de cambios
        positive_changes = [inf for inf in recent_inferences if inf.get('probability_change', 0) > 0]
        negative_changes = [inf for inf in recent_inferences if inf.get('probability_change', 0) < 0]
        
        patterns = {
            'total_decisions': len(recent_inferences),
            'positive_bias': len(positive_changes) / len(recent_inferences),
            'negative_bias': len(negative_changes) / len(recent_inferences),
            'average_confidence': np.mean([inf.get('confidence', 0) for inf in recent_inferences]),
            'decision_volatility': np.std([inf.get('probability_change', 0) for inf in recent_inferences]),
            'entropy_trend': np.mean([inf.get('entropy_change', 0) for inf in recent_inferences])
        }
        
        return patterns
    
    def _analyze_system_uncertainty(self) -> Dict[str, float]:
        """Analiza niveles de incertidumbre en el sistema"""
        
        uncertainties = {}
        
        for name, event in self.inference_engine.events.items():
            # Incertidumbre basada en entropía
            entropy = event.density_matrix.von_neumann_entropy()
            uncertainty = entropy / np.log2(event.density_matrix.matrix.shape[0])  # Normalizar
            uncertainties[name] = uncertainty
        
        if uncertainties:
            avg_uncertainty = np.mean(list(uncertainties.values()))
            max_uncertainty = max(uncertainties.values())
            min_uncertainty = min(uncertainties.values())
        else:
            avg_uncertainty = max_uncertainty = min_uncertainty = 0.0
        
        return {
            'average_uncertainty': avg_uncertainty,
            'max_uncertainty': max_uncertainty,
            'min_uncertainty': min_uncertainty,
            'uncertainty_variance': np.var(list(uncertainties.values())) if uncertainties else 0.0,
            'event_uncertainties': uncertainties
        }
    
    def _analyze_causal_network(self) -> Dict[str, Any]:
        """Analiza la red de relaciones causales"""
        
        network_stats = {
            'total_nodes': len(self.inference_engine.events),
            'total_edges': sum(len(relations) for relations in self.inference_engine.causal_relationships.values()),
            'network_density': 0.0,
            'strongly_connected_components': 0,
            'average_causal_strength': 0.0
        }
        
        # Calcular densidad de red
        max_possible_edges = len(self.inference_engine.events) * (len(self.inference_engine.events) - 1)
        if max_possible_edges > 0:
            network_stats['network_density'] = network_stats['total_edges'] / max_possible_edges
        
        # Calcular fuerza causal promedio
        all_strengths = []
        for relations in self.inference_engine.causal_relationships.values():
            all_strengths.extend([rel['strength'] for rel in relations])
        
        if all_strengths:
            network_stats['average_causal_strength'] = np.mean(all_strengths)
        
        return network_stats
    
    def integrate_with_ganst(self, ganst_activation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integra datos de activación de GANST con el sistema Bayesiano cuántico
        
        Args:
            ganst_activation_data: Datos de activación del núcleo GANST
        
        Returns:
            Resultado de la integración
        """
        # Almacenar activación GANST
        self.ganst_integration['activation_buffer'].append(ganst_activation_data)
        
        # Extraer información relevante
        activation_tensor = ganst_activation_data.get('activation_tensor')
        coherence_score = ganst_activation_data.get('coherence_score', 0.5)
        neural_state = ganst_activation_data.get('neural_state', 'active')
        
        # Convertir activación GANST a eventos bayesianos
        ganst_event_name = f"ganst_activation_{ganst_activation_data.get('activation_id', 'unknown')}"
        
        # Registrar evento de activación GANST
        if ganst_event_name not in self.inference_engine.events:
            # Crear estado cuántico inicial basado en tensor de activación
            if activation_tensor is not None:
                if hasattr(activation_tensor, 'numpy'):
                    activation_array = activation_tensor.numpy()
                else:
                    activation_array = np.array(activation_tensor)
                
                # Normalizar y usar primeras dimensiones como estado cuántico
                state_vector = activation_array[:4] / (np.linalg.norm(activation_array[:4]) + 1e-8)
                
                ganst_event = self.inference_engine.register_event(
                    ganst_event_name,
                    initial_state=state_vector,
                    prior_probability=coherence_score
                )
            else:
                ganst_event = self.inference_engine.register_event(ganst_event_name)
        
        # Crear hipótesis de consciencia basada en estado neural GANST
        consciousness_hypothesis = f"consciousness_state_{neural_state}"
        
        # Realizar actualización Bayesiana
        update_result = self.inference_engine.quantum_bayes_update(
            consciousness_hypothesis,
            ganst_event_name,
            evidence_data={
                'coherence_score': coherence_score,
                'neural_state': neural_state,
                'activation_magnitude': float(np.linalg.norm(activation_array)) if activation_tensor is not None else 0.0
            },
            confidence=coherence_score
        )
        
        # Actualizar sincronización de coherencia
        self.ganst_integration['coherence_sync'] = (
            self.ganst_integration['coherence_sync'] * 0.9 + coherence_score * 0.1
        )
        self.ganst_integration['last_ganst_sync'] = datetime.now()
        
        # Generar respuesta integrada
        integration_result = {
            'ganst_integration': {
                'event_created': ganst_event_name,
                'hypothesis_updated': consciousness_hypothesis,
                'bayes_update': update_result,
                'coherence_sync': self.ganst_integration['coherence_sync'],
                'integration_timestamp': datetime.now().isoformat()
            },
            'quantum_influence': self._calculate_quantum_influence_on_ganst(activation_tensor),
            'feedback_recommendations': self._generate_ganst_feedback(update_result)
        }
        
        return integration_result
    
    def _calculate_quantum_influence_on_ganst(self, activation_tensor) -> Dict[str, float]:
        """Calcula influencia cuántica en el sistema GANST"""
        if activation_tensor is None:
            return {'quantum_coherence': 0.0, 'entanglement_strength': 0.0}
        
        if hasattr(activation_tensor, 'numpy'):
            tensor_array = activation_tensor.numpy()
        else:
            tensor_array = np.array(activation_tensor)
        
        # Calcular métricas cuánticas
        tensor_norm = np.linalg.norm(tensor_array)
        tensor_entropy = -np.sum(np.abs(tensor_array) * np.log(np.abs(tensor_array) + 1e-8))
        
        quantum_coherence = min(1.0, tensor_norm / 10.0)  # Normalizar
        entanglement_strength = 1.0 / (1.0 + tensor_entropy)  # Inverso de entropía
        
        return {
            'quantum_coherence': quantum_coherence,
            'entanglement_strength': entanglement_strength,
            'tensor_magnitude': tensor_norm,
            'information_content': tensor_entropy
        }
    
    def _generate_ganst_feedback(self, bayes_update_result: Dict) -> List[str]:
        """Genera recomendaciones de retroalimentación para GANST"""
        feedback = []
        
        probability_change = bayes_update_result.get('probability_change', 0)
        entropy_change = bayes_update_result.get('entropy_change', 0)
        
        if probability_change > 0.3:
            feedback.append("Aumentar frecuencia de activaciones similares")
        elif probability_change < -0.3:
            feedback.append("Reducir patrones de activación similares")
        
        if entropy_change > 0.2:
            feedback.append("Incrementar diversidad en síntesis de tensores")
        elif entropy_change < -0.2:
            feedback.append("Enfocar síntesis hacia patrones más coherentes")
        
        if not feedback:
            feedback.append("Mantener configuración actual de activación")
        
        return feedback
    
    def get_ganst_integration_status(self) -> Dict[str, Any]:
        """Obtiene estado de integración con GANST"""
        return {
            'buffer_size': len(self.ganst_integration['activation_buffer']),
            'coherence_sync': self.ganst_integration['coherence_sync'],
            'last_sync': self.ganst_integration['last_ganst_sync'].isoformat() 
                         if self.ganst_integration['last_sync'] else None,
            'recent_activations': len([
                act for act in self.ganst_integration['activation_buffer']
                if (datetime.now() - act.get('timestamp', datetime.min)).seconds < 60
            ]) if self.ganst_integration['activation_buffer'] else 0
        }
