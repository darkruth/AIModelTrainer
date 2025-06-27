
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime
import math
from collections import deque
import logging

class KUbitMemoryRouter:
    """
    Submódulo k-ubit: Supernodo meta enrutador de recuerdos vividos
    
    Implementa la ecuación matemática del flujo de trabajo para memorias
    envueltas en emoción al momento de su grabado.
    
    Ecuación implementada:
    μᵢ(t) = MetaRoute_sin/cos(Θ(t)) · μᵢ(0)
    ξᵢ(t) = Decode(μᵢ(t))
    𝒩(t) = (Ξ(t), 𝒞)
    Θ(t) = ωt
    Mem_vol(t) = Σ ξᵢ(t) · wᵢⱼ(t) · sin(Θ(t))
    Contexto_final(t) = ZeroShot(Mem_vol(t), Prompt_meta)
    """
    
    def __init__(self, dim=768, omega=0.1):
        self.dim = dim
        self.omega = omega  # Frecuencia cognitiva interna
        self.start_time = datetime.now()
        
        # Almacenamiento de nodos semánticos iniciales
        self.semantic_nodes = {}  # μᵢ(0)
        self.active_contexts = {}  # 𝒞
        self.memory_cache = deque(maxlen=1000)
        
        # Red de conexiones contextuales
        self.contextual_weights = {}  # wᵢⱼ(t)
        
        # Proveedor de contexto emocional
        self.emotional_context_provider = None
        
        # Decodificador semántico
        self.semantic_decoder = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim),
            nn.Tanh()
        )
        
        logging.info("k-ubit Memory Router inicializado")
    
    def set_emotional_context_provider(self, provider: Callable):
        """Establece proveedor de contexto emocional"""
        self.emotional_context_provider = provider
    
    def register_memory_with_emotion(self, memory_content: Any, emotion_wrapper: Dict[str, float]) -> str:
        """
        Registra una memoria envuelta en emoción
        
        Args:
            memory_content: Contenido de la memoria
            emotion_wrapper: Estado emocional al momento del grabado
        
        Returns:
            ID del nodo semántico creado
        """
        # Generar vector semántico inicial μᵢ(0)
        memory_id = f"mem_{len(self.semantic_nodes)}_{int(datetime.now().timestamp())}"
        
        # Codificar memoria con contexto emocional
        if isinstance(memory_content, str):
            # Encodificación simple para texto
            content_vector = self._encode_text_memory(memory_content)
        elif isinstance(memory_content, (list, tuple)):
            content_vector = torch.tensor(memory_content[:self.dim], dtype=torch.float32)
            if len(content_vector) < self.dim:
                padding = torch.zeros(self.dim - len(content_vector))
                content_vector = torch.cat([content_vector, padding])
        else:
            content_vector = torch.randn(self.dim)
        
        # Aplicar modulación emocional
        emotion_modulation = self._create_emotion_modulation(emotion_wrapper)
        initial_semantic_vector = content_vector * emotion_modulation
        
        # Almacenar nodo semántico inicial
        self.semantic_nodes[memory_id] = {
            'mu_0': initial_semantic_vector,
            'emotion_wrapper': emotion_wrapper,
            'content': memory_content,
            'timestamp': datetime.now(),
            'access_count': 0
        }
        
        logging.info(f"Memoria registrada con k-ubit: {memory_id}")
        return memory_id
    
    def route_memory_query(self, query: str, emotional_context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Enruta consulta de memoria usando el sistema k-ubit
        
        Implementa el flujo completo de la ecuación matemática
        """
        current_time = (datetime.now() - self.start_time).total_seconds()
        
        # 1. Calcular fase de rotación contextual: Θ(t) = ωt
        theta_t = self.omega * current_time
        
        # 2. Obtener contexto emocional actual
        if emotional_context is None and self.emotional_context_provider:
            emotional_context = self.emotional_context_provider()
        
        # 3. Procesar cada nodo semántico
        active_projections = {}  # ξᵢ(t)
        
        for memory_id, memory_data in self.semantic_nodes.items():
            # Representación dinámica del nodo: μᵢ(t) = MetaRoute(Θ(t)) · μᵢ(0)
            mu_0 = memory_data['mu_0']
            mu_t = self._meta_route_transform(mu_0, theta_t)
            
            # Proyección cognitiva observable: ξᵢ(t) = Decode(μᵢ(t))
            xi_t = self._semantic_decode(mu_t)
            
            active_projections[memory_id] = {
                'xi_t': xi_t,
                'mu_t': mu_t,
                'memory_data': memory_data
            }
        
        # 4. Formar red activa: 𝒩(t) = (Ξ(t), 𝒞)
        active_network = self._form_active_network(active_projections, query)
        
        # 5. Calcular memoria volátil acumulativa
        volatile_memory = self._calculate_volatile_memory(active_projections, theta_t, query)
        
        # 6. Proceso de inferencia sin entrenamiento (zero-shot)
        final_context = self._zero_shot_inference(volatile_memory, query, emotional_context)
        
        # Actualizar estadísticas de acceso
        self._update_access_statistics(active_network['relevant_memories'])
        
        result = {
            'routed_memories': active_network['relevant_memories'],
            'volatile_memory_magnitude': float(torch.norm(volatile_memory)),
            'final_context': final_context,
            'theta_t': theta_t,
            'active_projections_count': len(active_projections),
            'emotional_influence': self._calculate_emotional_influence(emotional_context),
            'routing_timestamp': datetime.now().isoformat()
        }
        
        # Cachear resultado
        self.memory_cache.append({
            'query': query,
            'result': result,
            'timestamp': datetime.now()
        })
        
        return result
    
    def _encode_text_memory(self, text: str) -> torch.Tensor:
        """Codifica memoria textual a vector semántico"""
        # Codificación simple basada en hash y características del texto
        text_hash = hash(text) % (2**31)
        
        # Crear vector base desde hash
        np.random.seed(text_hash)
        base_vector = torch.tensor(np.random.randn(self.dim), dtype=torch.float32)
        
        # Añadir características semánticas simples
        word_count = len(text.split())
        char_count = len(text)
        sentiment_proxy = sum(ord(c) for c in text[:10]) / (10 * 255)  # Proxy simple
        
        # Modular vector con características
        modulation = torch.tensor([
            word_count / 100.0,
            char_count / 1000.0,
            sentiment_proxy
        ])
        
        # Aplicar modulación a las primeras dimensiones
        base_vector[:3] = base_vector[:3] * modulation
        
        return F.normalize(base_vector, dim=0)
    
    def _create_emotion_modulation(self, emotion_wrapper: Dict[str, float]) -> torch.Tensor:
        """Crea vector de modulación emocional"""
        modulation = torch.ones(self.dim)
        
        # Mapear emociones principales a modulaciones
        emotion_mappings = {
            'joy': (0.8, 1.2),
            'sadness': (0.5, 0.8),
            'anger': (1.1, 1.5),
            'fear': (0.6, 0.9),
            'surprise': (0.9, 1.4),
            'disgust': (0.4, 0.7),
            'neutral': (0.9, 1.1)
        }
        
        for emotion, intensity in emotion_wrapper.items():
            if emotion in emotion_mappings:
                low, high = emotion_mappings[emotion]
                emotion_factor = low + (high - low) * intensity
                
                # Aplicar modulación a segmentos específicos del vector
                start_idx = hash(emotion) % (self.dim - 100)
                end_idx = start_idx + 100
                modulation[start_idx:end_idx] *= emotion_factor
        
        return modulation
    
    def _meta_route_transform(self, mu_0: torch.Tensor, theta_t: float) -> torch.Tensor:
        """
        Aplica transformación MetaRoute_sin/cos(Θ(t))
        
        Implementa rotación posicional estilo RoPE
        """
        # Crear matriz de rotación
        cos_theta = math.cos(theta_t)
        sin_theta = math.sin(theta_t)
        
        # Aplicar rotación posicional a pares de dimensiones
        mu_t = mu_0.clone()
        
        for i in range(0, self.dim - 1, 2):
            x = mu_0[i]
            y = mu_0[i + 1]
            
            # Rotación 2D
            mu_t[i] = x * cos_theta - y * sin_theta
            mu_t[i + 1] = x * sin_theta + y * cos_theta
        
        return mu_t
    
    def _semantic_decode(self, mu_t: torch.Tensor) -> torch.Tensor:
        """Decodificador semántico de superficie"""
        with torch.no_grad():
            return self.semantic_decoder(mu_t.unsqueeze(0)).squeeze(0)
    
    def _form_active_network(self, active_projections: Dict, query: str) -> Dict:
        """Forma la red activa en tiempo real"""
        query_embedding = self._encode_text_memory(query)
        
        # Calcular similitudes y formar conexiones contextuales
        relevant_memories = []
        
        for memory_id, projection_data in active_projections.items():
            xi_t = projection_data['xi_t']
            memory_data = projection_data['memory_data']
            
            # Calcular similitud contextual
            similarity = float(F.cosine_similarity(query_embedding, xi_t, dim=0))
            
            # Filtrar por umbral de relevancia
            if similarity > 0.3:  # Umbral configurable
                relevant_memories.append({
                    'memory_id': memory_id,
                    'similarity': similarity,
                    'content': memory_data['content'],
                    'emotion_wrapper': memory_data['emotion_wrapper'],
                    'xi_t': xi_t
                })
        
        # Ordenar por relevancia
        relevant_memories.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {
            'relevant_memories': relevant_memories[:10],  # Top 10
            'total_activated': len(active_projections),
            'contextual_connections': self._update_contextual_weights(relevant_memories)
        }
    
    def _calculate_volatile_memory(self, active_projections: Dict, theta_t: float, query: str) -> torch.Tensor:
        """
        Calcula memoria volátil acumulativa:
        Mem_vol(t) = Σ ξᵢ(t) · wᵢⱼ(t) · sin(Θ(t))
        """
        volatile_memory = torch.zeros(self.dim)
        sin_theta = math.sin(theta_t)
        
        query_embedding = self._encode_text_memory(query)
        
        for memory_id, projection_data in active_projections.items():
            xi_t = projection_data['xi_t']
            
            # Peso contextual wᵢⱼ(t)
            contextual_weight = float(F.cosine_similarity(query_embedding, xi_t, dim=0))
            
            # Acumular en memoria volátil
            contribution = xi_t * contextual_weight * sin_theta
            volatile_memory += contribution
        
        return volatile_memory
    
    def _zero_shot_inference(self, volatile_memory: torch.Tensor, query: str, emotional_context: Optional[Dict]) -> str:
        """
        Proceso de inferencia sin entrenamiento:
        Contexto_final(t) = ZeroShot(Mem_vol(t), Prompt_meta)
        """
        # Analizar magnitud y dirección de la memoria volátil
        memory_magnitude = float(torch.norm(volatile_memory))
        memory_direction = F.normalize(volatile_memory, dim=0)
        
        # Generar contexto basado en patrones emergentes
        context_insights = []
        
        # Análisis de magnitud
        if memory_magnitude > 1.0:
            context_insights.append("Fuerte resonancia en memorias relacionadas")
        elif memory_magnitude > 0.5:
            context_insights.append("Resonancia moderada en memorias")
        else:
            context_insights.append("Resonancia débil en memorias")
        
        # Análisis emocional si está disponible
        if emotional_context:
            dominant_emotion = max(emotional_context.items(), key=lambda x: x[1])
            context_insights.append(f"Contexto emocional dominante: {dominant_emotion[0]}")
        
        # Análisis de patrón direccional
        direction_analysis = self._analyze_memory_direction(memory_direction)
        context_insights.extend(direction_analysis)
        
        final_context = f"Consulta: '{query}' | " + " | ".join(context_insights)
        
        return final_context
    
    def _analyze_memory_direction(self, direction: torch.Tensor) -> List[str]:
        """Analiza la dirección semántica de la memoria volátil"""
        insights = []
        
        # Analizar distribución de activaciones
        positive_ratio = float(torch.sum(direction > 0)) / len(direction)
        
        if positive_ratio > 0.7:
            insights.append("Patrón de activación predominantemente positivo")
        elif positive_ratio < 0.3:
            insights.append("Patrón de activación predominantemente negativo")
        else:
            insights.append("Patrón de activación equilibrado")
        
        # Analizar concentración
        std_dev = float(torch.std(direction))
        if std_dev > 0.5:
            insights.append("Alta variabilidad en activación")
        else:
            insights.append("Activación concentrada")
        
        return insights
    
    def _update_contextual_weights(self, relevant_memories: List[Dict]) -> Dict:
        """Actualiza pesos contextuales wᵢⱼ(t)"""
        connections = {}
        
        for i, mem1 in enumerate(relevant_memories):
            for j, mem2 in enumerate(relevant_memories[i+1:], i+1):
                connection_key = f"{mem1['memory_id']}-{mem2['memory_id']}"
                
                # Calcular peso basado en similitud emocional y semántica
                emotional_similarity = self._calculate_emotional_similarity(
                    mem1['emotion_wrapper'], 
                    mem2['emotion_wrapper']
                )
                
                semantic_similarity = float(F.cosine_similarity(
                    mem1['xi_t'], 
                    mem2['xi_t'], 
                    dim=0
                ))
                
                weight = (emotional_similarity + semantic_similarity) / 2.0
                connections[connection_key] = weight
                
                # Almacenar para uso futuro
                self.contextual_weights[connection_key] = weight
        
        return connections
    
    def _calculate_emotional_similarity(self, emotion1: Dict, emotion2: Dict) -> float:
        """Calcula similitud entre estados emocionales"""
        common_emotions = set(emotion1.keys()).intersection(set(emotion2.keys()))
        
        if not common_emotions:
            return 0.0
        
        similarities = []
        for emotion in common_emotions:
            val1 = emotion1[emotion]
            val2 = emotion2[emotion]
            similarity = 1.0 - abs(val1 - val2)
            similarities.append(similarity)
        
        return np.mean(similarities)
    
    def _calculate_emotional_influence(self, emotional_context: Optional[Dict]) -> float:
        """Calcula influencia emocional en el enrutamiento"""
        if not emotional_context:
            return 0.0
        
        # Intensidad emocional total
        total_intensity = sum(emotional_context.values())
        return min(1.0, total_intensity)
    
    def _update_access_statistics(self, relevant_memories: List[Dict]):
        """Actualiza estadísticas de acceso a memorias"""
        for memory_info in relevant_memories:
            memory_id = memory_info['memory_id']
            if memory_id in self.semantic_nodes:
                self.semantic_nodes[memory_id]['access_count'] += 1
                self.semantic_nodes[memory_id]['last_access'] = datetime.now()
    
    def get_system_state(self) -> Dict[str, Any]:
        """Obtiene estado del sistema k-ubit"""
        current_time = (datetime.now() - self.start_time).total_seconds()
        
        # Estadísticas de memorias
        total_memories = len(self.semantic_nodes)
        total_accesses = sum(mem['access_count'] for mem in self.semantic_nodes.values())
        
        # Análisis emocional de memorias almacenadas
        emotion_distribution = {}
        for memory_data in self.semantic_nodes.values():
            for emotion, intensity in memory_data['emotion_wrapper'].items():
                if emotion in emotion_distribution:
                    emotion_distribution[emotion].append(intensity)
                else:
                    emotion_distribution[emotion] = [intensity]
        
        # Promedio por emoción
        emotion_averages = {
            emotion: np.mean(intensities) 
            for emotion, intensities in emotion_distribution.items()
        }
        
        return {
            'system_name': 'k-ubit Memory Router',
            'uptime_seconds': current_time,
            'current_theta': self.omega * current_time,
            'total_memories': total_memories,
            'total_accesses': total_accesses,
            'contextual_connections': len(self.contextual_weights),
            'cache_size': len(self.memory_cache),
            'emotion_distribution': emotion_averages,
            'omega_frequency': self.omega,
            'timestamp': datetime.now().isoformat()
        }
    
    def optimize_omega_frequency(self, target_coherence: float = 0.8):
        """Optimiza la frecuencia omega para mejor coherencia"""
        current_coherence = self._measure_system_coherence()
        
        if current_coherence < target_coherence:
            # Ajustar omega basado en performance
            if current_coherence < 0.5:
                self.omega *= 1.1  # Incrementar frecuencia
            else:
                self.omega *= 0.95  # Decrementar ligeramente
            
            # Mantener omega en rango válido
            self.omega = max(0.01, min(1.0, self.omega))
            
            logging.info(f"Frecuencia omega ajustada a: {self.omega:.4f}")
    
    def _measure_system_coherence(self) -> float:
        """Mide coherencia del sistema basada en accesos recientes"""
        if len(self.memory_cache) < 2:
            return 0.5
        
        recent_queries = list(self.memory_cache)[-10:]
        coherence_scores = []
        
        for i in range(len(recent_queries) - 1):
            query1 = recent_queries[i]
            query2 = recent_queries[i + 1]
            
            # Medir consistencia en resultados
            mag1 = query1['result']['volatile_memory_magnitude']
            mag2 = query2['result']['volatile_memory_magnitude']
            
            if mag1 > 0 and mag2 > 0:
                consistency = 1.0 - abs(mag1 - mag2) / max(mag1, mag2)
                coherence_scores.append(consistency)
        
        return np.mean(coherence_scores) if coherence_scores else 0.5
