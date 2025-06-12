"""
Sistema de Memorias de Corto Plazo - Ruth R1
Buffer dinámico con decaimiento temporal y consolidación adaptativa
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
import math
from collections import defaultdict, deque
import json
import logging

class MemoryType(Enum):
    """Tipos de memoria de corto plazo"""
    SENSORY = "sensory"           # Memoria sensorial (300ms-2s)
    WORKING = "working"           # Memoria de trabajo (15-30s)
    BUFFER = "buffer"            # Buffer intermedio (2-15s)
    EPISODIC = "episodic"        # Episódica temporal (30s-5min)
    EMOTIONAL = "emotional"       # Emocional intensificada (variable)

class MemoryPriority(Enum):
    """Prioridades de memoria"""
    CRITICAL = 1.0
    HIGH = 0.8
    MEDIUM = 0.5
    LOW = 0.3
    MINIMAL = 0.1

@dataclass
class MemoryTrace:
    """Rastro de memoria individual"""
    content: torch.Tensor
    memory_type: MemoryType
    priority: float
    timestamp: float
    decay_rate: float
    consolidation_threshold: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    access_count: int = 0
    last_access: float = 0.0
    emotional_weight: float = 0.0
    
    def __post_init__(self):
        if self.last_access == 0.0:
            self.last_access = self.timestamp

class MemoryDecayFunction:
    """Funciones de decaimiento temporal para memoria"""
    
    @staticmethod
    def exponential_decay(initial_strength: float, 
                         elapsed_time: float, 
                         decay_rate: float) -> float:
        """Decaimiento exponencial clásico"""
        return initial_strength * math.exp(-decay_rate * elapsed_time)
    
    @staticmethod
    def power_law_decay(initial_strength: float, 
                       elapsed_time: float, 
                       decay_factor: float = 0.8) -> float:
        """Decaimiento por ley de potencias (más realista cognitivamente)"""
        if elapsed_time <= 0:
            return initial_strength
        return initial_strength * math.pow(elapsed_time + 1, -decay_factor)
    
    @staticmethod
    def forgetting_curve(initial_strength: float, 
                        elapsed_time: float, 
                        retention_factor: float = 1.84) -> float:
        """Curva de olvido de Ebbinghaus"""
        if elapsed_time <= 0:
            return initial_strength
        return initial_strength * math.exp(-elapsed_time / retention_factor)
    
    @staticmethod
    def emotional_modulated_decay(initial_strength: float,
                                 elapsed_time: float,
                                 decay_rate: float,
                                 emotional_weight: float) -> float:
        """Decaimiento modulado por peso emocional"""
        # Las memorias emocionales decaen más lentamente
        modulated_rate = decay_rate * (1.0 - emotional_weight * 0.7)
        return initial_strength * math.exp(-modulated_rate * elapsed_time)

class ShortTermMemoryBuffer:
    """Buffer individual de memoria de corto plazo"""
    
    def __init__(self, 
                 memory_type: MemoryType,
                 max_capacity: int = 100,
                 default_decay_rate: float = 0.1,
                 consolidation_threshold: float = 0.7):
        self.memory_type = memory_type
        self.max_capacity = max_capacity
        self.default_decay_rate = default_decay_rate
        self.consolidation_threshold = consolidation_threshold
        
        self.traces: Dict[str, MemoryTrace] = {}
        self.access_pattern = deque(maxlen=1000)
        self.consolidation_candidates = []
        
        self.lock = threading.Lock()
        
        # Configuración específica por tipo de memoria
        self._configure_memory_parameters()
    
    def _configure_memory_parameters(self):
        """Configura parámetros específicos según tipo de memoria"""
        config = {
            MemoryType.SENSORY: {
                'default_duration': 2.0,  # 2 segundos
                'decay_rate': 2.0,        # Decaimiento rápido
                'capacity': 20
            },
            MemoryType.BUFFER: {
                'default_duration': 15.0,  # 15 segundos
                'decay_rate': 0.3,
                'capacity': 50
            },
            MemoryType.WORKING: {
                'default_duration': 30.0,  # 30 segundos
                'decay_rate': 0.1,
                'capacity': 100
            },
            MemoryType.EPISODIC: {
                'default_duration': 300.0,  # 5 minutos
                'decay_rate': 0.05,
                'capacity': 200
            },
            MemoryType.EMOTIONAL: {
                'default_duration': 600.0,  # 10 minutos
                'decay_rate': 0.02,
                'capacity': 150
            }
        }
        
        params = config.get(self.memory_type, config[MemoryType.WORKING])
        self.default_duration = params['default_duration']
        self.default_decay_rate = params['decay_rate']
        self.max_capacity = params['capacity']
    
    def store(self, 
             content: torch.Tensor,
             trace_id: Optional[str] = None,
             priority: float = 0.5,
             metadata: Dict[str, Any] = None,
             emotional_weight: float = 0.0) -> str:
        """Almacena un nuevo rastro de memoria"""
        
        current_time = time.time()
        if trace_id is None:
            trace_id = f"{self.memory_type.value}_{int(current_time * 1000)}"
        
        # Crear rastro de memoria
        trace = MemoryTrace(
            content=content.clone(),
            memory_type=self.memory_type,
            priority=priority,
            timestamp=current_time,
            decay_rate=self.default_decay_rate,
            consolidation_threshold=self.consolidation_threshold,
            metadata=metadata or {},
            emotional_weight=emotional_weight
        )
        
        with self.lock:
            # Verificar capacidad y hacer espacio si es necesario
            if len(self.traces) >= self.max_capacity:
                self._make_space()
            
            self.traces[trace_id] = trace
            
            # Registrar patrón de acceso
            self.access_pattern.append({
                'trace_id': trace_id,
                'action': 'store',
                'timestamp': current_time,
                'priority': priority
            })
        
        return trace_id
    
    def retrieve(self, trace_id: str) -> Optional[torch.Tensor]:
        """Recupera un rastro de memoria específico"""
        with self.lock:
            if trace_id not in self.traces:
                return None
            
            trace = self.traces[trace_id]
            current_time = time.time()
            
            # Calcular fuerza actual considerando decaimiento
            elapsed_time = current_time - trace.timestamp
            
            if trace.emotional_weight > 0.0:
                current_strength = MemoryDecayFunction.emotional_modulated_decay(
                    1.0, elapsed_time, trace.decay_rate, trace.emotional_weight
                )
            else:
                current_strength = MemoryDecayFunction.power_law_decay(
                    1.0, elapsed_time, trace.decay_rate
                )
            
            # Si la memoria es demasiado débil, removerla
            if current_strength < 0.05:
                del self.traces[trace_id]
                return None
            
            # Actualizar estadísticas de acceso
            trace.access_count += 1
            trace.last_access = current_time
            
            # Registrar acceso
            self.access_pattern.append({
                'trace_id': trace_id,
                'action': 'retrieve',
                'timestamp': current_time,
                'strength': current_strength
            })
            
            # Fortalecer ligeramente por acceso (efecto de ensayo)
            if current_strength > 0.3:
                trace.priority = min(trace.priority * 1.05, 1.0)
            
            # Devolver contenido modulado por fuerza actual
            return trace.content * current_strength
    
    def search_by_similarity(self, 
                           query_tensor: torch.Tensor, 
                           top_k: int = 5,
                           threshold: float = 0.7) -> List[Tuple[str, float, torch.Tensor]]:
        """Busca memorias por similaridad de contenido"""
        
        candidates = []
        current_time = time.time()
        
        with self.lock:
            for trace_id, trace in self.traces.items():
                # Calcular similaridad
                similarity = F.cosine_similarity(
                    query_tensor.flatten().unsqueeze(0),
                    trace.content.flatten().unsqueeze(0)
                ).item()
                
                if similarity >= threshold:
                    # Calcular fuerza actual
                    elapsed_time = current_time - trace.timestamp
                    current_strength = MemoryDecayFunction.power_law_decay(
                        1.0, elapsed_time, trace.decay_rate
                    )
                    
                    # Combinar similaridad con fuerza y prioridad
                    combined_score = similarity * current_strength * trace.priority
                    
                    candidates.append((trace_id, combined_score, trace.content))
            
            # Ordenar por puntuación combinada
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            return candidates[:top_k]
    
    def get_active_memories(self, strength_threshold: float = 0.1) -> Dict[str, Dict[str, Any]]:
        """Obtiene memorias actualmente activas por encima del umbral"""
        
        active_memories = {}
        current_time = time.time()
        
        with self.lock:
            for trace_id, trace in list(self.traces.items()):
                elapsed_time = current_time - trace.timestamp
                
                if trace.emotional_weight > 0.0:
                    current_strength = MemoryDecayFunction.emotional_modulated_decay(
                        1.0, elapsed_time, trace.decay_rate, trace.emotional_weight
                    )
                else:
                    current_strength = MemoryDecayFunction.power_law_decay(
                        1.0, elapsed_time, trace.decay_rate
                    )
                
                if current_strength >= strength_threshold:
                    active_memories[trace_id] = {
                        'content': trace.content,
                        'strength': current_strength,
                        'priority': trace.priority,
                        'access_count': trace.access_count,
                        'age': elapsed_time,
                        'emotional_weight': trace.emotional_weight,
                        'metadata': trace.metadata
                    }
                else:
                    # Remover memorias muy débiles
                    del self.traces[trace_id]
        
        return active_memories
    
    def _make_space(self):
        """Libera espacio removiendo memorias más débiles"""
        current_time = time.time()
        
        # Evaluar todas las memorias
        memory_scores = []
        for trace_id, trace in self.traces.items():
            elapsed_time = current_time - trace.timestamp
            current_strength = MemoryDecayFunction.power_law_decay(
                1.0, elapsed_time, trace.decay_rate
            )
            
            # Puntuación combinada para decidir qué mantener
            combined_score = (
                current_strength * 0.4 +
                trace.priority * 0.3 +
                trace.emotional_weight * 0.2 +
                min(trace.access_count / 10.0, 1.0) * 0.1
            )
            
            memory_scores.append((trace_id, combined_score))
        
        # Remover las 20% memorias con menor puntuación
        memory_scores.sort(key=lambda x: x[1])
        to_remove = int(len(memory_scores) * 0.2)
        
        for trace_id, _ in memory_scores[:to_remove]:
            del self.traces[trace_id]
    
    def consolidate_memories(self) -> List[Dict[str, Any]]:
        """Identifica memorias candidatas para consolidación a largo plazo"""
        
        candidates = []
        current_time = time.time()
        
        with self.lock:
            for trace_id, trace in self.traces.items():
                # Criterios para consolidación
                age = current_time - trace.timestamp
                
                consolidation_score = (
                    trace.priority * 0.3 +
                    min(trace.access_count / 5.0, 1.0) * 0.3 +
                    trace.emotional_weight * 0.2 +
                    min(age / self.default_duration, 1.0) * 0.2
                )
                
                if consolidation_score >= self.consolidation_threshold:
                    candidates.append({
                        'trace_id': trace_id,
                        'content': trace.content,
                        'consolidation_score': consolidation_score,
                        'metadata': trace.metadata,
                        'memory_type': self.memory_type.value,
                        'age': age,
                        'access_count': trace.access_count
                    })
        
        # Ordenar por puntuación de consolidación
        candidates.sort(key=lambda x: x['consolidation_score'], reverse=True)
        
        return candidates
    
    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas del buffer de memoria"""
        current_time = time.time()
        
        stats = {
            'memory_type': self.memory_type.value,
            'total_traces': len(self.traces),
            'max_capacity': self.max_capacity,
            'utilization': len(self.traces) / self.max_capacity,
            'average_age': 0.0,
            'average_strength': 0.0,
            'high_priority_count': 0,
            'emotional_count': 0,
            'access_patterns': {}
        }
        
        if self.traces:
            ages = []
            strengths = []
            
            for trace in self.traces.values():
                age = current_time - trace.timestamp
                ages.append(age)
                
                strength = MemoryDecayFunction.power_law_decay(1.0, age, trace.decay_rate)
                strengths.append(strength)
                
                if trace.priority > 0.7:
                    stats['high_priority_count'] += 1
                    
                if trace.emotional_weight > 0.3:
                    stats['emotional_count'] += 1
            
            stats['average_age'] = np.mean(ages)
            stats['average_strength'] = np.mean(strengths)
        
        # Analizar patrones de acceso recientes
        recent_access = [a for a in self.access_pattern if current_time - a['timestamp'] < 60.0]
        access_types = defaultdict(int)
        for access in recent_access:
            access_types[access['action']] += 1
        
        stats['access_patterns'] = dict(access_types)
        
        return stats

class ShortTermMemorySystem:
    """Sistema completo de memoria de corto plazo"""
    
    def __init__(self):
        self.buffers: Dict[MemoryType, ShortTermMemoryBuffer] = {}
        self.cross_modal_associations = {}
        self.consolidation_queue = deque()
        self.system_statistics = {
            'total_stores': 0,
            'total_retrievals': 0,
            'consolidations_performed': 0,
            'cross_modal_links': 0
        }
        
        self.maintenance_thread = None
        self.is_running = False
        self.lock = threading.Lock()
        
        self._initialize_buffers()
        
    def _initialize_buffers(self):
        """Inicializa todos los buffers de memoria"""
        for memory_type in MemoryType:
            self.buffers[memory_type] = ShortTermMemoryBuffer(memory_type)
    
    def start(self):
        """Inicia el sistema de memoria"""
        if self.is_running:
            return
            
        self.is_running = True
        
        # Iniciar hilo de mantenimiento
        self.maintenance_thread = threading.Thread(
            target=self._maintenance_loop,
            daemon=True
        )
        self.maintenance_thread.start()
        
        logging.info("Sistema de memoria de corto plazo iniciado")
    
    def stop(self):
        """Detiene el sistema de memoria"""
        self.is_running = False
        if self.maintenance_thread:
            self.maintenance_thread.join(timeout=5.0)
            
        logging.info("Sistema de memoria de corto plazo detenido")
    
    def store_memory(self,
                    content: torch.Tensor,
                    memory_type: MemoryType = MemoryType.WORKING,
                    priority: float = 0.5,
                    metadata: Dict[str, Any] = None,
                    emotional_weight: float = 0.0,
                    trace_id: Optional[str] = None) -> str:
        """Almacena una memoria en el buffer apropiado"""
        
        buffer = self.buffers[memory_type]
        stored_id = buffer.store(
            content, trace_id, priority, metadata, emotional_weight
        )
        
        with self.lock:
            self.system_statistics['total_stores'] += 1
        
        # Crear asociaciones cross-modales si hay metadata
        if metadata and 'modality' in metadata:
            self._create_cross_modal_association(stored_id, metadata)
        
        return stored_id
    
    def retrieve_memory(self, 
                       trace_id: str, 
                       memory_type: Optional[MemoryType] = None) -> Optional[torch.Tensor]:
        """Recupera una memoria específica"""
        
        if memory_type:
            # Buscar en buffer específico
            result = self.buffers[memory_type].retrieve(trace_id)
        else:
            # Buscar en todos los buffers
            result = None
            for buffer in self.buffers.values():
                result = buffer.retrieve(trace_id)
                if result is not None:
                    break
        
        if result is not None:
            with self.lock:
                self.system_statistics['total_retrievals'] += 1
        
        return result
    
    def search_memories(self,
                       query_tensor: torch.Tensor,
                       memory_types: Optional[List[MemoryType]] = None,
                       top_k: int = 10,
                       threshold: float = 0.7) -> List[Tuple[str, float, torch.Tensor, MemoryType]]:
        """Busca memorias por similaridad en múltiples buffers"""
        
        search_types = memory_types or list(MemoryType)
        all_candidates = []
        
        for memory_type in search_types:
            buffer = self.buffers[memory_type]
            candidates = buffer.search_by_similarity(query_tensor, top_k, threshold)
            
            for trace_id, score, content in candidates:
                all_candidates.append((trace_id, score, content, memory_type))
        
        # Ordenar todos los candidatos por puntuación
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return all_candidates[:top_k]
    
    def get_working_memory_content(self) -> Dict[str, torch.Tensor]:
        """Obtiene el contenido actual de la memoria de trabajo"""
        working_buffer = self.buffers[MemoryType.WORKING]
        active_memories = working_buffer.get_active_memories()
        
        content = {}
        for trace_id, memory_info in active_memories.items():
            content[trace_id] = memory_info['content']
        
        return content
    
    def get_sensory_buffer_state(self) -> Dict[str, Any]:
        """Obtiene el estado del buffer sensorial"""
        sensory_buffer = self.buffers[MemoryType.SENSORY]
        return sensory_buffer.get_statistics()
    
    def consolidate_to_long_term(self) -> List[Dict[str, Any]]:
        """Realiza consolidación a memoria de largo plazo"""
        
        all_candidates = []
        
        # Recopilar candidatos de todos los buffers
        for memory_type, buffer in self.buffers.items():
            candidates = buffer.consolidate_memories()
            all_candidates.extend(candidates)
        
        # Priorizar candidatos para consolidación
        all_candidates.sort(key=lambda x: x['consolidation_score'], reverse=True)
        
        # Tomar los mejores candidatos (max 20 por ciclo)
        to_consolidate = all_candidates[:20]
        
        with self.lock:
            self.system_statistics['consolidations_performed'] += len(to_consolidate)
        
        # Agregar a cola de consolidación
        for candidate in to_consolidate:
            self.consolidation_queue.append(candidate)
        
        return to_consolidate
    
    def _create_cross_modal_association(self, trace_id: str, metadata: Dict[str, Any]):
        """Crea asociaciones entre modalidades"""
        modality = metadata.get('modality')
        if not modality:
            return
        
        # Buscar memorias relacionadas en otras modalidades
        current_time = time.time()
        time_window = 5.0  # 5 segundos de ventana temporal
        
        related_traces = []
        for memory_type, buffer in self.buffers.items():
            for tid, trace in buffer.traces.items():
                if tid != trace_id and abs(trace.timestamp - current_time) < time_window:
                    trace_modality = trace.metadata.get('modality')
                    if trace_modality and trace_modality != modality:
                        related_traces.append(tid)
        
        if related_traces:
            self.cross_modal_associations[trace_id] = related_traces
            with self.lock:
                self.system_statistics['cross_modal_links'] += len(related_traces)
    
    def _maintenance_loop(self):
        """Bucle de mantenimiento del sistema"""
        while self.is_running:
            try:
                time.sleep(10.0)  # Mantenimiento cada 10 segundos
                
                # Limpieza de memorias expiradas
                self._cleanup_expired_memories()
                
                # Consolidación automática
                if len(self.consolidation_queue) < 50:
                    self.consolidate_to_long_term()
                
                # Optimización de asociaciones cross-modales
                self._optimize_cross_modal_associations()
                
            except Exception as e:
                logging.error(f"Error en mantenimiento de memoria: {e}")
    
    def _cleanup_expired_memories(self):
        """Limpia memorias completamente expiradas"""
        for buffer in self.buffers.values():
            # Forzar limpieza de memorias débiles
            buffer.get_active_memories(strength_threshold=0.02)
    
    def _optimize_cross_modal_associations(self):
        """Optimiza asociaciones cross-modales"""
        current_time = time.time()
        expired_associations = []
        
        for trace_id, related_traces in self.cross_modal_associations.items():
            # Verificar si las asociaciones aún son válidas
            main_exists = any(
                trace_id in buffer.traces 
                for buffer in self.buffers.values()
            )
            
            if not main_exists:
                expired_associations.append(trace_id)
                continue
            
            # Filtrar trazas relacionadas que ya no existen
            valid_related = []
            for related_id in related_traces:
                related_exists = any(
                    related_id in buffer.traces 
                    for buffer in self.buffers.values()
                )
                if related_exists:
                    valid_related.append(related_id)
            
            if valid_related:
                self.cross_modal_associations[trace_id] = valid_related
            else:
                expired_associations.append(trace_id)
        
        # Remover asociaciones expiradas
        for trace_id in expired_associations:
            del self.cross_modal_associations[trace_id]
    
    def get_system_state(self) -> Dict[str, Any]:
        """Obtiene el estado completo del sistema de memoria"""
        
        state = {
            'is_running': self.is_running,
            'statistics': self.system_statistics.copy(),
            'buffers': {},
            'consolidation_queue_size': len(self.consolidation_queue),
            'cross_modal_associations': len(self.cross_modal_associations),
            'timestamp': time.time()
        }
        
        # Estadísticas de cada buffer
        for memory_type, buffer in self.buffers.items():
            state['buffers'][memory_type.value] = buffer.get_statistics()
        
        return state
    
    def get_memory_landscape(self) -> Dict[str, Any]:
        """Obtiene un panorama de toda la memoria activa"""
        
        landscape = {
            'active_memories': {},
            'memory_distribution': defaultdict(int),
            'strength_distribution': defaultdict(int),
            'emotional_memories': [],
            'high_priority_memories': [],
            'cross_modal_clusters': []
        }
        
        # Analizar cada buffer
        for memory_type, buffer in self.buffers.items():
            active_memories = buffer.get_active_memories()
            
            landscape['active_memories'][memory_type.value] = len(active_memories)
            
            for trace_id, memory_info in active_memories.items():
                # Distribución por fuerza
                strength_bucket = int(memory_info['strength'] * 10) / 10
                landscape['strength_distribution'][strength_bucket] += 1
                
                # Memorias emocionales
                if memory_info['emotional_weight'] > 0.5:
                    landscape['emotional_memories'].append({
                        'trace_id': trace_id,
                        'memory_type': memory_type.value,
                        'emotional_weight': memory_info['emotional_weight'],
                        'strength': memory_info['strength']
                    })
                
                # Memorias de alta prioridad
                if memory_info['priority'] > 0.8:
                    landscape['high_priority_memories'].append({
                        'trace_id': trace_id,
                        'memory_type': memory_type.value,
                        'priority': memory_info['priority'],
                        'strength': memory_info['strength']
                    })
        
        return landscape

# Instancia global del sistema de memoria
short_term_memory = ShortTermMemorySystem()

def initialize_short_term_memory() -> ShortTermMemorySystem:
    """Inicializa el sistema global de memoria de corto plazo"""
    short_term_memory.start()
    return short_term_memory

def get_short_term_memory() -> ShortTermMemorySystem:
    """Obtiene la instancia global del sistema de memoria"""
    return short_term_memory

# Funciones de conveniencia
def store_sensory_memory(content: torch.Tensor, metadata: Dict[str, Any] = None) -> str:
    """Almacena memoria sensorial"""
    return short_term_memory.store_memory(
        content, MemoryType.SENSORY, priority=0.3, metadata=metadata
    )

def store_working_memory(content: torch.Tensor, priority: float = 0.7, metadata: Dict[str, Any] = None) -> str:
    """Almacena memoria de trabajo"""
    return short_term_memory.store_memory(
        content, MemoryType.WORKING, priority=priority, metadata=metadata
    )

def store_emotional_memory(content: torch.Tensor, emotional_weight: float, metadata: Dict[str, Any] = None) -> str:
    """Almacena memoria emocional"""
    return short_term_memory.store_memory(
        content, MemoryType.EMOTIONAL, 
        priority=0.8, emotional_weight=emotional_weight, metadata=metadata
    )

def search_recent_memories(query: torch.Tensor, top_k: int = 5) -> List[Tuple[str, float, torch.Tensor, MemoryType]]:
    """Busca memorias recientes por similaridad"""
    return short_term_memory.search_memories(query, top_k=top_k, threshold=0.6)

def get_working_memory_state() -> Dict[str, Any]:
    """Obtiene estado actual de memoria de trabajo"""
    return short_term_memory.get_system_state()