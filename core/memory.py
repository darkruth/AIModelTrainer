import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from datetime import datetime, timedelta
import json

from utils.logger import Logger

class MemoriaCortoplazo(nn.Module):
    """
    Sistema de memoria a corto plazo basado en LSTM
    Implementa la arquitectura del documento original
    """
    
    def __init__(self, input_size=784, hidden_size=128, num_layers=1):
        super(MemoriaCortoplazo, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM principal para memoria a corto plazo
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        
        # Atención para enfoque selectivo
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True
        )
        
        # Normalización
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x, hidden_state=None):
        """
        Forward pass de la memoria a corto plazo
        
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            hidden_state: Estado oculto previo (opcional)
        
        Returns:
            output: Salida procesada
            hidden_state: Nuevo estado oculto
        """
        
        # Procesar con LSTM
        lstm_out, hidden_state = self.lstm(x, hidden_state)
        
        # Aplicar atención
        attended_out, attention_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual connection y normalización
        output = self.layer_norm(lstm_out + attended_out)
        
        return output, hidden_state, attention_weights

class MemoriaLargoplazo:
    """
    Sistema de memoria a largo plazo basado en consolidación y recuperación
    """
    
    def __init__(self, max_memories=10000):
        self.memories = deque(maxlen=max_memories)
        self.semantic_memory = {}  # Memoria semántica
        self.episodic_memory = deque(maxlen=1000)  # Memoria episódica
        self.consolidated_memories = {}  # Memorias consolidadas
        
        self.logger = Logger()
    
    def store_memory(self, content: Any, memory_type: str, importance: float = 0.5, 
                    context: Dict = None):
        """
        Almacena una memoria en el sistema a largo plazo
        
        Args:
            content: Contenido de la memoria
            memory_type: Tipo ('semantic', 'episodic', 'procedural')
            importance: Importancia (0.0-1.0)
            context: Contexto adicional
        """
        
        memory_entry = {
            'id': len(self.memories),
            'content': content,
            'type': memory_type,
            'importance': importance,
            'timestamp': datetime.now(),
            'access_count': 0,
            'last_accessed': datetime.now(),
            'context': context or {},
            'consolidated': False
        }
        
        if memory_type == 'semantic':
            # Extraer conceptos clave para memoria semántica
            key = self._extract_semantic_key(content)
            if key not in self.semantic_memory:
                self.semantic_memory[key] = []
            self.semantic_memory[key].append(memory_entry)
        
        elif memory_type == 'episodic':
            self.episodic_memory.append(memory_entry)
        
        self.memories.append(memory_entry)
        
        self.logger.log("DEBUG", f"Memory stored: {memory_type}, importance: {importance}")
    
    def retrieve_memory(self, query: str, memory_type: str = None, 
                       limit: int = 10) -> List[Dict]:
        """
        Recupera memorias basado en una consulta
        
        Args:
            query: Consulta de búsqueda
            memory_type: Tipo de memoria a buscar (opcional)
            limit: Número máximo de resultados
        
        Returns:
            Lista de memorias relevantes
        """
        
        relevant_memories = []
        
        # Buscar en diferentes tipos de memoria
        if memory_type is None or memory_type == 'semantic':
            relevant_memories.extend(self._search_semantic_memory(query, limit//2))
        
        if memory_type is None or memory_type == 'episodic':
            relevant_memories.extend(self._search_episodic_memory(query, limit//2))
        
        # Ordenar por relevancia y recencia
        relevant_memories.sort(key=lambda m: (
            m['relevance_score'] * 0.7 + 
            m['importance'] * 0.3
        ), reverse=True)
        
        # Actualizar estadísticas de acceso
        for memory in relevant_memories[:limit]:
            memory['access_count'] += 1
            memory['last_accessed'] = datetime.now()
        
        return relevant_memories[:limit]
    
    def _extract_semantic_key(self, content: Any) -> str:
        """Extrae una clave semántica del contenido"""
        
        if isinstance(content, str):
            # Extraer palabras clave principales
            words = content.lower().split()
            # Filtrar palabras comunes
            stop_words = {'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le'}
            key_words = [w for w in words if w not in stop_words and len(w) > 3]
            return '_'.join(key_words[:3])  # Usar las primeras 3 palabras clave
        else:
            return str(type(content).__name__)
    
    def _search_semantic_memory(self, query: str, limit: int) -> List[Dict]:
        """Busca en la memoria semántica"""
        
        results = []
        query_words = set(query.lower().split())
        
        for key, memories in self.semantic_memory.items():
            key_words = set(key.split('_'))
            
            # Calcular relevancia basada en solapamiento de palabras
            relevance = len(query_words.intersection(key_words)) / len(query_words.union(key_words))
            
            if relevance > 0.1:  # Umbral mínimo de relevancia
                for memory in memories:
                    memory_copy = memory.copy()
                    memory_copy['relevance_score'] = relevance
                    results.append(memory_copy)
        
        return sorted(results, key=lambda m: m['relevance_score'], reverse=True)[:limit]
    
    def _search_episodic_memory(self, query: str, limit: int) -> List[Dict]:
        """Busca en la memoria episódica"""
        
        results = []
        query_lower = query.lower()
        
        for memory in self.episodic_memory:
            if isinstance(memory['content'], str):
                content_lower = memory['content'].lower()
                
                # Calcular relevancia basada en coincidencias de texto
                relevance = 0.0
                for word in query_lower.split():
                    if word in content_lower:
                        relevance += 1.0 / len(query_lower.split())
                
                if relevance > 0.0:
                    memory_copy = memory.copy()
                    memory_copy['relevance_score'] = relevance
                    results.append(memory_copy)
        
        return sorted(results, key=lambda m: m['relevance_score'], reverse=True)[:limit]
    
    def consolidate_memories(self):
        """
        Consolida memorias importantes para almacenamiento a largo plazo
        Simula el proceso de consolidación durante el "sueño"
        """
        
        consolidation_threshold = 0.7
        access_threshold = 3
        
        for memory in list(self.memories):
            if (not memory['consolidated'] and 
                (memory['importance'] >= consolidation_threshold or 
                 memory['access_count'] >= access_threshold)):
                
                # Consolidar memoria
                consolidated_key = f"consolidated_{memory['id']}"
                self.consolidated_memories[consolidated_key] = {
                    'original_memory': memory,
                    'consolidation_date': datetime.now(),
                    'strength': memory['importance'] + (memory['access_count'] * 0.1)
                }
                
                memory['consolidated'] = True
                
                self.logger.log("DEBUG", f"Memory consolidated: {memory['id']}")
    
    def forget_old_memories(self, days_threshold: int = 30):
        """
        Olvida memorias antiguas y poco importantes
        Simula el proceso natural de olvido
        """
        
        cutoff_date = datetime.now() - timedelta(days=days_threshold)
        forgotten_count = 0
        
        # Filtrar memorias no consolidadas y antiguas
        filtered_memories = deque(maxlen=self.memories.maxlen)
        
        for memory in self.memories:
            should_keep = (
                memory['consolidated'] or
                memory['importance'] > 0.6 or
                memory['timestamp'] > cutoff_date or
                memory['access_count'] > 2
            )
            
            if should_keep:
                filtered_memories.append(memory)
            else:
                forgotten_count += 1
        
        self.memories = filtered_memories
        
        if forgotten_count > 0:
            self.logger.log("INFO", f"Forgot {forgotten_count} old memories")

class MemoriaSystem:
    """
    Sistema integrado de memoria que combina memoria a corto y largo plazo
    """
    
    def __init__(self, input_size=784, hidden_size=128):
        self.short_term = MemoriaCortoplazo(input_size, hidden_size)
        self.long_term = MemoriaLargoplazo()
        
        # Estado de memoria de trabajo
        self.working_memory = {}
        self.current_context = {}
        
        # Configuración
        self.consolidation_frequency = 100  # Cada N interacciones
        self.interaction_count = 0
        
        self.logger = Logger()
        
        self.logger.log("INFO", "MemoriaSystem initialized")
    
    def process_input(self, input_data: torch.Tensor, context: Dict = None) -> torch.Tensor:
        """
        Procesa input a través del sistema de memoria a corto plazo
        
        Args:
            input_data: Datos de entrada
            context: Contexto adicional
        
        Returns:
            Salida procesada por la memoria
        """
        
        # Procesar con memoria a corto plazo
        output, hidden_state, attention_weights = self.short_term(input_data)
        
        # Actualizar memoria de trabajo
        self.working_memory['last_output'] = output
        self.working_memory['hidden_state'] = hidden_state
        self.working_memory['attention_weights'] = attention_weights
        
        if context:
            self.current_context.update(context)
        
        return output
    
    def store_interaction(self, input_text: str, output_text: str, 
                         neurotransmitter_levels: Dict):
        """
        Almacena una interacción en la memoria a largo plazo
        
        Args:
            input_text: Texto de entrada
            output_text: Texto de salida
            neurotransmitter_levels: Niveles actuales de neurotransmisores
        """
        
        # Calcular importancia basada en factores múltiples
        importance = self._calculate_interaction_importance(
            input_text, output_text, neurotransmitter_levels
        )
        
        # Crear contexto de la interacción
        context = {
            'neurotransmitters': neurotransmitter_levels.copy(),
            'working_memory_state': {
                'attention_pattern': self.working_memory.get('attention_weights'),
                'context': self.current_context.copy()
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Almacenar como memoria episódica
        interaction_content = {
            'input': input_text,
            'output': output_text,
            'interaction_type': self._classify_interaction(input_text)
        }
        
        self.long_term.store_memory(
            content=interaction_content,
            memory_type='episodic',
            importance=importance,
            context=context
        )
        
        # Extraer y almacenar conceptos semánticos
        self._extract_and_store_concepts(input_text, output_text, importance)
        
        self.interaction_count += 1
        
        # Consolidación periódica
        if self.interaction_count % self.consolidation_frequency == 0:
            self.consolidate_memories()
        
        self.logger.log("DEBUG", f"Interaction stored with importance: {importance:.3f}")
    
    def _calculate_interaction_importance(self, input_text: str, output_text: str, 
                                        neurotransmitter_levels: Dict) -> float:
        """Calcula la importancia de una interacción"""
        
        base_importance = 0.5
        
        # Factor de longitud y complejidad
        complexity_factor = min(1.0, (len(input_text) + len(output_text)) / 1000.0)
        
        # Factor emocional basado en neurotransmisores
        emotion_factor = 0.0
        serotonin = neurotransmitter_levels.get('serotonin', 7.0)
        dopamine = neurotransmitter_levels.get('dopamine', 50.0)
        oxytocin = neurotransmitter_levels.get('oxytocin', 5.0)
        
        # Experiencias con alta dopamina son más importantes (aprendizaje)
        if dopamine > 60:
            emotion_factor += 0.2
        
        # Experiencias con alta oxitocina son importantes (sociales)
        if oxytocin > 8:
            emotion_factor += 0.15
        
        # Experiencias con serotonina muy alta o baja son memorables
        if serotonin > 12 or serotonin < 4:
            emotion_factor += 0.1
        
        # Factor de novedad (palabras poco comunes)
        novelty_factor = self._calculate_novelty(input_text, output_text)
        
        # Combinar factores
        importance = base_importance + complexity_factor * 0.3 + emotion_factor + novelty_factor * 0.2
        
        return min(1.0, importance)  # Limitar a 1.0
    
    def _classify_interaction(self, input_text: str) -> str:
        """Clasifica el tipo de interacción"""
        
        input_lower = input_text.lower()
        
        if '?' in input_text:
            return 'question'
        elif any(word in input_lower for word in ['siento', 'emoción', 'feliz', 'triste']):
            return 'emotional'
        elif any(word in input_lower for word in ['crear', 'inventar', 'imaginar']):
            return 'creative'
        elif any(word in input_lower for word in ['explicar', 'cómo', 'qué', 'por qué']):
            return 'educational'
        else:
            return 'general'
    
    def _extract_and_store_concepts(self, input_text: str, output_text: str, importance: float):
        """Extrae y almacena conceptos semánticos"""
        
        # Conceptos del input
        input_concepts = self._extract_concepts(input_text)
        for concept in input_concepts:
            self.long_term.store_memory(
                content=f"concepto: {concept}",
                memory_type='semantic',
                importance=importance * 0.8,
                context={'source': 'user_input', 'full_text': input_text}
            )
        
        # Conceptos del output
        output_concepts = self._extract_concepts(output_text)
        for concept in output_concepts:
            self.long_term.store_memory(
                content=f"concepto: {concept}",
                memory_type='semantic',
                importance=importance * 0.7,
                context={'source': 'ai_output', 'full_text': output_text}
            )
    
    def _extract_concepts(self, text: str) -> List[str]:
        """Extrae conceptos principales del texto"""
        
        # Implementación simplificada
        words = text.lower().split()
        
        # Filtrar palabras importantes
        important_words = []
        for word in words:
            if (len(word) > 4 and 
                word not in ['que', 'esta', 'esto', 'como', 'para', 'con', 'por']):
                important_words.append(word)
        
        return important_words[:5]  # Top 5 conceptos
    
    def _calculate_novelty(self, input_text: str, output_text: str) -> float:
        """Calcula la novedad de la interacción"""
        
        all_text = input_text + ' ' + output_text
        words = set(all_text.lower().split())
        
        # Buscar palabras en memoria semántica existente
        existing_concepts = set()
        for key in self.long_term.semantic_memory.keys():
            existing_concepts.update(key.split('_'))
        
        # Calcular ratio de palabras nuevas
        new_words = words - existing_concepts
        novelty_ratio = len(new_words) / max(len(words), 1)
        
        return min(1.0, novelty_ratio)
    
    def retrieve_relevant_memories(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Recupera memorias relevantes para una consulta
        
        Args:
            query: Consulta de búsqueda
            limit: Número máximo de memorias a recuperar
        
        Returns:
            Lista de memorias relevantes
        """
        
        return self.long_term.retrieve_memory(query, limit=limit)
    
    def get_memory_context(self, query: str) -> Dict:
        """
        Obtiene contexto de memoria para una consulta
        
        Args:
            query: Consulta actual
        
        Returns:
            Diccionario con contexto de memoria relevante
        """
        
        relevant_memories = self.retrieve_relevant_memories(query, limit=3)
        
        context = {
            'relevant_interactions': [],
            'relevant_concepts': [],
            'memory_summary': {}
        }
        
        for memory in relevant_memories:
            if memory['type'] == 'episodic':
                context['relevant_interactions'].append({
                    'input': memory['content'].get('input', ''),
                    'output': memory['content'].get('output', ''),
                    'timestamp': memory['timestamp'].isoformat(),
                    'importance': memory['importance']
                })
            elif memory['type'] == 'semantic':
                context['relevant_concepts'].append(memory['content'])
        
        # Estadísticas de memoria
        context['memory_summary'] = {
            'total_memories': len(self.long_term.memories),
            'consolidated_memories': len(self.long_term.consolidated_memories),
            'semantic_concepts': len(self.long_term.semantic_memory),
            'recent_interactions': len([m for m in self.long_term.episodic_memory 
                                      if (datetime.now() - m['timestamp']).days < 7])
        }
        
        return context
    
    def consolidate_memories(self):
        """Ejecuta consolidación de memorias"""
        self.long_term.consolidate_memories()
        self.logger.log("INFO", "Memory consolidation completed")
    
    def cleanup_old_memories(self):
        """Limpia memorias antiguas"""
        self.long_term.forget_old_memories()
        self.logger.log("INFO", "Old memory cleanup completed")
    
    def get_memory_stats(self) -> Dict:
        """Obtiene estadísticas del sistema de memoria"""
        
        return {
            'short_term': {
                'input_size': self.short_term.input_size,
                'hidden_size': self.short_term.hidden_size,
                'current_state': 'active' if self.working_memory else 'idle'
            },
            'long_term': {
                'total_memories': len(self.long_term.memories),
                'episodic_memories': len(self.long_term.episodic_memory),
                'semantic_concepts': len(self.long_term.semantic_memory),
                'consolidated_memories': len(self.long_term.consolidated_memories)
            },
            'interactions': {
                'total_interactions': self.interaction_count,
                'consolidations_performed': self.interaction_count // self.consolidation_frequency
            }
        }
    
    def check_health(self) -> bool:
        """Verifica el estado de salud del sistema de memoria"""
        
        try:
            # Verificar que los componentes principales funcionen
            test_input = torch.randn(1, 10, self.short_term.input_size)
            _ = self.short_term(test_input)
            
            # Verificar memoria a largo plazo
            _ = self.long_term.retrieve_memory("test", limit=1)
            
            return True
        except Exception as e:
            self.logger.log("ERROR", f"Memory system health check failed: {str(e)}")
            return False
