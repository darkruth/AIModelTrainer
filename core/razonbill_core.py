"""
RazonBill Core - Motor de Inferencia Local
Sistema de razonamiento avanzado con contexto dinámico y memoria persistente
"""

import os
import sys
import json
import time
import uuid
import asyncio
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import deque
import logging

# Imports principales
import numpy as np
import pandas as pd
import importlib.util
import select
from pathlib import Path

# ChromaDB para memoria vectorial
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("ChromaDB no disponible - instalando...")

# LangChain modificado
try:
    from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.agents import Tool, AgentExecutor, create_react_agent
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChain no disponible completamente")

# Sistema de voz
try:
    import speech_recognition as sr
    import pyttsx3
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    print("Componentes de voz no disponibles")

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ThoughtStep:
    """Representa un paso en el proceso de pensamiento"""
    step_id: str
    thought_type: str  # 'analysis', 'action', 'reflection', 'conclusion'
    content: str
    timestamp: datetime
    confidence: float
    parent_step: Optional[str] = None
    children_steps: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.children_steps is None:
            self.children_steps = []

@dataclass
class ContextualMemory:
    """Memoria contextual para persistencia"""
    memory_id: str
    content: str
    embedding: List[float]
    timestamp: datetime
    context_type: str  # 'conversation', 'knowledge', 'experience'
    relevance_score: float = 0.0
    access_count: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class PromptLoggingWrapper:
    """Wrapper para logging y análisis de prompts"""
    
    def __init__(self, log_dir: str = "logs/prompts"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = str(uuid.uuid4())
        self.prompt_history = []
        
    def log_prompt(self, prompt: str, response: str, metadata: Dict = None):
        """Registra un prompt y su respuesta"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'prompt': prompt,
            'response': response,
            'metadata': metadata or {},
            'prompt_id': str(uuid.uuid4())
        }
        
        self.prompt_history.append(entry)
        
        # Guardar en archivo
        log_file = self.log_dir / f"session_{self.session_id[:8]}.jsonl"
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    def analyze_prompt_patterns(self) -> Dict[str, Any]:
        """Analiza patrones en los prompts"""
        if not self.prompt_history:
            return {}
        
        patterns = {
            'total_prompts': len(self.prompt_history),
            'avg_prompt_length': np.mean([len(p['prompt']) for p in self.prompt_history]),
            'avg_response_length': np.mean([len(p['response']) for p in self.prompt_history]),
            'session_duration': (datetime.now() - datetime.fromisoformat(self.prompt_history[0]['timestamp'])).total_seconds(),
            'common_keywords': self._extract_common_keywords()
        }
        
        return patterns
    
    def _extract_common_keywords(self) -> List[str]:
        """Extrae palabras clave comunes"""
        all_text = ' '.join([p['prompt'] + ' ' + p['response'] for p in self.prompt_history])
        words = all_text.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 3:  # Solo palabras de más de 3 caracteres
                word_freq[word] = word_freq.get(word, 0) + 1
        
        return sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]

class TreeOfThoughtsProcessor:
    """Procesador de árbol de pensamientos para razonamiento estructurado"""
    
    def __init__(self, max_depth: int = 5, branching_factor: int = 3):
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.thought_tree = {}
        self.current_path = []
        
    def generate_thought_tree(self, initial_prompt: str) -> Dict[str, ThoughtStep]:
        """Genera un árbol de pensamientos para el prompt dado"""
        root_id = str(uuid.uuid4())
        
        # Paso raíz - análisis inicial
        root_step = ThoughtStep(
            step_id=root_id,
            thought_type='analysis',
            content=f"Pensamiento: Analizo paso a paso la entrada: '{initial_prompt}'",
            timestamp=datetime.now(),
            confidence=1.0
        )
        
        self.thought_tree[root_id] = root_step
        self.current_path = [root_id]
        
        # Generar ramas de pensamiento
        self._expand_thought_branch(root_id, initial_prompt, depth=0)
        
        return self.thought_tree
    
    def _expand_thought_branch(self, parent_id: str, context: str, depth: int):
        """Expande una rama del árbol de pensamientos"""
        if depth >= self.max_depth:
            return
        
        parent_step = self.thought_tree[parent_id]
        
        # Generar pasos de pensamiento basados en el tipo del padre
        if parent_step.thought_type == 'analysis':
            # Después del análisis, considerar acciones
            child_types = ['action', 'reflection']
        elif parent_step.thought_type == 'action':
            # Después de una acción, evaluar resultado
            child_types = ['reflection', 'conclusion']
        else:
            # Reflexión puede llevar a más análisis o conclusión
            child_types = ['analysis', 'conclusion']
        
        for i, child_type in enumerate(child_types[:self.branching_factor]):
            child_id = str(uuid.uuid4())
            
            child_step = ThoughtStep(
                step_id=child_id,
                thought_type=child_type,
                content=self._generate_thought_content(child_type, context, parent_step),
                timestamp=datetime.now(),
                confidence=max(0.1, parent_step.confidence - 0.1 * depth),
                parent_step=parent_id
            )
            
            self.thought_tree[child_id] = child_step
            parent_step.children_steps.append(child_id)
            
            # Expandir recursivamente si no es conclusión
            if child_type != 'conclusion' and depth < self.max_depth - 1:
                self._expand_thought_branch(child_id, context, depth + 1)
    
    def _generate_thought_content(self, thought_type: str, context: str, parent: ThoughtStep) -> str:
        """Genera contenido específico para cada tipo de pensamiento"""
        templates = {
            'analysis': [
                f"Pensamiento: Descompongo el problema en partes: {context}",
                f"Pensamiento: Identifico los elementos clave en: {context}",
                f"Pensamiento: Analizo las relaciones en: {context}"
            ],
            'action': [
                f"Acción: Busco información sobre: {context}",
                f"Acción: Evalúo opciones para: {context}",
                f"Acción: Implemento estrategia para: {context}"
            ],
            'reflection': [
                f"Reflexión: Considero si el enfoque anterior es correcto",
                f"Reflexión: Evalúo la calidad de la información obtenida",
                f"Reflexión: Analizo posibles mejoras al razonamiento"
            ],
            'conclusion': [
                f"Conclusión: Basándome en el análisis, la respuesta es...",
                f"Conclusión: Sintetizando la información, concluyo que...",
                f"Conclusión: La solución más apropiada es..."
            ]
        }
        
        import random
        return random.choice(templates.get(thought_type, [f"Pensamiento sobre: {context}"]))
    
    def get_best_path(self) -> List[ThoughtStep]:
        """Obtiene la mejor ruta de razonamiento basada en confianza"""
        if not self.thought_tree:
            return []
        
        # Comenzar desde la raíz
        root_id = next(iter(self.thought_tree.keys()))
        best_path = []
        current_id = root_id
        
        while current_id:
            current_step = self.thought_tree[current_id]
            best_path.append(current_step)
            
            # Encontrar el mejor hijo basado en confianza
            if current_step.children_steps:
                best_child_id = max(
                    current_step.children_steps,
                    key=lambda child_id: self.thought_tree[child_id].confidence
                )
                current_id = best_child_id
            else:
                current_id = None
        
        return best_path

class VectorMemoryStore:
    """Almacén de memoria vectorial usando ChromaDB"""
    
    def __init__(self, collection_name: str = "razonbill_memory"):
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._initialize_store()
        
    def _initialize_store(self):
        """Inicializa el almacén vectorial"""
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB no disponible - usando memoria temporal")
            self.memory_cache = {}
            return
        
        try:
            # Configurar ChromaDB persistente
            self.client = chromadb.PersistentClient(
                path="./memory_store",
                settings=Settings(allow_reset=True)
            )
            
            # Crear o obtener colección
            try:
                self.collection = self.client.get_collection(self.collection_name)
            except:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "RazonBill contextual memory"}
                )
                
            logger.info(f"ChromaDB inicializado con colección: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Error inicializando ChromaDB: {e}")
            self.memory_cache = {}
    
    def store_memory(self, memory: ContextualMemory):
        """Almacena una memoria en el vector store"""
        if self.collection:
            try:
                self.collection.add(
                    embeddings=[memory.embedding],
                    documents=[memory.content],
                    metadatas=[{
                        'memory_id': memory.memory_id,
                        'timestamp': memory.timestamp.isoformat(),
                        'context_type': memory.context_type,
                        'relevance_score': memory.relevance_score,
                        'access_count': memory.access_count,
                        **memory.metadata
                    }],
                    ids=[memory.memory_id]
                )
            except Exception as e:
                logger.error(f"Error almacenando memoria: {e}")
        else:
            # Fallback a cache en memoria
            self.memory_cache[memory.memory_id] = memory
    
    def retrieve_similar_memories(self, query_embedding: List[float], 
                                 top_k: int = 5) -> List[ContextualMemory]:
        """Recupera memorias similares basadas en embedding"""
        if self.collection:
            try:
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k
                )
                
                memories = []
                for i in range(len(results['documents'][0])):
                    metadata = results['metadatas'][0][i]
                    memory = ContextualMemory(
                        memory_id=metadata['memory_id'],
                        content=results['documents'][0][i],
                        embedding=query_embedding,  # Aproximación
                        timestamp=datetime.fromisoformat(metadata['timestamp']),
                        context_type=metadata['context_type'],
                        relevance_score=metadata['relevance_score'],
                        access_count=metadata['access_count'],
                        metadata={k: v for k, v in metadata.items() 
                                if k not in ['memory_id', 'timestamp', 'context_type', 
                                           'relevance_score', 'access_count']}
                    )
                    memories.append(memory)
                
                return memories
                
            except Exception as e:
                logger.error(f"Error recuperando memorias: {e}")
                return []
        else:
            # Fallback simple - retornar memorias recientes
            return list(self.memory_cache.values())[-top_k:]
    
    def update_memory_access(self, memory_id: str):
        """Actualiza contador de acceso de una memoria"""
        if self.collection:
            try:
                # ChromaDB no soporta updates directos, necesitamos re-insertar
                pass  # Implementar si es necesario
            except Exception as e:
                logger.error(f"Error actualizando acceso de memoria: {e}")
        else:
            if memory_id in self.memory_cache:
                self.memory_cache[memory_id].access_count += 1

class ReActAgent:
    """Agente estilo ReAct (Reasoning + Acting) con herramientas Python"""
    
    def __init__(self, tools_dir: str = "tools"):
        self.tools_dir = Path(tools_dir)
        self.tools_dir.mkdir(exist_ok=True)
        self.available_tools = {}
        self.execution_history = []
        self._load_tools()
        
    def _load_tools(self):
        """Carga herramientas disponibles"""
        # Herramientas básicas integradas
        self.available_tools = {
            'search': self._search_tool,
            'calculate': self._calculate_tool,
            'file_read': self._file_read_tool,
            'file_write': self._file_write_tool,
            'python_exec': self._python_exec_tool,
            'web_request': self._web_request_tool
        }
        
        # Cargar herramientas personalizadas desde directorio
        if self.tools_dir.exists():
            for tool_file in self.tools_dir.glob("*.py"):
                try:
                    tool_name = tool_file.stem
                    # Importar dinámicamente
                    spec = importlib.util.spec_from_file_location(tool_name, tool_file)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    if hasattr(module, 'execute'):
                        self.available_tools[tool_name] = module.execute
                        
                except Exception as e:
                    logger.error(f"Error cargando herramienta {tool_file}: {e}")
    
    def execute_thought_action_cycle(self, input_text: str, max_iterations: int = 5) -> str:
        """Ejecuta ciclo de pensamiento-acción estilo ReAct"""
        thoughts = []
        actions = []
        results = []
        
        current_context = input_text
        
        for iteration in range(max_iterations):
            # Fase de Pensamiento
            thought = self._generate_thought(current_context, thoughts, actions, results)
            thoughts.append(thought)
            
            # Determinar si necesita acción
            if self._needs_action(thought):
                # Fase de Acción
                action = self._determine_action(thought, current_context)
                actions.append(action)
                
                # Ejecutar acción
                result = self._execute_action(action)
                results.append(result)
                
                # Actualizar contexto
                current_context = f"{current_context}\n\nResultado de {action}: {result}"
                
                # Verificar si la acción resolvió el problema
                if self._is_problem_solved(result, input_text):
                    break
            else:
                # Si no necesita acción, hemos terminado
                break
        
        # Generar respuesta final
        final_response = self._generate_final_response(
            input_text, thoughts, actions, results
        )
        
        # Registrar ejecución
        self.execution_history.append({
            'input': input_text,
            'thoughts': thoughts,
            'actions': actions,
            'results': results,
            'final_response': final_response,
            'timestamp': datetime.now()
        })
        
        return final_response
    
    def _generate_thought(self, context: str, prev_thoughts: List[str], 
                         prev_actions: List[str], prev_results: List[str]) -> str:
        """Genera un pensamiento basado en el contexto actual"""
        # Plantillas de pensamiento
        thought_templates = [
            f"Pensamiento: Necesito entender mejor: {context}",
            f"Pensamiento: Para resolver esto, debo analizar...",
            f"Pensamiento: Basándome en la información previa, creo que...",
            f"Pensamiento: El siguiente paso sería..."
        ]
        
        # Seleccionar plantilla basada en historial
        if not prev_thoughts:
            return f"Pensamiento: Analizo la pregunta: {context}"
        elif prev_results and "error" in str(prev_results[-1]).lower():
            return "Pensamiento: El enfoque anterior falló, intentaré otra estrategia"
        else:
            import random
            return random.choice(thought_templates)
    
    def _needs_action(self, thought: str) -> bool:
        """Determina si un pensamiento requiere una acción"""
        action_keywords = ['buscar', 'calcular', 'obtener', 'verificar', 'ejecutar', 'leer']
        return any(keyword in thought.lower() for keyword in action_keywords)
    
    def _determine_action(self, thought: str, context: str) -> str:
        """Determina qué acción tomar basada en el pensamiento"""
        thought_lower = thought.lower()
        
        if 'buscar' in thought_lower or 'google' in thought_lower:
            return f"search(\"{context}\")"
        elif 'calcular' in thought_lower or 'matemática' in thought_lower:
            return f"calculate(\"{context}\")"
        elif 'archivo' in thought_lower or 'leer' in thought_lower:
            return f"file_read(\"data.txt\")"
        elif 'ejecutar' in thought_lower or 'código' in thought_lower:
            return f"python_exec(\"print('Ejecutando análisis...')\")"
        else:
            return f"search(\"{context}\")"
    
    def _execute_action(self, action: str) -> str:
        """Ejecuta una acción específica"""
        try:
            # Parsear la acción
            if '(' in action and ')' in action:
                tool_name = action.split('(')[0]
                tool_args = action.split('(')[1].rstrip(')')
                
                # Limpiar argumentos
                tool_args = tool_args.strip('"\'')
                
                if tool_name in self.available_tools:
                    return str(self.available_tools[tool_name](tool_args))
                else:
                    return f"Error: Herramienta '{tool_name}' no disponible"
            else:
                return f"Error: Formato de acción inválido: {action}"
                
        except Exception as e:
            return f"Error ejecutando acción: {str(e)}"
    
    def _is_problem_solved(self, result: str, original_input: str) -> bool:
        """Determina si el problema ha sido resuelto"""
        # Heurística simple - si el resultado no contiene errores y tiene contenido
        return (
            "error" not in result.lower() and 
            len(result) > 20 and
            any(word in result.lower() for word in ['resultado', 'respuesta', 'solución'])
        )
    
    def _generate_final_response(self, input_text: str, thoughts: List[str], 
                               actions: List[str], results: List[str]) -> str:
        """Genera respuesta final basada en todo el proceso"""
        if results and not any("error" in str(r).lower() for r in results):
            # Si hay resultados exitosos
            return f"Basándome en el análisis realizado: {results[-1]}"
        elif thoughts:
            # Si solo hay pensamientos
            return f"Después de analizar tu pregunta: {thoughts[-1].replace('Pensamiento: ', '')}"
        else:
            # Respuesta por defecto
            return f"He analizado tu consulta sobre: {input_text}. Necesitaría más información para dar una respuesta más específica."
    
    # Herramientas básicas
    def _search_tool(self, query: str) -> str:
        """Simula búsqueda (placeholder para integración real)"""
        return f"Resultados de búsqueda para '{query}': [Información relevante encontrada]"
    
    def _calculate_tool(self, expression: str) -> str:
        """Evalúa expresiones matemáticas seguras"""
        try:
            # Solo permitir operaciones matemáticas básicas
            allowed_chars = set('0123456789+-*/.() ')
            if all(c in allowed_chars for c in expression):
                result = eval(expression)
                return f"Resultado: {result}"
            else:
                return "Error: Expresión contiene caracteres no permitidos"
        except Exception as e:
            return f"Error en cálculo: {str(e)}"
    
    def _file_read_tool(self, filename: str) -> str:
        """Lee contenido de archivo"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()[:1000]  # Limitar a 1000 caracteres
                return f"Contenido de {filename}: {content}"
        except Exception as e:
            return f"Error leyendo archivo: {str(e)}"
    
    def _file_write_tool(self, content: str) -> str:
        """Escribe contenido a archivo temporal"""
        try:
            filename = f"temp_output_{int(time.time())}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Contenido guardado en: {filename}"
        except Exception as e:
            return f"Error escribiendo archivo: {str(e)}"
    
    def _python_exec_tool(self, code: str) -> str:
        """Ejecuta código Python seguro"""
        try:
            # Lista blanca de funciones permitidas
            safe_dict = {
                '__builtins__': {
                    'print': print,
                    'len': len,
                    'str': str,
                    'int': int,
                    'float': float,
                    'sum': sum,
                    'max': max,
                    'min': min,
                    'round': round
                }
            }
            
            # Capturar salida
            from io import StringIO
            import contextlib
            
            output = StringIO()
            with contextlib.redirect_stdout(output):
                exec(code, safe_dict)
            
            result = output.getvalue()
            return result if result else "Código ejecutado exitosamente"
            
        except Exception as e:
            return f"Error en ejecución Python: {str(e)}"
    
    def _web_request_tool(self, url: str) -> str:
        """Realiza solicitud web simple"""
        return f"Simulación de solicitud a: {url} - [Contenido web obtenido]"

class RLFeedbackSystem:
    """Sistema de retroalimentación con Reinforcement Learning"""
    
    def __init__(self):
        self.feedback_history = []
        self.response_scores = {}
        self.heuristic_penalties = {
            'vague_response': -0.5,
            'incorrect_information': -1.0,
            'incomplete_answer': -0.3,
            'off_topic': -0.8,
            'good_response': +1.0,
            'excellent_response': +1.5
        }
        
    def evaluate_response(self, response: str, user_input: str, 
                         user_feedback: Optional[str] = None) -> float:
        """Evalúa la calidad de una respuesta usando heurísticas"""
        score = 0.0
        penalties = []
        
        # Heurística 1: Longitud de respuesta
        if len(response) < 20:
            score += self.heuristic_penalties['vague_response']
            penalties.append('respuesta_vaga')
        elif len(response) > 50:
            score += 0.2  # Bonus por respuesta detallada
        
        # Heurística 2: Palabras clave relevantes
        input_words = set(user_input.lower().split())
        response_words = set(response.lower().split())
        
        relevance = len(input_words.intersection(response_words)) / len(input_words) if input_words else 0
        score += relevance * 0.5
        
        # Heurística 3: Estructuras de respuesta
        if any(phrase in response.lower() for phrase in ['no sé', 'no puedo', 'no estoy seguro']):
            score += self.heuristic_penalties['incomplete_answer']
            penalties.append('respuesta_incompleta')
        
        # Heurística 4: Palabras de calidad
        quality_indicators = ['porque', 'debido a', 'por ejemplo', 'específicamente', 'resultado']
        quality_score = sum(1 for indicator in quality_indicators if indicator in response.lower())
        score += quality_score * 0.1
        
        # Incorporar feedback del usuario si está disponible
        if user_feedback:
            user_score = self._parse_user_feedback(user_feedback)
            score += user_score
        
        # Registrar evaluación
        evaluation = {
            'response': response,
            'user_input': user_input,
            'score': score,
            'penalties': penalties,
            'timestamp': datetime.now(),
            'user_feedback': user_feedback
        }
        
        self.feedback_history.append(evaluation)
        return score
    
    def _parse_user_feedback(self, feedback: str) -> float:
        """Parsea feedback del usuario en puntaje numérico"""
        feedback_lower = feedback.lower()
        
        if any(word in feedback_lower for word in ['excelente', 'perfecto', 'muy bueno']):
            return self.heuristic_penalties['excellent_response']
        elif any(word in feedback_lower for word in ['bueno', 'correcto', 'útil']):
            return self.heuristic_penalties['good_response']
        elif any(word in feedback_lower for word in ['malo', 'incorrecto', 'inútil']):
            return self.heuristic_penalties['incorrect_information']
        elif 'vago' in feedback_lower or 'superficial' in feedback_lower:
            return self.heuristic_penalties['vague_response']
        else:
            return 0.0  # Neutral
    
    def get_improvement_suggestions(self) -> List[str]:
        """Genera sugerencias de mejora basadas en historial"""
        if not self.feedback_history:
            return []
        
        recent_evaluations = self.feedback_history[-10:]  # Últimas 10 evaluaciones
        avg_score = np.mean([e['score'] for e in recent_evaluations])
        
        suggestions = []
        
        # Análisis de patrones problemáticos
        common_penalties = {}
        for evaluation in recent_evaluations:
            for penalty in evaluation['penalties']:
                common_penalties[penalty] = common_penalties.get(penalty, 0) + 1
        
        if common_penalties.get('respuesta_vaga', 0) > 2:
            suggestions.append("Proporcionar respuestas más detalladas y específicas")
        
        if common_penalties.get('respuesta_incompleta', 0) > 2:
            suggestions.append("Evitar respuestas que indiquen incertidumbre sin alternativas")
        
        if avg_score < 0:
            suggestions.append("Mejorar la relevancia de las respuestas al contexto de la pregunta")
        
        return suggestions

class VoiceInterface:
    """Interfaz de voz para texto-a-voz y voz-a-texto"""
    
    def __init__(self):
        self.recognizer = None
        self.tts_engine = None
        self.microphone = None
        self._initialize_voice_components()
        
    def _initialize_voice_components(self):
        """Inicializa componentes de voz"""
        if not VOICE_AVAILABLE:
            logger.warning("Componentes de voz no disponibles")
            return
            
        try:
            # Inicializar reconocimiento de voz
            self.recognizer = sr.Recognizer()
            self.microphone = sr.Microphone()
            
            # Calibrar ruido ambiental
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
            
            # Inicializar text-to-speech
            self.tts_engine = pyttsx3.init()
            
            # Configurar voz
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Preferir voz femenina en español si está disponible
                for voice in voices:
                    if 'spanish' in voice.name.lower() or 'es' in voice.id.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
            
            # Configurar velocidad
            self.tts_engine.setProperty('rate', 150)
            
            logger.info("Interfaz de voz inicializada correctamente")
            
        except Exception as e:
            logger.error(f"Error inicializando interfaz de voz: {e}")
    
    def listen_for_speech(self, timeout: int = 5) -> Optional[str]:
        """Escucha entrada de voz del usuario"""
        if not self.recognizer or not self.microphone:
            return None
            
        try:
            with self.microphone as source:
                print("Escuchando... (habla ahora)")
                audio = self.recognizer.listen(source, timeout=timeout)
            
            # Reconocer speech
            text = self.recognizer.recognize_google(audio, language='es-ES')
            print(f"Escuchado: {text}")
            return text
            
        except sr.WaitTimeoutError:
            print("Tiempo de espera agotado")
            return None
        except sr.UnknownValueError:
            print("No se pudo entender el audio")
            return None
        except sr.RequestError as e:
            print(f"Error en el servicio de reconocimiento: {e}")
            return None
    
    def speak_text(self, text: str):
        """Convierte texto a voz"""
        if not self.tts_engine:
            print(f"RazonBill (texto): {text}")
            return
            
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            logger.error(f"Error en text-to-speech: {e}")
            print(f"RazonBill (texto): {text}")

class InteractiveShell:
    """Sistema shell interactivo con embedding TTY"""
    
    def __init__(self, razonbill_core):
        self.core = razonbill_core
        self.shell_history = []
        self.commands = {
            'help': self._show_help,
            'status': self._show_status,
            'memory': self._show_memory_stats,
            'clear': self._clear_screen,
            'history': self._show_history,
            'feedback': self._show_feedback,
            'voice': self._toggle_voice,
            'exit': self._exit_shell
        }
        
    def start_interactive_session(self):
        """Inicia sesión interactiva"""
        print("=" * 60)
        print("🧠 RazonBill Interactive Shell - Motor de Inferencia Local")
        print("=" * 60)
        print("Escriba 'help' para ver comandos disponibles")
        print("Escriba 'voice' para alternar modo de voz")
        print("Escriba 'exit' para salir")
        print("")
        
        while True:
            try:
                # Prompt personalizado
                prompt = f"RazonBill[{len(self.shell_history)}]> "
                user_input = input(prompt).strip()
                
                if not user_input:
                    continue
                
                # Registrar en historial
                self.shell_history.append({
                    'input': user_input,
                    'timestamp': datetime.now()
                })
                
                # Procesar comando o consulta
                if user_input.startswith('/'):
                    self._process_command(user_input[1:])
                else:
                    response = self.core.procesar_entrada(user_input)
                    print(f"\nRazonBill: {response}\n")
                    
            except KeyboardInterrupt:
                print("\n\nSesión interrumpida. ¡Hasta luego!")
                break
            except EOFError:
                print("\n\nSesión terminada. ¡Hasta luego!")
                break
    
    def _process_command(self, command: str):
        """Procesa comandos del shell"""
        cmd_parts = command.split()
        if not cmd_parts:
            return
            
        cmd_name = cmd_parts[0]
        cmd_args = cmd_parts[1:] if len(cmd_parts) > 1 else []
        
        if cmd_name in self.commands:
            self.commands[cmd_name](cmd_args)
        else:
            print(f"Comando desconocido: {cmd_name}. Use '/help' para ver comandos disponibles.")
    
    def _show_help(self, args):
        """Muestra ayuda de comandos"""
        help_text = """
Comandos disponibles:
  /help          - Muestra esta ayuda
  /status        - Muestra estado del sistema
  /memory        - Estadísticas de memoria
  /clear         - Limpia pantalla
  /history       - Historial de comandos
  /feedback      - Estadísticas de feedback
  /voice         - Alterna modo de voz
  /exit          - Salir del shell
        """
        print(help_text)
    
    def _show_status(self, args):
        """Muestra estado del sistema"""
        status = f"""
Estado del Sistema RazonBill:
- Memoria Vectorial: {'✓ Activa' if self.core.memory_store.collection else '✗ No disponible'}
- Interfaz de Voz: {'✓ Activa' if self.core.voice_interface.recognizer else '✗ No disponible'}
- Agente ReAct: ✓ Activo
- Comandos ejecutados: {len(self.shell_history)}
- Tiempo de sesión: {datetime.now().strftime('%H:%M:%S')}
        """
        print(status)
    
    def _show_memory_stats(self, args):
        """Muestra estadísticas de memoria"""
        print("Estadísticas de Memoria:")
        # Implementar estadísticas específicas
        print("- Memorias almacenadas: [Por implementar]")
        print("- Último acceso: [Por implementar]")
    
    def _clear_screen(self, args):
        """Limpia la pantalla"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def _show_history(self, args):
        """Muestra historial de comandos"""
        print("\nHistorial de comandos:")
        for i, entry in enumerate(self.shell_history[-10:], 1):  # Últimos 10
            print(f"{i:2d}. {entry['timestamp'].strftime('%H:%M:%S')} - {entry['input']}")
        print()
    
    def _show_feedback(self, args):
        """Muestra estadísticas de feedback"""
        feedback_stats = self.core.rl_feedback.feedback_history
        if feedback_stats:
            avg_score = np.mean([f['score'] for f in feedback_stats])
            print(f"\nEstadísticas de Feedback:")
            print(f"- Evaluaciones: {len(feedback_stats)}")
            print(f"- Puntuación promedio: {avg_score:.2f}")
            print("- Sugerencias de mejora:")
            for suggestion in self.core.rl_feedback.get_improvement_suggestions():
                print(f"  • {suggestion}")
        else:
            print("No hay estadísticas de feedback disponibles")
    
    def _toggle_voice(self, args):
        """Alterna modo de voz"""
        if self.core.voice_interface.recognizer:
            print("Modo de voz activado. Escuchando...")
            speech = self.core.voice_interface.listen_for_speech()
            if speech:
                response = self.core.procesar_entrada(speech)
                print(f"RazonBill: {response}")
                self.core.voice_interface.speak_text(response)
        else:
            print("Interfaz de voz no disponible")
    
    def _exit_shell(self, args):
        """Sale del shell"""
        print("Cerrando RazonBill Shell...")
        exit(0)

class RazonBillCore:
    """Motor principal de inferencia RazonBill"""
    
    def __init__(self):
        # Inicializar componentes
        self.prompt_logger = PromptLoggingWrapper()
        self.thought_processor = TreeOfThoughtsProcessor()
        self.memory_store = VectorMemoryStore()
        self.react_agent = ReActAgent()
        self.rl_feedback = RLFeedbackSystem()
        self.voice_interface = VoiceInterface()
        self.shell = InteractiveShell(self)
        
        # Estado del sistema
        self.system_profile = {
            'agent_id': str(uuid.uuid4()),
            'initialized_at': datetime.now(),
            'processing_mode': 'analytical',
            'context_depth': 'deep',
            'response_style': 'detailed'
        }
        
        logger.info("RazonBill Core inicializado correctamente")
    
    def procesar_entrada(self, entrada: str) -> str:
        """Función principal de procesamiento de entrada"""
        # 1. Recuperar contexto de memoria
        memoria = self.recuperar_contexto(entrada)
        
        # 2. Generar árbol de pensamientos
        thought_tree = self.thought_processor.generate_thought_tree(entrada)
        best_path = self.thought_processor.get_best_path()
        
        # 3. Ejecutar agente ReAct si es necesario
        react_response = self.react_agent.execute_thought_action_cycle(entrada)
        
        # 4. Combinar información y generar respuesta
        context_prompt = f"""
Sistema: RazonBill - Motor de inferencia local avanzado
Perfil del agente: Analítico, detallado, basado en evidencia

Contexto de memoria:
{memoria}

Proceso de pensamiento:
{self._format_thought_path(best_path)}

Análisis ReAct:
{react_response}

Usuario: {entrada}

Instrucciones: Proporciona una respuesta comprehensiva basada en todo el contexto disponible.
        """
        
        # 5. Generar respuesta final
        respuesta = self._generate_final_inference(context_prompt, entrada)
        
        # 6. Evaluar respuesta con RL
        score = self.rl_feedback.evaluate_response(respuesta, entrada)
        
        # 7. Registrar en prompt logger
        self.prompt_logger.log_prompt(
            context_prompt, 
            respuesta, 
            {'score': score, 'thought_steps': len(best_path)}
        )
        
        # 8. Guardar en memoria
        self.guardar_contexto(entrada, respuesta)
        
        return respuesta
    
    def recuperar_contexto(self, entrada: str) -> str:
        """Recupera contexto relevante de la memoria vectorial"""
        # Generar embedding simple (placeholder para embedding real)
        entrada_embedding = self._generate_simple_embedding(entrada)
        
        # Recuperar memorias similares
        memorias_similares = self.memory_store.retrieve_similar_memories(
            entrada_embedding, top_k=3
        )
        
        if not memorias_similares:
            return "No hay contexto previo relevante."
        
        # Formatear contexto
        contexto = "Contexto relevante:\n"
        for memoria in memorias_similares:
            contexto += f"- {memoria.content[:200]}...\n"
        
        return contexto
    
    def guardar_contexto(self, entrada: str, respuesta: str):
        """Guarda contexto en memoria vectorial"""
        # Crear memoria contextual
        memory_content = f"Usuario: {entrada}\nRazonBill: {respuesta}"
        memory_embedding = self._generate_simple_embedding(memory_content)
        
        memory = ContextualMemory(
            memory_id=str(uuid.uuid4()),
            content=memory_content,
            embedding=memory_embedding,
            timestamp=datetime.now(),
            context_type='conversation',
            relevance_score=1.0,
            metadata={'entrada_length': len(entrada), 'respuesta_length': len(respuesta)}
        )
        
        # Almacenar en vector store
        self.memory_store.store_memory(memory)
    
    def _generate_simple_embedding(self, text: str) -> List[float]:
        """Genera embedding simple basado en frecuencia de palabras"""
        # Implementación básica - en producción usar modelos como sentence-transformers
        words = text.lower().split()
        vocab = list(set(words))
        
        # Vector de frecuencias normalizado
        embedding = []
        for word in vocab[:100]:  # Limitar a 100 dimensiones
            freq = words.count(word) / len(words)
            embedding.append(freq)
        
        # Pad o truncar a 100 dimensiones
        while len(embedding) < 100:
            embedding.append(0.0)
        
        return embedding[:100]
    
    def _format_thought_path(self, thought_path: List[ThoughtStep]) -> str:
        """Formatea la ruta de pensamiento para contexto"""
        if not thought_path:
            return "No hay proceso de pensamiento registrado."
        
        formatted = "Proceso de pensamiento:\n"
        for i, step in enumerate(thought_path, 1):
            formatted += f"{i}. {step.content} (Confianza: {step.confidence:.2f})\n"
        
        return formatted
    
    def _generate_final_inference(self, context_prompt: str, entrada: str) -> str:
        """Genera respuesta final basada en todo el contexto"""
        # Esta es una implementación simplificada
        # En producción, aquí se integraría con un modelo de lenguaje
        
        # Análizar tipo de pregunta
        entrada_lower = entrada.lower()
        
        if '¿cómo' in entrada_lower:
            return self._generate_how_to_response(entrada)
        elif '¿qué es' in entrada_lower or '¿qué' in entrada_lower:
            return self._generate_what_is_response(entrada)
        elif '¿por qué' in entrada_lower:
            return self._generate_why_response(entrada)
        elif '¿cuándo' in entrada_lower:
            return self._generate_when_response(entrada)
        else:
            return self._generate_general_response(entrada)
    
    def _generate_how_to_response(self, entrada: str) -> str:
        """Genera respuesta tipo 'cómo hacer'"""
        return f"""Para responder a tu consulta sobre '{entrada}', he analizado el problema paso a paso:

1. **Análisis del problema**: He identificado que necesitas una solución práctica
2. **Evaluación de opciones**: Consideré múltiples enfoques posibles
3. **Recomendación**: Basándome en el análisis, sugiero un enfoque estructurado

La estrategia más efectiva sería abordar el problema de manera sistemática, considerando tanto los aspectos prácticos como las limitaciones del contexto específico."""
    
    def _generate_what_is_response(self, entrada: str) -> str:
        """Genera respuesta tipo 'qué es'"""
        return f"""Respecto a tu pregunta '{entrada}', puedo proporcionarte la siguiente información:

**Definición**: Se trata de un concepto que requiere análisis desde múltiples perspectivas.

**Características principales**:
- Tiene aspectos teóricos y prácticos
- Su comprensión depende del contexto
- Involucra múltiples factores interrelacionados

**Contexto relevante**: Para una comprensión completa, es importante considerar tanto el marco teórico como las aplicaciones prácticas."""
    
    def _generate_why_response(self, entrada: str) -> str:
        """Genera respuesta tipo 'por qué'"""
        return f"""Analizando tu pregunta '{entrada}', las razones principales incluyen:

**Factores causales**:
1. Factores históricos y contextuales
2. Mecanismos subyacentes que influyen en el fenómeno
3. Interacciones complejas entre múltiples variables

**Explicación**: Los fenómenos raramente tienen una sola causa. En este caso, la explicación más completa involucra una combinación de factores que interactúan de manera sistemática."""
    
    def _generate_when_response(self, entrada: str) -> str:
        """Genera respuesta tipo 'cuándo'"""
        return f"""Respecto al aspecto temporal de '{entrada}':

**Contexto temporal**:
- Los tiempos específicos dependen de múltiples variables
- Las condiciones del entorno influyen significativamente
- Hay patrones generales, pero también excepciones importantes

**Recomendación**: Para una respuesta más precisa, sería útil conocer más detalles sobre el contexto específico de tu situación."""
    
    def _generate_general_response(self, entrada: str) -> str:
        """Genera respuesta general"""
        return f"""He analizado tu consulta '{entrada}' desde múltiples perspectivas:

**Análisis realizado**:
- Procesamiento contextual de la información
- Evaluación de relevancia y precisión
- Consideración de múltiples enfoques posibles

**Respuesta**: Basándome en el análisis disponible, puedo decir que este tema tiene varias dimensiones importantes que vale la pena explorar. Para poder proporcionarte una respuesta más específica y útil, sería beneficioso conocer más detalles sobre qué aspecto particular te interesa más."""

# Función principal de inferencia (API simplificada)
def inferencia(prompt: str) -> str:
    """Función de inferencia principal para integración externa"""
    global _razonbill_instance
    
    if '_razonbill_instance' not in globals():
        _razonbill_instance = RazonBillCore()
    
    return _razonbill_instance.procesar_entrada(prompt)

# Funciones de utilidad para el loop principal
def escuchar_usuario() -> str:
    """Escucha entrada del usuario (texto o voz)"""
    global _razonbill_instance
    
    if '_razonbill_instance' not in globals():
        _razonbill_instance = RazonBillCore()
    
    # Intentar entrada por voz primero
    if _razonbill_instance.voice_interface.recognizer:
        print("Escuchando... (presiona Enter para usar texto)")
        
        # Non-blocking input attempt
        import select
        import sys
        
        if select.select([sys.stdin], [], [], 1):  # Timeout de 1 segundo
            return input("Usuario: ")
        else:
            speech = _razonbill_instance.voice_interface.listen_for_speech(timeout=3)
            if speech:
                return speech
    
    # Fallback a entrada de texto
    return input("Usuario: ")

def recuperar_contexto(entrada: str) -> str:
    """Recupera contexto para la entrada"""
    global _razonbill_instance
    
    if '_razonbill_instance' not in globals():
        _razonbill_instance = RazonBillCore()
    
    return _razonbill_instance.recuperar_contexto(entrada)

def guardar_contexto(entrada: str, respuesta: str):
    """Guarda contexto de la conversación"""
    global _razonbill_instance
    
    if '_razonbill_instance' not in globals():
        _razonbill_instance = RazonBillCore()
    
    _razonbill_instance.guardar_contexto(entrada, respuesta)

# Función principal para ejecutar el loop interactivo
def main_loop():
    """Loop principal como se especifica en el ejemplo"""
    print("🧠 Iniciando RazonBill Core - Motor de Inferencia Local")
    
    # Inicializar sistema
    core = RazonBillCore()
    
    # Usar shell interactivo
    core.shell.start_interactive_session()

if __name__ == "__main__":
    main_loop()