
"""
Módulo Axón Mealenizado - Ruth R1
Sistema de brincos neuroplásticos a acciones por impulsos
Integrado con la arquitectura de grafos axónicos mielinizados
"""

import torch
import torch.nn as nn
import random
import logging
import numpy as np
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from utils.logger import Logger

class NeuronaA:
    """Neurona A - Almacenamiento y gestión de capas de experiencia"""
    
    def __init__(self):
        self.capas = []
        self.id_capa = 1
        self.logger = Logger()

    def crear_capa(self, contexto: str, intensidad: float) -> Dict[str, Any]:
        """Crea una nueva capa de experiencia"""
        nueva_capa = {
            "id": self.id_capa,
            "contexto": contexto,
            "intensidad": intensidad,
            "frecuencia": 1,
            "similitud": 1.0,
            "peso_historico": intensidad,
            "timestamp": datetime.now()
        }
        
        self.logger.log("DEBUG", f"🔌 Capa creada ID: {nueva_capa['id']} | Peso histórico: {nueva_capa['peso_historico']}")
        self.capas.append(nueva_capa)
        self.id_capa += 1
        return nueva_capa

    def actualizar_capa(self, capa: Dict[str, Any], nueva_intensidad: float):
        """Actualiza una capa existente con nueva intensidad"""
        capa["intensidad"] = nueva_intensidad
        capa["frecuencia"] += 1
        capa["peso_historico"] = capa["intensidad"] * capa["frecuencia"]
        capa["timestamp"] = datetime.now()
        
        self.logger.log("DEBUG", f"🔄 Capa actualizada ID: {capa['id']} | Nuevo peso histórico: {capa['peso_historico']}")

    def calcular_similitud(self, contexto_actual: str) -> List[Tuple[Dict[str, Any], float]]:
        """Calcula similitud entre contexto actual y capas existentes"""
        scores = []
        for capa in self.capas:
            ratio = SequenceMatcher(None, capa["contexto"], contexto_actual).ratio()
            scores.append((capa, ratio))
        return scores

    def obtener_capa_relevante(self, contexto_actual: str) -> Optional[Dict[str, Any]]:
        """Obtiene la capa más relevante para el contexto actual"""
        similitudes = self.calcular_similitud(contexto_actual)
        if not similitudes:
            return None
        mejor_capa, _ = max(similitudes, key=lambda x: x[1])
        return mejor_capa

    def get_tensor_representation(self) -> torch.Tensor:
        """Convierte capas a representación tensorial para integración con Ruth R1"""
        if not self.capas:
            return torch.zeros(768)
        
        # Crear tensor basado en pesos históricos y contextos
        tensor_data = []
        for capa in self.capas:
            # Convertir contexto a hash numérico
            context_hash = hash(capa["contexto"]) % 1000000
            context_normalized = context_hash / 1000000.0
            
            # Combinar características
            features = [
                capa["peso_historico"],
                capa["frecuencia"],
                capa["intensidad"],
                context_normalized
            ]
            tensor_data.extend(features)
        
        # Padding o truncate a 768 dimensiones
        while len(tensor_data) < 768:
            tensor_data.append(0.0)
        tensor_data = tensor_data[:768]
        
        return torch.tensor(tensor_data, dtype=torch.float32)


class NeuronaB:
    """Neurona B - Sistema de decisiones por brinco cognitivo"""
    
    def __init__(self):
        self.peso_intensidad = 0.5
        self.peso_frecuencia = 0.3
        self.peso_similitud = 0.2
        self.logger = Logger()

    def brinco_cognitivo(self, opciones_actuales: Dict[str, float], neurona_a: NeuronaA) -> str:
        """Realiza brinco cognitivo para selección de acción"""
        if not neurona_a.capas:
            self.logger.log("WARNING", "⚠️ Sin historia previa. Decisión aleatoria.")
            return random.choice(list(opciones_actuales.keys()))

        puntuaciones = defaultdict(float)

        for capa in neurona_a.capas:
            accion_similar = self._encontrar_similar(capa["contexto"], opciones_actuales)
            peso = (
                self.peso_intensidad * capa["intensidad"] +
                self.peso_frecuencia * capa["frecuencia"] +
                self.peso_similitud * capa["similitud"]
            )
            puntuaciones[accion_similar] += peso

        mejor_accion = max(puntuaciones, key=puntuaciones.get)
        self.logger.log("INFO", f"🧠 Acción seleccionada por brinco cognitivo: {mejor_accion}")
        return mejor_accion

    def _encontrar_similar(self, contexto_previo: str, opciones_actuales: Dict[str, float]) -> str:
        """Encuentra acción similar basada en contexto previo"""
        mapeos_contexto = {
            "alimentación": "buscar_comida",
            "peligro": "huir",
            "exploración": "avanzar",
            "descanso": "esperar",
            "social": "comunicar",
            "aprendizaje": "analizar"
        }
        
        return mapeos_contexto.get(contexto_previo, random.choice(list(opciones_actuales.keys())))

    def _calcular_similitud(self, contexto_previo: str) -> float:
        """Calcula similitud contextual"""
        return random.uniform(0.6, 1.0)


class AxonMealenizado(nn.Module):
    """
    Sistema principal Axón Mealenizado integrado con Ruth R1
    Maneja brincos neuroplásticos y decisiones por impulsos
    """
    
    def __init__(self, dim: int = 768):
        super().__init__()
        self.dim = dim
        self.neurona_A = NeuronaA()
        self.neurona_B = NeuronaB()
        self.historial_estimulos = []
        
        # Opciones de acción expandidas para Ruth R1
        self.opciones_actuales = {
            "buscar_comida": 3.0,
            "huir": -2.0,
            "avanzar": 1.0,
            "esperar": 0.0,
            "comunicar": 2.0,
            "analizar": 1.5,
            "crear": 2.5,
            "reflexionar": 1.2
        }
        
        # Red neural para procesamiento tensorial
        self.tensor_processor = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, len(self.opciones_actuales)),
            nn.Softmax(dim=-1)
        )
        
        self.logger = Logger()
        self.logger.log("INFO", "AxonMealenizado inicializado para Ruth R1")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass para integración con arquitectura Ruth R1"""
        # Procesar entrada a través de la red tensorial
        action_probs = self.tensor_processor(x)
        
        # Combinar con representación de NeuronaA
        neurona_a_tensor = self.neurona_A.get_tensor_representation()
        if neurona_a_tensor.dim() == 1:
            neurona_a_tensor = neurona_a_tensor.unsqueeze(0)
        
        # Redimensionar para coincidir con batch
        if x.size(0) != neurona_a_tensor.size(0):
            neurona_a_tensor = neurona_a_tensor.expand(x.size(0), -1)
        
        # Procesar representación de memoria
        memory_influence = self.tensor_processor(neurona_a_tensor[:, :self.dim])
        
        # Combinar probabilidades
        combined_probs = (action_probs + memory_influence) / 2.0
        
        return combined_probs

    def recibir_estimulo(self, input_data: Dict[str, Any], reward: Optional[float] = None) -> Tuple[str, Dict[str, Any]]:
        """Recibe estímulo y genera respuesta con metadata"""
        self.logger.log("INFO", f"🧠 Axón Mealenizado recibiendo estímulo: {input_data}")
        
        contexto = input_data.get("contexto", "desconocido")
        intensidad = input_data.get("intensidad", 0.5)

        # Procesar con NeuronaA
        capa_relevante = self.neurona_A.obtener_capa_relevante(contexto)
        if capa_relevante:
            self.neurona_A.actualizar_capa(capa_relevante, intensidad)
        else:
            capa_relevante = self.neurona_A.crear_capa(contexto, intensidad)

        # Decisión con NeuronaB
        mejor_accion = self.neurona_B.brinco_cognitivo(self.opciones_actuales, self.neurona_A)
        
        # Crear metadata de decisión
        metadata_decision = {
            "accion": mejor_accion,
            "capa_usada": capa_relevante["id"],
            "contexto": contexto,
            "intensidad": intensidad,
            "peso_historico": capa_relevante["peso_historico"],
            "similitud": capa_relevante["similitud"],
            "timestamp": datetime.now().isoformat()
        }

        # Aplicar refuerzo si hay recompensa
        if reward is not None:
            learning_rate = 0.1
            capa_relevante["peso_historico"] += reward * learning_rate * intensidad
            self.logger.log("INFO", f"🧠 Recompensa aplicada +{reward}")

        self.historial_estimulos.append(input_data)
        return mejor_accion, metadata_decision

    def metacompilar(self, relevancia_umbral: float = 2.0, total_epocas: int = 3):
        """Metacompila y optimiza la red usando algoritmo amiloid"""
        self.logger.log("INFO", "🧠 Axón Mealenizado: Metacompilando redes...")
        
        self.neurona_A, self.historial_estimulos = self._amiloid_agent(
            self.neurona_A,
            self.historial_estimulos,
            relevancia_umbral=relevancia_umbral,
            total_epocas=total_epocas
        )

    def _amiloid_agent(self, neurona_a: NeuronaA, historial_estimulos: List[Dict], 
                      relevancia_umbral: float = 2.0, total_epocas: int = 3) -> Tuple[NeuronaA, List[Dict]]:
        """Algoritmo amiloid para optimización neuronal"""
        
        for epoca in range(total_epocas):
            self.logger.log("INFO", f"🔄 Época {epoca+1} / {total_epocas}")
            
            # Procesar estímulos históricos
            for estimulo in historial_estimulos:
                contexto = estimulo.get("contexto", "desconocido")
                intensidad = estimulo.get("intensidad", 0.5)

                capa_similar = neurona_a.obtener_capa_relevante(contexto)
                if capa_similar:
                    neurona_a.actualizar_capa(capa_similar, intensidad)
                else:
                    neurona_a.crear_capa(contexto, intensidad)

            # Evaluar y podar capas irrelevantes
            relevancias = self._evaluate_relevance(neurona_a)
            capas_a_eliminar = []
            
            for capa in neurona_a.capas:
                if relevancias[capa["id"]] < relevancia_umbral:
                    self.logger.log("INFO", f"✂️ Eliminando capa {capa['id']} - Contexto: {capa['contexto']}")
                    capas_a_eliminar.append(capa)
            
            for capa in capas_a_eliminar:
                neurona_a.capas.remove(capa)

            # Filtrar datos históricos por relevancia
            data_relevancias = self._evaluate_data_relevance(historial_estimulos)
            historial_estimulos = [
                e for e in historial_estimulos 
                if data_relevancias.get(e.get("contexto", ""), 0) >= relevancia_umbral
            ]

        return neurona_a, historial_estimulos

    def _evaluate_relevance(self, neurona_a: NeuronaA) -> Dict[int, float]:
        """Evalúa relevancia de capas"""
        scores = {}
        for capa in neurona_a.capas:
            score = self._calculate_importance(capa)
            scores[capa["id"]] = score
        return scores

    def _evaluate_data_relevance(self, data: List[Dict]) -> Dict[str, float]:
        """Evalúa relevancia de datos históricos"""
        scores = {}
        for sample in data:
            contexto = sample.get("contexto", "desconocido")
            score = self._calculate_data_importance(sample)
            scores[contexto] = scores.get(contexto, 0) + score
        return scores

    def _calculate_importance(self, capa: Dict[str, Any]) -> float:
        """Calcula importancia de una capa"""
        return capa["intensidad"] * capa["frecuencia"]

    def _calculate_data_importance(self, sample: Dict[str, Any]) -> float:
        """Calcula importancia de una muestra de datos"""
        return sample.get("intensidad", 0.5)

    def get_system_state(self) -> Dict[str, Any]:
        """Obtiene estado actual del sistema"""
        return {
            "total_capas": len(self.neurona_A.capas),
            "total_estimulos": len(self.historial_estimulos),
            "peso_total_historico": sum(capa["peso_historico"] for capa in self.neurona_A.capas),
            "opciones_disponibles": list(self.opciones_actuales.keys()),
            "ultima_actualizacion": datetime.now().isoformat()
        }

    def conectar_con_grafo_neuronal(self, grafo_neuronal):
        """Conecta con el grafo neuronal principal de Ruth R1"""
        if hasattr(grafo_neuronal, 'nodes'):
            # Crear nodo para el axón mealenizado
            axon_node = type('AxonNode', (), {
                'embedding': self.neurona_A.get_tensor_representation(),
                'name': 'axon_mealenizado',
                'activation_threshold': 0.15,
                'connections': {},
                'activation_level': 0.0,
                'last_activation': 0.0
            })()
            
            grafo_neuronal.add_node(axon_node)
            
            # Conectar con nodos existentes
            for node in grafo_neuronal.nodes[:3]:  # Conectar con primeros 3 nodos
                if hasattr(node, 'connect'):
                    node.connect(axon_node, weight=0.6)
            
            self.logger.log("INFO", "Axón Mealenizado conectado al grafo neuronal principal")
