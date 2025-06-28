"""
Visualizador Interactivo de Flujos Neuronales en Tiempo Real
Sistema Ruth R1 - Mapas de Conexiones Neuronales Dinámicos
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List, Tuple, Any, Optional
import random
from dataclasses import dataclass
from enum import Enum

class FlowType(Enum):
    """Tipos de flujo neural"""
    ACTIVATION = "activation"
    INHIBITION = "inhibition"
    MODULATION = "modulation"
    MEMORY = "memory"
    ATTENTION = "attention"
    EMOTIONAL = "emotional"
    INTROSPECTIVE = "introspective"

class NodeType(Enum):
    """Tipos de nodos neurales"""
    SENSOR = "sensor"
    PROCESSOR = "processor"
    MEMORY = "memory"
    OUTPUT = "output"
    MODULATOR = "modulator"
    INTEGRATOR = "integrator"
    CONSCIOUSNESS = "consciousness"

@dataclass
class NeuralNode:
    """Representa un nodo en la red neural"""
    id: str
    type: NodeType
    position: Tuple[float, float, float]
    activation: float
    connections: List[str]
    properties: Dict[str, Any]
    last_update: datetime

@dataclass
class NeuralFlow:
    """Representa un flujo entre nodos"""
    source: str
    target: str
    flow_type: FlowType
    intensity: float
    speed: float
    timestamp: datetime
    properties: Dict[str, Any]

class NeuralNetworkTopology:
    """Gestiona la topología de la red neural"""
    
    def __init__(self):
        self.nodes: Dict[str, NeuralNode] = {}
        self.flows: List[NeuralFlow] = []
        self.topology_history: List[Dict] = []
        self.max_history = 1000
    
    def create_ruth_r1_topology(self) -> Dict[str, NeuralNode]:
        """Crea la topología específica de Ruth R1"""
        nodes = {}
        
        # Nodos del GANST Core
        ganst_positions = self._generate_sphere_positions(10, radius=5.0, center=(0, 0, 0))
        for i, pos in enumerate(ganst_positions):
            nodes[f"ganst_core_{i}"] = NeuralNode(
                id=f"ganst_core_{i}",
                type=NodeType.PROCESSOR,
                position=pos,
                activation=0.5,
                connections=[],
                properties={
                    "module": "ganst_core",
                    "tensor_dim": 768,
                    "processing_type": "tensor_synthesis"
                },
                last_update=datetime.now()
            )
        
        # Nodos de Moduladores
        modulator_types = ["amplitude", "frequency", "phase", "attention", 
                          "emotional", "temporal", "contextual"]
        modulator_positions = self._generate_circle_positions(7, radius=8.0, center=(0, 0, 3))
        for i, (mod_type, pos) in enumerate(zip(modulator_types, modulator_positions)):
            nodes[f"modulator_{mod_type}"] = NeuralNode(
                id=f"modulator_{mod_type}",
                type=NodeType.MODULATOR,
                position=pos,
                activation=0.3,
                connections=[],
                properties={
                    "module": "moduladores",
                    "modulation_type": mod_type,
                    "base_frequency": 1.0 + i * 0.2
                },
                last_update=datetime.now()
            )
        
        # Nodos de Memoria de Corto Plazo
        memory_types = ["sensorial", "buffer", "trabajo", "episodica", "emocional"]
        memory_positions = self._generate_line_positions(5, start=(-6, -6, 0), end=(6, -6, 0))
        for i, (mem_type, pos) in enumerate(zip(memory_types, memory_positions)):
            nodes[f"memory_{mem_type}"] = NeuralNode(
                id=f"memory_{mem_type}",
                type=NodeType.MEMORY,
                position=pos,
                activation=0.4,
                connections=[],
                properties={
                    "module": "memoria_corto_plazo",
                    "memory_type": mem_type,
                    "decay_rate": 2.0 - i * 0.3,
                    "capacity": 100 - i * 10
                },
                last_update=datetime.now()
            )
        
        # Nodos del Meta-Enrutador Ruth R1
        meta_positions = self._generate_grid_positions(4, 4, spacing=2.0, center=(10, 0, 0))
        for i, pos in enumerate(meta_positions):
            nodes[f"meta_router_{i}"] = NeuralNode(
                id=f"meta_router_{i}",
                type=NodeType.INTEGRATOR,
                position=pos,
                activation=0.6,
                connections=[],
                properties={
                    "module": "meta_enrutador",
                    "layer": i // 4,
                    "transformer_head": i % 4,
                    "attention_dim": 512
                },
                last_update=datetime.now()
            )
        
        # Nodos de Consciencia Bayesiana
        consciousness_positions = self._generate_sphere_positions(14, radius=3.0, center=(0, 8, 2))
        consciousness_modules = [
            "sensory_integration", "working_memory", "attention_control",
            "emotional_processing", "self_awareness", "temporal_integration",
            "abstract_reasoning", "meta_cognition", "decision_making",
            "language_processing", "pattern_recognition", "memory_consolidation",
            "executive_control", "consciousness_unity"
        ]
        
        for i, (module, pos) in enumerate(zip(consciousness_modules, consciousness_positions)):
            nodes[f"consciousness_{module}"] = NeuralNode(
                id=f"consciousness_{module}",
                type=NodeType.CONSCIOUSNESS,
                position=pos,
                activation=0.7,
                connections=[],
                properties={
                    "module": "consciousness_network",
                    "consciousness_type": module,
                    "bayesian_weight": 0.1 + i * 0.05,
                    "integration_level": "high"
                },
                last_update=datetime.now()
            )
        
        # Nodos del Bucle Introspectivo
        introspective_positions = self._generate_spiral_positions(8, radius=4.0, height=6.0, center=(0, 0, 8))
        for i, pos in enumerate(introspective_positions):
            nodes[f"introspective_{i}"] = NeuralNode(
                id=f"introspective_{i}",
                type=NodeType.PROCESSOR,
                position=pos,
                activation=0.5,
                connections=[],
                properties={
                    "module": "introspective_loop",
                    "loop_cycle": i,
                    "observation_depth": 0.1 + i * 0.1,
                    "insight_generation": True
                },
                last_update=datetime.now()
            )
        
        # Establecer conexiones entre nodos
        self._establish_connections(nodes)
        
        self.nodes = nodes
        return nodes
    
    def _generate_sphere_positions(self, count: int, radius: float, center: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """Genera posiciones en una esfera"""
        positions = []
        phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle
        
        for i in range(count):
            y = 1 - (i / float(count - 1)) * 2
            radius_at_y = np.sqrt(1 - y * y)
            theta = phi * i
            
            x = np.cos(theta) * radius_at_y * radius
            z = np.sin(theta) * radius_at_y * radius
            y = y * radius
            
            positions.append((x + center[0], y + center[1], z + center[2]))
        
        return positions
    
    def _generate_circle_positions(self, count: int, radius: float, center: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """Genera posiciones en un círculo"""
        positions = []
        for i in range(count):
            angle = 2 * np.pi * i / count
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            positions.append((x + center[0], y + center[1], center[2]))
        return positions
    
    def _generate_line_positions(self, count: int, start: Tuple[float, float, float], end: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """Genera posiciones en una línea"""
        positions = []
        for i in range(count):
            t = i / (count - 1) if count > 1 else 0
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])
            z = start[2] + t * (end[2] - start[2])
            positions.append((x, y, z))
        return positions
    
    def _generate_grid_positions(self, rows: int, cols: int, spacing: float, center: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """Genera posiciones en una grilla"""
        positions = []
        start_x = center[0] - (cols - 1) * spacing / 2
        start_y = center[1] - (rows - 1) * spacing / 2
        
        for i in range(rows):
            for j in range(cols):
                x = start_x + j * spacing
                y = start_y + i * spacing
                positions.append((x, y, center[2]))
        return positions
    
    def _generate_spiral_positions(self, count: int, radius: float, height: float, center: Tuple[float, float, float]) -> List[Tuple[float, float, float]]:
        """Genera posiciones en una espiral"""
        positions = []
        for i in range(count):
            t = i / count
            angle = 4 * np.pi * t
            r = radius * (1 - t * 0.5)
            x = r * np.cos(angle)
            y = r * np.sin(angle)
            z = height * t
            positions.append((x + center[0], y + center[1], z + center[2]))
        return positions
    
    def _establish_connections(self, nodes: Dict[str, NeuralNode]):
        """Establece conexiones realistas entre nodos"""
        node_list = list(nodes.keys())
        
        # Conexiones principales del sistema
        connection_rules = [
            # GANST Core se conecta con todos los moduladores
            ("ganst_core_", "modulator_", 0.8),
            # Moduladores se conectan con memoria
            ("modulator_", "memory_", 0.6),
            # Memoria se conecta con meta-enrutador
            ("memory_", "meta_router_", 0.7),
            # Meta-enrutador se conecta con consciencia
            ("meta_router_", "consciousness_", 0.9),
            # Consciencia se conecta con introspección
            ("consciousness_", "introspective_", 0.7),
            # Introspección retroalimenta al GANST
            ("introspective_", "ganst_core_", 0.5),
            # Conexiones internas de consciencia
            ("consciousness_", "consciousness_", 0.4),
            # Conexiones internas de meta-enrutador
            ("meta_router_", "meta_router_", 0.6)
        ]
        
        for source_prefix, target_prefix, probability in connection_rules:
            source_nodes = [n for n in node_list if n.startswith(source_prefix)]
            target_nodes = [n for n in node_list if n.startswith(target_prefix)]
            
            for source in source_nodes:
                for target in target_nodes:
                    if source != target and random.random() < probability:
                        nodes[source].connections.append(target)
    
    def update_activations(self, ganst_core=None, modulators=None, memory_system=None, consciousness_network=None):
        """Actualiza las activaciones basado en el estado real del sistema"""
        current_time = datetime.now()
        
        # Actualizar activaciones del GANST Core
        if ganst_core:
            try:
                ganst_state = ganst_core.get_system_state()
                activations = ganst_state.get('current_activations', [])
                
                ganst_nodes = [n for n in self.nodes.keys() if n.startswith('ganst_core_')]
                for i, node_id in enumerate(ganst_nodes):
                    if i < len(activations):
                        activation_data = activations[i]
                        intensity = activation_data.get('intensity', 0.5)
                        self.nodes[node_id].activation = min(1.0, max(0.0, intensity))
                        self.nodes[node_id].last_update = current_time
            except Exception as e:
                # Simulación si no hay datos reales
                self._simulate_ganst_activations()
        else:
            self._simulate_ganst_activations()
        
        # Actualizar moduladores
        if modulators:
            try:
                modulator_status = modulators.get_modulator_status()
                for mod_name, status in modulator_status.items():
                    node_id = f"modulator_{mod_name}"
                    if node_id in self.nodes:
                        intensity = status.get('current_intensity', 0.5)
                        self.nodes[node_id].activation = intensity
                        self.nodes[node_id].last_update = current_time
            except:
                self._simulate_modulator_activations()
        else:
            self._simulate_modulator_activations()
        
        # Actualizar memoria
        if memory_system:
            try:
                memory_state = memory_system.get_system_state()
                buffers = memory_state.get('buffers', {})
                
                for buffer_name, buffer_data in buffers.items():
                    node_id = f"memory_{buffer_name}"
                    if node_id in self.nodes:
                        usage = buffer_data.get('current_usage', 0.0)
                        self.nodes[node_id].activation = usage
                        self.nodes[node_id].last_update = current_time
            except:
                self._simulate_memory_activations()
        else:
            self._simulate_memory_activations()
        
        # Actualizar consciencia
        if consciousness_network:
            try:
                # Simular activaciones de consciencia basadas en el estado de la red
                consciousness_nodes = [n for n in self.nodes.keys() if n.startswith('consciousness_')]
                for node_id in consciousness_nodes:
                    # Activación basada en coherencia del sistema
                    activation = 0.6 + 0.3 * np.sin(time.time() + hash(node_id) % 100)
                    self.nodes[node_id].activation = max(0.0, min(1.0, activation))
                    self.nodes[node_id].last_update = current_time
            except:
                self._simulate_consciousness_activations()
        else:
            self._simulate_consciousness_activations()
        
        # Actualizar introspección
        self._simulate_introspective_activations()
        
        # Actualizar meta-enrutador
        self._simulate_meta_router_activations()
    
    def _simulate_ganst_activations(self):
        """Simula activaciones del GANST Core"""
        ganst_nodes = [n for n in self.nodes.keys() if n.startswith('ganst_core_')]
        base_time = time.time()
        
        for i, node_id in enumerate(ganst_nodes):
            # Patrón de activación con ondas sinusoidales y ruido
            phase = i * 0.5
            frequency = 0.8 + i * 0.1
            activation = 0.5 + 0.3 * np.sin(base_time * frequency + phase) + 0.1 * np.random.normal()
            self.nodes[node_id].activation = max(0.0, min(1.0, activation))
            self.nodes[node_id].last_update = datetime.now()
    
    def _simulate_modulator_activations(self):
        """Simula activaciones de moduladores"""
        modulator_nodes = [n for n in self.nodes.keys() if n.startswith('modulator_')]
        base_time = time.time()
        
        for i, node_id in enumerate(modulator_nodes):
            # Modulación basada en tipo de modulador
            mod_type = node_id.split('_')[1]
            if mod_type == "attention":
                activation = 0.7 + 0.2 * np.sin(base_time * 2.0)
            elif mod_type == "emotional":
                activation = 0.4 + 0.4 * np.sin(base_time * 0.5)
            elif mod_type == "temporal":
                activation = 0.6 + 0.3 * np.sin(base_time * 1.5)
            else:
                activation = 0.5 + 0.2 * np.sin(base_time * (1.0 + i * 0.2))
            
            self.nodes[node_id].activation = max(0.0, min(1.0, activation))
            self.nodes[node_id].last_update = datetime.now()
    
    def _simulate_memory_activations(self):
        """Simula activaciones de memoria"""
        memory_nodes = [n for n in self.nodes.keys() if n.startswith('memory_')]
        base_time = time.time()
        
        for i, node_id in enumerate(memory_nodes):
            # Patrón de uso de memoria con decaimiento
            mem_type = node_id.split('_')[1]
            if mem_type == "sensorial":
                activation = 0.8 + 0.2 * np.sin(base_time * 5.0)  # Alta frecuencia
            elif mem_type == "trabajo":
                activation = 0.6 + 0.3 * np.sin(base_time * 1.0)  # Frecuencia media
            elif mem_type == "episodica":
                activation = 0.3 + 0.2 * np.sin(base_time * 0.3)  # Baja frecuencia
            else:
                activation = 0.4 + 0.2 * np.sin(base_time * (0.5 + i * 0.2))
            
            self.nodes[node_id].activation = max(0.0, min(1.0, activation))
            self.nodes[node_id].last_update = datetime.now()
    
    def _simulate_consciousness_activations(self):
        """Simula activaciones de consciencia"""
        consciousness_nodes = [n for n in self.nodes.keys() if n.startswith('consciousness_')]
        base_time = time.time()
        
        for i, node_id in enumerate(consciousness_nodes):
            # Activación coherente de consciencia
            module_type = node_id.split('_')[1]
            if module_type in ["self_awareness", "meta_cognition"]:
                activation = 0.8 + 0.15 * np.sin(base_time * 0.8 + i)
            elif module_type in ["attention_control", "executive_control"]:
                activation = 0.7 + 0.2 * np.sin(base_time * 1.2 + i)
            else:
                activation = 0.6 + 0.25 * np.sin(base_time * 1.0 + i * 0.5)
            
            self.nodes[node_id].activation = max(0.0, min(1.0, activation))
            self.nodes[node_id].last_update = datetime.now()
    
    def _simulate_introspective_activations(self):
        """Simula activaciones introspectivas"""
        introspective_nodes = [n for n in self.nodes.keys() if n.startswith('introspective_')]
        base_time = time.time()
        
        for i, node_id in enumerate(introspective_nodes):
            # Patrón de bucle introspectivo
            cycle_phase = (base_time + i * 2.0) % 10.0  # Ciclo de 10 segundos
            if cycle_phase < 3.0:  # Fase de observación
                activation = 0.8 + 0.1 * np.sin(base_time * 3.0)
            elif cycle_phase < 7.0:  # Fase de análisis
                activation = 0.6 + 0.2 * np.sin(base_time * 2.0)
            else:  # Fase de insight
                activation = 0.9 + 0.1 * np.sin(base_time * 4.0)
            
            self.nodes[node_id].activation = max(0.0, min(1.0, activation))
            self.nodes[node_id].last_update = datetime.now()
    
    def _simulate_meta_router_activations(self):
        """Simula activaciones del meta-enrutador"""
        meta_router_nodes = [n for n in self.nodes.keys() if n.startswith('meta_router_')]
        base_time = time.time()
        
        for i, node_id in enumerate(meta_router_nodes):
            # Activación por capas del transformer
            layer = i // 4
            head = i % 4
            
            # Diferente intensidad por capa
            layer_factor = 0.6 + layer * 0.1
            head_phase = head * np.pi / 2
            
            activation = layer_factor + 0.2 * np.sin(base_time * 1.5 + head_phase)
            self.nodes[node_id].activation = max(0.0, min(1.0, activation))
            self.nodes[node_id].last_update = datetime.now()
    
    def generate_flows(self) -> List[NeuralFlow]:
        """Genera flujos neuronales basados en activaciones y conexiones"""
        flows = []
        current_time = datetime.now()
        
        for source_id, source_node in self.nodes.items():
            for target_id in source_node.connections:
                if target_id in self.nodes:
                    target_node = self.nodes[target_id]
                    
                    # Intensidad basada en activaciones de ambos nodos
                    intensity = (source_node.activation + target_node.activation) / 2.0
                    
                    # Añadir variabilidad temporal
                    time_factor = 0.8 + 0.4 * np.sin(time.time() * 2.0 + hash(source_id + target_id) % 100)
                    intensity *= time_factor
                    
                    # Determinar tipo de flujo basado en módulos
                    flow_type = self._determine_flow_type(source_node, target_node)
                    
                    # Velocidad basada en tipo de flujo
                    speed = self._calculate_flow_speed(flow_type, intensity)
                    
                    # Solo crear flujo si la intensidad es significativa
                    if intensity > 0.2:
                        flow = NeuralFlow(
                            source=source_id,
                            target=target_id,
                            flow_type=flow_type,
                            intensity=intensity,
                            speed=speed,
                            timestamp=current_time,
                            properties={
                                "distance": self._calculate_distance(source_node.position, target_node.position),
                                "source_module": source_node.properties.get("module", "unknown"),
                                "target_module": target_node.properties.get("module", "unknown")
                            }
                        )
                        flows.append(flow)
        
        self.flows = flows
        return flows
    
    def _determine_flow_type(self, source: NeuralNode, target: NeuralNode) -> FlowType:
        """Determina el tipo de flujo basado en los nodos"""
        source_module = source.properties.get("module", "")
        target_module = target.properties.get("module", "")
        
        if "modulator" in source_module or "modulator" in target_module:
            return FlowType.MODULATION
        elif "memoria" in source_module or "memoria" in target_module:
            return FlowType.MEMORY
        elif "consciousness" in source_module or "consciousness" in target_module:
            if source.activation > target.activation:
                return FlowType.ACTIVATION
            else:
                return FlowType.ATTENTION
        elif "introspective" in source_module:
            return FlowType.INTROSPECTIVE
        elif source.type == NodeType.MODULATOR:
            return FlowType.MODULATION
        else:
            return FlowType.ACTIVATION if source.activation > 0.5 else FlowType.INHIBITION
    
    def _calculate_flow_speed(self, flow_type: FlowType, intensity: float) -> float:
        """Calcula la velocidad del flujo"""
        base_speeds = {
            FlowType.ACTIVATION: 1.5,
            FlowType.INHIBITION: 1.0,
            FlowType.MODULATION: 0.8,
            FlowType.MEMORY: 0.6,
            FlowType.ATTENTION: 2.0,
            FlowType.EMOTIONAL: 0.7,
            FlowType.INTROSPECTIVE: 0.5
        }
        
        base_speed = base_speeds.get(flow_type, 1.0)
        return base_speed * (0.5 + intensity)
    
    def _calculate_distance(self, pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
        """Calcula la distancia euclidiana entre dos posiciones"""
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))

class NeuralFlowVisualizer:
    """Visualizador principal de flujos neuronales"""
    
    def __init__(self, consciousness_network=None):
        self.topology = NeuralNetworkTopology()
        self.topology.create_ruth_r1_topology()
        self.flow_history = []
        self.max_flow_history = 100
        self.color_scheme = self._setup_color_scheme()
        self.last_update = datetime.now()
        self.consciousness_network = consciousness_network
        self.is_running = False
        self.flow_data = []
        self.connection_strengths = {}
        
        # Simular datos iniciales para demostración inmediata
        self._initialize_sample_data()
        
    def _setup_color_scheme(self) -> Dict:
        """Define el esquema de colores para visualización"""
        return {
            "node_types": {
                NodeType.SENSOR: "#FF6B6B",
                NodeType.PROCESSOR: "#4ECDC4", 
                NodeType.MEMORY: "#45B7D1",
                NodeType.OUTPUT: "#96CEB4",
                NodeType.MODULATOR: "#FFEAA7",
                NodeType.INTEGRATOR: "#DDA0DD",
                NodeType.CONSCIOUSNESS: "#FFD93D"
            },
            "flow_types": {
                FlowType.ACTIVATION: "#00FF00",
                FlowType.INHIBITION: "#FF4444",
                FlowType.MODULATION: "#FFB347",
                FlowType.MEMORY: "#87CEEB",
                FlowType.ATTENTION: "#FF69B4",
                FlowType.EMOTIONAL: "#FFA500",
                FlowType.INTROSPECTIVE: "#9370DB"
            },
            "background": "#0F0F23",
            "grid": "#2A2A2A"
        }
    
    def update_real_time_data(self, ganst_core=None, modulators=None, memory_system=None, consciousness_network=None):
        """Actualiza los datos en tiempo real del sistema"""
        self.topology.update_activations(ganst_core, modulators, memory_system, consciousness_network)
        flows = self.topology.generate_flows()
        
        # Mantener historial de flujos
        self.flow_history.append({
            'timestamp': datetime.now(),
            'flows': flows,
            'total_activity': sum(node.activation for node in self.topology.nodes.values())
        })
        
        if len(self.flow_history) > self.max_flow_history:
            self.flow_history.pop(0)
        
        self.last_update = datetime.now()
        return flows
    
    def create_3d_network_plot(self) -> go.Figure:
        """Crea visualización 3D de la red neural"""
        fig = go.Figure()
        
        # Preparar datos de nodos
        node_x, node_y, node_z = [], [], []
        node_colors, node_sizes, node_text = [], [], []
        node_customdata = []
        
        for node_id, node in self.topology.nodes.items():
            node_x.append(node.position[0])
            node_y.append(node.position[1])
            node_z.append(node.position[2])
            
            # Color basado en tipo de nodo
            node_colors.append(self.color_scheme["node_types"][node.type])
            
            # Tamaño basado en activación
            node_sizes.append(5 + node.activation * 15)
            
            # Texto informativo
            module = node.properties.get("module", "unknown")
            node_text.append(f"{node_id}<br>Módulo: {module}<br>Activación: {node.activation:.3f}")
            
            # Datos personalizados para hover
            node_customdata.append({
                'id': node_id,
                'type': node.type.value,
                'module': module,
                'activation': node.activation,
                'connections': len(node.connections)
            })
        
        # Añadir nodos al gráfico
        fig.add_trace(go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                opacity=0.8,
                line=dict(width=1, color='white')
            ),
            text=[node_id.replace('_', '<br>') for node_id in self.topology.nodes.keys()],
            textposition="top center",
            textfont=dict(size=8, color='white'),
            hovertemplate='<b>%{text}</b><br>Activación: %{customdata[3]:.3f}<br>Conexiones: %{customdata[4]}<extra></extra>',
            customdata=node_customdata,
            name="Nodos Neurales"
        ))
        
        # Añadir conexiones (flujos)
        flows = self.topology.flows
        for flow in flows:
            if flow.intensity > 0.3:  # Solo mostrar flujos significativos
                source_pos = self.topology.nodes[flow.source].position
                target_pos = self.topology.nodes[flow.target].position
                
                # Color basado en tipo de flujo
                flow_color = self.color_scheme["flow_types"][flow.flow_type]
                
                # Línea de conexión
                fig.add_trace(go.Scatter3d(
                    x=[source_pos[0], target_pos[0]],
                    y=[source_pos[1], target_pos[1]],
                    z=[source_pos[2], target_pos[2]],
                    mode='lines',
                    line=dict(
                        color=flow_color,
                        width=max(1, flow.intensity * 8),
                        opacity=0.6
                    ),
                    hovertemplate=f'<b>{flow.flow_type.value.title()}</b><br>Intensidad: {flow.intensity:.3f}<br>Velocidad: {flow.speed:.3f}<extra></extra>',
                    name=f"Flujo {flow.flow_type.value}",
                    showlegend=False
                ))
        
        # Configurar layout
        fig.update_layout(
            title={
                'text': "🧠 Red Neural Ruth R1 - Visualización 3D en Tiempo Real",
                'x': 0.5,
                'font': {'size': 20, 'color': 'white'}
            },
            scene=dict(
                xaxis=dict(title="X", gridcolor="#2A2A2A", zerolinecolor="#2A2A2A"),
                yaxis=dict(title="Y", gridcolor="#2A2A2A", zerolinecolor="#2A2A2A"), 
                zaxis=dict(title="Z", gridcolor="#2A2A2A", zerolinecolor="#2A2A2A"),
                bgcolor=self.color_scheme["background"],
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    up=dict(x=0, y=0, z=1)
                )
            ),
            paper_bgcolor=self.color_scheme["background"],
            plot_bgcolor=self.color_scheme["background"],
            font=dict(color='white'),
            height=700,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        return fig
    
    def create_flow_intensity_heatmap(self) -> go.Figure:
        """Crea mapa de calor de intensidades de flujo"""
        # Crear matriz de adyacencia
        node_ids = list(self.topology.nodes.keys())
        n_nodes = len(node_ids)
        intensity_matrix = np.zeros((n_nodes, n_nodes))
        
        node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}
        
        for flow in self.topology.flows:
            if flow.source in node_to_idx and flow.target in node_to_idx:
                source_idx = node_to_idx[flow.source]
                target_idx = node_to_idx[flow.target]
                intensity_matrix[source_idx][target_idx] = flow.intensity
        
        # Crear heatmap
        fig = go.Figure(data=go.Heatmap(
            z=intensity_matrix,
            x=[node_id.replace('_', '<br>') for node_id in node_ids],
            y=[node_id.replace('_', '<br>') for node_id in node_ids],
            colorscale='Viridis',
            hovertemplate='Origen: %{y}<br>Destino: %{x}<br>Intensidad: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="🔥 Mapa de Calor - Intensidades de Flujo Neural",
            xaxis_title="Nodo Destino",
            yaxis_title="Nodo Origen",
            font=dict(color='white', size=10),
            paper_bgcolor=self.color_scheme["background"],
            plot_bgcolor=self.color_scheme["background"],
            height=600
        )
        
        return fig
    
    def create_temporal_flow_analysis(self) -> go.Figure:
        """Crea análisis temporal de flujos"""
        if not self.flow_history:
            return go.Figure()
        
        # Extraer datos temporales
        timestamps = [entry['timestamp'] for entry in self.flow_history]
        activities = [entry['total_activity'] for entry in self.flow_history]
        
        # Análisis por tipo de flujo
        flow_type_activities = {flow_type: [] for flow_type in FlowType}
        
        for entry in self.flow_history:
            type_counts = {flow_type: 0 for flow_type in FlowType}
            for flow in entry['flows']:
                type_counts[flow.flow_type] += flow.intensity
            
            for flow_type in FlowType:
                flow_type_activities[flow_type].append(type_counts[flow_type])
        
        # Crear subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Actividad Neural Total', 'Actividad por Tipo de Flujo'],
            vertical_spacing=0.1
        )
        
        # Gráfico de actividad total
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=activities,
                mode='lines+markers',
                name='Actividad Total',
                line=dict(color='#FFD93D', width=2),
                marker=dict(size=4)
            ),
            row=1, col=1
        )
        
        # Gráficos por tipo de flujo
        for flow_type in FlowType:
            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=flow_type_activities[flow_type],
                    mode='lines',
                    name=flow_type.value.title(),
                    line=dict(color=self.color_scheme["flow_types"][flow_type], width=1.5),
                    opacity=0.8
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title="📊 Análisis Temporal de Flujos Neuronales",
            paper_bgcolor=self.color_scheme["background"],
            plot_bgcolor=self.color_scheme["background"],
            font=dict(color='white'),
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_network_statistics_plot(self) -> go.Figure:
        """Crea gráfico de estadísticas de la red"""
        # Calcular estadísticas por módulo
        module_stats = {}
        for node in self.topology.nodes.values():
            module = node.properties.get("module", "unknown")
            if module not in module_stats:
                module_stats[module] = {
                    'count': 0,
                    'avg_activation': 0,
                    'total_connections': 0,
                    'activation_values': []
                }
            
            module_stats[module]['count'] += 1
            module_stats[module]['activation_values'].append(node.activation)
            module_stats[module]['total_connections'] += len(node.connections)
        
        # Calcular promedios
        for module, stats in module_stats.items():
            stats['avg_activation'] = np.mean(stats['activation_values'])
            stats['std_activation'] = np.std(stats['activation_values'])
            stats['avg_connections'] = stats['total_connections'] / stats['count']
        
        # Crear gráfico de barras
        modules = list(module_stats.keys())
        activations = [module_stats[m]['avg_activation'] for m in modules]
        connections = [module_stats[m]['avg_connections'] for m in modules]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Activación Promedio por Módulo', 'Conexiones Promedio por Módulo'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Gráfico de activaciones
        fig.add_trace(
            go.Bar(
                x=modules,
                y=activations,
                name='Activación Promedio',
                marker_color='#4ECDC4',
                text=[f'{a:.3f}' for a in activations],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Gráfico de conexiones
        fig.add_trace(
            go.Bar(
                x=modules,
                y=connections,
                name='Conexiones Promedio',
                marker_color='#FFB347',
                text=[f'{c:.1f}' for c in connections],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="📈 Estadísticas de Red Neural por Módulo",
            paper_bgcolor=self.color_scheme["background"],
            plot_bgcolor=self.color_scheme["background"],
            font=dict(color='white'),
            height=500,
            showlegend=False
        )
        
        return fig
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas del sistema neural"""
        total_nodes = len(self.topology.nodes)
        total_flows = len(self.topology.flows)
        avg_activation = np.mean([node.activation for node in self.topology.nodes.values()])
        avg_flow_intensity = np.mean([flow.intensity for flow in self.topology.flows]) if self.topology.flows else 0
        
        # Métricas por tipo de nodo
        node_type_counts = {}
        for node in self.topology.nodes.values():
            if node.type not in node_type_counts:
                node_type_counts[node.type] = 0
            node_type_counts[node.type] += 1
        
        # Métricas por tipo de flujo
        flow_type_counts = {}
        for flow in self.topology.flows:
            if flow.flow_type not in flow_type_counts:
                flow_type_counts[flow.flow_type] = 0
            flow_type_counts[flow.flow_type] += 1
        
        return {
            'total_nodes': total_nodes,
            'total_flows': total_flows,
            'average_activation': avg_activation,
            'average_flow_intensity': avg_flow_intensity,
            'node_type_distribution': {nt.value: count for nt, count in node_type_counts.items()},
            'flow_type_distribution': {ft.value: count for ft, count in flow_type_counts.items()},
            'last_update': self.last_update.isoformat(),
            'network_coherence': self._calculate_network_coherence(),
            'information_flow_rate': self._calculate_information_flow_rate()
        }
    
    def _calculate_network_coherence(self) -> float:
        """Calcula la coherencia de la red neural"""
        if not self.topology.nodes:
            return 0.0
        
        activations = [node.activation for node in self.topology.nodes.values()]
        # Coherencia basada en la varianza de activaciones (menor varianza = mayor coherencia)
        variance = np.var(activations)
        coherence = 1.0 / (1.0 + variance)
        return coherence
    
    def _calculate_information_flow_rate(self) -> float:
        """Calcula la tasa de flujo de información"""
        if not self.topology.flows:
            return 0.0
        
        total_flow = sum(flow.intensity * flow.speed for flow in self.topology.flows)
        return total_flow / len(self.topology.flows)
    
    def start_real_time_monitoring(self):
        """Inicia el monitoreo en tiempo real"""
        self.is_running = True
        
    def stop_real_time_monitoring(self):
        """Detiene el monitoreo en tiempo real"""
        self.is_running = False
        
    def get_flow_statistics(self):
        """Obtiene estadísticas de flujos"""
        if not self.flow_data:
            return {
                'total_modules': 0,
                'overall_activity': 0.0,
                'most_active_module': 'None',
                'data_points_collected': 0
            }
        
        last_data = self.flow_data[-1]
        activations = last_data.get('activations', {})
        
        most_active = max(activations.items(), key=lambda x: x[1]) if activations else ('None', 0)
        
        return {
            'total_modules': len(activations),
            'overall_activity': np.mean(list(activations.values())) if activations else 0.0,
            'most_active_module': most_active[0],
            'data_points_collected': len(self.flow_data)
        }
    
    def create_real_time_flow_visualization(self):
        """Crea visualización de flujos en tiempo real"""
        fig = self.create_3d_network_plot()
        fig.update_layout(title="Flujos Neuronales en Tiempo Real")
        return fig
    
    def create_coherence_heatmap(self):
        """Crea mapa de coherencia simplificado"""
        modules = ['ganst_core', 'moduladores', 'memoria_corto_plazo', 'meta_enrutador', 'consciousness_network']
        coherence_matrix = np.random.rand(len(modules), len(modules)) * 0.5 + 0.5
        
        fig = go.Figure(data=go.Heatmap(
            z=coherence_matrix,
            x=modules,
            y=modules,
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title="Mapa de Coherencia",
            height=300,
            paper_bgcolor=self.color_scheme["background"],
            font=dict(color='white', size=10)
        )
        
        return fig
    
    def create_activation_timeline(self):
        """Crea timeline de activaciones"""
        if not self.flow_data:
            # Datos simulados para demostración
            timestamps = [datetime.now() - timedelta(seconds=i*5) for i in range(20, 0, -1)]
            activity_levels = [0.3 + 0.4 * np.sin(i * 0.5) + 0.1 * np.random.random() for i in range(20)]
        else:
            timestamps = [entry.get('timestamp', datetime.now()) for entry in self.flow_data[-20:]]
            activity_levels = [np.mean(list(entry.get('activations', {}).values())) for entry in self.flow_data[-20:]]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=activity_levels,
            mode='lines+markers',
            name='Actividad Neural',
            line=dict(color='#4ECDC4', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title="Timeline de Actividad Neural",
            xaxis_title="Tiempo",
            yaxis_title="Nivel de Activación",
            paper_bgcolor=self.color_scheme["background"],
            plot_bgcolor=self.color_scheme["background"],
            font=dict(color='white'),
            height=250
        )
        
        return fig
    
    def _initialize_sample_data(self):
        """Inicializa datos de muestra para demostración inmediata"""
        current_time = datetime.now()
        
        # Generar datos de activaciones simuladas para todos los módulos
        sample_activations = {}
        for node_id, node in self.topology.nodes.items():
            module = node.properties.get("module", "unknown")
            base_activation = 0.4 + 0.3 * np.sin(time.time() * 0.5 + hash(node_id) % 100)
            sample_activations[node_id] = max(0.0, min(1.0, base_activation))
        
        # Agregar datos iniciales al historial
        self.flow_data.append({
            'timestamp': current_time,
            'activations': sample_activations,
            'total_activity': np.mean(list(sample_activations.values()))
        })
        
        # Generar fortalezas de conexión iniciales
        for node_id, node in self.topology.nodes.items():
            for target_id in node.connections:
                connection_key = f"{node_id}->{target_id}"
                self.connection_strengths[connection_key] = 0.3 + 0.4 * np.random.random()

# Función principal para la interfaz de Streamlit
def create_neural_flow_interface():
    """Crea la interfaz principal de visualización de flujos neuronales"""
    st.header("🌊 Visualización Interactiva de Flujos Neuronales Ruth R1")
    
    # Inicializar visualizador si no existe
    if 'neural_visualizer' not in st.session_state:
        st.session_state.neural_visualizer = NeuralFlowVisualizer()
    
    visualizer = st.session_state.neural_visualizer
    
    # Panel de control
    st.sidebar.subheader("🎛️ Control de Visualización")
    
    # Opciones de actualización
    auto_update = st.sidebar.checkbox("Actualización Automática", value=True)
    update_interval = st.sidebar.slider("Intervalo de Actualización (segundos)", 1, 10, 3)
    
    # Filtros de visualización
    st.sidebar.subheader("🔍 Filtros")
    min_flow_intensity = st.sidebar.slider("Intensidad Mínima de Flujo", 0.0, 1.0, 0.3, 0.1)
    show_node_labels = st.sidebar.checkbox("Mostrar Etiquetas de Nodos", value=True)
    
    # Selección de módulos a mostrar
    available_modules = list(set(node.properties.get("module", "unknown") 
                               for node in visualizer.topology.nodes.values()))
    selected_modules = st.sidebar.multiselect(
        "Módulos a Visualizar",
        available_modules,
        default=available_modules
    )
    
    # Botón de actualización manual
    if st.sidebar.button("🔄 Actualizar Ahora"):
        # Intentar obtener datos reales del sistema
        try:
            from core.ganst_core import get_ganst_core
            from core.moduladores import get_modulation_manager
            from core.memorias_corto_plazo import get_short_term_memory
            
            ganst_core = get_ganst_core()
            modulators = get_modulation_manager()
            memory_system = get_short_term_memory()
            
            flows = visualizer.update_real_time_data(ganst_core, modulators, memory_system)
            st.sidebar.success(f"✅ Datos actualizados: {len(flows)} flujos activos")
        except Exception as e:
            flows = visualizer.update_real_time_data()
            st.sidebar.info("📊 Usando datos simulados")
    
    # Métricas del sistema
    st.subheader("📊 Métricas del Sistema Neural")
    metrics = visualizer.get_system_metrics()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Nodos Totales", metrics['total_nodes'])
    with col2:
        st.metric("Flujos Activos", metrics['total_flows'])
    with col3:
        st.metric("Activación Promedio", f"{metrics['average_activation']:.3f}")
    with col4:
        st.metric("Coherencia de Red", f"{metrics['network_coherence']:.3f}")
    
    # Visualizaciones principales
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌐 Red 3D", "🔥 Mapa de Calor", "📊 Análisis Temporal", "📈 Estadísticas"
    ])
    
    with tab1:
        st.subheader("🌐 Visualización 3D de la Red Neural")
        
        # Filtrar nodos por módulos seleccionados
        if selected_modules:
            # Actualizar datos antes de mostrar
            flows = visualizer.update_real_time_data()
            
            # Crear y mostrar gráfico 3D
            fig_3d = visualizer.create_3d_network_plot()
            st.plotly_chart(fig_3d, use_container_width=True)
            
            # Información adicional
            with st.expander("ℹ️ Información de la Red"):
                st.write("**Tipos de Nodos:**")
                for node_type, color in visualizer.color_scheme["node_types"].items():
                    st.markdown(f"• **{node_type.value.title()}**: <span style='color:{color}'>●</span> {metrics['node_type_distribution'].get(node_type.value, 0)} nodos", unsafe_allow_html=True)
                
                st.write("**Tipos de Flujo:**")
                for flow_type, color in visualizer.color_scheme["flow_types"].items():
                    st.markdown(f"• **{flow_type.value.title()}**: <span style='color:{color}'>—</span> {metrics['flow_type_distribution'].get(flow_type.value, 0)} flujos", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Selecciona al menos un módulo para visualizar")
    
    with tab2:
        st.subheader("🔥 Mapa de Calor de Intensidades")
        
        if selected_modules:
            flows = visualizer.update_real_time_data()
            fig_heatmap = visualizer.create_flow_intensity_heatmap()
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Análisis de intensidades
            st.write("**Análisis de Intensidades:**")
            if visualizer.topology.flows:
                max_intensity = max(flow.intensity for flow in visualizer.topology.flows)
                avg_intensity = np.mean([flow.intensity for flow in visualizer.topology.flows])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Intensidad Máxima", f"{max_intensity:.3f}")
                with col2:
                    st.metric("Intensidad Promedio", f"{avg_intensity:.3f}")
        else:
            st.warning("⚠️ Selecciona al menos un módulo para visualizar")
    
    with tab3:
        st.subheader("📊 Análisis Temporal de Flujos")
        
        if len(visualizer.flow_history) > 1:
            fig_temporal = visualizer.create_temporal_flow_analysis()
            st.plotly_chart(fig_temporal, use_container_width=True)
            
            # Estadísticas temporales
            st.write("**Estadísticas Temporales:**")
            recent_activities = [entry['total_activity'] for entry in visualizer.flow_history[-10:]]
            if recent_activities:
                trend = "🔼 Aumentando" if recent_activities[-1] > recent_activities[0] else "🔽 Disminuyendo"
                st.write(f"Tendencia reciente: {trend}")
                st.write(f"Actividad máxima: {max(recent_activities):.3f}")
                st.write(f"Actividad mínima: {min(recent_activities):.3f}")
        else:
            st.info("📈 Acumulando datos temporales... Espera unos momentos para ver el análisis.")
    
    with tab4:
        st.subheader("📈 Estadísticas de la Red")
        
        flows = visualizer.update_real_time_data()
        fig_stats = visualizer.create_network_statistics_plot()
        st.plotly_chart(fig_stats, use_container_width=True)
        
        # Tabla de estadísticas detalladas
        st.write("**Métricas Detalladas:**")
        metrics_df = pd.DataFrame([
            {"Métrica": "Tasa de Flujo de Información", "Valor": f"{metrics['information_flow_rate']:.3f}"},
            {"Métrica": "Última Actualización", "Valor": metrics['last_update']},
            {"Métrica": "Nodos por Tipo", "Valor": str(metrics['node_type_distribution'])},
            {"Métrica": "Flujos por Tipo", "Valor": str(metrics['flow_type_distribution'])}
        ])
        st.dataframe(metrics_df, use_container_width=True)
    
    # Auto-actualización
    if auto_update:
        time.sleep(update_interval)
        st.rerun()

if __name__ == "__main__":
    create_neural_flow_interface()