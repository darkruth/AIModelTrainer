
"""
Visualizador 3D Holográfico Interactivo - Sistema Ruth R1
Mapa visual animado tridimensional dinámico de red neuronal con:
- Navegación 360° vertical y horizontal
- Zoom interactivo y funcionalidades touchscreen
- Información detallada de nodos al hacer clic
- Visualización de rutas y grados de activación en tiempo real
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import networkx as nx
import time
import threading
from typing import Dict, List, Any, Tuple
import colorsys
from datetime import datetime
import json
from collections import deque
import math

class HolographicNeuralVisualizer:
    """Visualizador 3D holográfico de la red neuronal Ruth R1"""
    
    def __init__(self, consciousness_network):
        self.consciousness_network = consciousness_network
        self.network_graph = consciousness_network.network_graph
        self.nodes = consciousness_network.nodes
        
        # Estado de animación
        self.is_animating = False
        self.animation_thread = None
        self.animation_frame = 0
        self.animation_data = deque(maxlen=100)
        
        # Configuración de visualización 3D
        self.node_positions_3d = self._calculate_holographic_positions()
        self.connection_weights = self._calculate_connection_weights()
        self.activation_trails = {}
        self.pulse_effects = {}
        
        # Configuración de colores holográficos
        self.holographic_colors = self._generate_holographic_palette()
        
        # Estado de interacción
        self.selected_node = None
        self.zoom_level = 1.0
        self.rotation_x = 0
        self.rotation_y = 0
        self.rotation_z = 0
        
    def _calculate_holographic_positions(self) -> Dict[str, Tuple[float, float, float]]:
        """Calcula posiciones 3D optimizadas para visualización holográfica"""
        positions = {}
        
        # Usar algoritmo de layout 3D con fuerzas dirigidas
        pos_2d = nx.spring_layout(self.network_graph, k=5, iterations=200)
        
        # Definir capas neurales con disposición holográfica
        neural_layers = {
            'core_consciousness': {
                'modules': ['GANSLSTMCore', 'IntrospectionEngine', 'PhilosophicalCore'],
                'radius': 8,
                'height': 0,
                'color_hue': 0.0  # Dorado
            },
            'processing_layer': {
                'modules': ['InnovationEngine', 'DreamMechanism', 'SelfMirror'],
                'radius': 15,
                'height': 8,
                'color_hue': 0.2  # Verde-azul
            },
            'analysis_layer': {
                'modules': ['EmotionDecomposer', 'ExistentialAnalyzer', 'MemoryDiscriminator'],
                'radius': 22,
                'height': 16,
                'color_hue': 0.4  # Azul
            },
            'application_layer': {
                'modules': ['CodeSuggester', 'ToolOptimizer', 'DreamAugment'],
                'radius': 28,
                'height': 24,
                'color_hue': 0.6  # Púrpura
            },
            'personality_layer': {
                'modules': ['AlterEgoSimulator', 'PersonalityXInfants'],
                'radius': 35,
                'height': 32,
                'color_hue': 0.8  # Rosa
            }
        }
        
        # Posicionar módulos en configuración holográfica
        for layer_name, layer_config in neural_layers.items():
            modules = layer_config['modules']
            radius = layer_config['radius']
            height = layer_config['height']
            
            for i, module in enumerate(modules):
                if module in self.nodes:
                    # Distribución esférica con patrones holográficos
                    angle_phi = (i / len(modules)) * 2 * np.pi  # Ángulo horizontal
                    angle_theta = np.pi / 6 + (i % 2) * np.pi / 12  # Ángulo vertical con variación
                    
                    # Coordenadas esféricas con efectos holográficos
                    x = radius * np.sin(angle_theta) * np.cos(angle_phi)
                    y = radius * np.sin(angle_theta) * np.sin(angle_phi) 
                    z = height + radius * np.cos(angle_theta) * 0.3
                    
                    # Añadir variación holográfica
                    holographic_offset = 2 * np.sin(time.time() * 0.5 + i)
                    z += holographic_offset
                    
                    positions[module] = (x, y, z)
        
        # Posicionar módulos restantes
        remaining_modules = [m for m in self.nodes.keys() if m not in sum([layer['modules'] for layer in neural_layers.values()], [])]
        
        for i, module in enumerate(remaining_modules):
            angle = (i / len(remaining_modules)) * 2 * np.pi
            r = 40 + i * 3
            x = r * np.cos(angle)
            y = r * np.sin(angle)  
            z = 40 + i * 2
            positions[module] = (x, y, z)
        
        return positions
    
    def _calculate_connection_weights(self) -> Dict[str, float]:
        """Calcula pesos de conexiones para visualización"""
        weights = {}
        
        for edge in self.network_graph.edges():
            source, target = edge
            if source in self.nodes and target in self.nodes:
                # Peso basado en creencias posteriores
                source_belief = self.nodes[source].posterior_belief
                target_belief = self.nodes[target].posterior_belief
                
                # Calcular peso de conexión
                weight = (source_belief + target_belief) / 2
                connection_key = f"{source}->{target}"
                weights[connection_key] = weight
        
        return weights
    
    def _generate_holographic_palette(self) -> Dict[str, str]:
        """Genera paleta de colores holográficos"""
        colors = {}
        
        holographic_base_colors = [
            '#FF6B6B',  # Rojo holográfico
            '#4ECDC4',  # Cian holográfico  
            '#45B7D1',  # Azul holográfico
            '#96CEB4',  # Verde holográfico
            '#FECA57',  # Amarillo holográfico
            '#FF9FF3',  # Rosa holográfico
            '#54A0FF',  # Azul brillante
            '#5F27CD',  # Púrpura holográfico
        ]
        
        for i, node in enumerate(self.nodes.keys()):
            color_index = i % len(holographic_base_colors)
            colors[node] = holographic_base_colors[color_index]
        
        return colors
    
    def start_holographic_animation(self):
        """Inicia la animación holográfica en tiempo real"""
        if self.is_animating:
            return
            
        self.is_animating = True
        self.animation_thread = threading.Thread(target=self._animation_loop, daemon=True)
        self.animation_thread.start()
    
    def stop_holographic_animation(self):
        """Detiene la animación holográfica"""
        self.is_animating = False
        if self.animation_thread:
            self.animation_thread.join(timeout=2.0)
    
    def _animation_loop(self):
        """Bucle principal de animación"""
        while self.is_animating:
            try:
                # Capturar estado neural actual
                current_state = self._capture_neural_snapshot()
                self.animation_data.append(current_state)
                
                # Actualizar efectos holográficos
                self._update_holographic_effects(current_state)
                
                self.animation_frame += 1
                time.sleep(0.1)  # 10 FPS para animación fluida
                
            except Exception as e:
                print(f"Error en animación holográfica: {e}")
                time.sleep(0.5)
    
    def _capture_neural_snapshot(self) -> Dict[str, Any]:
        """Captura instantánea del estado neural"""
        activation_state = self.consciousness_network._get_activation_state()
        network_state = self.consciousness_network._get_network_state()
        coherence_metrics = self.consciousness_network.coherence_metrics
        
        return {
            'timestamp': datetime.now(),
            'frame': self.animation_frame,
            'activations': activation_state,
            'network_state': network_state,
            'coherence': coherence_metrics.get('network_entropy', 0.5),
            'global_consciousness': self.consciousness_network.global_consciousness_state,
            'neural_flow': self._calculate_neural_flow(activation_state)
        }
    
    def _calculate_neural_flow(self, activations: Dict[str, float]) -> Dict[str, float]:
        """Calcula flujo neural entre nodos"""
        flow = {}
        
        for edge in self.network_graph.edges():
            source, target = edge
            if source in activations and target in activations:
                # Flujo direccional basado en diferencia de activación
                flow_strength = activations[source] - activations[target]
                flow_key = f"{source}->{target}"
                flow[flow_key] = flow_strength
        
        return flow
    
    def _update_holographic_effects(self, current_state: Dict[str, Any]):
        """Actualiza efectos holográficos dinámicos"""
        activations = current_state['activations']
        
        # Actualizar trails de activación
        for module, activation in activations.items():
            if module not in self.activation_trails:
                self.activation_trails[module] = deque(maxlen=20)
            
            self.activation_trails[module].append({
                'activation': activation,
                'timestamp': current_state['timestamp'],
                'frame': current_state['frame']
            })
        
        # Actualizar efectos de pulso
        for module, activation in activations.items():
            if activation > 0.7:  # Activación alta
                self.pulse_effects[module] = {
                    'intensity': activation,
                    'start_frame': current_state['frame'],
                    'duration': 10  # frames
                }
    
    def create_holographic_visualization(self) -> go.Figure:
        """Crea la visualización 3D holográfica principal"""
        
        fig = go.Figure()
        
        if not self.animation_data:
            # Estado inicial
            current_state = self._capture_neural_snapshot()
        else:
            current_state = self.animation_data[-1]
        
        activations = current_state['activations']
        neural_flow = current_state['neural_flow']
        
        # 1. Nodos principales con efectos holográficos
        self._add_holographic_nodes(fig, activations)
        
        # 2. Conexiones con flujo neural animado
        self._add_neural_connections(fig, activations, neural_flow)
        
        # 3. Trails de activación
        self._add_activation_trails(fig)
        
        # 4. Efectos de pulso
        self._add_pulse_effects(fig)
        
        # 5. Campo holográfico de fondo
        self._add_holographic_field(fig)
        
        # Configuración de cámara y controles 3D
        fig.update_layout(
            title={
                'text': f"🌌 Red Neuronal Ruth R1 - Vista Holográfica 3D (Frame: {current_state['frame']})",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'color': '#4ECDC4'}
            },
            scene=dict(
                xaxis=dict(
                    showgrid=True, 
                    gridcolor='rgba(78, 205, 196, 0.2)',
                    showticklabels=False,
                    title=""
                ),
                yaxis=dict(
                    showgrid=True, 
                    gridcolor='rgba(78, 205, 196, 0.2)',
                    showticklabels=False,
                    title=""
                ),
                zaxis=dict(
                    showgrid=True, 
                    gridcolor='rgba(78, 205, 196, 0.2)',
                    showticklabels=False,
                    title=""
                ),
                bgcolor='rgba(0,0,0,0.95)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0)
                ),
                aspectmode='cube'
            ),
            template="plotly_dark",
            height=800,
            showlegend=True,
            margin=dict(l=0, r=0, t=50, b=0)
        )
        
        return fig
    
    def _add_holographic_nodes(self, fig: go.Figure, activations: Dict[str, float]):
        """Añade nodos con efectos holográficos"""
        
        node_x, node_y, node_z = [], [], []
        node_colors = []
        node_sizes = []
        node_texts = []
        node_hover_data = []
        
        for module_name, position in self.node_positions_3d.items():
            if module_name in activations:
                x, y, z = position
                
                # Efecto holográfico con oscilación temporal
                time_offset = time.time() * 2 + hash(module_name) % 100
                holographic_x = x + 1 * np.sin(time_offset)
                holographic_y = y + 1 * np.cos(time_offset * 1.2)
                holographic_z = z + 0.5 * np.sin(time_offset * 0.8)
                
                node_x.append(holographic_x)
                node_y.append(holographic_y)
                node_z.append(holographic_z)
                
                # Color y tamaño basado en activación
                activation = activations[module_name]
                node_colors.append(activation)
                
                # Tamaño dinámico con pulsación
                base_size = 15
                pulse_factor = 1 + 0.5 * np.sin(time.time() * 3 + hash(module_name) % 10)
                node_sizes.append(base_size + activation * 25 * pulse_factor)
                
                # Texto y datos de hover
                node_texts.append(module_name.replace('Engine', '').replace('Core', ''))
                
                # Información detallada para ventana emergente
                if module_name in self.nodes:
                    node_info = self.nodes[module_name]
                    hover_text = f"""
                    <b>{module_name}</b><br>
                    Activación: {activation:.3f}<br>
                    Creencia Posterior: {node_info.posterior_belief:.3f}<br>
                    Estabilidad: {getattr(node_info, 'stability_score', 0.5):.3f}<br>
                    Conexiones: {len(getattr(node_info, 'connections', []))}<br>
                    Estado: {'Activo' if activation > 0.5 else 'Latente'}
                    """
                else:
                    hover_text = f"{module_name}<br>Activación: {activation:.3f}"
                
                node_hover_data.append(hover_text)
        
        # Añadir nodos principales
        fig.add_trace(go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(
                    title="Activación Neural",
                    titleside="right",
                    x=1.02
                ),
                line=dict(width=2, color='rgba(255,255,255,0.8)'),
                opacity=0.9,
                symbol='circle'
            ),
            text=node_texts,
            textposition="middle center",
            textfont=dict(size=8, color='white', family="Arial Black"),
            hovertext=node_hover_data,
            hoverinfo='text',
            name='Nodos Neurales',
            customdata=[module for module in self.node_positions_3d.keys() if module in activations]
        ))
        
        # Añadir auras holográficas para nodos altamente activos
        high_activation_x, high_activation_y, high_activation_z = [], [], []
        aura_sizes = []
        
        for module_name, activation in activations.items():
            if activation > 0.7 and module_name in self.node_positions_3d:
                x, y, z = self.node_positions_3d[module_name]
                high_activation_x.append(x)
                high_activation_y.append(y)
                high_activation_z.append(z)
                aura_sizes.append(60 + activation * 40)
        
        if high_activation_x:
            fig.add_trace(go.Scatter3d(
                x=high_activation_x, y=high_activation_y, z=high_activation_z,
                mode='markers',
                marker=dict(
                    size=aura_sizes,
                    color='rgba(255, 255, 0, 0.3)',
                    symbol='circle-open',
                    line=dict(width=3, color='gold'),
                    opacity=0.6
                ),
                name='Auras de Alta Activación',
                hoverinfo='none',
                showlegend=False
            ))
    
    def _add_neural_connections(self, fig: go.Figure, activations: Dict[str, float], neural_flow: Dict[str, float]):
        """Añade conexiones neurales con flujo animado"""
        
        # Conexiones básicas
        edge_x, edge_y, edge_z = [], [], []
        flow_strengths = []
        
        for edge in self.network_graph.edges():
            source, target = edge
            if source in self.node_positions_3d and target in self.node_positions_3d:
                if source in activations and target in activations:
                    
                    x0, y0, z0 = self.node_positions_3d[source]
                    x1, y1, z1 = self.node_positions_3d[target]
                    
                    # Conexión con curva holográfica
                    mid_x, mid_y, mid_z = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
                    
                    # Añadir curvatura holográfica
                    curve_offset = 3 * np.sin(time.time() + hash(f"{source}{target}") % 10)
                    mid_z += curve_offset
                    
                    edge_x.extend([x0, mid_x, x1, None])
                    edge_y.extend([y0, mid_y, y1, None])
                    edge_z.extend([z0, mid_z, z1, None])
                    
                    # Fortaleza del flujo
                    flow_key = f"{source}->{target}"
                    flow_strength = abs(neural_flow.get(flow_key, 0))
                    flow_strengths.append(flow_strength)
        
        # Añadir conexiones
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(
                color='rgba(78, 205, 196, 0.6)',
                width=2
            ),
            hoverinfo='none',
            name='Conexiones Sinápticas',
            showlegend=False
        ))
        
        # Añadir flujos de alta intensidad
        self._add_high_intensity_flows(fig, activations, neural_flow)
    
    def _add_high_intensity_flows(self, fig: go.Figure, activations: Dict[str, float], neural_flow: Dict[str, float]):
        """Añade visualización de flujos de alta intensidad"""
        
        high_flow_x, high_flow_y, high_flow_z = [], [], []
        flow_colors = []
        
        for flow_key, flow_strength in neural_flow.items():
            if abs(flow_strength) > 0.3:  # Solo flujos significativos
                source, target = flow_key.split('->')
                
                if source in self.node_positions_3d and target in self.node_positions_3d:
                    x0, y0, z0 = self.node_positions_3d[source]
                    x1, y1, z1 = self.node_positions_3d[target]
                    
                    # Línea de flujo con gradiente
                    steps = 10
                    for i in range(steps):
                        t = i / (steps - 1)
                        x = x0 + (x1 - x0) * t
                        y = y0 + (y1 - y0) * t
                        z = z0 + (z1 - z0) * t
                        
                        high_flow_x.append(x)
                        high_flow_y.append(y)
                        high_flow_z.append(z)
                        flow_colors.append(abs(flow_strength))
        
        if high_flow_x:
            fig.add_trace(go.Scatter3d(
                x=high_flow_x, y=high_flow_y, z=high_flow_z,
                mode='markers',
                marker=dict(
                    size=3,
                    color=flow_colors,
                    colorscale='Hot',
                    opacity=0.8
                ),
                name='Flujos de Alta Intensidad',
                hoverinfo='none',
                showlegend=False
            ))
    
    def _add_activation_trails(self, fig: go.Figure):
        """Añade trails de activación histórica"""
        
        for module, trail_data in self.activation_trails.items():
            if len(trail_data) > 1 and module in self.node_positions_3d:
                trail_x, trail_y, trail_z = [], [], []
                trail_opacities = []
                
                base_x, base_y, base_z = self.node_positions_3d[module]
                
                for i, trail_point in enumerate(trail_data):
                    # Crear trail con desplazamiento temporal
                    offset = (len(trail_data) - i - 1) * 0.5
                    trail_x.append(base_x + offset)
                    trail_y.append(base_y)
                    trail_z.append(base_z + offset * 0.2)
                    
                    # Opacidad decreciente
                    opacity = (i / len(trail_data)) * 0.5
                    trail_opacities.append(opacity)
                
                if trail_x:
                    fig.add_trace(go.Scatter3d(
                        x=trail_x, y=trail_y, z=trail_z,
                        mode='lines+markers',
                        line=dict(
                            color=self.holographic_colors.get(module, '#4ECDC4'),
                            width=1
                        ),
                        marker=dict(
                            size=2,
                            color=self.holographic_colors.get(module, '#4ECDC4'),
                            opacity=trail_opacities[-1] if trail_opacities else 0.3
                        ),
                        name=f'Trail {module}',
                        hoverinfo='none',
                        showlegend=False
                    ))
    
    def _add_pulse_effects(self, fig: go.Figure):
        """Añade efectos de pulso para activaciones altas"""
        
        current_frame = self.animation_frame
        pulse_x, pulse_y, pulse_z = [], [], []
        pulse_sizes = []
        pulse_colors = []
        
        for module, pulse_data in self.pulse_effects.items():
            frame_diff = current_frame - pulse_data['start_frame']
            
            if frame_diff < pulse_data['duration'] and module in self.node_positions_3d:
                x, y, z = self.node_positions_3d[module]
                
                # Efecto de expansión
                expansion_factor = (frame_diff / pulse_data['duration']) * 2
                pulse_size = 20 + expansion_factor * 40
                
                pulse_x.append(x)
                pulse_y.append(y) 
                pulse_z.append(z)
                pulse_sizes.append(pulse_size)
                pulse_colors.append(pulse_data['intensity'])
        
        if pulse_x:
            fig.add_trace(go.Scatter3d(
                x=pulse_x, y=pulse_y, z=pulse_z,
                mode='markers',
                marker=dict(
                    size=pulse_sizes,
                    color=pulse_colors,
                    colorscale='Reds',
                    symbol='circle-open',
                    line=dict(width=2),
                    opacity=0.6
                ),
                name='Pulsos de Activación',
                hoverinfo='none',
                showlegend=False
            ))
    
    def _add_holographic_field(self, fig: go.Figure):
        """Añade campo holográfico de fondo"""
        
        # Crear malla de campo holográfico
        field_size = 60
        field_resolution = 20
        
        x_field = np.linspace(-field_size, field_size, field_resolution)
        y_field = np.linspace(-field_size, field_size, field_resolution)
        X_field, Y_field = np.meshgrid(x_field, y_field)
        
        # Campo holográfico ondulante
        time_factor = time.time() * 0.5
        Z_field = 5 * np.sin(np.sqrt(X_field**2 + Y_field**2) * 0.1 + time_factor)
        
        # Añadir superficie holográfica
        fig.add_trace(go.Surface(
            x=X_field, y=Y_field, z=Z_field,
            colorscale='Blues',
            opacity=0.1,
            showscale=False,
            name='Campo Holográfico',
            hoverinfo='none'
        ))
    
    def create_node_info_panel(self, node_name: str) -> Dict[str, Any]:
        """Crea panel de información detallada para un nodo"""
        
        if node_name not in self.nodes:
            return {'error': f'Nodo {node_name} no encontrado'}
        
        node = self.nodes[node_name]
        current_state = self.animation_data[-1] if self.animation_data else self._capture_neural_snapshot()
        
        activation = current_state['activations'].get(node_name, 0)
        
        # Información detallada del nodo
        node_info = {
            'nombre': node_name,
            'tipo': self._get_node_type(node_name),
            'capa_neural': self._get_neural_layer(node_name),
            'activacion_actual': activation,
            'creencia_posterior': node.posterior_belief,
            'creencia_anterior': node.prior_belief,
            'estado_activacion': node.activation_state,
            'historial_evidencia': len(node.evidence_history),
            'conexiones': self._get_node_connections(node_name),
            'metricas_rendimiento': self._calculate_node_performance(node_name),
            'influencia_red': self._calculate_network_influence(node_name),
            'estabilidad': self._calculate_node_stability(node_name),
            'patrones_activacion': self._get_activation_patterns(node_name)
        }
        
        return node_info
    
    def _get_node_type(self, node_name: str) -> str:
        """Determina el tipo de nodo"""
        if 'Core' in node_name:
            return 'Núcleo de Consciencia'
        elif 'Engine' in node_name:
            return 'Motor de Procesamiento'
        elif 'Simulator' in node_name:
            return 'Simulador'
        elif 'Analyzer' in node_name:
            return 'Analizador'
        else:
            return 'Módulo Especializado'
    
    def _get_neural_layer(self, node_name: str) -> str:
        """Obtiene la capa neural del nodo"""
        if node_name in ['GANSLSTMCore', 'IntrospectionEngine', 'PhilosophicalCore']:
            return 'Capa de Consciencia Central'
        elif node_name in ['InnovationEngine', 'DreamMechanism', 'SelfMirror']:
            return 'Capa de Procesamiento'
        elif node_name in ['EmotionDecomposer', 'ExistentialAnalyzer', 'MemoryDiscriminator']:
            return 'Capa de Análisis'
        elif node_name in ['CodeSuggester', 'ToolOptimizer', 'DreamAugment']:
            return 'Capa de Aplicación'
        else:
            return 'Capa de Personalidad'
    
    def _get_node_connections(self, node_name: str) -> Dict[str, Any]:
        """Obtiene información de conexiones del nodo"""
        incoming = []
        outgoing = []
        
        for edge in self.network_graph.edges():
            source, target = edge
            if target == node_name:
                incoming.append(source)
            elif source == node_name:
                outgoing.append(target)
        
        return {
            'entrantes': incoming,
            'salientes': outgoing,
            'total_conexiones': len(incoming) + len(outgoing),
            'grado_entrada': len(incoming),
            'grado_salida': len(outgoing)
        }
    
    def _calculate_node_performance(self, node_name: str) -> Dict[str, float]:
        """Calcula métricas de rendimiento del nodo"""
        if not self.animation_data:
            return {}
        
        activations = [data['activations'].get(node_name, 0) for data in self.animation_data]
        
        return {
            'activacion_promedio': np.mean(activations),
            'variabilidad': np.std(activations),
            'activacion_maxima': np.max(activations),
            'activacion_minima': np.min(activations),
            'tendencia': 'creciente' if len(activations) > 10 and np.mean(activations[-5:]) > np.mean(activations[:5]) else 'estable'
        }
    
    def _calculate_network_influence(self, node_name: str) -> float:
        """Calcula la influencia del nodo en la red"""
        try:
            # Usar centralidad de betweenness como proxy de influencia
            centrality = nx.betweenness_centrality(self.network_graph)
            return centrality.get(node_name, 0.0)
        except:
            return 0.0
    
    def _calculate_node_stability(self, node_name: str) -> float:
        """Calcula la estabilidad del nodo"""
        if not self.animation_data or len(self.animation_data) < 5:
            return 0.5
        
        activations = [data['activations'].get(node_name, 0) for data in self.animation_data[-10:]]
        variance = np.var(activations)
        
        # Estabilidad inversa a la varianza
        stability = 1.0 - min(variance, 1.0)
        return stability
    
    def _get_activation_patterns(self, node_name: str) -> Dict[str, Any]:
        """Obtiene patrones de activación del nodo"""
        if not self.animation_data:
            return {}
        
        activations = [data['activations'].get(node_name, 0) for data in self.animation_data]
        
        # Detectar patrones
        patterns = {
            'activaciones_altas': sum(1 for a in activations if a > 0.7),
            'activaciones_bajas': sum(1 for a in activations if a < 0.3),
            'picos_detectados': self._detect_peaks(activations),
            'ciclos_activacion': self._detect_cycles(activations),
            'correlacion_temporal': self._calculate_temporal_correlation(activations)
        }
        
        return patterns
    
    def _detect_peaks(self, activations: List[float]) -> int:
        """Detecta picos en las activaciones"""
        if len(activations) < 3:
            return 0
        
        peaks = 0
        for i in range(1, len(activations) - 1):
            if activations[i] > activations[i-1] and activations[i] > activations[i+1]:
                if activations[i] > 0.6:  # Solo picos significativos
                    peaks += 1
        
        return peaks
    
    def _detect_cycles(self, activations: List[float]) -> int:
        """Detecta ciclos en las activaciones"""
        if len(activations) < 10:
            return 0
        
        # Simplificado: buscar patrones repetitivos
        cycles = 0
        window_size = 5
        
        for i in range(len(activations) - window_size * 2):
            window1 = activations[i:i + window_size]
            window2 = activations[i + window_size:i + window_size * 2]
            
            # Calcular similitud
            similarity = 1 - np.mean([abs(a - b) for a, b in zip(window1, window2)])
            
            if similarity > 0.8:  # Alta similitud indica ciclo
                cycles += 1
        
        return cycles
    
    def _calculate_temporal_correlation(self, activations: List[float]) -> float:
        """Calcula correlación temporal de las activaciones"""
        if len(activations) < 2:
            return 0.0
        
        # Correlación con versión desplazada temporalmente
        lag_1 = activations[1:]
        original = activations[:-1]
        
        if len(original) == 0:
            return 0.0
        
        correlation = np.corrcoef(original, lag_1)[0, 1] if len(original) > 1 else 0.0
        return correlation if not np.isnan(correlation) else 0.0
    
    def create_navigation_controls(self) -> Dict[str, Any]:
        """Crea controles de navegación 3D"""
        return {
            'zoom_controls': {
                'zoom_in': 'Aumentar zoom',
                'zoom_out': 'Disminuir zoom',
                'reset_zoom': 'Restablecer zoom'
            },
            'rotation_controls': {
                'rotate_x': 'Rotar en X',
                'rotate_y': 'Rotar en Y', 
                'rotate_z': 'Rotar en Z',
                'auto_rotate': 'Rotación automática'
            },
            'view_presets': {
                'top_view': 'Vista superior',
                'side_view': 'Vista lateral',
                'front_view': 'Vista frontal',
                'isometric': 'Vista isométrica'
            },
            'animation_controls': {
                'play_pause': 'Reproducir/Pausar',
                'speed_control': 'Velocidad de animación',
                'frame_step': 'Avanzar frame'
            }
        }
    
    def export_holographic_data(self) -> Dict[str, Any]:
        """Exporta datos de la visualización holográfica"""
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'animation_frames': len(self.animation_data),
            'node_positions': self.node_positions_3d,
            'connection_weights': self.connection_weights,
            'holographic_colors': self.holographic_colors,
            'current_state': self.animation_data[-1] if self.animation_data else None,
            'performance_metrics': {
                'total_nodes': len(self.nodes),
                'total_connections': len(self.connection_weights),
                'animation_fps': 10,
                'rendering_mode': 'holographic_3d'
            }
        }
        
        return export_data
