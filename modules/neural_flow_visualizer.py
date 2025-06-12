
"""
Visualizador Interactivo de Flujos Neuronales en Tiempo Real
Sistema Ruth R1 - Visualización dinámica de activaciones neurales
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import time
import threading
from collections import deque
from typing import Dict, List, Any, Tuple
import colorsys
from datetime import datetime, timedelta

class NeuralFlowVisualizer:
    """Visualizador interactivo de flujos neuronales en tiempo real"""
    
    def __init__(self, consciousness_network):
        self.consciousness_network = consciousness_network
        self.is_running = False
        self.flow_data = deque(maxlen=100)
        self.activation_history = deque(maxlen=200)
        self.connection_strengths = {}
        self.update_thread = None
        
        # Configuración de visualización
        self.node_positions = self._calculate_optimal_positions()
        self.color_palette = self._generate_color_palette()
        
    def start_real_time_monitoring(self):
        """Inicia el monitoreo en tiempo real"""
        if self.is_running:
            return
            
        self.is_running = True
        self.update_thread = threading.Thread(target=self._real_time_update_loop, daemon=True)
        self.update_thread.start()
        
    def stop_real_time_monitoring(self):
        """Detiene el monitoreo en tiempo real"""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=2.0)
    
    def _real_time_update_loop(self):
        """Bucle de actualización en tiempo real"""
        while self.is_running:
            try:
                # Capturar estado actual
                current_state = self._capture_neural_state()
                
                # Almacenar datos
                self.flow_data.append(current_state)
                self.activation_history.append({
                    'timestamp': datetime.now(),
                    'activations': current_state['activations'],
                    'coherence': current_state['coherence']
                })
                
                # Actualizar fortalezas de conexión
                self._update_connection_strengths(current_state)
                
                time.sleep(0.5)  # Actualizar cada 500ms
                
            except Exception as e:
                print(f"Error en actualización de flujo neural: {e}")
                time.sleep(1.0)
    
    def _capture_neural_state(self) -> Dict[str, Any]:
        """Captura el estado neural actual"""
        activation_state = self.consciousness_network._get_activation_state()
        network_state = self.consciousness_network._get_network_state()
        coherence_metrics = self.consciousness_network.coherence_metrics
        
        return {
            'timestamp': datetime.now(),
            'activations': activation_state,
            'network_state': network_state,
            'coherence': coherence_metrics.get('network_entropy', 0.5),
            'global_consciousness': self.consciousness_network.global_consciousness_state
        }
    
    def _calculate_optimal_positions(self) -> Dict[str, Tuple[float, float]]:
        """Calcula posiciones óptimas para los nodos"""
        nodes = list(self.consciousness_network.nodes.keys())
        
        # Crear disposición circular con capas
        positions = {}
        
        # Núcleo central
        core_modules = ['GANSLSTMCore', 'IntrospectionEngine', 'PhilosophicalCore']
        
        # Módulos de procesamiento
        processing_modules = ['InnovationEngine', 'DreamMechanism', 'SelfMirror']
        
        # Módulos de análisis
        analysis_modules = ['EmotionDecomposer', 'ExistentialAnalyzer', 'MemoryDiscriminator']
        
        # Resto de módulos
        other_modules = [node for node in nodes if node not in core_modules + processing_modules + analysis_modules]
        
        layers = [core_modules, processing_modules, analysis_modules, other_modules]
        radii = [2, 5, 8, 11]
        
        for layer_idx, (layer, radius) in enumerate(zip(layers, radii)):
            angle_step = 2 * np.pi / max(len(layer), 1)
            for i, module in enumerate(layer):
                angle = i * angle_step + layer_idx * 0.3  # Offset entre capas
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                positions[module] = (x, y)
        
        return positions
    
    def _generate_color_palette(self) -> Dict[str, str]:
        """Genera paleta de colores para módulos"""
        nodes = list(self.consciousness_network.nodes.keys())
        colors = {}
        
        for i, node in enumerate(nodes):
            hue = i / len(nodes)
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            colors[node] = f"rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})"
        
        return colors
    
    def _update_connection_strengths(self, current_state: Dict[str, Any]):
        """Actualiza las fortalezas de conexión"""
        activations = current_state['activations']
        
        for edge in self.consciousness_network.network_graph.edges():
            source, target = edge
            if source in activations and target in activations:
                # Calcular fortaleza basada en diferencia de activación
                strength = 1.0 - abs(activations[source] - activations[target])
                connection_key = f"{source}->{target}"
                
                # Suavizar cambios
                if connection_key in self.connection_strengths:
                    self.connection_strengths[connection_key] = (
                        self.connection_strengths[connection_key] * 0.7 + strength * 0.3
                    )
                else:
                    self.connection_strengths[connection_key] = strength

    def create_real_time_flow_visualization(self) -> go.Figure:
        """Crea visualización de flujo en tiempo real"""
        fig = go.Figure()
        
        if not self.flow_data:
            # Mostrar estado inicial
            fig.add_annotation(
                text="Iniciando captura de flujos neuronales...",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="white")
            )
            fig.update_layout(
                title="Flujos Neuronales en Tiempo Real - Inicializando",
                template="plotly_dark",
                height=600
            )
            return fig
        
        current_data = self.flow_data[-1]
        activations = current_data['activations']
        
        # Nodos con activación actual
        node_x, node_y = [], []
        node_colors = []
        node_sizes = []
        node_texts = []
        
        for module, (x, y) in self.node_positions.items():
            if module in activations:
                node_x.append(x)
                node_y.append(y)
                
                activation = activations[module]
                node_colors.append(activation)
                node_sizes.append(20 + activation * 40)
                node_texts.append(f"{module}<br>Act: {activation:.3f}")
        
        # Añadir nodos
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Activación", x=1.02),
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            text=[module.split('Engine')[0].split('Core')[0] for module in self.node_positions.keys() if module in activations],
            textposition="middle center",
            textfont=dict(size=10, color='white'),
            hovertext=node_texts,
            hoverinfo='text',
            name='Módulos Neurales'
        ))
        
        # Conexiones con flujo
        edge_x, edge_y = [], []
        edge_colors = []
        edge_widths = []
        
        for edge in self.consciousness_network.network_graph.edges():
            source, target = edge
            if source in self.node_positions and target in self.node_positions and source in activations and target in activations:
                x0, y0 = self.node_positions[source]
                x1, y1 = self.node_positions[target]
                
                connection_key = f"{source}->{target}"
                strength = self.connection_strengths.get(connection_key, 0.3)
                
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_colors.append(strength)
                edge_widths.append(max(1, strength * 5))
        
        # Añadir conexiones
        if edge_x:
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(
                    color='rgba(78, 205, 196, 0.6)',
                    width=2
                ),
                hoverinfo='none',
                showlegend=False
            ))
        
        # Añadir pulsos de activación
        self._add_activation_pulses(fig, current_data)
        
        fig.update_layout(
            title=f"Flujos Neuronales Ruth R1 - {current_data['timestamp'].strftime('%H:%M:%S')}",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_dark",
            height=700,
            showlegend=True
        )
        
        return fig
    
    def _add_activation_pulses(self, fig: go.Figure, current_data: Dict[str, Any]):
        """Añade pulsos de activación visual"""
        activations = current_data['activations']
        
        # Crear pulsos para módulos altamente activos
        pulse_x, pulse_y = [], []
        pulse_sizes = []
        
        for module, activation in activations.items():
            if activation > 0.6 and module in self.node_positions:  # Solo módulos muy activos
                x, y = self.node_positions[module]
                pulse_x.append(x)
                pulse_y.append(y)
                pulse_sizes.append(60 + activation * 30)
        
        if pulse_x:
            fig.add_trace(go.Scatter(
                x=pulse_x, y=pulse_y,
                mode='markers',
                marker=dict(
                    size=pulse_sizes,
                    color='rgba(255, 255, 0, 0.3)',
                    symbol='circle-open',
                    line=dict(width=3, color='yellow')
                ),
                name='Pulsos de Alta Activación',
                hoverinfo='none'
            ))
    
    def create_activation_timeline(self) -> go.Figure:
        """Crea timeline de activaciones"""
        if len(self.activation_history) < 2:
            fig = go.Figure()
            fig.add_annotation(
                text="Recopilando datos de activación...",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        # Convertir a DataFrame
        timeline_data = []
        for entry in self.activation_history:
            for module, activation in entry['activations'].items():
                timeline_data.append({
                    'timestamp': entry['timestamp'],
                    'module': module,
                    'activation': activation,
                    'coherence': entry['coherence']
                })
        
        df = pd.DataFrame(timeline_data)
        
        # Crear gráfico de líneas para módulos principales
        main_modules = ['GANSLSTMCore', 'IntrospectionEngine', 'PhilosophicalCore', 'InnovationEngine']
        
        fig = go.Figure()
        
        for module in main_modules:
            module_data = df[df['module'] == module]
            if not module_data.empty:
                fig.add_trace(go.Scatter(
                    x=module_data['timestamp'],
                    y=module_data['activation'],
                    mode='lines+markers',
                    name=module,
                    line=dict(width=2),
                    marker=dict(size=4)
                ))
        
        fig.update_layout(
            title="Timeline de Activaciones Neurales",
            xaxis_title="Tiempo",
            yaxis_title="Nivel de Activación",
            template="plotly_dark",
            height=400
        )
        
        return fig
    
    def create_coherence_heatmap(self) -> go.Figure:
        """Crea mapa de calor de coherencia"""
        if not self.flow_data:
            return go.Figure()
        
        # Calcular matriz de coherencia
        modules = list(self.consciousness_network.nodes.keys())
        coherence_matrix = np.zeros((len(modules), len(modules)))
        
        current_activations = self.flow_data[-1]['activations']
        
        for i, mod1 in enumerate(modules):
            for j, mod2 in enumerate(modules):
                if mod1 in current_activations and mod2 in current_activations:
                    # Coherencia basada en similaridad de activación
                    coherence = 1.0 - abs(current_activations[mod1] - current_activations[mod2])
                    coherence_matrix[i, j] = coherence
        
        fig = go.Figure(data=go.Heatmap(
            z=coherence_matrix,
            x=[m.split('Engine')[0].split('Core')[0] for m in modules],
            y=[m.split('Engine')[0].split('Core')[0] for m in modules],
            colorscale='Viridis',
            text=np.round(coherence_matrix, 2),
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="Mapa de Coherencia Neural",
            template="plotly_dark",
            height=500
        )
        
        return fig
    
    def get_flow_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas de flujo"""
        if not self.flow_data:
            return {}
        
        recent_data = list(self.flow_data)[-10:]
        
        # Calcular estadísticas
        avg_activations = {}
        for data in recent_data:
            for module, activation in data['activations'].items():
                if module not in avg_activations:
                    avg_activations[module] = []
                avg_activations[module].append(activation)
        
        stats = {
            'total_modules': len(avg_activations),
            'avg_activation_per_module': {
                module: np.mean(activations) 
                for module, activations in avg_activations.items()
            },
            'most_active_module': max(avg_activations.items(), 
                                    key=lambda x: np.mean(x[1]))[0] if avg_activations else None,
            'least_active_module': min(avg_activations.items(), 
                                     key=lambda x: np.mean(x[1]))[0] if avg_activations else None,
            'overall_activity': np.mean([
                np.mean(list(data['activations'].values())) 
                for data in recent_data
            ]) if recent_data else 0.0,
            'data_points_collected': len(self.flow_data)
        }
        
        return stats
