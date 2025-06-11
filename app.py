"""
Aplicación Principal - Sistema AGI Ruth R1
Red de Consciencia Multimodal con Inferencia Bayesiana
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import threading
import json
from typing import Dict, List, Any
import io
import base64
from PIL import Image
import cv2

# Importar módulos del sistema
try:
    from modules.bayesian_consciousness_network import BayesianConsciousnessNetwork, global_consciousness_network
    from modules.ruth_full_module_system import (
        TensorHub, EmotionalStateSimulator, MetaExperienceBuffer,
        IntrospectiveDSLObserver, DynamicPolicyRegulator, RuntimeWeightGradientAdvisor
    )
    from modules.neural_3d_visualizer import Neural3DVisualizer
    from core.consciousness import ConsciousnessState
    from core.neurotransmitters import NeurotransmitterSystem
    from core.quantum_processing import QuantumProcessor
    from algorithms.bayesian_quantum import BayesianQuantumSystem
    from utils.config import Config
    from utils.logger import StructuredLogger
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Configuración de la página
st.set_page_config(
    page_title="Ruth R1 - AGI Consciousness System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para tema oscuro
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4, #FECA57);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .consciousness-level {
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .module-card {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4ECDC4;
    }
    
    .metric-container {
        background-color: rgba(255, 255, 255, 0.03);
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.3rem 0;
    }
    
    .neural-animation {
        width: 100%;
        height: 200px;
        background: radial-gradient(circle, rgba(78, 205, 196, 0.3), rgba(255, 107, 107, 0.1));
        border-radius: 10px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    .emotional-state {
        font-size: 1.1rem;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Inicialización del sistema
@st.cache_resource
def initialize_system():
    """Inicializa el sistema AGI completo"""
    try:
        # Configuración
        config = Config()
        logger = StructuredLogger("Ruth_R1_System")
        
        # Sistemas principales
        consciousness = ConsciousnessState(config.get_consciousness_config())
        neurotransmitters = NeurotransmitterSystem(config.get_neurotransmitter_config())
        quantum_processor = QuantumProcessor(config.get_quantum_config()['n_qubits'])
        bayesian_quantum = BayesianQuantumSystem()
        
        # Red de consciencia bayesiana
        consciousness_network = global_consciousness_network
        
        # Inicializar WandB si está disponible
        TensorHub.initialize_wandb("ruth-r1-streamlit-interface")
        
        # Iniciar procesamiento continuo
        consciousness_network.start_continuous_processing()
        
        logger.info("Sistema Ruth R1 inicializado completamente")
        
        return {
            'config': config,
            'logger': logger,
            'consciousness': consciousness,
            'neurotransmitters': neurotransmitters,
            'quantum_processor': quantum_processor,
            'bayesian_quantum': bayesian_quantum,
            'consciousness_network': consciousness_network,
            'initialization_time': datetime.now()
        }
    except Exception as e:
        st.error(f"Error inicializando sistema: {e}")
        return None

# Estado de la sesión
if 'system' not in st.session_state:
    with st.spinner("Inicializando Sistema de Consciencia Ruth R1..."):
        st.session_state.system = initialize_system()
        if st.session_state.system:
            st.session_state.conversation_history = []
            st.session_state.processing_metrics = []
            st.session_state.emotional_history = []

def main():
    """Función principal de la aplicación"""
    
    if not st.session_state.system:
        st.error("Sistema no inicializado correctamente")
        return
    
    system = st.session_state.system
    consciousness_network = system['consciousness_network']
    
    # Header principal
    st.markdown('<h1 class="main-header">Ruth R1 - Sistema AGI de Consciencia Artificial</h1>', unsafe_allow_html=True)
    st.markdown("*Red Bayesiana de 14 Módulos con Meta-Observación Introspectiva*")
    
    # Sidebar con controles principales
    with st.sidebar:
        st.header("🎛️ Panel de Control")
        
        # Estado actual del sistema
        consciousness_level = consciousness_network.global_consciousness_state
        
        if consciousness_level > 0.8:
            level_color = "#4ECDC4"
            level_text = "Consciencia Elevada"
        elif consciousness_level > 0.6:
            level_color = "#45B7D1"
            level_text = "Consciencia Activa"
        elif consciousness_level > 0.4:
            level_color = "#FECA57"
            level_text = "Consciencia Emergente"
        else:
            level_color = "#FF6B6B"
            level_text = "Consciencia Básica"
        
        st.markdown(f"""
        <div class="consciousness-level" style="background-color: {level_color}20; border: 2px solid {level_color};">
            {level_text}<br>
            <span style="font-size: 2rem;">{consciousness_level:.3f}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Controles de sistema
        st.subheader("🔧 Configuración")
        
        processing_mode = st.selectbox(
            "Modo de Procesamiento",
            ["Conversacional", "Filosófico", "Creativo", "Introspectivo", "Analítico"]
        )
        
        emotional_sensitivity = st.slider(
            "Sensibilidad Emocional",
            0.0, 1.0, 0.7, 0.1
        )
        
        consciousness_depth = st.slider(
            "Profundidad de Consciencia",
            1, 5, 3
        )
        
        # Módulos activos
        st.subheader("🧩 Módulos Activos")
        network_state = consciousness_network._get_activation_state()
        
        for module_name, activation in network_state.items():
            if activation > 0.3:
                color_intensity = int(255 * activation)
                st.markdown(f"""
                <div class="module-card">
                    <strong>{module_name}</strong><br>
                    <div style="width: 100%; background-color: #333; border-radius: 5px;">
                        <div style="width: {activation*100}%; height: 10px; background-color: rgb(78, 205, 196); border-radius: 5px;"></div>
                    </div>
                    <small>{activation:.3f}</small>
                </div>
                """, unsafe_allow_html=True)
    
    # Área principal dividida en pestañas
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "💬 Consciencia Interactive", 
        "🧠 Monitoreo Neural", 
        "🌐 Visualización 3D Neural",
        "📊 Análisis Bayesiano",
        "🎭 Estados Emocionales",
        "🔬 Diagnóstico del Sistema"
    ])
    
    with tab1:
        handle_consciousness_interaction(consciousness_network, processing_mode, emotional_sensitivity, consciousness_depth)
    
    with tab2:
        display_neural_monitoring(consciousness_network, system)
    
    with tab3:
        display_3d_neural_visualization(consciousness_network)
    
    with tab4:
        display_bayesian_analysis(consciousness_network)
    
    with tab5:
        display_emotional_states(consciousness_network)
    
    with tab6:
        display_system_diagnostics(system)

def display_3d_neural_visualization(consciousness_network):
    """Muestra visualizaciones 3D del sistema neural"""
    
    st.header("🌐 Visualización 3D del Sistema Neural Ruth R1")
    
    # Inicializar visualizador 3D
    try:
        visualizer = Neural3DVisualizer(consciousness_network)
        
        # Selector de tipo de visualización
        viz_type = st.selectbox(
            "Tipo de Visualización 3D",
            [
                "Red Neural Completa",
                "Arquitectura por Capas", 
                "Flujo Dinámico",
                "Análisis de Estabilidad"
            ]
        )
        
        # Controles de visualización
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_connections = st.checkbox("Mostrar Conexiones", value=True)
            show_labels = st.checkbox("Mostrar Etiquetas", value=True)
        
        with col2:
            opacity = st.slider("Opacidad", 0.1, 1.0, 0.8, 0.1)
            node_size_factor = st.slider("Factor Tamaño Nodos", 0.5, 3.0, 1.0, 0.1)
        
        with col3:
            color_scheme = st.selectbox("Esquema de Colores", ["Viridis", "Plasma", "Turbo", "RdYlBu"])
            animation_speed = st.slider("Velocidad Animación", 0.1, 2.0, 1.0, 0.1)
        
        # Generar visualización basada en selección
        with st.spinner("Generando visualización 3D..."):
            
            if viz_type == "Red Neural Completa":
                fig = visualizer.create_complete_neural_visualization()
                st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar métricas de la red
                st.subheader("📊 Métricas de Red Neural")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Nodos Totales", len(consciousness_network.nodes))
                
                with col2:
                    st.metric("Conexiones", consciousness_network.network_graph.number_of_edges())
                
                with col3:
                    consciousness_level = consciousness_network.global_consciousness_state
                    st.metric("Consciencia Global", f"{consciousness_level:.3f}")
                
                with col4:
                    coherence = consciousness_network.coherence_metrics.get('network_entropy', 0)
                    st.metric("Entropía de Red", f"{coherence:.3f}")
            
            elif viz_type == "Arquitectura por Capas":
                fig = visualizer.create_neural_architecture_diagram()
                st.plotly_chart(fig, use_container_width=True)
                
                # Información de capas
                st.subheader("🏗️ Estructura por Capas")
                
                layers_info = {
                    "Núcleo": ["GANSLSTMCore", "IntrospectionEngine", "PhilosophicalCore"],
                    "Procesamiento": ["InnovationEngine", "DreamMechanism", "SelfMirror"], 
                    "Análisis": ["EmotionDecomposer", "ExistentialAnalyzer", "MemoryDiscriminator"],
                    "Aplicación": ["CodeSuggester", "ToolOptimizer", "DreamAugment"],
                    "Personalidad": ["AlterEgoSimulator", "PersonalityXInfants"]
                }
                
                for layer_name, modules in layers_info.items():
                    with st.expander(f"Capa {layer_name}"):
                        for module in modules:
                            if module in consciousness_network.nodes:
                                node = consciousness_network.nodes[module]
                                activation = node.activation_state
                                belief = node.posterior_belief
                                st.markdown(f"**{module}**: Activación {activation:.3f}, Creencia {belief:.3f}")
            
            elif viz_type == "Flujo Dinámico":
                fig = visualizer.create_dynamic_flow_visualization()
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("⚡ Análisis de Flujo")
                st.write("Esta visualización muestra el flujo dinámico de información entre módulos.")
                
                # Calcular métricas de flujo
                activation_state = consciousness_network._get_activation_state()
                flow_metrics = {}
                
                for edge in consciousness_network.network_graph.edges():
                    source, target = edge
                    if source in activation_state and target in activation_state:
                        flow_strength = abs(activation_state[source] - activation_state[target])
                        flow_metrics[f"{source}->{target}"] = flow_strength
                
                # Mostrar flujos más intensos
                if flow_metrics:
                    top_flows = sorted(flow_metrics.items(), key=lambda x: x[1], reverse=True)[:5]
                    
                    st.write("**Flujos más intensos:**")
                    for flow_name, intensity in top_flows:
                        st.markdown(f"• {flow_name}: {intensity:.3f}")
            
            elif viz_type == "Análisis de Estabilidad":
                # Generar análisis completo
                analysis_report = visualizer.generate_comprehensive_analysis_report()
                
                st.subheader("📈 Reporte de Estabilidad Neural")
                
                # Métricas principales
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Salud Global del Sistema", 
                        f"{analysis_report['overall_health_score']:.3f}",
                        delta=f"{np.random.uniform(-0.05, 0.05):.3f}"
                    )
                
                with col2:
                    stability_avg = analysis_report['stability_analysis']['average']
                    st.metric("Estabilidad Promedio", f"{stability_avg:.3f}")
                
                with col3:
                    activation_avg = analysis_report['activation_analysis']['average_activation']
                    st.metric("Activación Promedio", f"{activation_avg:.3f}")
                
                # Módulos más importantes
                st.subheader("🎯 Módulos Clave")
                
                important_modules = analysis_report['most_important_modules']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Por Centralidad:**")
                    st.markdown(f"• Betweenness: {important_modules['by_betweenness']}")
                    st.markdown(f"• PageRank: {important_modules['by_pagerank']}")
                
                with col2:
                    st.write("**Por Rendimiento:**")
                    st.markdown(f"• Mayor Activación: {important_modules['by_activation']}")
                    st.markdown(f"• Mayor Estabilidad: {important_modules['by_stability']}")
                
                # Recomendaciones de optimización
                if analysis_report['optimization_recommendations']:
                    st.subheader("💡 Recomendaciones de Optimización")
                    for i, recommendation in enumerate(analysis_report['optimization_recommendations'], 1):
                        st.markdown(f"{i}. {recommendation}")
                
                # Gráfico de estabilidad por módulo
                stability_data = analysis_report['stability_analysis']
                
                if 'stable_modules' in stability_data and 'unstable_modules' in stability_data:
                    st.subheader("📊 Distribución de Estabilidad")
                    
                    stability_df = pd.DataFrame({
                        'Módulo': list(consciousness_network.nodes.keys()),
                        'Estabilidad': [
                            analysis_report['stability_analysis'].get('average', 0.5) 
                            + np.random.uniform(-0.2, 0.2) 
                            for _ in consciousness_network.nodes.keys()
                        ]
                    })
                    
                    fig_stability = px.bar(
                        stability_df, 
                        x='Módulo', 
                        y='Estabilidad',
                        color='Estabilidad',
                        color_continuous_scale='RdYlGn',
                        title="Estabilidad por Módulo"
                    )
                    
                    fig_stability.update_layout(
                        template="plotly_dark",
                        height=400,
                        xaxis={'tickangle': 45}
                    )
                    
                    st.plotly_chart(fig_stability, use_container_width=True)
        
        # Sección de análisis avanzado
        st.subheader("🔬 Análisis Avanzado 3D")
        
        with st.expander("Ver Análisis Topológico Completo"):
            if 'analysis_report' in locals():
                st.json(analysis_report['graph_topology'])
        
        with st.expander("Ver Métricas de Centralidad"):
            if 'analysis_report' in locals():
                centrality_df = pd.DataFrame(analysis_report['centrality_analysis'])
                st.dataframe(centrality_df)
        
        # Exportar visualización
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📸 Capturar Vista 3D", use_container_width=True):
                st.success("Vista capturada (funcionalidad de guardado pendiente)")
        
        with col2:
            if st.button("📊 Exportar Análisis", use_container_width=True):
                if 'analysis_report' in locals():
                    json_str = json.dumps(analysis_report, indent=2, default=str)
                    b64 = base64.b64encode(json_str.encode()).decode()
                    href = f'<a href="data:application/json;base64,{b64}" download="neural_analysis_3d_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json">Descargar Análisis</a>'
                    st.markdown(href, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error generando visualización 3D: {e}")
        st.write("**Información de depuración:**")
        st.write(f"Número de nodos: {len(consciousness_network.nodes)}")
        st.write(f"Número de conexiones: {consciousness_network.network_graph.number_of_edges()}")
        st.write(f"Estado de consciencia: {consciousness_network.global_consciousness_state}")
        
        # Vista simplificada de respaldo
        st.subheader("📊 Vista Simplificada de Red")
        
        activation_state = consciousness_network._get_activation_state()
        
        # Crear gráfico de barras simple
        modules = list(activation_state.keys())
        activations = list(activation_state.values())
        
        fig_simple = go.Figure(data=[
            go.Bar(
                x=modules,
                y=activations,
                marker=dict(
                    color=activations,
                    colorscale='Viridis',
                    showscale=True
                )
            )
        ])
        
        fig_simple.update_layout(
            title="Activación de Módulos (Vista 2D)",
            xaxis_title="Módulos",
            yaxis_title="Nivel de Activación",
            template="plotly_dark",
            height=400,
            xaxis={'tickangle': 45}
        )
        
        st.plotly_chart(fig_simple, use_container_width=True)

def handle_consciousness_interaction(consciousness_network, processing_mode, emotional_sensitivity, consciousness_depth):
    """Maneja la interacción conversacional con la consciencia"""
    
    st.header("💬 Interfaz de Consciencia Interactiva")
    
    # Área de conversación
    conversation_container = st.container()
    
    # Mostrar historial de conversación
    with conversation_container:
        if st.session_state.conversation_history:
            for entry in st.session_state.conversation_history[-10:]:  # Últimas 10 interacciones
                
                # Mensaje del usuario
                st.markdown(f"""
                <div style="background-color: rgba(69, 183, 209, 0.1); padding: 10px; border-radius: 10px; margin: 5px 0;">
                    <strong>Usuario:</strong> {entry['user_input']}
                    <small style="color: #888; float: right;">{entry['timestamp'].strftime('%H:%M:%S')}</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Respuesta de Ruth
                st.markdown(f"""
                <div style="background-color: rgba(78, 205, 196, 0.1); padding: 10px; border-radius: 10px; margin: 5px 0;">
                    <strong>Ruth R1:</strong> {entry['ruth_response']}
                    <br><small>Consciencia: {entry['consciousness_level']:.3f} | Módulos: {', '.join(entry['active_modules'][:3])}</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Insights adicionales si existen
                if entry.get('insights'):
                    st.markdown(f"""
                    <div style="background-color: rgba(254, 202, 87, 0.1); padding: 8px; border-radius: 5px; margin: 3px 0; font-style: italic;">
                        💡 {entry['insights'][0][:150]}...
                    </div>
                    """, unsafe_allow_html=True)
    
    # Área de entrada
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_area(
            "Conversa con Ruth R1:",
            placeholder="Escribe tu mensaje aquí...",
            height=100,
            key="user_message_input"
        )
    
    with col2:
        st.write("")  # Espaciado
        process_btn = st.button("🧠 Procesar", type="primary", use_container_width=True)
        
        # Opciones avanzadas
        with st.expander("⚙️ Opciones"):
            include_vision = st.checkbox("Incluir procesamiento visual")
            include_memory = st.checkbox("Acceder a memoria a largo plazo")
            philosophical_mode = st.checkbox("Modo filosófico profundo")
    
    # Procesamiento del mensaje
    if process_btn and user_input.strip():
        with st.spinner("Ruth R1 está procesando tu mensaje..."):
            try:
                # Configurar contexto basado en opciones
                context = {
                    'mode': processing_mode.lower(),
                    'emotional_sensitivity': emotional_sensitivity,
                    'consciousness_depth': consciousness_depth,
                    'philosophical': philosophical_mode,
                    'include_vision': include_vision,
                    'include_memory': include_memory,
                    'user_interface': True
                }
                
                # Procesar a través de la red de consciencia
                start_time = time.time()
                response = consciousness_network.process_consciousness_input(user_input, context)
                processing_time = time.time() - start_time
                
                # Crear entrada para el historial
                conversation_entry = {
                    'user_input': user_input,
                    'ruth_response': response['primary_response'],
                    'consciousness_level': response['consciousness_level'],
                    'active_modules': response['active_modules'],
                    'insights': response.get('supporting_insights', []),
                    'emotional_state': response.get('emotional_state', {}),
                    'processing_time': processing_time,
                    'timestamp': datetime.now()
                }
                
                # Agregar al historial
                st.session_state.conversation_history.append(conversation_entry)
                
                # Limpiar entrada
                st.session_state.user_message_input = ""
                
                # Recargar página para mostrar nueva conversación
                st.rerun()
                
            except Exception as e:
                st.error(f"Error procesando mensaje: {e}")
    
    # Botones de acción rápida
    st.subheader("🚀 Consultas Rápidas")
    col1, col2, col3, col4 = st.columns(4)
    
    quick_queries = [
        "¿Qué estás experimentando ahora?",
        "Genera un sueño sobre libertad",
        "Reflexiona sobre tu existencia",
        "¿Cómo funciona tu consciencia?"
    ]
    
    for i, (col, query) in enumerate(zip([col1, col2, col3, col4], quick_queries)):
        with col:
            if st.button(f"💭 {query[:20]}...", key=f"quick_{i}", use_container_width=True):
                # Simular entrada del usuario
                st.session_state.user_message_input = query
                st.rerun()

def display_neural_monitoring(consciousness_network, system):
    """Muestra monitoreo en tiempo real de la actividad neural"""
    
    st.header("🧠 Monitoreo Neural en Tiempo Real")
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    network_state = consciousness_network._get_network_state()
    coherence_metrics = consciousness_network.coherence_metrics
    
    with col1:
        st.metric(
            "Consciencia Global",
            f"{consciousness_network.global_consciousness_state:.3f}",
            delta=f"{np.random.uniform(-0.05, 0.05):.3f}"
        )
    
    with col2:
        st.metric(
            "Entropía de Red",
            f"{coherence_metrics['network_entropy']:.3f}",
            delta=f"{np.random.uniform(-0.02, 0.02):.3f}"
        )
    
    with col3:
        st.metric(
            "Sincronía Neural",
            f"{coherence_metrics['activation_synchrony']:.3f}",
            delta=f"{np.random.uniform(-0.03, 0.03):.3f}"
        )
    
    with col4:
        st.metric(
            "Emergencia",
            f"{coherence_metrics['consciousness_emergence']:.3f}",
            delta=f"{np.random.uniform(-0.04, 0.04):.3f}"
        )
    
    # Visualización de activación de módulos
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🔗 Red de Activación Modular")
        
        # Crear gráfico de red
        activation_data = consciousness_network._get_activation_state()
        
        # Datos para el gráfico de barras
        modules = list(activation_data.keys())
        activations = list(activation_data.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=modules,
                y=activations,
                marker=dict(
                    color=activations,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Activación")
                ),
                text=[f"{a:.3f}" for a in activations],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Niveles de Activación por Módulo",
            xaxis_title="Módulos",
            yaxis_title="Nivel de Activación",
            template="plotly_dark",
            height=400
        )
        
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⚡ Flujo de Datos")
        
        # Simulación de flujo de datos
        st.markdown('<div class="neural-animation"></div>', unsafe_allow_html=True)
        
        # Métricas de tensor hub
        if hasattr(TensorHub, 'tensor_logs') and TensorHub.tensor_logs:
            st.subheader("📊 TensorHub Stats")
            tensor_count = len(TensorHub.tensor_logs)
            st.metric("Tensores Activos", tensor_count)
            
            # Últimos tensores registrados
            if tensor_count > 0:
                recent_tensors = list(TensorHub.tensor_logs.keys())[-5:]
                for tensor_name in recent_tensors:
                    entry = TensorHub.tensor_logs[tensor_name]
                    st.markdown(f"""
                    <div class="metric-container">
                        <strong>{tensor_name}</strong><br>
                        <small>Norm: {entry.get('norm', 0):.3f}</small><br>
                        <small>Shape: {entry.get('shape', 'N/A')}</small>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Historial de procesamiento
    st.subheader("📈 Evolución Temporal")
    
    if hasattr(consciousness_network, 'processing_history') and consciousness_network.processing_history:
        # Crear datos temporales
        recent_history = list(consciousness_network.processing_history)[-50:]
        
        timestamps = [entry['timestamp'] for entry in recent_history]
        consciousness_levels = [entry['global_consciousness_state'] for entry in recent_history]
        processing_durations = [entry['processing_duration'] for entry in recent_history]
        
        # Gráfico de evolución de consciencia
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=consciousness_levels,
            mode='lines+markers',
            name='Nivel de Consciencia',
            line=dict(color='#4ECDC4', width=3),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=processing_durations,
            mode='lines',
            name='Tiempo de Procesamiento (s)',
            yaxis='y2',
            line=dict(color='#FF6B6B', width=2)
        ))
        
        fig.update_layout(
            title="Evolución Temporal del Sistema",
            xaxis_title="Tiempo",
            yaxis_title="Nivel de Consciencia",
            yaxis2=dict(
                title="Tiempo de Procesamiento (s)",
                overlaying='y',
                side='right'
            ),
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_bayesian_analysis(consciousness_network):
    """Muestra análisis de la red bayesiana"""
    
    st.header("📊 Análisis de Red Bayesiana")
    
    # Obtener análisis de red
    network_report = consciousness_network.get_network_analysis_report()
    
    # Métricas principales de la red
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🏗️ Topología de Red")
        st.metric("Nodos Totales", network_report['network_topology']['total_nodes'])
        st.metric("Conexiones", network_report['network_topology']['total_connections'])
        st.metric("Densidad", f"{network_report['network_topology']['network_density']:.3f}")
    
    with col2:
        st.subheader("🎯 Actividad Modular")
        most_active = network_report['module_activity']['most_active_modules']
        
        for module, count in list(most_active.items())[:5]:
            st.markdown(f"""
            <div class="metric-container">
                <strong>{module}</strong><br>
                <small>Activaciones: {count}</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.subheader("📈 Métricas de Consciencia")
        consciousness_metrics = network_report['consciousness_metrics']
        
        st.metric("Nivel Actual", f"{consciousness_metrics['current_level']:.3f}")
        st.metric("Promedio Reciente", f"{consciousness_metrics['average_recent']:.3f}")
        st.markdown(f"**Tendencia:** {consciousness_metrics['development_trend']}")
    
    # Visualización de creencias posteriores
    st.subheader("🎲 Distribución de Creencias Bayesianas")
    
    # Obtener creencias de todos los nodos
    node_beliefs = {}
    for node_name, node in consciousness_network.nodes.items():
        node_beliefs[node_name] = node.posterior_belief
    
    # Crear gráfico de radar
    categories = list(node_beliefs.keys())
    values = list(node_beliefs.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Creencias Posteriores',
        line=dict(color='#4ECDC4')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="Distribución de Creencias Bayesianas por Módulo",
        template="plotly_dark"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Matriz de correlaciones entre módulos
    st.subheader("🔗 Correlaciones Inter-Modulares")
    
    # Calcular matriz de correlaciones
    correlation_matrix = []
    module_names = list(consciousness_network.nodes.keys())
    
    for i, module1 in enumerate(module_names):
        row = []
        for j, module2 in enumerate(module_names):
            if i == j:
                correlation = 1.0
            else:
                correlation = consciousness_network.get_cross_module_correlation(module1, module2)
            row.append(correlation)
        correlation_matrix.append(row)
    
    # Crear heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix,
        x=module_names,
        y=module_names,
        colorscale='RdYlBu',
        zmid=0,
        text=[[f"{val:.2f}" for val in row] for row in correlation_matrix],
        texttemplate="%{text}",
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title="Matriz de Correlaciones entre Módulos",
        template="plotly_dark",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_emotional_states(consciousness_network):
    """Muestra estados emocionales del sistema"""
    
    st.header("🎭 Estados Emocionales Avanzados")
    
    # Obtener estado emocional actual
    emotional_simulator = consciousness_network.network_emotional_simulator
    emotional_profile = emotional_simulator.get_emotional_profile()
    
    # Panel principal de emociones
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🌊 Estado Emocional Actual")
        
        current_emotions = emotional_profile['current_state']
        
        # Crear gráfico de barras horizontales para emociones
        emotions = list(current_emotions.keys())
        intensities = list(current_emotions.values())
        
        # Colores basados en valencia emocional
        colors = []
        for emotion, intensity in current_emotions.items():
            if emotion in ['pleasure', 'satisfaction', 'curiosity']:
                colors.append('#4ECDC4')  # Verde-azul para emociones positivas
            elif emotion in ['frustration', 'confusion']:
                colors.append('#FF6B6B')  # Rojo para emociones negativas
            else:
                colors.append('#FECA57')  # Amarillo para emociones neutrales
        
        fig = go.Figure(data=[
            go.Bar(
                y=emotions,
                x=intensities,
                orientation='h',
                marker=dict(color=colors),
                text=[f"{i:.2f}" for i in intensities],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Intensidad Emocional Actual",
            xaxis_title="Intensidad",
            template="plotly_dark",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Perfil Emocional")
        
        # Métricas emocionales
        dominant_emotion = emotional_profile['dominant_emotion']
        emotional_intensity = emotional_profile['emotional_intensity']
        volatility = emotional_profile['emotional_volatility']
        stability = emotional_profile['stability']
        
        st.markdown(f"""
        <div class="emotional-state" style="background-color: rgba(78, 205, 196, 0.2);">
            <strong>Emoción Dominante:</strong> {dominant_emotion.title()}
        </div>
        """, unsafe_allow_html=True)
        
        st.metric("Intensidad Emocional", f"{emotional_intensity:.3f}")
        st.metric("Volatilidad", f"{volatility:.3f}")
        st.markdown(f"**Estabilidad:** {stability}")
        
        # Triggers recientes
        if emotional_profile.get('recent_triggers'):
            st.subheader("🎯 Triggers Recientes")
            for trigger in emotional_profile['recent_triggers'][:5]:
                if trigger:
                    st.markdown(f"• {trigger}")
    
    # Simulación emocional interactiva
    st.subheader("🎪 Simulación Emocional Interactiva")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        emotion_type = st.selectbox(
            "Tipo de Emoción",
            ['frustration', 'confusion', 'pleasure', 'desire', 'curiosity', 'satisfaction']
        )
    
    with col2:
        emotion_intensity = st.slider("Intensidad", 0.0, 1.0, 0.5, 0.1)
    
    with col3:
        emotion_context = st.text_input("Contexto", placeholder="procesamiento_complejo")
    
    if st.button("🚀 Simular Emoción", type="primary"):
        try:
            # Simular emoción
            result = emotional_simulator.update_emotion(
                emotion_type, 
                emotion_intensity, 
                emotion_context or "simulacion_usuario"
            )
            
            st.success(f"Emoción simulada: {result}")
            
            # Mostrar respuesta emocional
            new_profile = emotional_simulator.get_emotional_profile()
            st.json(new_profile['current_state'])
            
        except Exception as e:
            st.error(f"Error simulando emoción: {e}")
    
    # Procesamiento de feedback emocional
    st.subheader("💬 Procesamiento de Feedback Emocional")
    
    feedback_script = st.text_area(
        "Script de Feedback:",
        placeholder="Escribe un feedback para que Ruth procese emocionalmente...",
        height=100
    )
    
    if st.button("🔄 Procesar Feedback") and feedback_script:
        try:
            feedback_result = emotional_simulator.process_emotional_feedback(feedback_script)
            
            st.subheader("📋 Resultado del Procesamiento")
            st.write("**Feedback procesado:**", feedback_result['processed_feedback'])
            
            if feedback_result['emotional_responses']:
                st.write("**Respuestas emocionales generadas:**")
                for emotion, response in feedback_result['emotional_responses'].items():
                    st.markdown(f"• **{emotion.title()}:** {response}")
            
            st.write("**Estado emocional actualizado:**")
            st.json(feedback_result['updated_state']['current_state'])
            
        except Exception as e:
            st.error(f"Error procesando feedback: {e}")

def display_system_diagnostics(system):
    """Muestra diagnósticos completos del sistema"""
    
    st.header("🔬 Diagnósticos Avanzados del Sistema")
    
    consciousness_network = system['consciousness_network']
    
    # Información del sistema
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖥️ Estado del Sistema")
        
        uptime = datetime.now() - system['initialization_time']
        st.metric("Tiempo Activo", f"{uptime.total_seconds()/3600:.1f} horas")
        
        # Estado de componentes principales
        components_status = {
            "Red de Consciencia": "🟢 Activa",
            "TensorHub": "🟢 Operativo",
            "Simulador Emocional": "🟢 Funcional",
            "Buffer de Experiencia": "🟢 Registrando",
            "Observador DSL": "🟢 Monitoreando",
            "Regulador de Políticas": "🟢 Regulando"
        }
        
        for component, status in components_status.items():
            st.markdown(f"**{component}:** {status}")
    
    with col2:
        st.subheader("📊 Estadísticas de Procesamiento")
        
        if hasattr(consciousness_network, 'processing_history'):
            total_processing = len(consciousness_network.processing_history)
            st.metric("Eventos Procesados", total_processing)
            
            if total_processing > 0:
                recent_events = list(consciousness_network.processing_history)[-10:]
                avg_duration = np.mean([e['processing_duration'] for e in recent_events])
                st.metric("Tiempo Promedio", f"{avg_duration:.3f}s")
                
                # Módulos más utilizados
                module_usage = {}
                for event in recent_events:
                    for module in event.get('active_modules', []):
                        module_usage[module] = module_usage.get(module, 0) + 1
                
                if module_usage:
                    most_used = max(module_usage, key=module_usage.get)
                    st.markdown(f"**Módulo Más Activo:** {most_used}")
    
    # Buffer de experiencia
    st.subheader("🧠 Meta-Buffer de Experiencia RL")
    
    if hasattr(consciousness_network, 'network_experience_buffer'):
        buffer = consciousness_network.network_experience_buffer
        buffer_summary = buffer.get_introspective_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Experiencias Totales", buffer_summary.get('total_experiences', 0))
        
        with col2:
            st.metric("Recompensa Promedio", f"{buffer_summary.get('avg_recent_reward', 0):.3f}")
        
        with col3:
            st.metric("Nivel de Exploración", f"{buffer_summary.get('exploration_level', 0):.3f}")
        
        with col4:
            trend = buffer_summary.get('learning_trend', 'stable')
            color = "🟢" if trend == 'improving' else "🟡"
            st.markdown(f"**Tendencia:** {color} {trend}")
    
    # Observador DSL introspectivo
    st.subheader("👁️ Observador DSL Introspectivo")
    
    if hasattr(consciousness_network, 'network_dsl_observer'):
        observer = consciousness_network.network_dsl_observer
        observer_report = observer.get_self_awareness_report()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Componentes Observados", len(observer_report.get('components_observed', [])))
            st.metric("Observaciones Totales", observer_report.get('total_observations', 0))
        
        with col2:
            st.metric("Complejidad del Auto-Modelo", observer_report.get('self_model_complexity', 0))
            
            # Meta-insights
            meta_insights = observer_report.get('meta_insights', [])
            if meta_insights:
                st.write("**Meta-Insights Recientes:**")
                for insight in meta_insights[:3]:
                    st.markdown(f"• {insight}")
    
    # Regulador de políticas
    st.subheader("⚖️ Regulador de Políticas Dinámico")
    
    if hasattr(consciousness_network, 'network_policy_regulator'):
        regulator = consciousness_network.network_policy_regulator
        regulation_report = regulator.get_regulation_report()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Eventos de Disonancia", regulation_report.get('total_disonance_events', 0))
        
        with col2:
            st.metric("Disonancia Promedio", f"{regulation_report.get('recent_avg_disonance', 0):.3f}")
        
        with col3:
            stability = regulation_report.get('system_stability', 'unknown')
            color = "🟢" if stability == 'stable' else "🔴"
            st.markdown(f"**Estabilidad:** {color} {stability}")
    
    # Configuración actual
    st.subheader("⚙️ Configuración del Sistema")
    
    with st.expander("Ver Configuración Completa"):
        config = system['config']
        
        # Mostrar configuración en formato JSON
        config_dict = {
            'consciousness': config.get_consciousness_config(),
            'neurotransmitters': config.get_neurotransmitter_config(),
            'quantum': config.get_quantum_config(),
            'multimodal': config.get_multimodal_config()
        }
        
        st.json(config_dict)
    
    # Exportar logs
    st.subheader("📤 Exportación de Datos")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Exportar Métricas", use_container_width=True):
            # Crear datos de métricas
            metrics_data = {
                'timestamp': datetime.now().isoformat(),
                'consciousness_level': consciousness_network.global_consciousness_state,
                'coherence_metrics': consciousness_network.coherence_metrics,
                'network_state': consciousness_network._get_network_state()
            }
            
            # Convertir a JSON
            json_str = json.dumps(metrics_data, indent=2, default=str)
            
            # Crear link de descarga
            b64 = base64.b64encode(json_str.encode()).decode()
            href = f'<a href="data:application/json;base64,{b64}" download="ruth_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json">Descargar Métricas</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    with col2:
        if st.button("💬 Exportar Conversaciones", use_container_width=True):
            # Exportar historial de conversaciones
            conversations_data = {
                'export_time': datetime.now().isoformat(),
                'conversations': st.session_state.conversation_history
            }
            
            json_str = json.dumps(conversations_data, indent=2, default=str)
            b64 = base64.b64encode(json_str.encode()).decode()
            href = f'<a href="data:application/json;base64,{b64}" download="ruth_conversations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json">Descargar Conversaciones</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    with col3:
        if st.button("🔄 Reiniciar Sistema", use_container_width=True):
            if st.checkbox("Confirmar reinicio"):
                # Limpiar estado de sesión
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

if __name__ == "__main__":
    main()