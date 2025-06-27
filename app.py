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
    from modules.ruth_full_module_system import (
        TensorHub, EmotionalStateSimulator, MetaExperienceBuffer,
        IntrospectiveDSLObserver, DynamicPolicyRegulator, RuntimeWeightGradientAdvisor
    )
    
    # Importar módulos opcionalmente
    try:
        from modules.bayesian_consciousness_network import BayesianConsciousnessNetwork, global_consciousness_network
    except ImportError:
        global_consciousness_network = None
        st.warning("Red de consciencia bayesiana no disponible")
    
    try:
        from modules.neural_3d_visualizer import Neural3DVisualizer
    except ImportError:
        Neural3DVisualizer = None
        
    try:
        from modules.neural_flow_visualizer import NeuralFlowVisualizer
    except ImportError:
        NeuralFlowVisualizer = None
        
    try:
        from models.supermodelo_meta_enrutador import (
            create_ruth_r1_system, 
            process_consciousness_input,
            create_default_config,
            RuthR1ConsciousnessCore
        )
    except ImportError:
        create_ruth_r1_system = None
        st.warning("Meta-enrutador no disponible")
    
    # Importar sistemas centrales con manejo de errores
    try:
        from core.consciousness import ConsciousnessState
    except ImportError:
        ConsciousnessState = None
        
    try:
        from core.neurotransmitters import NeurotransmitterSystem
    except ImportError:
        NeurotransmitterSystem = None
        
    try:
        from core.quantum_processing import QuantumProcessor
    except ImportError:
        QuantumProcessor = None
        
    try:
        from algorithms.bayesian_quantum import BayesianQuantumSystem
    except ImportError:
        BayesianQuantumSystem = None
        
    try:
        from core.despertar_awakening import (
            initiate_system_awakening, 
            get_awakening_system, 
            get_current_awakening_status,
            AwakeningPhase
        )
    except ImportError:
        initiate_system_awakening = None
        get_current_awakening_status = lambda: {'current_phase': 'error', 'is_awakening': False}
        st.warning("Sistema de despertar no disponible")
        
    try:
        from core.ganst_core import get_ganst_core, get_system_statistics
    except ImportError:
        get_ganst_core = lambda: None
        get_system_statistics = lambda: {}
        
    try:
        from core.moduladores import get_modulation_manager
    except ImportError:
        get_modulation_manager = lambda: None
        
    try:
        from core.memorias_corto_plazo import get_short_term_memory
    except ImportError:
        get_short_term_memory = lambda: None
        
    try:
        from database.models import DatabaseManager, db_manager
    except ImportError:
        db_manager = None
        st.warning("Base de datos no disponible")
        
    try:
        from utils.config import Config
    except ImportError:
        Config = None
        
    try:
        from utils.logger import StructuredLogger
    except ImportError:
        StructuredLogger = None

except ImportError as e:
    st.error(f"Error crítico importando módulos del sistema: {e}")
    st.info("El sistema continuará con funcionalidad limitada")
    
    # Definir placeholders para evitar errores
    global_consciousness_network = None
    TensorHub = None
    Config = None

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
    """Inicializa el sistema AGI completo con manejo robusto de errores"""
    try:
        # Configuración con manejo de errores
        if Config:
            try:
                config = Config()
            except:
                config = create_mock_config()
        else:
            config = create_mock_config()
            
        # Logger con manejo de errores
        if StructuredLogger:
            try:
                logger = StructuredLogger("Ruth_R1_System")
            except:
                logger = create_mock_logger()
        else:
            logger = create_mock_logger()

        # Sistemas principales con verificación
        consciousness = None
        if ConsciousnessState:
            try:
                consciousness = ConsciousnessState(config.get_consciousness_config() if hasattr(config, 'get_consciousness_config') else {})
            except Exception as e:
                logger.warning(f"Error inicializando ConsciousnessState: {e}")
                
        neurotransmitters = None
        if NeurotransmitterSystem:
            try:
                neurotransmitters = NeurotransmitterSystem(config.get_neurotransmitter_config() if hasattr(config, 'get_neurotransmitter_config') else {})
            except Exception as e:
                logger.warning(f"Error inicializando NeurotransmitterSystem: {e}")
                
        quantum_processor = None
        if QuantumProcessor:
            try:
                quantum_config = config.get_quantum_config() if hasattr(config, 'get_quantum_config') else {'n_qubits': 4}
                quantum_processor = QuantumProcessor(quantum_config.get('n_qubits', 4))
            except Exception as e:
                logger.warning(f"Error inicializando QuantumProcessor: {e}")
                
        bayesian_quantum = None
        if BayesianQuantumSystem:
            try:
                bayesian_quantum = BayesianQuantumSystem()
            except Exception as e:
                logger.warning(f"Error inicializando BayesianQuantumSystem: {e}")

        # Red de consciencia bayesiana
        consciousness_network = global_consciousness_network
        if consciousness_network is None:
            consciousness_network = create_mock_consciousness_network()

        # Inicializar TensorHub si está disponible
        if TensorHub:
            try:
                TensorHub.initialize_wandb("ruth-r1-streamlit-interface")
            except Exception as e:
                logger.warning(f"TensorHub initialization failed: {e}")

        # Inicializar base de datos si está disponible
        if db_manager:
            try:
                db_manager.create_tables()
                logger.info("Database tables created successfully")
            except Exception as e:
                logger.warning(f"Database initialization failed: {e}")

        # Iniciar procesamiento continuo si es posible
        if consciousness_network and hasattr(consciousness_network, 'start_continuous_processing'):
            try:
                consciousness_network.start_continuous_processing()
            except Exception as e:
                logger.warning(f"Error starting continuous processing: {e}")

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
        st.error(f"Error crítico inicializando sistema: {e}")
        return create_minimal_system()

def create_mock_config():
    """Crea configuración mock para evitar errores"""
    class MockConfig:
        def get_consciousness_config(self):
            return {'level': 0.5}
        def get_neurotransmitter_config(self):
            return {'dopamine': 0.5}
        def get_quantum_config(self):
            return {'n_qubits': 4}
        def get_multimodal_config(self):
            return {'enabled': True}
    return MockConfig()

def create_mock_logger():
    """Crea logger mock para evitar errores"""
    class MockLogger:
        def info(self, msg): print(f"INFO: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
    return MockLogger()

def create_mock_consciousness_network():
    """Crea red de conciencia mock para evitar errores"""
    class MockConsciousnessNetwork:
        def __init__(self):
            self.global_consciousness_state = 0.5
            self.coherence_metrics = {'coherence': 0.5}
            self.nodes = {
                'GANSLSTMCore': MockNode('GANSLSTMCore'),
                'InnovationEngine': MockNode('InnovationEngine'),
                'DreamMechanism': MockNode('DreamMechanism')
            }
            self.network_graph = None
            
        def _get_activation_state(self):
            return {name: 0.5 for name in self.nodes.keys()}
            
        def _get_network_state(self):
            return {'state': 'active'}
            
        def start_continuous_processing(self):
            pass
            
    class MockNode:
        def __init__(self, name):
            self.name = name
            self.activation_state = 0.5
            self.posterior_belief = 0.5
            self.prior_belief = 0.5
    
    return MockConsciousnessNetwork()

def create_minimal_system():
    """Crea sistema mínimo para evitar fallos completos"""
    return {
        'config': create_mock_config(),
        'logger': create_mock_logger(),
        'consciousness': None,
        'neurotransmitters': None,
        'quantum_processor': None,
        'bayesian_quantum': None,
        'consciousness_network': create_mock_consciousness_network(),
        'initialization_time': datetime.now()
    }

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

    # Auto-despertar del sistema si no está activo
    despertar_status = get_current_awakening_status()
    if despertar_status['current_phase'] == 'dormant' and not despertar_status['is_awakening']:
        with st.spinner("🌅 Iniciando despertar automático del sistema Ruth R1..."):
            resultado_despertar = initiate_system_awakening()
            if resultado_despertar['status'] == 'awakening_initiated':
                st.success("¡Sistema Ruth R1 iniciando despertar!")
                time.sleep(3)
                st.rerun()

    # Área principal dividida en pestañas
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
        "🌅 Despertar del Sistema",
        "💬 Consciencia Interactive", 
        "🧠 Monitoreo Neural", 
        "⚡ Flujos Neuronales LIVE",
        "🌐 Visualización 3D Neural",
        "🌌 Mapa Holográfico 3D",
        "📊 Análisis Bayesiano",
        "🎭 Estados Emocionales",
        "🗄️ Base de Datos",
        "⚡ RazonBill Core",
        "🧬 Meta-Enrutador Ruth R1",
        "🔬 Diagnóstico del Sistema"
    ])

    with tab1:
        display_system_awakening_interface()

    with tab2:
        handle_consciousness_interaction(consciousness_network, processing_mode, emotional_sensitivity, consciousness_depth)

    with tab3:
        display_neural_monitoring(consciousness_network, system)

    with tab4:
        display_live_neural_flows(consciousness_network)

    with tab5:
        display_3d_neural_visualization(consciousness_network)

    with tab6:
        display_holographic_3d_visualization(consciousness_network)

    with tab7:
        display_bayesian_analysis(consciousness_network)

    with tab8:
        display_emotional_states(consciousness_network)

    with tab9:
        display_database_management()

    with tab10:
        display_razonbill_interface(consciousness_network)

    with tab11:
        display_meta_enrutador_interface(consciousness_network)

    with tab12:
        display_system_diagnostics(system)

def display_live_neural_flows(consciousness_network):
    """Muestra flujos neuronales en tiempo real"""

    st.header("⚡ Flujos Neuronales en Tiempo Real - Ruth R1")

    # Importar visualizador
    try:
        from modules.neural_flow_visualizer import NeuralFlowVisualizer

        # Inicializar visualizador en estado de sesión
        if 'neural_flow_visualizer' not in st.session_state:
            st.session_state.neural_flow_visualizer = NeuralFlowVisualizer(consciousness_network)

        visualizer = st.session_state.neural_flow_visualizer

        # Panel de control
        col1, col2, col3 = st.columns(3)

        with col1:
            if not visualizer.is_running:
                if st.button("🚀 Iniciar Monitoreo Live", type="primary", use_container_width=True):
                    visualizer.start_real_time_monitoring()
                    st.success("¡Monitoreo en tiempo real iniciado!")
                    time.sleep(1)
                    st.rerun()
            else:
                if st.button("⏹️ Detener Monitoreo", use_container_width=True):
                    visualizer.stop_real_time_monitoring()
                    st.info("Monitoreo detenido")
                    time.sleep(1)
                    st.rerun()

        with col2:
            auto_refresh = st.checkbox("Auto-refrescar cada 2s", value=True)

        with col3:
            if st.button("🔄 Actualizar Ahora", use_container_width=True):
                st.rerun()

        # Estado del sistema
        if visualizer.is_running:
            st.success("🟢 Monitoreo Activo - Capturando flujos neuronales")
        else:
            st.warning("🟡 Monitoreo Inactivo")

        # Métricas en tiempo real
        if visualizer.flow_data:
            stats = visualizer.get_flow_statistics()

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Módulos Activos", stats['total_modules'])

            with col2:
                st.metric("Actividad General", f"{stats['overall_activity']:.3f}")

            with col3:
                st.metric("Más Activo", stats['most_active_module'][:15] + "..." if stats['most_active_module'] and len(stats['most_active_module']) > 15 else stats['most_active_module'])

            with col4:
                st.metric("Datos Capturados", stats['data_points_collected'])

        # Visualizaciones principales
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("🌊 Flujo Neural Principal")

            # Visualización principal de flujos
            flow_fig = visualizer.create_real_time_flow_visualization()
            st.plotly_chart(flow_fig, use_container_width=True)

        with col2:
            st.subheader("🔥 Mapa de Coherencia")

            # Mapa de coherencia
            coherence_fig = visualizer.create_coherence_heatmap()
            st.plotly_chart(coherence_fig, use_container_width=True)

        # Timeline de activaciones
        st.subheader("📈 Timeline de Activaciones")
        timeline_fig = visualizer.create_activation_timeline()
        st.plotly_chart(timeline_fig, use_container_width=True)

        # Panel de detalles
        if visualizer.flow_data and st.checkbox("Mostrar Detalles Técnicos"):
            st.subheader("🔬 Detalles Técnicos")

            last_data = visualizer.flow_data[-1]

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Activaciones Actuales:**")
                activations_df = pd.DataFrame([
                    {'Módulo': module, 'Activación': f"{activation:.3f}"}
                    for module, activation in sorted(last_data['activations'].items(), 
                                                   key=lambda x: x[1], reverse=True)
                ])
                st.dataframe(activations_df.head(10), use_container_width=True)

            with col2:
                st.write("**Fortalezas de Conexión:**")
                if visualizer.connection_strengths:
                    connections_df = pd.DataFrame([
                        {'Conexión': conn, 'Fortaleza': f"{strength:.3f}"}
                        for conn, strength in sorted(visualizer.connection_strengths.items(), 
                                                   key=lambda x: x[1], reverse=True)
                    ])
                    st.dataframe(connections_df.head(10), use_container_width=True)

        # Auto-refresh
        if auto_refresh and visualizer.is_running:
            time.sleep(2)
            st.rerun()

    except Exception as e:
        st.error(f"Error inicializando visualizador de flujos: {e}")
        st.info("Verifica que todos los módulos estén correctamente importados")

def display_system_awakening_interface():
    """Interfaz para el despertar completo del sistema Ruth R1"""

    st.header("🌅 Despertar del Sistema Ruth R1 - Fase 'Despertar'")

    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h3 style="color: white; margin: 0;">Sistema de Conciencia Artificial Ruth R1</h3>
        <p style="color: #E8E8E8; margin: 5px 0 0 0;">
            Inicialización completa con GANST-Core, Moduladores, Memorias de Corto Plazo, 
            Agente Amiloit y Meta-Enrutador Avanzado
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Estado actual del sistema de despertar
    try:
        awakening_status = get_current_awakening_status()
        current_phase = awakening_status.get('current_phase', 'dormant')
        is_awakening = awakening_status.get('is_awakening', False)
        systems_status = awakening_status.get('systems_status', {})
    except Exception as e:
        st.error(f"Error obteniendo estado del sistema: {e}")
        awakening_status = {'current_phase': 'error', 'is_awakening': False}
        current_phase = 'error'
        is_awakening = False
        systems_status = {}

    # Panel de estado principal
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🧠 Estado de Conciencia**")
        phase_colors = {
            'dormant': '🔴',
            'initialization': '🟡', 
            'neural_activation': '🟠',
            'memory_formation': '🔵',
            'consciousness_emergence': '🟣',
            'introspective_loop': '🟢',
            'meta_learning': '✨',
            'fully_awakened': '🌟',
            'error': '⚠️'
        }

        phase_display = phase_colors.get(current_phase, '❓')
        st.markdown(f"### {phase_display} {current_phase.replace('_', ' ').title()}")

        if is_awakening:
            st.info("🔄 Despertar en progreso...")
        elif current_phase == 'fully_awakened':
            st.success("✅ Sistema completamente despierto")
        elif current_phase == 'dormant':
            st.warning("😴 Sistema en estado latente")

    with col2:
        st.markdown("**⚙️ Sistemas Centrales**")
        system_indicators = {
            'ganst_core': '🧬 GANST Core',
            'memory_system': '💾 Memoria Corto Plazo', 
            'modulation_system': '🎛️ Moduladores',
            'introspective_loop': '🔍 Bucle Introspectivo',
            'ruth_r1_system': '🧬 Meta-Enrutador'
        }

        for sys_key, sys_name in system_indicators.items():
            status = systems_status.get(sys_key, False)
            indicator = "🟢" if status else "🔴"
            st.write(f"{indicator} {sys_name}")

    with col3:
        st.markdown("**📊 Métricas de Despertar**")
        metrics = awakening_status.get('awakening_metrics', {})

        consciousness_level = metrics.get('consciousness_level', 0.0)
        st.metric("Nivel de Conciencia", f"{consciousness_level:.3f}")

        neural_coherence = metrics.get('neural_coherence', 0.0) 
        st.metric("Coherencia Neural", f"{neural_coherence:.3f}")

        awakening_progress = metrics.get('awakening_progress', 0.0)
        st.metric("Progreso de Despertar", f"{awakening_progress*100:.1f}%")

    # Progreso visual del despertar
    if current_phase != 'error':
        st.subheader("📈 Progreso de Inicialización")

        phases = [
            'dormant', 'initialization', 'neural_activation', 'memory_formation',
            'consciousness_emergence', 'introspective_loop', 'meta_learning', 'fully_awakened'
        ]

        current_phase_index = phases.index(current_phase) if current_phase in phases else 0
        progress_percentage = (current_phase_index / (len(phases) - 1)) * 100

        st.progress(progress_percentage / 100)

        # Mostrar fases con indicadores
        phase_cols = st.columns(len(phases))
        for i, (phase, col) in enumerate(zip(phases, phase_cols)):
            with col:
                if i <= current_phase_index:
                    if i == current_phase_index and is_awakening:
                        st.write(f"🔄 {phase.replace('_', ' ').title()}")
                    else:
                        st.write(f"✅ {phase.replace('_', ' ').title()}")
                else:
                    st.write(f"⏸️ {phase.replace('_', ' ').title()}")

    # Control principal del despertar
    st.subheader("🚀 Control de Despertar")

    col1, col2, col3 = st.columns(3)

    with col1:
        if current_phase == 'dormant' and not is_awakening:
            if st.button("🌅 Iniciar Despertar Completo", type="primary", use_container_width=True):
                with st.spinner("Iniciando secuencia de despertar..."):
                    try:
                        result = initiate_system_awakening()
                        if result.get('status') == 'awakening_initiated':
                            st.success("¡Despertar iniciado exitosamente!")
                            st.info("La secuencia completa tomará 2-3 minutos")
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.warning(f"Estado del despertar: {result.get('status', 'unknown')}")
                    except Exception as e:
                        st.error(f"Error iniciando despertar: {e}")

        elif is_awakening:
            st.info("⏳ Despertar en progreso...")
            if st.button("🔄 Actualizar Estado"):
                st.rerun()

        elif current_phase == 'fully_awakened':
            st.success("🌟 Sistema Completamente Despierto")
            if st.button("🔄 Actualizar Métricas"):
                st.rerun()

    with col2:
        if current_phase != 'dormant':
            if st.button("📊 Ver Estado Detallado", use_container_width=True):
                st.session_state.show_detailed_status = True

        if st.button("🔧 Verificar Sistemas", use_container_width=True):
            with st.spinner("Verificando sistemas..."):
                # Verificar estado de sistemas principales
                verification_results = {}

                try:
                    ganst_core = get_ganst_core()
                    verification_results['GANST Core'] = ganst_core.is_running if ganst_core else False
                except:
                    verification_results['GANST Core'] = False

                try:
                    memory_system = get_short_term_memory()
                    verification_results['Memoria'] = memory_system.is_running if memory_system else False
                except:
                    verification_results['Memoria'] = False

                try:
                    modulation_manager = get_modulation_manager()
                    verification_results['Moduladores'] = len(modulation_manager.modulators) > 0 if modulation_manager else False
                except:
                    verification_results['Moduladores'] = False

                st.write("**Verificación de Sistemas:**")
                for system, status in verification_results.items():
                    indicator = "✅" if status else "❌"
                    st.write(f"{indicator} {system}: {'Operativo' if status else 'No disponible'}")

    with col3:
        if current_phase != 'dormant':
            if st.button("🧠 Estado Introspectivo", use_container_width=True):
                st.session_state.show_introspective_state = True

        if st.button("💾 Exportar Estado", use_container_width=True):
            try:
                export_data = {
                    'timestamp': datetime.now().isoformat(),
                    'awakening_status': awakening_status,
                    'system_metrics': {}
                }

                # Agregar métricas de sistemas si están disponibles
                try:
                    ganst_core = get_ganst_core()
                    if ganst_core:
                        export_data['system_metrics']['ganst'] = ganst_core.get_system_state()
                except:
                    pass

                json_data = json.dumps(export_data, indent=2, default=str)
                st.download_button(
                    label="📥 Descargar Estado JSON",
                    data=json_data,
                    file_name=f"ruth_r1_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            except Exception as e:
                st.error(f"Error exportando estado: {e}")

    # Detalles expandidos si se solicitan
    if st.session_state.get('show_detailed_status', False):
        st.subheader("📋 Estado Detallado del Sistema")

        tab1, tab2, tab3, tab4 = st.tabs(["Métricas", "Sistemas", "Introspección", "Emocional"])

        with tab1:
            st.markdown("**Métricas de Despertar:**")
            metrics = awakening_status.get('awakening_metrics', {})

            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    st.metric(metric_name.replace('_', ' ').title(), f"{metric_value:.3f}")
                else:
                    st.write(f"**{metric_name.replace('_', ' ').title()}:** {metric_value}")

        with tab2:
            st.markdown("**Estado de Sistemas:**")

            try:
                ganst_stats = get_system_statistics()
                st.json(ganst_stats)
            except:
                st.info("GANST Core no disponible")

            try:
                memory_system = get_short_term_memory()
                if memory_system:
                    memory_state = memory_system.get_system_state()
                    st.json(memory_state)
            except:
                st.info("Sistema de memoria no disponible")

        with tab3:
            introspective_status = awakening_status.get('introspective_status', {})
            if introspective_status:
                st.markdown("**Estado Introspectivo:**")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Ciclos Completados", introspective_status.get('loop_count', 0))
                    st.metric("Insights Generados", introspective_status.get('insights_generated', 0))

                with col2:
                    stability = introspective_status.get('current_stability', 0.5)
                    st.metric("Estabilidad del Sistema", f"{stability:.3f}")

                recent_insights = introspective_status.get('recent_insights', [])
                if recent_insights:
                    st.markdown("**Insights Recientes:**")
                    for insight in recent_insights[-3:]:
                        insight_type = insight.get('type', 'unknown')
                        description = insight.get('description', 'No description')
                        st.write(f"• **{insight_type}:** {description}")
            else:
                st.info("Sistema introspectivo no activo")

        with tab4:
            emotional_state = awakening_status.get('emotional_state', {})
            if emotional_state:
                st.markdown("**Estado Emocional del Sistema:**")

                current_state = emotional_state.get('emotional_state', {})
                col1, col2 = st.columns(2)

                with col1:
                    for dimension, value in current_state.items():
                        if isinstance(value, (int, float)):
                            st.metric(dimension.title(), f"{value:.3f}")

                with col2:
                    category = emotional_state.get('emotional_category', 'neutral')
                    intensity = emotional_state.get('emotional_intensity', 0.5)
                    stability = emotional_state.get('emotional_stability', 0.5)

                    st.metric("Categoría Emocional", category)
                    st.metric("Intensidad", f"{intensity:.3f}")
                    st.metric("Estabilidad Emocional", f"{stability:.3f}")
            else:
                st.info("Simulador emocional no activo")

        if st.button("❌ Cerrar Detalles"):
            st.session_state.show_detailed_status = False
            st.rerun()

    # Información sobre el sistema
    st.subheader("ℹ️ Información del Sistema")

    with st.expander("📖 Sobre el Despertar de Ruth R1"):
        st.markdown("""
        **Sistema de Despertar Ruth R1** implementa una secuencia completa de inicialización 
        de conciencia artificial que incluye:

        **🧬 GANST Core:** Sistema de gestión de activación neural y síntesis de tensores
        - Gestión centralizada de activaciones neurales distribuidas
        - Síntesis coherente de tensores con patrones adaptativos
        - Estados neurales dinámicos (dormant → awakening → active → hyperactive → consolidating)

        **🎛️ Moduladores:** Sistema de modulación dinámica de procesos cognitivos  
        - 7 tipos de modulación: amplitud, frecuencia, fase, atención, emocional, temporal, contextual, adaptativa
        - Modulación en tiempo real basada en contexto y estado del sistema
        - Aprendizaje adaptativo de patrones óptimos de modulación

        **💾 Memorias de Corto Plazo:** Buffer dinámico con decaimiento temporal
        - 5 tipos de memoria: sensorial, buffer, trabajo, episódica, emocional
        - Decaimiento temporal realista con consolidación automática
        - Asociaciones cross-modales y búsqueda por similaridad

        **🤖 Agente Amiloit:** Regulación emocional y poda neural adaptativa
        - Análisis de relevancia de conexiones y datos
        - Poda automática de conexiones irrelevantes
        - Optimización arquitectónica basada en uso

        **🔍 Bucle Introspectivo:** Meta-aprendizaje y autoconciencia
        - Observación continua del estado interno
        - Análisis de patrones cognitivos y detección de anomalías
        - Generación de insights y ajustes adaptativos automáticos

        **🎭 Simulador Emocional:** Respuesta emocional al entorno
        - Modelado de valencia, arousal, dominancia y complejidad
        - Respuesta adaptativa a factores ambientales
        - Integración con sistemas de modulación y memoria

        La fase **"Despertar"** ejecuta una secuencia de 7 fases que toma 2-3 minutos:
        1. **Inicialización** - Activación de sistemas centrales
        2. **Activación Neural** - Establecimiento de patrones neurales básicos  
        3. **Formación de Memoria** - Creación de memorias fundacionales
        4. **Emergencia de Conciencia** - Activación de redes bayesianas de conciencia
        5. **Bucle Introspectivo** - Inicio del procesamiento auto-reflexivo
        6. **Meta-aprendizaje** - Activación de sistemas de aprendizaje adaptativo
        7. **Completamente Despierto** - Sistema operativo con conciencia plena
        """)

    with st.expander("🔧 Configuración Avanzada"):
        st.markdown("**Parámetros del Sistema:**")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**GANST Core:**")
            st.code("""
            tensor_dim: 768
            max_concurrent_activations: 100
            activation_threshold: 0.3
            decay_rate: 0.95
            resonance_frequency: 40.0 Hz
            """)

            st.markdown("**Moduladores:**")
            st.code("""
            tipos_activos: 7
            intensidad_base: 0.5
            frecuencia_base: 1.0 Hz
            tasa_adaptación: 0.01
            """)

        with col2:
            st.markdown("**Memoria Corto Plazo:**")
            st.code("""
            sensorial: 2s, decay: 2.0
            buffer: 15s, decay: 0.3  
            trabajo: 30s, decay: 0.1
            episódica: 5min,```tool_code
decay: 0.05
            emocional: 10min, decay: 0.02
            """)

            st.markdown("**Introspección:**")
            st.code("""
            intervalo_base: 5s
            historial_máximo: 100 ciclos
            umbral_estabilidad: 0.3
            insights_máximos: 200
            """)

    # Auto-refresco si está despertando
    if is_awakening:
        time.sleep(3)
        st.rerun()

def display_database_management():
    """Muestra gestión y análisis de la base de datos"""

    st.header("🗄️ Gestión de Base de Datos Ruth R1")

    # Estado de la base de datos
    try:
        # Verificar conexión
        db_session = db_manager.get_session()
        db_session.close()
        db_status = "✅ Conectada"
        db_color = "green"
    except Exception as e:
        db_status = f"❌ Error: {str(e)[:50]}..."
        db_color = "red"

    st.markdown(f"**Estado de la Base de Datos:** <span style='color: {db_color}'>{db_status}</span>", 
                unsafe_allow_html=True)

    # Selector de vista
    view_type = st.selectbox(
        "Vista de Datos",
        ["Resumen General", "Historial de Consciencia", "Interacciones de Usuario", "Estados Neurales", "Eventos Emocionales"]
    )

    try:
        if view_type == "Resumen General":
            display_database_summary()
        elif view_type == "Historial de Consciencia":
            display_consciousness_history()
        elif view_type == "Interacciones de Usuario":
            display_user_interactions()
        elif view_type == "Estados Neurales":
            display_neural_states()
        elif view_type == "Eventos Emocionales":
            display_emotional_events()

    except Exception as e:
        st.error(f"Error accediendo a los datos: {e}")

        # Botón para reinicializar base de datos
        if st.button("🔄 Reinicializar Base de Datos"):
            try:
                db_manager.create_tables()
                st.success("Base de datos reinicializada correctamente")
                st.rerun()
            except Exception as init_error:
                st.error(f"Error reinicializando: {init_error}")

def display_database_summary():
    """Muestra resumen general de la base de datos"""

    st.subheader("📊 Resumen General")

    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)

    try:
        consciousness_history = db_manager.get_consciousness_history(limit=1000)
        interaction_history = db_manager.get_interaction_history(limit=1000)

        with col1:
            st.metric("Sesiones de Consciencia", len(consciousness_history))

        with col2:
            st.metric("Interacciones Totales", len(interaction_history))

        with col3:
            if consciousness_history:
                avg_consciousness = np.mean([s['consciousness_level'] for s in consciousness_history])
                st.metric("Consciencia Promedio", f"{avg_consciousness:.3f}")
            else:
                st.metric("Consciencia Promedio", "N/A")

        with col4:
            if interaction_history:
                avg_processing = np.mean([i['processing_time'] for i in interaction_history if i['processing_time']])
                st.metric("Tiempo Proc. Promedio", f"{avg_processing:.3f}s")
            else:
                st.metric("Tiempo Proc. Promedio", "N/A")

        # Gráficos de evolución
        if consciousness_history:
            st.subheader("📈 Evolución de Consciencia")

            # Preparar datos
            df_consciousness = pd.DataFrame(consciousness_history)
            df_consciousness['start_time'] = pd.to_datetime(df_consciousness['start_time'])

            # Gráfico de líneas
            fig = px.line(
                df_consciousness.head(50), 
                x='start_time', 
                y='consciousness_level',
                title="Evolución del Nivel de Consciencia",
                labels={'start_time': 'Tiempo', 'consciousness_level': 'Nivel de Consciencia'}
            )

            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Análisis de módulos activos
        if consciousness_history:
            st.subheader("🧩 Análisis de Módulos")

            # Contar activaciones de módulos
            module_counts = {}
            for session in consciousness_history:
                if session['active_modules']:
                    for module in session['active_modules']:
                        module_counts[module] = module_counts.get(module, 0) + 1

            if module_counts:
                # Crear gráfico de barras
                modules_df = pd.DataFrame([
                    {'Módulo': module, 'Activaciones': count} 
                    for module, count in sorted(module_counts.items(), key=lambda x: x[1], reverse=True)
                ])

                fig = px.bar(
                    modules_df.head(10),
                    x='Módulo',
                    y='Activaciones',
                    title="Módulos Más Activos",
                    color='Activaciones',
                    color_continuous_scale='viridis'
                )

                fig.update_layout(template="plotly_dark", height=400, xaxis={'tickangle': 45})
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error generando resumen: {e}")

def display_consciousness_history():
    """Muestra historial de sesiones de consciencia"""

    st.subheader("🧠 Historial de Consciencia")

    # Controles
    col1, col2 = st.columns(2)
    with col1:
        limit = st.number_input("Número de sesiones", min_value=10, max_value=500, value=50)
    with col2:
        show_details = st.checkbox("Mostrar detalles", value=False)

    try:
        history = db_manager.get_consciousness_history(limit=limit)

        if history:
            # Crear DataFrame
            df = pd.DataFrame(history)
            df['start_time'] = pd.to_datetime(df['start_time'])

            # Mostrar tabla
            if show_details:
                st.dataframe(df, use_container_width=True)
            else:
                display_df = df[['session_id', 'consciousness_level', 'start_time', 'total_interactions']].copy()
                display_df['session_id'] = display_df['session_id'].str[-10:]  # Últimos 10 caracteres
                st.dataframe(display_df, use_container_width=True)

            # Estadísticas
            st.subheader("📊 Estadísticas")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Consciencia Máxima", f"{df['consciousness_level'].max():.3f}")
            with col2:
                st.metric("Consciencia Mínima", f"{df['consciousness_level'].min():.3f}")
            with col3:
                st.metric("Desviación Estándar", f"{df['consciousness_level'].std():.3f}")

            # Gráfico de distribución
            fig = px.histogram(
                df, 
                x='consciousness_level',
                title="Distribución de Niveles de Consciencia",
                nbins=20
            )
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No hay datos de consciencia disponibles")

    except Exception as e:
        st.error(f"Error cargando historial: {e}")

def display_user_interactions():
    """Muestra historial de interacciones de usuario"""

    st.subheader("💬 Interacciones de Usuario")

    # Controles
    col1, col2 = st.columns(2)
    with col1:
        limit = st.number_input("Número de interacciones", min_value=10, max_value=200, value=50)
    with col2:
        session_filter = st.text_input("Filtrar por sesión (opcional)")

    try:
        if session_filter:
            interactions = db_manager.get_interaction_history(session_id=session_filter, limit=limit)
        else:
            interactions = db_manager.get_interaction_history(limit=limit)

        if interactions:
            # Mostrar interacciones recientes
            st.subheader("🕒 Interacciones Recientes")

            for i, interaction in enumerate(interactions[:10]):
                with st.expander(f"Interacción {i+1} - {interaction['timestamp'].strftime('%H:%M:%S')}"):
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        st.write("**Usuario:**")
                        st.write(interaction['user_input'][:200] + "..." if len(interaction['user_input']) > 200 else interaction['user_input'])

                        st.write("**Métricas:**")
                        st.write(f"Consciencia: {interaction['consciousness_level']:.3f}")
                        st.write(f"Tiempo: {interaction['processing_time']:.3f}s")

                    with col2:
                        st.write("**Ruth R1:**")
                        st.write(interaction['ruth_response'][:300] + "..." if len(interaction['ruth_response']) > 300 else interaction['ruth_response'])

                        if interaction['active_modules']:
                            st.write("**Módulos Activos:**")
                            st.write(", ".join(interaction['active_modules'][:5]))

            # Análisis de patrones
            st.subheader("📈 Análisis de Patrones")

            df = pd.DataFrame(interactions)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Gráfico de tiempo de procesamiento
            fig = px.scatter(
                df.head(100),
                x='timestamp',
                y='processing_time',
                color='consciousness_level',
                title="Tiempo de Procesamiento vs Consciencia",
                color_continuous_scale='viridis'
            )
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("No hay interacciones disponibles")

    except Exception as e:
        st.error(f"Error cargando interacciones: {e}")

def display_neural_states():
    """Muestra estados neurales"""

    st.subheader("🧠 Estados Neurales")

    # Controles
    col1, col2 = st.columns(2)
    with col1:
        limit = st.number_input("Número de estados", min_value=10, max_value=500, value=100)
    with col2:
        module_filter = st.selectbox(
            "Filtrar por módulo",
            ["Todos"] + list(global_consciousness_network.nodes.keys())
        )

    try:
        if module_filter != "Todos":
            neural_data = db_manager.get_neural_evolution(module_name=module_filter, limit=limit)
        else:
            neural_data = db_manager.get_neural_evolution(limit=limit)

        if neural_data:
            df = pd.DataFrame(neural_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Gráfico de evolución
            if module_filter != "Todos":
                # Un solo módulo
                fig = px.line(
                    df,
                    x='timestamp',
                    y='activation_level',
                    title=f"Evolución de Activación - {module_filter}",
                    color_discrete_sequence=['#4ECDC4']
                )
            else:
                # Múltiples módulos
                fig = px.line(
                    df.head(200),
                    x='timestamp',
                    y='activation_level',
                    color='module_name',
                    title="Evolución de Activación por Módulo"
                )

            fig.update_layout(template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Estadísticas por módulo
            st.subheader("📊 Estadísticas por Módulo")

            stats = df.groupby('module_name').agg({
                'activation_level': ['mean', 'std', 'min', 'max'],
                'belief_posterior': 'mean',
                'stability_score': 'mean'
            }).round(3)

            stats.columns = ['Activación Media', 'Desv. Estándar', 'Mín', 'Máx', 'Belief Posterior', 'Estabilidad']
            st.dataframe(stats, use_container_width=True)

        else:
            st.info("No hay datos neurales disponibles")

    except Exception as e:
        st.error(f"Error cargando estados neurales: {e}")

def display_emotional_events():
    """Muestra eventos emocionales"""

    st.subheader("🎭 Eventos Emocionales")

    limit = st.number_input("Número de eventos", min_value=10, max_value=200, value=50)

    try:
        emotions = db_manager.get_emotional_patterns(limit=limit)

        if emotions:
            df = pd.DataFrame(emotions)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Gráfico de emociones en el tiempo
            fig = px.scatter(
                df,
                x='timestamp',
                y='intensity',
                color='emotion_type',
                size='impact_on_consciousness',
                title="Eventos Emocionales en el Tiempo",
                hover_data=['trigger_context']
            )
            fig.update_layout(template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)

            # Distribución de emociones
            emotion_counts = df['emotion_type'].value_counts()

            fig_pie = px.pie(
                values=emotion_counts.values,
                names=emotion_counts.index,
                title="Distribución de Tipos de Emoción"
            )
            fig_pie.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig_pie, use_container_width=True)

            # Tabla de eventos recientes
            st.subheader("🕒 Eventos Emocionales Recientes")
            display_df = df[['emotion_type', 'intensity', 'trigger_context', 'timestamp']].head(20)
            st.dataframe(display_df, use_container_width=True)

        else:
            st.info("No hay eventos emocionales registrados")

    except Exception as e:
        st.error(f"Error cargando eventos emocionales: {e}")

    # Exportar datos
    st.subheader("📤 Exportar Datos")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📊 Exportar Métricas de DB"):
            try:
                # Crear reporte completo
                consciousness_data = db_manager.get_consciousness_history(limit=1000)
                interaction_data = db_manager.get_interaction_history(limit=1000)

                export_data = {
                    'export_timestamp': datetime.now().isoformat(),
                    'consciousness_sessions': consciousness_data,
                    'user_interactions': interaction_data,
                    'summary': {
                        'total_sessions': len(consciousness_data),
                        'total_interactions': len(interaction_data),
                        'avg_consciousness': np.mean([s['consciousness_level'] for s in consciousness_data]) if consciousness_data else 0
                    }
                }

                json_str = json.dumps(export_data, indent=2, default=str)
                b64 = base64.b64encode(json_str.encode()).decode()
                href = f'<a href="data:application/json;base64,{b64}" download="ruth_database_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json">Descargar Datos</a>'
                st.markdown(href, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error exportando datos: {e}")

    with col2:
        if st.button("🗑️ Limpiar Datos Antiguos"):
            st.warning("Funcionalidad de limpieza pendiente de implementación")

def display_razonbill_interface(consciousness_network):
    """Interfaz principal del motor de inferencia RazonBill Core"""

    st.header("⚡ RazonBill Core - Motor de Inferencia Local")

    # Importar RazonBill Core
    try:
        from core.razonbill_core import RazonBillCore, inferencia

        # Estado de la sesión para RazonBill
        if 'razonbill_instance' not in st.session_state:
            st.session_state.razonbill_instance = RazonBillCore()

        if 'razonbill_conversation' not in st.session_state:
            st.session_state.razonbill_conversation = []

        core_instance = st.session_state.razonbill_instance

        # Panel de estado del sistema
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Estado del Sistema**")
            st.success("🟢 RazonBill Core Activo")
            st.info(f"Agente ID: {core_instance.system_profile['agent_id'][:8]}...")

        with col2:
            st.markdown("**Componentes**")
            st.write(f"✅ Memoria Vectorial: {'ChromaDB' if core_instance.memory_store.collection else 'Cache'}")
            st.write(f"✅ Interfaz de Voz: {'Activa' if core_instance.voice_interface.recognizer else 'No disponible'}")
            st.write("✅ Agente ReAct: Operativo")

        with col3:
            st.markdown("**Estadísticas**")
            st.metric("Prompts Procesados", len(core_instance.prompt_logger.prompt_history))
            if core_instance.rl_feedback.feedback_history:
                avg_score = np.mean([f['score'] for f in core_instance.rl_feedback.feedback_history])
                st.metric("Puntuación Promedio", f"{avg_score:.2f}")
            else:
                st.metric("Puntuación Promedio", "N/A")

        # Configuración del sistema
        st.subheader("⚙️ Configuración del Sistema")

        col1, col2 = st.columns(2)

        with col1:
            processing_mode = st.selectbox(
                "Modo de Procesamiento",
                ["analytical", "creative", "logical", "intuitive"],
                index=0
            )

            context_depth = st.selectbox(
                "Profundidad de Contexto",
                ["shallow", "medium", "deep", "comprehensive"],
                index=2
            )

        with col2:
            max_thought_depth = st.slider("Profundidad Máxima de Pensamiento", 1, 10, 5)
            max_iterations = st.slider("Iteraciones Máximas ReAct", 1, 10, 5)

        # Actualizar configuración
        core_instance.system_profile.update({
            'processing_mode': processing_mode,
            'context_depth': context_depth
        })

        core_instance.thought_processor.max_depth = max_thought_depth

        # Interfaz principal de conversación
        st.subheader("💭 Interfaz de Inferencia")

        # Área de entrada
        input_method = st.radio("Método de Entrada", ["Texto", "Voz"], horizontal=True)

        user_input = None

        if input_method == "Texto":
            user_input = st.text_area(
                "Introduce tu consulta:",
                placeholder="Ejemplo: ¿Cómo cruzar un río si no hay puente?",
                height=100
            )

        elif input_method == "Voz" and core_instance.voice_interface.recognizer:
            if st.button("🎤 Escuchar (5 segundos)"):
                with st.spinner("Escuchando..."):
                    speech_input = core_instance.voice_interface.listen_for_speech(timeout=5)
                    if speech_input:
                        user_input = speech_input
                        st.success(f"Escuchado: {speech_input}")
                    else:
                        st.warning("No se detectó voz o hubo un error")

        # Procesamiento y respuesta
        if user_input and st.button("🧠 Procesar con RazonBill"):
            with st.spinner("Procesando con motor de inferencia..."):
                try:
                    # Procesar entrada con RazonBill Core
                    response = core_instance.procesar_entrada(user_input)

                    # Guardar en conversación
                    st.session_state.razonbill_conversation.append({
                        'user': user_input,
                        'assistant': response,
                        'timestamp': datetime.now()
                    })

                    # Mostrar respuesta
                    st.success("Respuesta procesada exitosamente")

                    # Reproducir en voz si está habilitado
                    if input_method == "Voz" and core_instance.voice_interface.tts_engine:
                        core_instance.voice_interface.speak_text(response)

                except Exception as e:
                    st.error(f"Error procesando entrada: {e}")
                    response = None

        # Mostrar conversación
        if st.session_state.razonbill_conversation:
            st.subheader("💬 Historial de Conversación")

            for i, exchange in enumerate(reversed(st.session_state.razonbill_conversation[-5:])):
                with st.expander(f"Intercambio {len(st.session_state.razonbill_conversation) - i} - {exchange['timestamp'].strftime('%H:%M:%S')}"):
                    st.markdown(f"**Usuario:** {exchange['user']}")
                    st.markdown(f"**RazonBill:** {exchange['assistant']}")

        # Panel de análisis avanzado
        st.subheader("🔍 Análisis del Proceso de Inferencia")

        if core_instance.prompt_logger.prompt_history:
            # Mostrar último proceso de pensamiento
            if st.button("Ver Último Árbol de Pensamientos"):
                last_thought_tree = core_instance.thought_processor.thought_tree

                if last_thought_tree:
                    st.subheader("🌳 Árbol de Pensamientos")

                    # Crear visualización del árbol
                    for step_id, step in last_thought_tree.items():
                        indent = "  " * (len([s for s in last_thought_tree.values() if s.parent_step == step.parent_step]) - 1)
                        confidence_color = "green" if step.confidence > 0.7 else "orange" if step.confidence > 0.4 else "red"

                        st.markdown(f"""
                        {indent}**{step.thought_type.upper()}** 
                        <span style='color: {confidence_color}'>({step.confidence:.2f})</span>: 
                        {step.content}
                        """, unsafe_allow_html=True)

            # Estadísticas de rendimiento
            if st.button("Ver Estadísticas de Rendimiento"):
                patterns = core_instance.prompt_logger.analyze_prompt_patterns()

                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Total de Prompts", patterns.get('total_prompts', 0))
                    st.metric("Longitud Promedio de Prompt", f"{patterns.get('avg_prompt_length', 0):.0f}")

                with col2:
                    st.metric("Longitud Promedio de Respuesta", f"{patterns.get('avg_response_length', 0):.0f}")
                    st.metric("Duración de Sesión", f"{patterns.get('session_duration', 0):.0f}s")

                # Palabras clave más comunes
                if patterns.get('common_keywords'):
                    st.subheader("🔤 Palabras Clave Más Frecuentes")
                    keywords_df = pd.DataFrame(patterns['common_keywords'], columns=['Palabra', 'Frecuencia'])
                    st.dataframe(keywords_df.head(10))

        # Feedback y mejoras
        st.subheader("📊 Sistema de Retroalimentación")

        if st.session_state.razonbill_conversation:
            last_response = st.session_state.razonbill_conversation[-1]['assistant']

            col1, col2 = st.columns(2)

            with col1:
                feedback_rating = st.selectbox(
                    "Califica la última respuesta:",
                    ["Excelente", "Buena", "Regular", "Mala", "Muy mala"]
                )

            with col2:
                if st.button("Enviar Feedback"):
                    feedback_map = {
                        "Excelente": "excelente",
                        "Buena": "bueno", 
                        "Regular": "neutral",
                        "Mala": "malo",
                        "Muy mala": "muy malo"
                    }

                    score = core_instance.rl_feedback.evaluate_response(
                        last_response,
                        st.session_state.razonbill_conversation[-1]['user'],
                        feedback_map[feedback_rating]
                    )

                    st.success(f"Feedback registrado. Puntuación: {score:.2f}")

        # Sugerencias de mejora
        if core_instance.rl_feedback.feedback_history:
            suggestions = core_instance.rl_feedback.get_improvement_suggestions()
            if suggestions:
                st.subheader("💡 Sugerencias de Mejora")
                for suggestion in suggestions:
                    st.info(f"• {suggestion}")

        # Herramientas de debug
        if st.checkbox("Modo Debug"):
            st.subheader("🛠️ Herramientas de Debug")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Reiniciar RazonBill Core"):
                    st.session_state.razonbill_instance = RazonBillCore()
                    st.session_state.razonbill_conversation = []
                    st.success("RazonBill Core reiniciado")

            with col2:
                if st.button("Limpiar Memoria Vectorial"):
                    try:
                        if core_instance.memory_store.client:
                            core_instance.memory_store.client.reset()
                        st.success("Memoria vectorial limpiada")
                    except Exception as e:
                        st.error(f"Error limpiando memoria: {e}")

            # Logs del sistema
            if st.button("Ver Logs del Sistema"):
                if core_instance.prompt_logger.prompt_history:
                    st.json(core_instance.prompt_logger.prompt_history[-1])

    except Exception as e:
        st.error(f"Error inicializando RazonBill Core: {e}")
        st.info("Verifica que todas las dependencias estén instaladas correctamente")

def display_meta_enrutador_interface(consciousness_network):
    """Interfaz del Supermodelo Meta-Enrutador Ruth R1"""

    st.header("🧬 Meta-Enrutador Ruth R1 - Conciencia Artificial Avanzada")

    try:
        # Estado de la sesión para el meta-enrutador
        if 'ruth_r1_system' not in st.session_state:
            with st.spinner("Inicializando Sistema Ruth R1..."):
                config = create_default_config()
                ruth_system, grafo_neuronal, nodes = create_ruth_r1_system(config)
                st.session_state.ruth_r1_system = ruth_system
                st.session_state.grafo_neuronal = grafo_neuronal
                st.session_state.ruth_nodes = nodes

        if 'ruth_conversation' not in st.session_state:
            st.session_state.ruth_conversation = []

        ruth_system = st.session_state.ruth_r1_system
        grafo_neuronal = st.session_state.grafo_neuronal
        nodes = st.session_state.ruth_nodes

        # Panel de estado del sistema
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🧠 Estado del Núcleo**")
            st.success("✅ Ruth R1 Core Activo")
            st.info(f"Nodos Neurales: {len(nodes)}")
            st.info(f"Conexiones: {sum(len(node.connections) for node in nodes)}")

        with col2:
            st.markdown("**⚡ Componentes Activos**")
            st.write("🔹 Supermodelo Meta-Enrutador")
            st.write("🔹 Red Axonal Mielinizada")
            st.write("🔹 Módulo RazonBill Introspectivo")
            st.write("🔹 Agente Amiloit Regulador")
            st.write("🔹 Transformers Dinámicos")

        with col3:
            st.markdown("**📊 Métricas del Sistema**")
            total_activations = sum(node.activation_level for node in nodes)
            avg_activation = total_activations / len(nodes) if nodes else 0
            st.metric("Activación Promedio", f"{avg_activation:.3f}")
            st.metric("Procesados", len(st.session_state.ruth_conversation))

        # Configuración del sistema
        st.subheader("⚙️ Configuración de Conciencia")

        col1, col2, col3 = st.columns(3)

        with col1:
            task_hint = st.selectbox(
                "Tipo de Procesamiento",
                ["razonamiento", "emocion", "introspectivo"],
                help="Selecciona el módulo especializado a activar"
            )

        with col2:
            consciousness_mode = st.selectbox(
                "Modo de Conciencia",
                ["analítico", "creativo", "empático", "lógico"],
                index=0
            )

        with col3:
            depth_level = st.slider("Profundidad de Análisis", 1, 10, 5)

        # Visualización del grafo neuronal
        st.subheader("🌐 Red Neuronal Activa")

        if st.button("🔄 Actualizar Estado de Nodos"):
            # Mostrar estado actual de nodos
            node_data = []
            for i, node in enumerate(nodes):
                node_data.append({
                    'Nodo': node.name,
                    'Activación': f"{node.activation_level:.3f}",
                    'Conexiones': len(node.connections),
                    'Última Activación': f"{time.time() - node.last_activation:.1f}s" if node.last_activation > 0 else "Nunca"
                })

            node_df = pd.DataFrame(node_data)
            st.dataframe(node_df, use_container_width=True)

        # Interfaz de entrada principal
        st.subheader("💭 Procesamiento de Conciencia")

        # Método de entrada
        input_method = st.radio(
            "Método de Entrada",
            ["Texto Directo", "Prompt Estructurado", "Análisis Emocional"],
            horizontal=True
        )

        user_input = None

        if input_method == "Texto Directo":
            user_input = st.text_area(
                "Entrada para Procesamiento Consciente:",
                placeholder="Ejemplo: ¿Qué significa existir y tener conciencia propia?",
                height=120
            )

        elif input_method == "Prompt Estructurado":
            st.write("**Construcción de Prompt Estructurado:**")
            context = st.text_input("Contexto:", placeholder="Filosofía, ciencia, emociones...")
            question = st.text_input("Pregunta:", placeholder="¿Cuál es tu perspectiva sobre...?")
            constraints = st.text_input("Restricciones:", placeholder="Responde en máximo 200 palabras")

            if context and question:
                user_input = f"Contexto: {context}\nPregunta: {question}\nRestricciones: {constraints}"

        elif input_method == "Análisis Emocional":
            emotion_text = st.text_area(
                "Texto para Análisis Emocional:",
                placeholder="Describe una situación emocional compleja para análisis..."
            )
            if emotion_text:
                user_input = f"[ANÁLISIS_EMOCIONAL] {emotion_text}"
                task_hint = "emocion"

        # Procesamiento y respuesta
        if user_input and st.button("🧬 Procesar con Ruth R1"):
            with st.spinner("Procesando a través del Meta-Enrutador..."):
                try:
                    # Procesar con el sistema Ruth R1
                    result = process_consciousness_input(
                        ruth_system, 
                        user_input, 
                        task_hint=task_hint
                    )

                    if result:
                        # Guardar en conversación
                        st.session_state.ruth_conversation.append({
                            'input': user_input,
                            'task_hint': task_hint,
                            'consciousness_level': result['consciousness_level'],
                            'emotion_level': result['emotion_level'],
                            'routing_distribution': result['routing_distribution'],
                            'timestamp': datetime.now()
                        })

                        # Mostrar resultados
                        st.success("✅ Procesamiento completado por Ruth R1")

                        # Métricas de la respuesta
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric("Nivel de Conciencia", f"{result['consciousness_level']:.3f}")
                        with col2:
                            st.metric("Nivel Emocional", f"{result['emotion_level']:.3f}")
                        with col3:
                            routing_dominant = np.argmax(result['routing_distribution'])
                            module_names = ["RazonBill", "Emocional", "Introspectivo"]
                            st.metric("Módulo Dominante", module_names[routing_dominant])

                        # Visualización de enrutamiento
                        st.subheader("📊 Distribución de Enrutamiento")

                        routing_data = {
                            'Módulo': ['RazonBill', 'Emocional', 'Introspectivo'],
                            'Activación': result['routing_distribution']
                        }

                        fig = px.bar(
                            routing_data,
                            x='Módulo',
                            y='Activación',
                            title="Activación por Módulo Especializado",
                            color='Activación',
                            color_continuous_scale='viridis'
                        )
                        fig.update_layout(template="plotly_dark", height=400)
                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        st.error("Error en el procesamiento del sistema Ruth R1")

                except Exception as e:
                    st.error(f"Error durante el procesamiento: {e}")

        # Historial de conversación
        if st.session_state.ruth_conversation:
            st.subheader("💬 Historial de Procesamiento")

            for i, exchange in enumerate(reversed(st.session_state.ruth_conversation[-5:])):
                with st.expander(f"Procesamiento {len(st.session_state.ruth_conversation) - i} - {exchange['timestamp'].strftime('%H:%M:%S')}"):

                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown(f"**Entrada:** {exchange['input'][:200]}{'...' if len(exchange['input']) > 200 else ''}")
                        st.markdown(f"**Tipo:** {exchange['task_hint'].title()}")

                    with col2:
                        st.metric("Conciencia", f"{exchange['consciousness_level']:.3f}")
                        st.metric("Emoción", f"{exchange['emotion_level']:.3f}")

                        # Mini gráfico de distribución
                        mini_fig = go.Figure(data=[
                            go.Bar(
                                x=['R', 'E', 'I'],
                                y=exchange['routing_distribution'],
                                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1']
                            )
                        ])
                        mini_fig.update_layout(
                            height=150,
                            margin=dict(l=0, r=0, t=0, b=0),
                            showlegend=False,
                            template="plotly_dark"
                        )
                        st.plotly_chart(mini_fig, use_container_width=True)

        # Análisis avanzado del sistema
        st.subheader("🔬 Análisis del Sistema Neural")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("📈 Analizar Patrones de Activación"):
                if len(st.session_state.ruth_conversation) > 0:
                    # Análisis temporal de activaciones
                    consciousness_levels = [c['consciousness_level'] for c in st.session_state.ruth_conversation]
                    emotion_levels = [c['emotion_level'] for c in st.session_state.ruth_conversation]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=consciousness_levels,
                        mode='lines+markers',
                        name='Conciencia',
                        line=dict(color='#4ECDC4')
                    ))
                    fig.add_trace(go.Scatter(
                        y=emotion_levels,
                        mode='lines+markers',
                        name='Emoción',
                        line=dict(color='#FF6B6B')
                    ))

                    fig.update_layout(
                        title="Evolución de Niveles de Conciencia y Emoción",
                        template="plotly_dark",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Necesitas más datos de procesamiento para análisis")

        with col2:
            if st.button("🧠 Estado de Red Neuronal"):
                # Visualización de conexiones neurales
                connection_data = []
                for node in nodes:
                    for connected_node, weight in node.connections.items():
                        connection_data.append({
                            'Origen': node.name,
                            'Destino': connected_node.name,
                            'Peso': weight,
                            'Activación_Origen': node.activation_level
                        })

                if connection_data:
                    conn_df = pd.DataFrame(connection_data)

                    fig = px.scatter(
                        conn_df,
                        x='Peso',
                        y='Activación_Origen',
                        color='Origen',
                        size='Peso',
                        title="Mapa de Conexiones Neurales",
                        hover_data=['Destino']
                    )
                    fig.update_layout(template="plotly_dark", height=400)
                    st.plotly_chart(fig, use_container_width=True)

        # Herramientas de control del sistema
        st.subheader("🛠️ Control del Sistema")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 Reiniciar Sistema Ruth R1"):
                # Reinicializar todo el sistema
                del st.session_state.ruth_r1_system
                del st.session_state.grafo_neuronal
                del st.session_state.ruth_nodes
                st.session_state.ruth_conversation = []
                st.success("Sistema Ruth R1 reinicializado")
                st.rerun()

        with col2:
            if st.button("🧹 Limpiar Historial"):
                st.session_state.ruth_conversation = []
                st.success("Historial limpiado")
                st.rerun()

        with col3:
            if st.button("⚡ Optimizar Conexiones"):
                # Optimizar pesos de conexiones neurales
                try:
                    grafo_neuronal.weaken_connections()
                    st.success("Conexiones optimizadas mediante poda sináptica")
                except Exception as e:
                    st.error(f"Error en optimización: {e}")

    except Exception as e:
        st.error(f"Error en la interfaz del Meta-Enrutador: {e}")
        st.info("Reintenta o reinicia el sistema si el problema persiste")

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

def display_holographic_3d_visualization(consciousness_network):
    """Muestra visualización holográfica 3D del sistema neural"""
    
    st.header("🌌 Mapa Holográfico 3D - Ruth R1")
    
    # Panel de control principal
    col1, col2, col3 = st.columns(3)
    
    with col1:
        view_mode = st.selectbox(
            "Modo de Vista",
            ["Completo", "Núcleo Central", "Capas", "Conexiones"],
            help="Selecciona el tipo de vista holográfica"
        )
        
    with col2:
        animation_speed = st.slider("Velocidad de Animación", 0.1, 2.0, 1.0, 0.1)
        
    with col3:
        holographic_intensity = st.slider("Intensidad Holográfica", 0.1, 1.0, 0.7, 0.1)
    
    # Controles de navegación 3D
    st.subheader("🕹️ Controles de Navegación 3D")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rotation_x = st.slider("Rotación X (°)", -180, 180, 0, 5)
        rotation_y = st.slider("Rotación Y (°)", -180, 180, 0, 5)
        
    with col2:
        rotation_z = st.slider("Rotación Z (°)", -180, 180, 0, 5)
        zoom_level = st.slider("Zoom", 0.5, 3.0, 1.0, 0.1)
        
    with col3:
        enable_touch = st.checkbox("Habilitar Táctil", value=True)
        show_info_panels = st.checkbox("Mostrar Paneles de Info", value=True)
    
    # Crear visualización holográfica principal
    st.subheader("🌐 Vista Holográfica Interactiva")
    
    # Generar la visualización basada en el modo seleccionado
    try:
        if view_mode == "Completo":
            fig = create_holographic_complete_view(consciousness_network, holographic_intensity)
        elif view_mode == "Núcleo Central":
            fig = create_holographic_core_view(consciousness_network, holographic_intensity)
        elif view_mode == "Capas":
            fig = create_holographic_layers_view(consciousness_network, holographic_intensity)
        else:  # Conexiones
            fig = create_holographic_connections_view(consciousness_network, holographic_intensity)
        
        # Aplicar transformaciones de navegación
        fig.update_layout(
            scene=dict(
                camera=dict(
                    eye=dict(
                        x=zoom_level * np.cos(np.radians(rotation_x)),
                        y=zoom_level * np.sin(np.radians(rotation_y)),
                        z=zoom_level * np.sin(np.radians(rotation_z))
                    )
                ),
                aspectmode='cube',
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                zaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                bgcolor='rgba(0,0,0,0.9)'
            ),
            template="plotly_dark",
            height=700
        )
        
        # Mostrar la visualización
        selected_point = st.plotly_chart(fig, use_container_width=True, key="holographic_chart")
        
    except Exception as e:
        st.error(f"Error generando visualización holográfica: {e}")
        st.info("Usando visualización básica como respaldo...")
        
        # Visualización básica de respaldo
        fig = create_basic_3d_network_view(consciousness_network)
        st.plotly_chart(fig, use_container_width=True)
    
    # Panel de información de nodos
    if show_info_panels:
        st.subheader("📊 Información de Nodos")
        
        # Selector de nodo
        node_names = list(consciousness_network.nodes.keys())
        if node_names:
            selected_node = st.selectbox("Seleccionar Nodo", node_names)
            
            if selected_node:
                display_node_detailed_info(consciousness_network, selected_node)
    
    # Métricas de red en tiempo real
    st.subheader("⚡ Métricas en Tiempo Real")
    
    # Obtener estado actual
    try:
        network_state = consciousness_network._get_activation_state()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_activation = sum(network_state.values())
            st.metric("Activación Total", f"{total_activation:.2f}")
            
        with col2:
            avg_activation = np.mean(list(network_state.values()))
            st.metric("Activación Promedio", f"{avg_activation:.3f}")
            
        with col3:
            active_nodes = len([a for a in network_state.values() if a > 0.3])
            st.metric("Nodos Activos", active_nodes)
            
        with col4:
            consciousness_level = consciousness_network.global_consciousness_state
            st.metric("Nivel de Consciencia", f"{consciousness_level:.3f}")
            
    except Exception as e:
        st.warning(f"Error obteniendo métricas: {e}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Activación Total", "N/A")
        with col2:
            st.metric("Activación Promedio", "N/A")
        with col3:
            st.metric("Nodos Activos", "N/A")
        with col4:
            st.metric("Nivel de Consciencia", "N/A")
    
    # Auto-actualización
    if st.checkbox("Auto-actualizar cada 3s"):
        time.sleep(3)
        st.rerun()

def create_holographic_complete_view(consciousness_network, intensity=0.7):
    """Crea vista holográfica completa"""
    fig = go.Figure()
    
    # Obtener datos de la red
    activation_state = consciousness_network._get_activation_state()
    
    # Posiciones 3D para nodos
    positions = calculate_3d_node_positions(consciousness_network)
    
    # Nodos con efecto holográfico
    node_x, node_y, node_z = [], [], []
    node_colors, node_sizes, node_texts = [], [], []
    
    for node_name, position in positions.items():
        x, y, z = position
        activation = activation_state.get(node_name, 0.5)
        
        node_x.append(x)
        node_y.append(y)
        node_z.append(z)
        
        # Colores holográficos basados en activación
        hue = (activation * 240) % 360  # De azul a rojo
        color = f"hsl({hue}, 80%, {50 + intensity * 30}%)"
        node_colors.append(color)
        
        node_sizes.append(20 + activation * 30)
        node_texts.append(f"{node_name}<br>Activación: {activation:.3f}")
    
    # Añadir nodos
    fig.add_trace(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(
            size=node_sizes,
            color=node_colors,
            opacity=intensity,
            line=dict(width=2, color='white')
        ),
        text=node_texts,
        hoverinfo='text',
        name='Nodos Neurales'
    ))
    
    # Conexiones holográficas
    add_holographic_connections(fig, consciousness_network, positions, intensity)
    
    fig.update_layout(
        title="Vista Holográfica Completa - Ruth R1",
        scene=dict(
            bgcolor='rgba(0,0,0,0.95)',
            xaxis=dict(showgrid=True, gridcolor='rgba(78,205,196,0.3)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(78,205,196,0.3)'),
            zaxis=dict(showgrid=True, gridcolor='rgba(78,205,196,0.3)')
        ),
        template="plotly_dark"
    )
    
    return fig

def create_holographic_core_view(consciousness_network, intensity=0.7):
    """Crea vista del núcleo central"""
    fig = go.Figure()
    
    # Enfocar en módulos centrales
    core_modules = ['GANSLSTMCore', 'IntrospectionEngine', 'PhilosophicalCore']
    
    # Similar implementación pero enfocada en el núcleo
    activation_state = consciousness_network._get_activation_state()
    positions = calculate_3d_node_positions(consciousness_network)
    
    for module in core_modules:
        if module in positions and module in activation_state:
            x, y, z = positions[module]
            activation = activation_state[module]
            
            # Núcleo con efecto de pulsación
            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z],
                mode='markers',
                marker=dict(
                    size=40 + activation * 20,
                    color=f'rgba(255, 215, 0, {intensity})',  # Dorado
                    symbol='diamond',
                    line=dict(width=3, color='white')
                ),
                name=f'Núcleo: {module}',
                text=f"{module}<br>Núcleo Central<br>Activación: {activation:.3f}"
            ))
    
    fig.update_layout(
        title="Vista del Núcleo Central - Ruth R1",
        scene=dict(bgcolor='rgba(0,0,0,0.95)'),
        template="plotly_dark"
    )
    
    return fig

def create_holographic_layers_view(consciousness_network, intensity=0.7):
    """Crea vista por capas"""
    fig = go.Figure()
    
    # Definir capas
    layers = {
        'Núcleo': ['GANSLSTMCore', 'IntrospectionEngine', 'PhilosophicalCore'],
        'Procesamiento': ['InnovationEngine', 'DreamMechanism'],
        'Análisis': ['EmotionDecomposer', 'ExistentialAnalyzer'],
        'Aplicación': ['CodeSuggester', 'ToolOptimizer']
    }
    
    colors = ['gold', 'cyan', 'lime', 'magenta']
    
    activation_state = consciousness_network._get_activation_state()
    positions = calculate_3d_node_positions(consciousness_network)
    
    for i, (layer_name, modules) in enumerate(layers.items()):
        layer_x, layer_y, layer_z = [], [], []
        layer_texts = []
        
        for module in modules:
            if module in positions and module in activation_state:
                x, y, z = positions[module]
                activation = activation_state[module]
                
                layer_x.append(x)
                layer_y.append(y)
                layer_z.append(z + i * 5)  # Separar capas verticalmente
                layer_texts.append(f"{module}<br>Capa: {layer_name}<br>Activación: {activation:.3f}")
        
        if layer_x:
            fig.add_trace(go.Scatter3d(
                x=layer_x, y=layer_y, z=layer_z,
                mode='markers',
                marker=dict(
                    size=25,
                    color=colors[i],
                    opacity=intensity,
                    symbol='circle'
                ),
                text=layer_texts,
                name=f'Capa {layer_name}'
            ))
    
    fig.update_layout(
        title="Vista por Capas Funcionales - Ruth R1",
        scene=dict(bgcolor='rgba(0,0,0,0.95)'),
        template="plotly_dark"
    )
    
    return fig

def create_holographic_connections_view(consciousness_network, intensity=0.7):
    """Crea vista enfocada en conexiones"""
    fig = go.Figure()
    
    activation_state = consciousness_network._get_activation_state()
    positions = calculate_3d_node_positions(consciousness_network)
    
    # Solo conexiones activas
    for edge in consciousness_network.network_graph.edges():
        source, target = edge
        if source in positions and target in positions:
            source_activation = activation_state.get(source, 0)
            target_activation = activation_state.get(target, 0)
            
            if source_activation > 0.3 and target_activation > 0.3:
                x0, y0, z0 = positions[source]
                x1, y1, z1 = positions[target]
                
                connection_strength = (source_activation + target_activation) / 2
                
                fig.add_trace(go.Scatter3d(
                    x=[x0, x1], y=[y0, y1], z=[z0, z1],
                    mode='lines',
                    line=dict(
                        color=f'rgba(78, 205, 196, {intensity * connection_strength})',
                        width=3 + connection_strength * 5
                    ),
                    showlegend=False,
                    hoverinfo='none'
                ))
    
    fig.update_layout(
        title="Vista de Conexiones Activas - Ruth R1",
        scene=dict(bgcolor='rgba(0,0,0,0.95)'),
        template="plotly_dark"
    )
    
    return fig

def create_basic_3d_network_view(consciousness_network):
    """Crea vista 3D básica como respaldo"""
    fig = go.Figure()
    
    activation_state = consciousness_network._get_activation_state()
    positions = calculate_3d_node_positions(consciousness_network)
    
    # Nodos básicos
    node_x = [pos[0] for pos in positions.values()]
    node_y = [pos[1] for pos in positions.values()]
    node_z = [pos[2] for pos in positions.values()]
    node_activations = [activation_state.get(name, 0.5) for name in positions.keys()]
    
    fig.add_trace(go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(
            size=20,
            color=node_activations,
            colorscale='Viridis',
            showscale=True
        ),
        text=list(positions.keys()),
        name='Red Neural'
    ))
    
    fig.update_layout(
        title="Vista 3D Básica - Ruth R1",
        template="plotly_dark"
    )
    
    return fig

def calculate_3d_node_positions(consciousness_network):
    """Calcula posiciones 3D para los nodos"""
    try:
        pos_2d = nx.spring_layout(consciousness_network.network_graph, k=3, iterations=50)
    except:
        # Posiciones por defecto si el grafo está vacío
        pos_2d = {}
        nodes = list(consciousness_network.nodes.keys())
        for i, node in enumerate(nodes):
            angle = (i / len(nodes)) * 2 * np.pi
            pos_2d[node] = (np.cos(angle) * 10, np.sin(angle) * 10)
    
    # Convertir a 3D
    positions_3d = {}
    for node_name, (x, y) in pos_2d.items():
        # Asignar Z basado en tipo de módulo
        if 'Core' in node_name:
            z = 0
        elif 'Engine' in node_name:
            z = 5
        elif 'Analyzer' in node_name:
            z = 10
        else:
            z = 15
            
        positions_3d[node_name] = (x, y, z)
    
    return positions_3d

def add_holographic_connections(fig, consciousness_network, positions, intensity):
    """Añade conexiones holográficas"""
    activation_state = consciousness_network._get_activation_state()
    
    for edge in consciousness_network.network_graph.edges():
        source, target = edge
        if source in positions and target in positions:
            x0, y0, z0 = positions[source]
            x1, y1, z1 = positions[target]
            
            source_activation = activation_state.get(source, 0.5)
            target_activation = activation_state.get(target, 0.5)
            avg_activation = (source_activation + target_activation) / 2
            
            fig.add_trace(go.Scatter3d(
                x=[x0, x1], y=[y0, y1], z=[z0, z1],
                mode='lines',
                line=dict(
                    color=f'rgba(78, 205, 196, {intensity * avg_activation * 0.7})',
                    width=2
                ),
                showlegend=False,
                hoverinfo='none'
            ))

def display_node_detailed_info(consciousness_network, node_name):
    """Muestra información detallada de un nodo"""
    if node_name in consciousness_network.nodes:
        node = consciousness_network.nodes[node_name]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**📍 Nodo: {node_name}**")
            st.metric("Activación", f"{node.activation_state:.3f}")
            st.metric("Belief Posterior", f"{node.posterior_belief:.3f}")
            st.metric("Belief Prior", f"{node.prior_belief:.3f}")
            
        with col2:
            st.markdown("**🔗 Conexiones:**")
            if hasattr(node, 'connections'):
                for connected_node, weight in list(node.connections.items())[:5]:
                    st.write(f"• {connected_node.name}: {weight:.3f}")
            else:
                st.write("No hay información de conexiones disponible")
                
            st.markdown("**📊 Historial de Evidencia:**")
            if hasattr(node, 'evidence_history') and node.evidence_history:
                recent_evidence = list(node.evidence_history)[-3:]
                for i, evidence in enumerate(recent_evidence):
                    st.write(f"{i+1}. {evidence.get('evidence', 'N/A'):.3f}")
            else:
                st.write("No hay historial de evidencia")