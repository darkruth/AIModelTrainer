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
    from models.supermodelo_meta_enrutador import (
        create_ruth_r1_system, 
        process_consciousness_input,
        create_default_config,
        RuthR1ConsciousnessCore
    )
    from core.consciousness import ConsciousnessState
    from core.neurotransmitters import NeurotransmitterSystem
    from core.quantum_processing import QuantumProcessor
    from algorithms.bayesian_quantum import BayesianQuantumSystem
    from core.despertar_awakening import (
        initiate_system_awakening, 
        get_awakening_system, 
        get_current_awakening_status,
        AwakeningPhase
    )
    from core.ganst_core import get_ganst_core, get_system_statistics
    from core.moduladores import get_modulation_manager
    from core.memorias_corto_plazo import get_short_term_memory
    from database.models import DatabaseManager, db_manager
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
        
        # Inicializar base de datos
        try:
            db_manager.create_tables()
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.warning(f"Database initialization failed: {e}")
        
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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "🌅 Despertar del Sistema",
        "💬 Consciencia Interactive", 
        "🧠 Monitoreo Neural", 
        "🌐 Visualización 3D Neural",
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
        display_3d_neural_visualization(consciousness_network)
    
    with tab5:
        display_bayesian_analysis(consciousness_network)
    
    with tab6:
        display_emotional_states(consciousness_network)
    
    with tab7:
        display_database_management()
    
    with tab8:
        display_razonbill_interface(consciousness_network)
    
    with tab9:
        display_meta_enrutador_interface(consciousness_network)
    
    with tab10:
        display_system_diagnostics(system)

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
            episódica: 5min, decay: 0.05
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
                
                # Guardar en base de datos
                try:
                    # Crear sesión si no existe
                    if 'current_session_id' not in st.session_state:
                        st.session_state.current_session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        
                        # Crear sesión en DB
                        session_data = {
                            'session_id': st.session_state.current_session_id,
                            'consciousness_level': response['consciousness_level'],
                            'coherence_metrics': response.get('network_coherence', {}),
                            'active_modules': response['active_modules'],
                            'emotional_state': response.get('emotional_state', {})
                        }
                        db_manager.save_consciousness_session(session_data)
                    
                    # Guardar interacción
                    interaction_data = {
                        'session_id': st.session_state.current_session_id,
                        'user_input': user_input,
                        'ruth_response': response['primary_response'],
                        'processing_mode': processing_mode.lower(),
                        'consciousness_level': response['consciousness_level'],
                        'emotional_sensitivity': emotional_sensitivity,
                        'active_modules': response['active_modules'],
                        'processing_time': processing_time,
                        'context_data': context
                    }
                    db_manager.save_user_interaction(interaction_data)
                    
                except Exception as e:
                    st.sidebar.warning(f"Database save failed: {str(e)[:50]}...")
                
                # Agregar al historial en memoria
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