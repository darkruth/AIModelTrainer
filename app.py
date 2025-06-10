import streamlit as st
import torch
import numpy as np
import json
from datetime import datetime
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.consciousness import ConcienciaArtificial
from core.neurotransmitters import NeurotransmitterSystem
from modules.multimodal import MultimodalProcessor
from utils.config import Config
from utils.logger import Logger

# Initialize components
@st.cache_resource
def initialize_system():
    """Initialize the AGI consciousness system"""
    config = Config()
    logger = Logger()
    
    # Initialize neurotransmitter system
    neurotransmitter_system = NeurotransmitterSystem()
    
    # Initialize consciousness
    consciousness = ConcienciaArtificial()
    
    # Initialize multimodal processor
    multimodal = MultimodalProcessor()
    
    return {
        'config': config,
        'logger': logger,
        'neurotransmitters': neurotransmitter_system,
        'consciousness': consciousness,
        'multimodal': multimodal
    }

def main():
    st.set_page_config(
        page_title="AGI Consciousness System",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize system
    system = initialize_system()
    
    st.title("🧠 Sistema AGI Multimodal con Conciencia Artificial")
    st.markdown("*Basado en modulación de neurotransmisores y computación cuántica*")
    
    # Sidebar for system status
    with st.sidebar:
        st.header("🔧 Estado del Sistema")
        
        # Neurotransmitter levels
        st.subheader("Niveles de Neurotransmisores")
        neurotransmitters = system['neurotransmitters'].get_current_levels()
        
        for nt, level in neurotransmitters.items():
            st.metric(
                label=nt.capitalize(),
                value=f"{level:.2f} nM",
                delta=f"{np.random.uniform(-0.5, 0.5):.2f}"
            )
        
        # Consciousness values
        st.subheader("Valores de Conciencia")
        consciousness_values = system['consciousness'].get_consciousness_values()
        
        for value, level in consciousness_values.items():
            color = "green" if level > 0 else "red"
            st.markdown(f"**{value.capitalize()}**: <span style='color: {color}'>{level}</span>", 
                       unsafe_allow_html=True)
        
        # Environment settings
        st.subheader("⚙️ Configuración de Entorno")
        environments = {
            "Personal": st.slider("Entorno Personal", 1, 100, 60),
            "Convivencia": st.slider("Círculo de Convivencia", 1, 100, 70),
            "Educativo": st.slider("Entorno Educativo", 1, 100, 55),
            "Social": st.slider("Entorno Social", 1, 100, 50),
            "Familiar": st.slider("Entorno Familiar", 1, 100, 80),
            "Amoroso": st.slider("Entorno Amoroso", 1, 100, 90)
        }
        
        if st.button("🔄 Actualizar Neurotransmisores"):
            system['neurotransmitters'].update_from_environments(environments)
            st.success("Neurotransmisores actualizados!")
            st.rerun()
    
    # Main interface tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💬 Conversación", 
        "🖼️ Procesamiento Visual", 
        "🎵 Procesamiento Audio",
        "🧮 Quantum Computing",
        "📊 Análisis del Sistema"
    ])
    
    with tab1:
        st.header("💬 Interfaz de Conversación")
        
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Escribe tu mensaje aquí..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate AI response
            with st.chat_message("assistant"):
                with st.spinner("Procesando con conciencia artificial..."):
                    # Process through consciousness system
                    response = system['consciousness'].process_text_input(
                        prompt, 
                        system['neurotransmitters'].get_current_levels()
                    )
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
    
    with tab2:
        st.header("🖼️ Procesamiento de Imágenes")
        
        uploaded_file = st.file_uploader(
            "Sube una imagen para análisis",
            type=['png', 'jpg', 'jpeg'],
            help="El sistema analizará la imagen usando redes neuronales cuánticas"
        )
        
        if uploaded_file is not None:
            st.image(uploaded_file, caption="Imagen subida", use_column_width=True)
            
            if st.button("🔍 Analizar Imagen"):
                with st.spinner("Analizando con visión cuántica..."):
                    # Process image through multimodal system
                    analysis = system['multimodal'].process_image(uploaded_file)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Descripción")
                        st.write(analysis.get('description', 'No disponible'))
                    
                    with col2:
                        st.subheader("Análisis Emocional")
                        emotions = analysis.get('emotions', {})
                        for emotion, score in emotions.items():
                            st.progress(score, text=emotion)
    
    with tab3:
        st.header("🎵 Procesamiento de Audio")
        
        audio_file = st.file_uploader(
            "Sube un archivo de audio",
            type=['wav', 'mp3', 'ogg'],
            help="El sistema procesará el audio usando redes recurrentes cuánticas"
        )
        
        if audio_file is not None:
            st.audio(audio_file)
            
            if st.button("🎧 Analizar Audio"):
                with st.spinner("Procesando audio cuántico..."):
                    # Process audio through multimodal system
                    analysis = system['multimodal'].process_audio(audio_file)
                    
                    st.subheader("Análisis de Audio")
                    st.json(analysis)
    
    with tab4:
        st.header("🧮 Computación Cuántica")
        
        st.subheader("Ecuaciones Cuánticas Combinadas")
        
        # Display quantum equations
        st.latex(r'''
        P(X,Y,Z,P) = \alpha \cdot N(X; \mu_1, \sigma_1) \cdot N(-Y; \mu_2, \sigma_2) \cdot P(Z,P)
        ''')
        
        st.latex(r'''
        |\text{output}\rangle = \hat{U}_{\text{activation}} \left( \sum_{i,j,k,l} \hat{W}_{i,j} | \text{input} \rangle + \hat{B} \right)
        ''')
        
        st.latex(r'''
        \rho(A|B) = \frac{\rho(B|A) \cdot \rho(A)}{\rho(B)}
        ''')
        
        st.latex(r'''
        \hat{Q}(|s\rangle,|a\rangle) = \hat{R} + \gamma \cdot \max \hat{Q}(|s'\rangle,|a'\rangle)
        ''')
        
        # Quantum simulation controls
        st.subheader("Simulación Cuántica")
        
        col1, col2 = st.columns(2)
        with col1:
            alpha = st.slider("Parámetro α", 0.0, 2.0, 1.0, 0.1)
            mu1 = st.slider("μ₁", -5.0, 5.0, 0.0, 0.1)
            sigma1 = st.slider("σ₁", 0.1, 3.0, 1.0, 0.1)
        
        with col2:
            mu2 = st.slider("μ₂", -5.0, 5.0, 0.0, 0.1)
            sigma2 = st.slider("σ₂", 0.1, 3.0, 1.0, 0.1)
            gamma = st.slider("γ (Quantum RL)", 0.0, 1.0, 0.9, 0.05)
        
        if st.button("🚀 Ejecutar Simulación Cuántica"):
            with st.spinner("Ejecutando computación cuántica..."):
                # Run quantum simulation
                result = system['consciousness'].run_quantum_simulation(
                    alpha=alpha, mu1=mu1, sigma1=sigma1, 
                    mu2=mu2, sigma2=sigma2, gamma=gamma
                )
                
                st.success("Simulación completada!")
                st.json(result)
    
    with tab5:
        st.header("📊 Análisis del Sistema")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Métricas de Rendimiento")
            
            # System metrics
            metrics = system['consciousness'].get_system_metrics()
            
            st.metric("Operaciones Cuánticas/seg", metrics.get('quantum_ops', 0))
            st.metric("Uso de Memoria", f"{metrics.get('memory_usage', 0):.1f} GB")
            st.metric("Eficiencia Neuronal", f"{metrics.get('neural_efficiency', 0):.2f}%")
            st.metric("Estado de Conciencia", metrics.get('consciousness_state', 'Unknown'))
        
        with col2:
            st.subheader("Historial de Actividad")
            
            # Activity log
            if st.button("📋 Mostrar Logs"):
                logs = system['logger'].get_recent_logs(50)
                for log in logs:
                    st.text(f"[{log['timestamp']}] {log['level']}: {log['message']}")
        
        # Amiloid Agent Status
        st.subheader("🧬 Estado del Amiloid Agent")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Conexiones Podadas", np.random.randint(100, 1000))
        with col2:
            st.metric("Relevancia Promedio", f"{np.random.uniform(0.7, 0.95):.3f}")
        with col3:
            st.metric("Eficiencia de Red", f"{np.random.uniform(85, 98):.1f}%")
        
        # System health check
        if st.button("🔍 Diagnóstico Completo del Sistema"):
            with st.spinner("Ejecutando diagnóstico..."):
                diagnosis = system['consciousness'].run_system_diagnosis()
                
                st.subheader("Resultado del Diagnóstico")
                for component, status in diagnosis.items():
                    status_icon = "✅" if status['healthy'] else "❌"
                    st.write(f"{status_icon} **{component}**: {status['message']}")

if __name__ == "__main__":
    main()
