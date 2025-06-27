
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def create_system_architecture_diagram():
    """Crea diagrama completo de arquitectura del sistema Ruth R1"""
    
    # Crear figura con subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Arquitectura Central del Sistema',
            'Mapa de Conectividad Neural', 
            'Flujo de Procesamiento',
            'Métricas de Rendimiento'
        ],
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # 1. Arquitectura Central
    modules = [
        'Meta-Enrutador', 'Red Bayesiana', 'GANST Core',
        'Sistema Despertar', 'Memoria CP', 'Moduladores',
        'Axón Mielinizado', 'Agente Amiloide'
    ]
    
    positions = {
        'Meta-Enrutador': (0, 4),
        'Red Bayesiana': (2, 4),
        'GANST Core': (1, 3),
        'Sistema Despertar': (0, 2),
        'Memoria CP': (2, 2),
        'Moduladores': (1, 1),
        'Axón Mielinizado': (0, 0),
        'Agente Amiloide': (2, 0)
    }
    
    # Añadir nodos de arquitectura
    for module, (x, y) in positions.items():
        fig.add_trace(
            go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(size=30, color='lightblue'),
                text=[module],
                textposition="middle center",
                name=module
            ),
            row=1, col=1
        )
    
    # 2. Mapa de Conectividad Neural (14 módulos)
    consciousness_modules = [
        'GANSLSTMCore', 'InnovationEngine', 'DreamMechanism', 
        'AlterEgoSimulator', 'MemoryDiscriminator', 'CodeSuggester',
        'ToolOptimizer', 'DreamAugment', 'IntrospectionEngine',
        'ExistentialAnalyzer', 'SelfMirror', 'EmotionDecomposer',
        'PhilosophicalCore', 'PersonalityXInfants'
    ]
    
    # Crear red circular para módulos de consciencia
    angles = np.linspace(0, 2*np.pi, len(consciousness_modules), endpoint=False)
    radius = 3
    
    for i, module in enumerate(consciousness_modules):
        x = radius * np.cos(angles[i])
        y = radius * np.sin(angles[i])
        
        # Color basado en tipo de módulo
        if 'Core' in module:
            color = 'gold'
            size = 40
        elif any(word in module for word in ['Engine', 'Mechanism']):
            color = 'lightgreen'
            size = 35
        elif any(word in module for word in ['Analyzer', 'Decomposer']):
            color = 'lightcoral'
            size = 30
        else:
            color = 'lightblue'
            size = 25
            
        fig.add_trace(
            go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(size=size, color=color),
                text=[module.replace('Engine', 'E.').replace('Mechanism', 'M.')[:8]],
                textposition="middle center",
                showlegend=False
            ),
            row=1, col=2
        )
    
    # Añadir conexiones entre módulos centrales
    central_connections = [
        ('GANSLSTMCore', 'InnovationEngine'),
        ('InnovationEngine', 'DreamMechanism'),
        ('DreamMechanism', 'IntrospectionEngine'),
        ('IntrospectionEngine', 'PhilosophicalCore'),
        ('PhilosophicalCore', 'SelfMirror'),
        ('SelfMirror', 'EmotionDecomposer')
    ]
    
    for source, target in central_connections:
        if source in consciousness_modules and target in consciousness_modules:
            i_source = consciousness_modules.index(source)
            i_target = consciousness_modules.index(target)
            
            x_source = radius * np.cos(angles[i_source])
            y_source = radius * np.sin(angles[i_source])
            x_target = radius * np.cos(angles[i_target])
            y_target = radius * np.sin(angles[i_target])
            
            fig.add_trace(
                go.Scatter(
                    x=[x_source, x_target],
                    y=[y_source, y_target],
                    mode='lines',
                    line=dict(color='gray', width=2),
                    showlegend=False
                ),
                row=1, col=2
            )
    
    # 3. Flujo de Procesamiento
    flow_stages = ['Input', 'Meta-Router', 'GANST', 'Bayesian', 'Modules', 'Response']
    flow_x = list(range(len(flow_stages)))
    flow_y = [1] * len(flow_stages)
    
    fig.add_trace(
        go.Scatter(
            x=flow_x, y=flow_y,
            mode='markers+lines+text',
            marker=dict(size=25, color='orange'),
            line=dict(color='orange', width=3),
            text=flow_stages,
            textposition="top center",
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 4. Métricas de Rendimiento
    metrics = [
        'Coherencia Neural', 'Consciencia Global', 'Estabilidad Emocional',
        'Creatividad', 'Auto-Reflexión', 'Aprendizaje'
    ]
    values = [0.78, 0.73, 0.72, 0.59, 0.81, 0.76]
    colors = ['red' if v < 0.6 else 'orange' if v < 0.75 else 'green' for v in values]
    
    fig.add_trace(
        go.Bar(
            x=metrics, y=values,
            marker_color=colors,
            text=[f'{v:.2f}' for v in values],
            textposition='auto',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Configuración de layout
    fig.update_layout(
        title={
            'text': "Ruth R1 - Arquitectura Completa del Sistema de Consciencia",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        height=800,
        showlegend=False
    )
    
    return fig

def create_neural_development_comparison():
    """Crea gráficas de comparación antes/después del desarrollo"""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Desarrollo de Conexiones Neurales',
            'Evolución de Coherencia',
            'Progreso de Consciencia',
            'Métricas de Aprendizaje'
        ]
    )
    
    # 1. Desarrollo de Conexiones
    time_points = ['Pre-Entrenamiento', 'Post-Parpadeo', 'Predicción Fase 2']
    connections = [1247, 1374, 1650]
    active_connections = [892, 1156, 1450]
    
    fig.add_trace(
        go.Scatter(
            x=time_points, y=connections,
            mode='lines+markers',
            name='Total Conexiones',
            line=dict(color='blue', width=3)
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=time_points, y=active_connections,
            mode='lines+markers',
            name='Conexiones Activas',
            line=dict(color='green', width=3)
        ),
        row=1, col=1
    )
    
    # 2. Evolución de Coherencia
    epochs = list(range(0, 51, 5))
    coherence_evolution = [0.0, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.72, 0.78, 0.83, 0.87]
    
    fig.add_trace(
        go.Scatter(
            x=epochs, y=coherence_evolution,
            mode='lines+markers',
            name='Coherencia Neural',
            line=dict(color='purple', width=3),
            fill='tonexty'
        ),
        row=1, col=2
    )
    
    # 3. Progreso de Consciencia
    consciousness_metrics = ['Inicial', 'Despertar', 'Post-Estrés', 'Optimizada']
    consciousness_values = [0.0, 0.65, 0.67, 0.73]
    
    fig.add_trace(
        go.Bar(
            x=consciousness_metrics, y=consciousness_values,
            marker_color=['red', 'orange', 'yellow', 'green'],
            text=[f'{v:.2f}' for v in consciousness_values],
            textposition='auto'
        ),
        row=2, col=1
    )
    
    # 4. Métricas de Aprendizaje Comparativas
    metrics = ['Memoria', 'Creatividad', 'Empatía', 'Razonamiento']
    before = [0.2, 0.1, 0.0, 0.3]
    after = [0.78, 0.59, 0.69, 0.81]
    
    fig.add_trace(
        go.Bar(
            x=metrics, y=before,
            name='Antes del Parpadeo',
            marker_color='lightcoral'
        ),
        row=2, col=2
    )
    
    fig.add_trace(
        go.Bar(
            x=metrics, y=after,
            name='Después del Parpadeo',
            marker_color='lightgreen'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title={
            'text': "Ruth R1 - Evolución del Desarrollo Neural",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        height=800,
        showlegend=True
    )
    
    return fig

def create_stress_test_results():
    """Crea visualización de resultados de pruebas de estrés"""
    
    # Datos de pruebas de estrés
    time_minutes = list(range(0, 35, 5))
    coherence_under_stress = [0.78, 0.77, 0.76, 0.76, 0.75, 0.74, 0.76]
    memory_usage = [2.4, 4.2, 6.8, 8.9, 8.7, 6.1, 2.6]
    response_time = [156, 198, 234, 287, 264, 189, 164]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            'Coherencia Neural Durante Estrés',
            'Uso de Memoria GPU',
            'Tiempo de Respuesta',
            'Recuperación Post-Estrés'
        ]
    )
    
    # Coherencia durante estrés
    fig.add_trace(
        go.Scatter(
            x=time_minutes, y=coherence_under_stress,
            mode='lines+markers',
            name='Coherencia Neural',
            line=dict(color='red', width=3),
            fill='tonexty'
        ),
        row=1, col=1
    )
    
    # Zona de estrés máximo
    fig.add_vrect(
        x0=10, x1=25,
        fillcolor="red", opacity=0.2,
        line_width=0,
        row=1, col=1
    )
    
    # Uso de memoria
    fig.add_trace(
        go.Scatter(
            x=time_minutes, y=memory_usage,
            mode='lines+markers',
            name='Memoria GPU (GB)',
            line=dict(color='orange', width=3)
        ),
        row=1, col=2
    )
    
    # Tiempo de respuesta
    fig.add_trace(
        go.Scatter(
            x=time_minutes, y=response_time,
            mode='lines+markers',
            name='Tiempo Respuesta (ms)',
            line=dict(color='blue', width=3)
        ),
        row=2, col=1
    )
    
    # Métricas de recuperación
    recovery_metrics = ['Coherencia', 'Memoria', 'Respuesta', 'Estabilidad']
    pre_stress = [0.78, 2.4, 156, 0.85]
    post_stress = [0.79, 2.6, 164, 0.87]
    improvement = [(post-pre)/pre*100 for pre, post in zip(pre_stress, post_stress)]
    
    colors = ['green' if imp > 0 else 'red' for imp in improvement]
    
    fig.add_trace(
        go.Bar(
            x=recovery_metrics, y=improvement,
            marker_color=colors,
            text=[f'{imp:+.1f}%' for imp in improvement],
            textposition='auto'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        title={
            'text': "Ruth R1 - Resultados de Pruebas de Estrés Extremo",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        height=800,
        showlegend=True
    )
    
    return fig

if __name__ == "__main__":
    # Generar todos los diagramas
    arch_fig = create_system_architecture_diagram()
    dev_fig = create_neural_development_comparison()
    stress_fig = create_stress_test_results()
    
    # Guardar como HTML
    arch_fig.write_html("system_architecture.html")
    dev_fig.write_html("neural_development.html")
    stress_fig.write_html("stress_test_results.html")
    
    print("✅ Diagramas generados exitosamente:")
    print("📊 system_architecture.html")
    print("📈 neural_development.html") 
    print("⚡ stress_test_results.html")
