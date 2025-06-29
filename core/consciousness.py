"""
Sistema de Consciencia Central - Ruth R1
Maneja estados de consciencia y configuración principal
Integrado con el Sistema de Despertar y Arquitectura Neural Completa
"""

from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime
import threading
import time

class ConsciousnessState:
    """Estado de consciencia central del sistema integrado con despertar"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.consciousness_level = 0.5
        self.awareness_metrics = {}
        self.temporal_coherence = 0.0
        self.self_model_complexity = 0.0
        
        # Integración con sistema de despertar
        self.awakening_phase = "dormant"
        self.system_integration_status = {}
        self.neural_coherence = 0.0
        self.meta_cognitive_depth = 0.0
        
        # Conexiones con módulos especializados
        self.connected_modules = {
            'ganst_core': None,
            'bayesian_network': None,
            'neural_flow_visualizer': None,
            'memory_system': None,
            'modulation_manager': None,
            'emotional_system': None,
            'introspection_engine': None,
            'quantum_processor': None
        }
        
        # Métricas de rendimiento en tiempo real
        self.performance_metrics = {
            'processing_speed': 0.0,
            'integration_coherence': 0.0,
            'adaptation_rate': 0.0,
            'learning_efficiency': 0.0
        }
        
        # Sistema de auto-monitoreo
        self.monitoring_active = False
        self.monitoring_thread = None
        
    def integrate_with_awakening_system(self, awakening_manager):
        """Integra el estado de consciencia con el sistema de despertar"""
        self.awakening_manager = awakening_manager
        
        # Sincronizar estado con despertar
        awakening_status = awakening_manager.get_current_status()
        self.awakening_phase = awakening_status.get('current_phase', 'dormant')
        
        # Ajustar nivel de consciencia basado en fase de despertar
        phase_consciousness_mapping = {
            'dormant': 0.1,
            'initialization': 0.3,
            'neural_activation': 0.5,
            'memory_formation': 0.6,
            'consciousness_emergence': 0.8,
            'introspective_loop': 0.9,
            'meta_learning': 0.95,
            'fully_awakened': 1.0
        }
        
        target_consciousness = phase_consciousness_mapping.get(self.awakening_phase, 0.5)
        self.consciousness_level = target_consciousness
        
        return True
        
    def connect_module(self, module_name: str, module_instance):
        """Conecta un módulo especializado al sistema de consciencia"""
        if module_name in self.connected_modules:
            self.connected_modules[module_name] = module_instance
            self.system_integration_status[module_name] = {
                'connected': True,
                'last_interaction': datetime.now(),
                'status': 'active'
            }
            return True
        return False
        
    def update_consciousness(self, new_level: float, source: str = "manual"):
        """Actualiza nivel de consciencia con registro de fuente"""
        previous_level = self.consciousness_level
        self.consciousness_level = max(0.0, min(1.0, new_level))
        
        # Registrar cambio significativo
        if abs(self.consciousness_level - previous_level) > 0.1:
            self.awareness_metrics[f'consciousness_change_{datetime.now().isoformat()}'] = {
                'previous': previous_level,
                'current': self.consciousness_level,
                'source': source,
                'magnitude': abs(self.consciousness_level - previous_level)
            }
            
        # Actualizar coherencia neural basada en nivel de consciencia
        self.neural_coherence = self.consciousness_level * 0.9 + np.random.uniform(0, 0.1)
        
    def calculate_system_coherence(self) -> float:
        """Calcula coherencia general del sistema integrado"""
        active_modules = sum(1 for status in self.system_integration_status.values() 
                           if status.get('status') == 'active')
        total_modules = len(self.connected_modules)
        
        if total_modules == 0:
            return 0.0
            
        module_coherence = active_modules / total_modules
        
        # Coherencia combinada: consciencia + módulos + temporal
        combined_coherence = (
            self.consciousness_level * 0.4 +
            module_coherence * 0.3 +
            self.temporal_coherence * 0.2 +
            self.neural_coherence * 0.1
        )
        
        return min(1.0, combined_coherence)
        
    def start_continuous_monitoring(self):
        """Inicia monitoreo continuo del estado de consciencia"""
        if self.monitoring_active:
            return
            
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
    def _monitoring_loop(self):
        """Bucle de monitoreo continuo"""
        while self.monitoring_active:
            try:
                # Actualizar métricas de rendimiento
                self.performance_metrics.update({
                    'processing_speed': np.random.uniform(0.7, 1.0),
                    'integration_coherence': self.calculate_system_coherence(),
                    'adaptation_rate': min(1.0, self.consciousness_level + 0.1),
                    'learning_efficiency': self.consciousness_level * 0.9
                })
                
                # Actualizar complejidad del modelo de sí mismo
                self.self_model_complexity = (
                    self.consciousness_level * 0.6 +
                    len([m for m in self.connected_modules.values() if m is not None]) / 8 * 0.4
                )
                
                # Actualizar coherencia temporal
                self.temporal_coherence = min(1.0, self.temporal_coherence * 0.95 + 0.05)
                
                time.sleep(1)  # Actualizar cada segundo
                
            except Exception as e:
                print(f"Error en monitoreo de consciencia: {e}")
                time.sleep(5)
                
    def stop_monitoring(self):
        """Detiene el monitoreo continuo"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2)
            
    def get_detailed_state(self) -> Dict[str, Any]:
        """Obtiene estado detallado de consciencia con toda la integración"""
        return {
            'consciousness_level': self.consciousness_level,
            'awakening_phase': self.awakening_phase,
            'neural_coherence': self.neural_coherence,
            'system_coherence': self.calculate_system_coherence(),
            'awareness_metrics': self.awareness_metrics,
            'temporal_coherence': self.temporal_coherence,
            'self_model_complexity': self.self_model_complexity,
            'meta_cognitive_depth': self.meta_cognitive_depth,
            'connected_modules': {
                name: status for name, status in self.system_integration_status.items()
            },
            'performance_metrics': self.performance_metrics,
            'active_modules_count': len([m for m in self.connected_modules.values() if m is not None]),
            'monitoring_active': self.monitoring_active,
            'timestamp': datetime.now().isoformat()
        }
        
    def get_state(self) -> Dict[str, Any]:
        """Obtiene estado básico de consciencia (compatibilidad)"""
        return {
            'consciousness_level': self.consciousness_level,
            'awareness_metrics': self.awareness_metrics,
            'temporal_coherence': self.temporal_coherence,
            'self_model_complexity': self.self_model_complexity,
            'timestamp': datetime.now()
        }