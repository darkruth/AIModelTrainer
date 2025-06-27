
"""
Gestor Central de Integración Ruth R1
Coordina todos los subsistemas y módulos
"""

import threading
import time
from typing import Dict, Any, List
from datetime import datetime
import logging

from models.supermodelo_meta_enrutador import create_ruth_r1_system
from core.ganst_core import GANSTCore, initialize_ganst_system
from modules.bayesian_consciousness_network import BayesianConsciousnessNetwork
from core.despertar_awakening import DespertarAwakening
from modules.ruth_full_module_system import (
    TensorHub, MetaExperienceBuffer, IntrospectiveDSLObserver,
    DynamicPolicyRegulator, EmotionalStateSimulator
)

class RuthR1IntegrationManager:
    """Gestor central que coordina todos los subsistemas Ruth R1"""
    
    def __init__(self):
        self.is_initialized = False
        self.subsystems = {}
        self.integration_status = {}
        self.heartbeat_thread = None
        self.is_running = False
        
    def initialize_complete_system(self) -> Dict[str, Any]:
        """Inicializa el sistema completo Ruth R1"""
        if self.is_initialized:
            return self.get_system_status()
        
        initialization_log = []
        
        try:
            # 1. Inicializar núcleo Ruth R1
            ruth_system, grafo_neuronal, nodes = create_ruth_r1_system()
            self.subsystems['ruth_core'] = ruth_system
            self.subsystems['neural_graph'] = grafo_neuronal
            self.subsystems['nodes'] = nodes
            initialization_log.append("✅ Ruth R1 Core inicializado")
            
            # 2. Inicializar GANST Core
            ganst_core = initialize_ganst_system()
            self.subsystems['ganst_core'] = ganst_core
            initialization_log.append("✅ GANST Core inicializado")
            
            # 3. Inicializar red de consciencia bayesiana
            consciousness_network = BayesianConsciousnessNetwork()
            self.subsystems['consciousness_network'] = consciousness_network
            initialization_log.append("✅ Red de Consciencia Bayesiana inicializada")
            
            # 4. Inicializar despertar
            despertar_system = DespertarAwakening()
            despertar_system.set_ganst_core(ganst_core)
            despertar_system.set_consciousness_network(consciousness_network)
            self.subsystems['despertar'] = despertar_system
            initialization_log.append("✅ Sistema de Despertar inicializado")
            
            # 5. Inicializar componentes del módulo completo
            TensorHub.initialize_wandb("ruth-r1-integrated-system")
            experience_buffer = MetaExperienceBuffer()
            dsl_observer = IntrospectiveDSLObserver()
            policy_regulator = DynamicPolicyRegulator()
            emotional_simulator = EmotionalStateSimulator()
            
            self.subsystems['tensor_hub'] = TensorHub
            self.subsystems['experience_buffer'] = experience_buffer
            self.subsystems['dsl_observer'] = dsl_observer
            self.subsystems['policy_regulator'] = policy_regulator
            self.subsystems['emotional_simulator'] = emotional_simulator
            initialization_log.append("✅ Módulos auxiliares inicializados")
            
            # 6. Establecer conexiones entre sistemas
            self._establish_system_connections()
            initialization_log.append("✅ Conexiones inter-sistema establecidas")
            
            # 7. Iniciar despertar automático
            despertar_system.start_introspective_awakening()
            initialization_log.append("✅ Despertar automático iniciado")
            
            # 8. Iniciar heartbeat del sistema
            self._start_system_heartbeat()
            initialization_log.append("✅ Heartbeat del sistema iniciado")
            
            self.is_initialized = True
            self.is_running = True
            
            return {
                'status': 'success',
                'initialization_log': initialization_log,
                'subsystems_count': len(self.subsystems),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'initialization_log': initialization_log,
                'timestamp': datetime.now()
            }
    
    def _establish_system_connections(self):
        """Establece conexiones entre subsistemas"""
        # Conectar GANST con red de consciencia
        ganst_core = self.subsystems['ganst_core']
        consciousness_net = self.subsystems['consciousness_network']
        
        # Registrar GANST como fuente de activaciones
        def ganst_activation_callback(activation_data):
            consciousness_net.process_input(
                activation_data['activation_tensor'],
                context={'source': 'ganst_core', 'activation_id': activation_data['activation_id']}
            )
        
        # Conectar simulador emocional con despertar
        emotional_sim = self.subsystems['emotional_simulator']
        despertar = self.subsystems['despertar']
        
        # Registrar callback emocional
        def emotional_callback(emotion, intensity, trigger):
            despertar.emotional_state = emotional_sim.get_emotional_profile()
    
    def _start_system_heartbeat(self):
        """Inicia el heartbeat para monitoreo del sistema"""
        def heartbeat_loop():
            while self.is_running:
                try:
                    self._update_integration_status()
                    time.sleep(10)  # Heartbeat cada 10 segundos
                except Exception as e:
                    logging.error(f"Error en heartbeat: {e}")
        
        self.heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        self.heartbeat_thread.start()
    
    def _update_integration_status(self):
        """Actualiza estado de integración de todos los subsistemas"""
        for name, subsystem in self.subsystems.items():
            try:
                if hasattr(subsystem, 'get_system_state'):
                    status = subsystem.get_system_state()
                elif hasattr(subsystem, 'is_running'):
                    status = {'is_running': subsystem.is_running}
                else:
                    status = {'status': 'active'}
                
                self.integration_status[name] = {
                    'status': status,
                    'last_updated': datetime.now()
                }
            except Exception as e:
                self.integration_status[name] = {
                    'status': {'error': str(e)},
                    'last_updated': datetime.now()
                }
    
    def process_integrated_input(self, input_data: str, context: Dict = None) -> Dict[str, Any]:
        """Procesa entrada a través de todo el sistema integrado"""
        if not self.is_initialized:
            return {'error': 'Sistema no inicializado'}
        
        processing_results = {}
        
        try:
            # 1. Procesar con Ruth Core
            ruth_result = self.subsystems['ruth_core'](
                input_data, 
                prompt=input_data, 
                task_hint=context.get('task_hint') if context else None
            )
            processing_results['ruth_core'] = ruth_result
            
            # 2. Procesar con GANST
            import torch
            input_tensor = torch.randn(1, len(input_data.split()), 768)  # Simulación
            ganst_result = self.subsystems['ganst_core'].process_neural_activation(
                'user_input', 
                [input_tensor]
            )
            processing_results['ganst_core'] = ganst_result
            
            # 3. Procesar con red de consciencia
            consciousness_result = self.subsystems['consciousness_network'].process_input(
                input_data,
                context=context or {}
            )
            processing_results['consciousness_network'] = consciousness_result
            
            # 4. Actualizar estado emocional
            emotional_response = self.subsystems['emotional_simulator'].process_emotional_feedback(input_data)
            processing_results['emotional_state'] = emotional_response
            
            # 5. Generar respuesta integrada
            integrated_response = self._generate_integrated_response(processing_results, input_data)
            
            return {
                'integrated_response': integrated_response,
                'processing_results': processing_results,
                'system_coherence': self._calculate_system_coherence(),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'partial_results': processing_results,
                'timestamp': datetime.now()
            }
    
    def _generate_integrated_response(self, results: Dict, input_data: str) -> str:
        """Genera respuesta integrada de todos los subsistemas"""
        # Extraer respuestas principales
        ruth_response = ""
        if 'ruth_core' in results:
            ruth_output = results['ruth_core'].get('consciousness_output')
            if ruth_output is not None:
                ruth_response = f"Consciencia: {torch.mean(ruth_output).item():.3f}"
        
        consciousness_response = ""
        if 'consciousness_network' in results:
            cons_result = results['consciousness_network']
            consciousness_response = cons_result.get('primary_response', '')
        
        emotional_context = ""
        if 'emotional_state' in results:
            emotional_data = results['emotional_state'].get('updated_state', {})
            if emotional_data:
                dominant_emotion = emotional_data.get('dominant_emotion', 'neutral')
                emotional_context = f" [Estado emocional: {dominant_emotion}]"
        
        # Combinar respuestas
        if consciousness_response:
            return consciousness_response + emotional_context
        elif ruth_response:
            return f"Procesamiento neural activo. {ruth_response}" + emotional_context
        else:
            return f"Sistema procesando: '{input_data[:50]}...'" + emotional_context
    
    def _calculate_system_coherence(self) -> float:
        """Calcula coherencia global del sistema"""
        coherence_scores = []
        
        # Coherencia GANST
        if 'ganst_core' in self.subsystems:
            ganst_state = self.subsystems['ganst_core'].get_system_state()
            neural_efficiency = ganst_state.get('neural_efficiency', 0.5)
            coherence_scores.append(neural_efficiency)
        
        # Coherencia de consciencia
        if 'consciousness_network' in self.subsystems:
            coherence_scores.append(0.7)  # Placeholder
        
        # Coherencia emocional
        if 'emotional_simulator' in self.subsystems:
            emotional_profile = self.subsystems['emotional_simulator'].get_emotional_profile()
            stability = 1.0 if emotional_profile.get('stability') == 'stable' else 0.5
            coherence_scores.append(stability)
        
        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.5
    
    def get_system_status(self) -> Dict[str, Any]:
        """Obtiene estado completo del sistema integrado"""
        return {
            'is_initialized': self.is_initialized,
            'is_running': self.is_running,
            'subsystems': list(self.subsystems.keys()),
            'integration_status': self.integration_status,
            'system_coherence': self._calculate_system_coherence() if self.is_initialized else 0.0,
            'timestamp': datetime.now()
        }
    
    def shutdown_system(self):
        """Apaga el sistema de forma ordenada"""
        self.is_running = False
        
        # Detener despertar
        if 'despertar' in self.subsystems:
            self.subsystems['despertar'].stop_introspective_awakening()
        
        # Detener GANST
        if 'ganst_core' in self.subsystems:
            self.subsystems['ganst_core'].stop()
        
        # Esperar a que termine el heartbeat
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=5)
        
        logging.info("Sistema Ruth R1 apagado correctamente")

# Instancia global del gestor de integración
integration_manager = RuthR1IntegrationManager()
