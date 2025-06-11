"""
Configuración del Sistema Ruth R1
"""

from typing import Dict, Any

class Config:
    """Configuración central del sistema"""
    
    def __init__(self):
        self.consciousness_config = {
            'default_level': 0.5,
            'update_rate': 0.1,
            'coherence_threshold': 0.7
        }
        
        self.neurotransmitter_config = {
            'update_frequency': 100,
            'default_levels': {
                'dopamine': 0.5,
                'serotonin': 0.5,
                'acetylcholine': 0.5,
                'norepinephrine': 0.5,
                'gaba': 0.5
            }
        }
        
        self.quantum_config = {
            'n_qubits': 8,
            'coherence_time': 1000,
            'entanglement_threshold': 0.7
        }
        
        self.multimodal_config = {
            'vision_enabled': True,
            'audio_enabled': True,
            'text_processing': True
        }
    
    def get_consciousness_config(self) -> Dict[str, Any]:
        return self.consciousness_config
    
    def get_neurotransmitter_config(self) -> Dict[str, Any]:
        return self.neurotransmitter_config
    
    def get_quantum_config(self) -> Dict[str, Any]:
        return self.quantum_config
    
    def get_multimodal_config(self) -> Dict[str, Any]:
        return self.multimodal_config