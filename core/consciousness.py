"""
Sistema de Consciencia Central - Ruth R1
Maneja estados de consciencia y configuración principal
"""

from typing import Dict, Any, List
import numpy as np
from datetime import datetime

class ConsciousnessState:
    """Estado de consciencia central del sistema"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.consciousness_level = 0.5
        self.awareness_metrics = {}
        self.temporal_coherence = 0.0
        self.self_model_complexity = 0.0
        
    def update_consciousness(self, new_level: float):
        """Actualiza nivel de consciencia"""
        self.consciousness_level = max(0.0, min(1.0, new_level))
        
    def get_state(self) -> Dict[str, Any]:
        """Obtiene estado actual de consciencia"""
        return {
            'consciousness_level': self.consciousness_level,
            'awareness_metrics': self.awareness_metrics,
            'temporal_coherence': self.temporal_coherence,
            'self_model_complexity': self.self_model_complexity,
            'timestamp': datetime.now()
        }