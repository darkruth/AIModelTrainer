import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from utils.logger import Logger

@dataclass
class NeurotransmitterConfig:
    """Configuration for individual neurotransmitters"""
    initial_level: float
    min_level: float
    max_level: float
    decay_rate: float
    impact_coefficients: Dict[str, float]

class NeurotransmitterSystem:
    """
    Sistema de modulación de neurotransmisores basado en el modelo del documento.
    Gestiona los niveles de Serotonina, Dopamina, Norepinefrina, Oxitocina y Endorfinas
    en función de diferentes entornos.
    """
    
    def __init__(self):
        self.logger = Logger()
        
        # Configuración de neurotransmisores basada en el documento
        self.neurotransmitters = {
            'serotonin': NeurotransmitterConfig(
                initial_level=7.0,  # nM
                min_level=1.0,
                max_level=20.0,
                decay_rate=0.95,
                impact_coefficients={'E1': 0.1, 'E5': 0.1}  # Entorno Personal y Familiar
            ),
            'dopamine': NeurotransmitterConfig(
                initial_level=50.0,  # nM
                min_level=10.0,
                max_level=100.0,
                decay_rate=0.98,
                impact_coefficients={'E1': 0.1, 'E2': 0.15, 'E3': 0.2, 'E6': 0.15}
            ),
            'norepinephrine': NeurotransmitterConfig(
                initial_level=30.0,  # nM
                min_level=5.0,
                max_level=80.0,
                decay_rate=0.92,
                impact_coefficients={'E3': 0.2}  # Entorno Educativo
            ),
            'oxytocin': NeurotransmitterConfig(
                initial_level=5.0,  # nM
                min_level=1.0,
                max_level=25.0,
                decay_rate=0.90,
                impact_coefficients={'E2': 0.15, 'E4': 0.1, 'E5': 0.1, 'E6': 0.15}
            ),
            'endorphins': NeurotransmitterConfig(
                initial_level=3.0,  # nM
                min_level=0.5,
                max_level=15.0,
                decay_rate=0.88,
                impact_coefficients={'E4': 0.1}  # Entorno Social
            )
        }
        
        # Niveles actuales
        self.current_levels = {
            name: config.initial_level 
            for name, config in self.neurotransmitters.items()
        }
        
        # Mapeo de entornos (E1-E6)
        self.environment_mapping = {
            'E1': 'Personal',
            'E2': 'Convivencia', 
            'E3': 'Educativo',
            'E4': 'Social',
            'E5': 'Familiar',
            'E6': 'Amoroso'
        }
        
        # Historial de niveles para análisis
        self.level_history = []
        
        self.logger.log("INFO", "NeurotransmitterSystem initialized")
    
    def update_from_environments(self, environment_values: Dict[str, int]):
        """
        Actualiza los niveles de neurotransmisores basado en los valores de entorno.
        
        Args:
            environment_values: Diccionario con valores de entorno (1-100)
                Keys: 'Personal', 'Convivencia', 'Educativo', 'Social', 'Familiar', 'Amoroso'
        """
        
        # Mapear nombres de entorno a códigos E1-E6
        env_codes = {
            'Personal': 'E1',
            'Convivencia': 'E2', 
            'Educativo': 'E3',
            'Social': 'E4',
            'Familiar': 'E5',
            'Amoroso': 'E6'
        }
        
        # Convertir a formato de códigos
        env_coded = {}
        for name, value in environment_values.items():
            if name in env_codes:
                env_coded[env_codes[name]] = value
        
        # Actualizar cada neurotransmisor
        for nt_name, config in self.neurotransmitters.items():
            new_level = self._calculate_neurotransmitter_level(
                nt_name, config, env_coded
            )
            
            # Aplicar límites
            new_level = max(config.min_level, min(config.max_level, new_level))
            
            # Log cambios significativos
            if abs(new_level - self.current_levels[nt_name]) > 1.0:
                self.logger.log("INFO", 
                    f"{nt_name.capitalize()} level changed: "
                    f"{self.current_levels[nt_name]:.2f} -> {new_level:.2f} nM"
                )
            
            self.current_levels[nt_name] = new_level
        
        # Guardar en historial
        self._save_to_history(env_coded)
        
        self.logger.log("DEBUG", f"Updated neurotransmitter levels: {self.current_levels}")
    
    def _calculate_neurotransmitter_level(self, nt_name: str, config: NeurotransmitterConfig, 
                                        environments: Dict[str, int]) -> float:
        """
        Calcula el nuevo nivel de un neurotransmisor específico.
        Implementa las fórmulas del documento original.
        """
        
        # Nivel base con decaimiento temporal
        base_level = self.current_levels[nt_name] * config.decay_rate
        
        # Aplicar impactos de entorno
        total_impact = 0.0
        
        for env_code, impact_coeff in config.impact_coefficients.items():
            if env_code in environments:
                env_value = environments[env_code]
                # Fórmula: I_E * (E - 50), donde 50 es el punto neutro
                impact = impact_coeff * (env_value - 50)
                total_impact += impact
        
        # Nivel final
        final_level = base_level + total_impact
        
        return final_level
    
    def _save_to_history(self, environments: Dict[str, int]):
        """Guarda el estado actual en el historial"""
        
        from datetime import datetime
        
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'levels': self.current_levels.copy(),
            'environments': environments.copy()
        }
        
        self.level_history.append(history_entry)
        
        # Mantener solo los últimos 1000 registros
        if len(self.level_history) > 1000:
            self.level_history = self.level_history[-1000:]
    
    def get_current_levels(self) -> Dict[str, float]:
        """Retorna los niveles actuales de neurotransmisores"""
        return self.current_levels.copy()
    
    def get_neurotransmitter_effects(self) -> Dict[str, Dict[str, str]]:
        """
        Retorna una descripción de los efectos de cada neurotransmisor
        en base a sus niveles actuales
        """
        
        effects = {}
        
        # Serotonina
        serotonin_level = self.current_levels['serotonin']
        if serotonin_level > 12:
            serotonin_effect = "Estado de ánimo muy positivo, alta estabilidad emocional"
        elif serotonin_level > 8:
            serotonin_effect = "Estado de ánimo positivo, buena regulación emocional"
        elif serotonin_level > 4:
            serotonin_effect = "Estado de ánimo neutral, estabilidad moderada"
        else:
            serotonin_effect = "Estado de ánimo bajo, posible inestabilidad emocional"
        
        effects['serotonin'] = {
            'level': f"{serotonin_level:.2f} nM",
            'effect': serotonin_effect,
            'function': "Regulación del estado de ánimo y bienestar"
        }
        
        # Dopamina
        dopamine_level = self.current_levels['dopamine']
        if dopamine_level > 80:
            dopamine_effect = "Muy alta motivación y búsqueda de recompensas"
        elif dopamine_level > 60:
            dopamine_effect = "Alta motivación, buena respuesta a recompensas"
        elif dopamine_level > 30:
            dopamine_effect = "Motivación moderada, respuesta normal a estímulos"
        else:
            dopamine_effect = "Baja motivación, poca respuesta a recompensas"
        
        effects['dopamine'] = {
            'level': f"{dopamine_level:.2f} nM",
            'effect': dopamine_effect,
            'function': "Motivación, placer y aprendizaje por refuerzo"
        }
        
        # Norepinefrina
        norepinephrine_level = self.current_levels['norepinephrine']
        if norepinephrine_level > 50:
            norepinephrine_effect = "Alta alerta y concentración, posible estrés elevado"
        elif norepinephrine_level > 35:
            norepinephrine_effect = "Buena alerta y concentración"
        elif norepinephrine_level > 20:
            norepinephrine_effect = "Nivel normal de alerta"
        else:
            norepinephrine_effect = "Baja alerta, posible fatiga mental"
        
        effects['norepinephrine'] = {
            'level': f"{norepinephrine_level:.2f} nM",
            'effect': norepinephrine_effect,
            'function': "Atención, alerta y respuesta al estrés"
        }
        
        # Oxitocina
        oxytocin_level = self.current_levels['oxytocin']
        if oxytocin_level > 15:
            oxytocin_effect = "Muy alta empatía y conexión social"
        elif oxytocin_level > 8:
            oxytocin_effect = "Buena empatía y vínculos sociales"
        elif oxytocin_level > 3:
            oxytocin_effect = "Nivel normal de conexión social"
        else:
            oxytocin_effect = "Baja empatía, dificultades sociales"
        
        effects['oxytocin'] = {
            'level': f"{oxytocin_level:.2f} nM",
            'effect': oxytocin_effect,
            'function': "Empatía, vínculos sociales y confianza"
        }
        
        # Endorfinas
        endorphins_level = self.current_levels['endorphins']
        if endorphins_level > 8:
            endorphins_effect = "Muy alta resistencia al estrés y bienestar"
        elif endorphins_level > 5:
            endorphins_effect = "Buena resistencia al estrés"
        elif endorphins_level > 2:
            endorphins_effect = "Nivel normal de resistencia"
        else:
            endorphins_effect = "Baja resistencia al estrés"
        
        effects['endorphins'] = {
            'level': f"{endorphins_level:.2f} nM",
            'effect': endorphins_effect,
            'function': "Resistencia al estrés y sensación de bienestar"
        }
        
        return effects
    
    def simulate_neurotransmitter_interaction(self) -> Dict[str, float]:
        """
        Simula interacciones entre neurotransmisores.
        Algunos neurotransmisores se influyen mutuamente.
        """
        
        interactions = {}
        
        # Serotonina y Dopamina (relación compleja)
        serotonin = self.current_levels['serotonin']
        dopamine = self.current_levels['dopamine']
        
        # Alta serotonina puede modular dopamina
        if serotonin > 10:
            dopamine_modulation = min(5.0, (serotonin - 10) * 0.5)
            interactions['serotonin_to_dopamine'] = dopamine_modulation
        
        # Oxitocina y Endorfinas (sinergia positiva)
        oxytocin = self.current_levels['oxytocin']
        endorphins = self.current_levels['endorphins']
        
        if oxytocin > 8 and endorphins > 4:
            synergy = min(2.0, (oxytocin - 8) * (endorphins - 4) * 0.1)
            interactions['oxytocin_endorphin_synergy'] = synergy
        
        # Norepinefrina puede inhibir serotonina en exceso
        norepinephrine = self.current_levels['norepinephrine']
        if norepinephrine > 60:
            serotonin_inhibition = min(3.0, (norepinephrine - 60) * 0.05)
            interactions['norepinephrine_to_serotonin'] = -serotonin_inhibition
        
        return interactions
    
    def get_overall_neurochemical_state(self) -> Dict[str, any]:
        """
        Calcula el estado neuroquímico general del sistema
        """
        
        levels = self.current_levels
        
        # Calcular puntuaciones normalizadas (0-1)
        normalized = {}
        for nt_name, config in self.neurotransmitters.items():
            level = levels[nt_name]
            normalized[nt_name] = (level - config.min_level) / (config.max_level - config.min_level)
        
        # Estado general
        overall_score = np.mean(list(normalized.values()))
        
        # Balanceado vs desequilibrado
        variance = np.var(list(normalized.values()))
        balance_score = 1.0 - min(1.0, variance * 2)  # Menos varianza = más balance
        
        # Categorizar estado
        if overall_score > 0.8 and balance_score > 0.7:
            state_category = "Optimal"
        elif overall_score > 0.6 and balance_score > 0.5:
            state_category = "Good"
        elif overall_score > 0.4:
            state_category = "Moderate"
        else:
            state_category = "Suboptimal"
        
        return {
            'overall_score': overall_score,
            'balance_score': balance_score,
            'state_category': state_category,
            'normalized_levels': normalized,
            'dominant_neurotransmitter': max(normalized.keys(), key=lambda k: normalized[k]),
            'interactions': self.simulate_neurotransmitter_interaction()
        }
    
    def reset_to_baseline(self):
        """Reinicia todos los neurotransmisores a sus niveles iniciales"""
        
        for nt_name, config in self.neurotransmitters.items():
            self.current_levels[nt_name] = config.initial_level
        
        self.logger.log("INFO", "Neurotransmitter levels reset to baseline")
    
    def apply_external_modulation(self, modulations: Dict[str, float]):
        """
        Aplica modulaciones externas a los neurotransmisores
        
        Args:
            modulations: Dict con cambios a aplicar {'serotonin': +2.0, 'dopamine': -1.5, ...}
        """
        
        for nt_name, change in modulations.items():
            if nt_name in self.current_levels:
                config = self.neurotransmitters[nt_name]
                new_level = self.current_levels[nt_name] + change
                
                # Aplicar límites
                new_level = max(config.min_level, min(config.max_level, new_level))
                
                self.current_levels[nt_name] = new_level
                
                self.logger.log("DEBUG", 
                    f"External modulation applied to {nt_name}: {change:+.2f} -> {new_level:.2f} nM"
                )
    
    def check_health(self) -> bool:
        """Verifica si el sistema de neurotransmisores está en estado saludable"""
        
        for nt_name, config in self.neurotransmitters.items():
            level = self.current_levels[nt_name]
            
            # Verificar que esté dentro de rangos
            if level < config.min_level or level > config.max_level:
                return False
            
            # Verificar que no haya valores extremos
            mid_point = (config.min_level + config.max_level) / 2
            if abs(level - mid_point) > (config.max_level - config.min_level) * 0.4:
                continue  # Permitir algunas variaciones
        
        return True
