import os
import json
import yaml
from typing import Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime

class Config:
    """
    Sistema de configuración centralizado para el sistema AGI de conciencia artificial
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Inicializa el sistema de configuración
        
        Args:
            config_file: Ruta al archivo de configuración (opcional)
        """
        
        # Configuración por defecto
        self._default_config = {
            # Configuración de conciencia
            "consciousness": {
                "valores_conciencia": {
                    "creatividad": 0.8,
                    "innovación": 0.7,
                    "originalidad": 0.9,
                    "sorpresa": 0.8,
                    "miedo": -0.8,
                    "ansiedad": -0.7,
                    "alegría": 0.9,
                    "tristeza": -0.8,
                    "realidad": 1.0,
                    "evolución": 1.0,
                    "humanidad": 1.0
                },
                "update_frequency": 100,
                "consciousness_threshold": 0.7,
                "state_transition_sensitivity": 0.1
            },
            
            # Configuración de neurotransmisores
            "neurotransmitters": {
                "initial_levels": {
                    "serotonin": 7.0,
                    "dopamine": 50.0,
                    "norepinephrine": 30.0,
                    "oxytocin": 5.0,
                    "endorphins": 3.0
                },
                "decay_rates": {
                    "serotonin": 0.95,
                    "dopamine": 0.98,
                    "norepinephrine": 0.92,
                    "oxytocin": 0.90,
                    "endorphins": 0.88
                },
                "impact_coefficients": {
                    "serotonin": {"E1": 0.1, "E5": 0.1},
                    "dopamine": {"E1": 0.1, "E2": 0.15, "E3": 0.2, "E6": 0.15},
                    "norepinephrine": {"E3": 0.2},
                    "oxytocin": {"E2": 0.15, "E4": 0.1, "E5": 0.1, "E6": 0.15},
                    "endorphins": {"E4": 0.1}
                },
                "environment_mapping": {
                    "E1": "Personal",
                    "E2": "Convivencia",
                    "E3": "Educativo",
                    "E4": "Social",
                    "E5": "Familiar",
                    "E6": "Amoroso"
                }
            },
            
            # Configuración de procesamiento cuántico
            "quantum": {
                "n_qubits": 4,
                "coherence_threshold": 0.8,
                "decoherence_rate": 0.01,
                "max_entanglement": 0.9,
                "quantum_noise_level": 0.05,
                "measurement_precision": 1e-6
            },
            
            # Configuración de memoria
            "memory": {
                "short_term": {
                    "input_size": 784,
                    "hidden_size": 128,
                    "num_layers": 1,
                    "dropout": 0.1
                },
                "long_term": {
                    "max_memories": 10000,
                    "consolidation_threshold": 0.7,
                    "access_threshold": 3,
                    "consolidation_frequency": 100,
                    "forgetting_threshold_days": 30
                },
                "working_memory_size": 7,
                "attention_heads": 8
            },
            
            # Configuración multimodal
            "multimodal": {
                "text": {
                    "vocab_size": 10000,
                    "embedding_dim": 768,
                    "max_sequence_length": 512,
                    "attention_heads": 12,
                    "hidden_size": 256
                },
                "vision": {
                    "input_size": 224,
                    "feature_dim": 512,
                    "conv_layers": 4,
                    "batch_norm": True,
                    "dropout": 0.1
                },
                "audio": {
                    "sample_rate": 22050,
                    "n_mels": 128,
                    "n_fft": 2048,
                    "hop_length": 512,
                    "feature_dim": 256
                },
                "fusion": {
                    "hidden_dim": 512,
                    "attention_heads": 8,
                    "fusion_layers": 2
                }
            },
            
            # Configuración de algoritmos
            "algorithms": {
                "amiloid_agent": {
                    "relevance_threshold": 0.3,
                    "pruning_rate": 0.1,
                    "evaluation_frequency": 10,
                    "decay_rate": 0.95
                },
                "quantum_rl": {
                    "n_states": 8,
                    "n_actions": 4,
                    "learning_rate": 0.1,
                    "discount_factor": 0.9,
                    "exploration_rate": 0.1,
                    "episodes_per_update": 5
                },
                "bayesian_quantum": {
                    "default_dimension": 4,
                    "decision_threshold": 0.7,
                    "uncertainty_tolerance": 0.3,
                    "significance_level": 0.05
                }
            },
            
            # Configuración de logging
            "logging": {
                "level": "INFO",
                "max_log_size_mb": 100,
                "max_log_files": 5,
                "log_to_console": True,
                "log_to_file": True,
                "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "date_format": "%Y-%m-%d %H:%M:%S"
            },
            
            # Configuración de la aplicación
            "app": {
                "title": "Sistema AGI Multimodal con Conciencia Artificial",
                "description": "Basado en modulación de neurotransmisores y computación cuántica",
                "version": "1.0.0",
                "author": "AGI Consciousness Research",
                "max_file_size_mb": 50,
                "supported_image_formats": ["png", "jpg", "jpeg", "gif", "bmp"],
                "supported_audio_formats": ["wav", "mp3", "ogg", "m4a"],
                "session_timeout_hours": 24
            },
            
            # Configuración de rendimiento
            "performance": {
                "batch_size": 32,
                "num_workers": 4,
                "device": "auto",  # "cpu", "cuda", "auto"
                "memory_limit_gb": 8,
                "cache_size_mb": 512,
                "optimization_level": "O2"
            },
            
            # Configuración de seguridad
            "security": {
                "enable_input_validation": True,
                "max_input_length": 10000,
                "sanitize_uploads": True,
                "enable_rate_limiting": True,
                "rate_limit_requests_per_minute": 60
            }
        }
        
        # Configuración actual
        self._config = self._default_config.copy()
        
        # Cargar configuración desde archivo si se especifica
        if config_file:
            self.load_config(config_file)
        
        # Cargar configuración desde variables de entorno
        self._load_from_environment()
        
        # Validar configuración
        self._validate_config()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtiene un valor de configuración usando notación de punto
        
        Args:
            key: Clave de configuración (ej: "consciousness.valores_conciencia.creatividad")
            default: Valor por defecto si la clave no existe
        
        Returns:
            Valor de configuración
        """
        try:
            keys = key.split('.')
            value = self._config
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
        except Exception:
            return default
    
    def set(self, key: str, value: Any) -> None:
        """
        Establece un valor de configuración usando notación de punto
        
        Args:
            key: Clave de configuración
            value: Valor a establecer
        """
        try:
            keys = key.split('.')
            config = self._config
            
            # Navegar hasta el penúltimo nivel
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            
            # Establecer el valor final
            config[keys[-1]] = value
            
        except Exception as e:
            raise ValueError(f"Error setting config key '{key}': {str(e)}")
    
    def load_config(self, config_file: str) -> None:
        """
        Carga configuración desde un archivo
        
        Args:
            config_file: Ruta al archivo de configuración (JSON o YAML)
        """
        try:
            config_path = Path(config_file)
            
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_file}")
            
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yml', '.yaml']:
                    file_config = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    file_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            # Fusionar configuración
            self._deep_merge(self._config, file_config)
            
        except Exception as e:
            raise RuntimeError(f"Error loading config file '{config_file}': {str(e)}")
    
    def save_config(self, config_file: str, format: str = 'json') -> None:
        """
        Guarda la configuración actual a un archivo
        
        Args:
            config_file: Ruta del archivo de destino
            format: Formato del archivo ('json' o 'yaml')
        """
        try:
            config_path = Path(config_file)
            config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                if format.lower() == 'yaml':
                    yaml.dump(self._config, f, default_flow_style=False, indent=2)
                else:
                    json.dump(self._config, f, indent=2, ensure_ascii=False)
                    
        except Exception as e:
            raise RuntimeError(f"Error saving config file '{config_file}': {str(e)}")
    
    def _load_from_environment(self) -> None:
        """Carga configuración desde variables de entorno"""
        
        # Mapeo de variables de entorno a configuración
        env_mappings = {
            'AGI_LOG_LEVEL': 'logging.level',
            'AGI_DEVICE': 'performance.device',
            'AGI_BATCH_SIZE': 'performance.batch_size',
            'AGI_MEMORY_LIMIT': 'performance.memory_limit_gb',
            'AGI_QUANTUM_QUBITS': 'quantum.n_qubits',
            'AGI_CONSCIOUSNESS_THRESHOLD': 'consciousness.consciousness_threshold',
            'AGI_MAX_MEMORIES': 'memory.long_term.max_memories',
            'AGI_SAMPLE_RATE': 'multimodal.audio.sample_rate',
            'AGI_MAX_FILE_SIZE': 'app.max_file_size_mb'
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convertir tipo si es necesario
                try:
                    # Intentar convertir a número
                    if '.' in env_value:
                        converted_value = float(env_value)
                    else:
                        converted_value = int(env_value)
                except ValueError:
                    # Mantener como string
                    converted_value = env_value
                
                self.set(config_key, converted_value)
    
    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> None:
        """
        Fusiona dos diccionarios recursivamente
        
        Args:
            dict1: Diccionario base (se modifica)
            dict2: Diccionario a fusionar
        """
        for key, value in dict2.items():
            if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
                self._deep_merge(dict1[key], value)
            else:
                dict1[key] = value
    
    def _validate_config(self) -> None:
        """Valida la configuración actual"""
        
        # Validaciones críticas
        validations = [
            ("consciousness.consciousness_threshold", lambda x: 0 <= x <= 1, "must be between 0 and 1"),
            ("quantum.n_qubits", lambda x: x > 0 and isinstance(x, int), "must be positive integer"),
            ("quantum.coherence_threshold", lambda x: 0 <= x <= 1, "must be between 0 and 1"),
            ("memory.short_term.hidden_size", lambda x: x > 0, "must be positive"),
            ("performance.batch_size", lambda x: x > 0, "must be positive"),
            ("algorithms.amiloid_agent.relevance_threshold", lambda x: 0 <= x <= 1, "must be between 0 and 1"),
            ("algorithms.quantum_rl.learning_rate", lambda x: 0 < x <= 1, "must be between 0 and 1"),
            ("logging.level", lambda x: x in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
             "must be valid log level"),
        ]
        
        for config_key, validator, error_msg in validations:
            value = self.get(config_key)
            if value is not None and not validator(value):
                raise ValueError(f"Invalid config value for '{config_key}': {error_msg}")
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Obtiene una sección completa de configuración
        
        Args:
            section: Nombre de la sección
        
        Returns:
            Diccionario con la configuración de la sección
        """
        return self.get(section, {})
    
    def update_section(self, section: str, updates: Dict[str, Any]) -> None:
        """
        Actualiza múltiples valores en una sección
        
        Args:
            section: Nombre de la sección
            updates: Diccionario con actualizaciones
        """
        current_section = self.get_section(section)
        if isinstance(current_section, dict):
            self._deep_merge(current_section, updates)
            self.set(section, current_section)
        else:
            self.set(section, updates)
    
    def reset_to_defaults(self) -> None:
        """Reinicia la configuración a valores por defecto"""
        self._config = self._default_config.copy()
    
    def get_consciousness_config(self) -> Dict[str, Any]:
        """Obtiene configuración específica de conciencia"""
        return self.get_section("consciousness")
    
    def get_neurotransmitter_config(self) -> Dict[str, Any]:
        """Obtiene configuración específica de neurotransmisores"""
        return self.get_section("neurotransmitters")
    
    def get_quantum_config(self) -> Dict[str, Any]:
        """Obtiene configuración específica de computación cuántica"""
        return self.get_section("quantum")
    
    def get_multimodal_config(self) -> Dict[str, Any]:
        """Obtiene configuración específica de procesamiento multimodal"""
        return self.get_section("multimodal")
    
    def get_algorithm_config(self, algorithm: str) -> Dict[str, Any]:
        """
        Obtiene configuración de un algoritmo específico
        
        Args:
            algorithm: Nombre del algoritmo
        
        Returns:
            Configuración del algoritmo
        """
        return self.get(f"algorithms.{algorithm}", {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Obtiene configuración de rendimiento"""
        return self.get_section("performance")
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Obtiene configuración de logging"""
        return self.get_section("logging")
    
    def get_app_config(self) -> Dict[str, Any]:
        """Obtiene configuración de la aplicación"""
        return self.get_section("app")
    
    def is_debug_mode(self) -> bool:
        """Verifica si está en modo debug"""
        return self.get("logging.level", "INFO") == "DEBUG"
    
    def get_device(self) -> str:
        """Obtiene el dispositivo de cómputo configurado"""
        device = self.get("performance.device", "auto")
        
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        
        return device
    
    def get_memory_limit_bytes(self) -> int:
        """Obtiene el límite de memoria en bytes"""
        limit_gb = self.get("performance.memory_limit_gb", 8)
        return int(limit_gb * 1024 * 1024 * 1024)
    
    def get_max_file_size_bytes(self) -> int:
        """Obtiene el tamaño máximo de archivo en bytes"""
        size_mb = self.get("app.max_file_size_mb", 50)
        return int(size_mb * 1024 * 1024)
    
    def export_config(self) -> Dict[str, Any]:
        """
        Exporta la configuración actual
        
        Returns:
            Diccionario con toda la configuración
        """
        return {
            "config": self._config.copy(),
            "export_timestamp": datetime.now().isoformat(),
            "version": self.get("app.version", "1.0.0")
        }
    
    def import_config(self, config_data: Dict[str, Any]) -> None:
        """
        Importa configuración desde un diccionario
        
        Args:
            config_data: Datos de configuración a importar
        """
        if "config" in config_data:
            self._config = config_data["config"].copy()
            self._validate_config()
        else:
            raise ValueError("Invalid config data format")
    
    def create_environment_template(self, output_file: str) -> None:
        """
        Crea un archivo de plantilla de variables de entorno
        
        Args:
            output_file: Ruta del archivo de salida
        """
        env_template = [
            "# Configuración del Sistema AGI de Conciencia Artificial",
            "# Establece estas variables de entorno para personalizar el comportamiento",
            "",
            "# Configuración de logging",
            "# AGI_LOG_LEVEL=INFO",
            "",
            "# Configuración de rendimiento",
            "# AGI_DEVICE=auto",
            "# AGI_BATCH_SIZE=32",
            "# AGI_MEMORY_LIMIT=8",
            "",
            "# Configuración cuántica",
            "# AGI_QUANTUM_QUBITS=4",
            "",
            "# Configuración de conciencia",
            "# AGI_CONSCIOUSNESS_THRESHOLD=0.7",
            "",
            "# Configuración de memoria",
            "# AGI_MAX_MEMORIES=10000",
            "",
            "# Configuración de audio",
            "# AGI_SAMPLE_RATE=22050",
            "",
            "# Configuración de aplicación",
            "# AGI_MAX_FILE_SIZE=50",
            ""
        ]
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(env_template))
        except Exception as e:
            raise RuntimeError(f"Error creating environment template: {str(e)}")
    
    def __str__(self) -> str:
        """Representación string de la configuración"""
        return f"Config(sections={list(self._config.keys())}, version={self.get('app.version')})"
    
    def __repr__(self) -> str:
        """Representación detallada de la configuración"""
        return f"Config({json.dumps(self._config, indent=2)})"
