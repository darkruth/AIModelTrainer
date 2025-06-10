import logging
import logging.handlers
import os
import sys
from typing import Optional, Any, Dict
from datetime import datetime
from pathlib import Path
import json
import threading
from contextlib import contextmanager

class CustomFormatter(logging.Formatter):
    """
    Formateador personalizado para logs con colores y formato mejorado
    """
    
    # Códigos de color ANSI
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Verde
        'WARNING': '\033[33m',    # Amarillo
        'ERROR': '\033[31m',      # Rojo
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def __init__(self, use_colors: bool = True, include_thread: bool = True):
        self.use_colors = use_colors
        self.include_thread = include_thread
        
        # Formato base
        base_format = "%(asctime)s - %(name)s - %(levelname)s"
        
        if include_thread:
            base_format += " - [%(threadName)s]"
        
        base_format += " - %(message)s"
        
        super().__init__(base_format, datefmt="%Y-%m-%d %H:%M:%S")
    
    def format(self, record):
        """Formatea el registro de log"""
        
        # Formatear el mensaje base
        formatted = super().format(record)
        
        # Aplicar colores si está habilitado y es una terminal
        if self.use_colors and hasattr(sys.stderr, 'isatty') and sys.stderr.isatty():
            level_name = record.levelname
            color = self.COLORS.get(level_name, self.COLORS['RESET'])
            reset = self.COLORS['RESET']
            
            # Colorear solo el nivel de log
            formatted = formatted.replace(
                f" - {level_name} - ",
                f" - {color}{level_name}{reset} - "
            )
        
        return formatted

class StructuredLogger:
    """
    Logger estructurado para el sistema AGI que soporta logging JSON y texto
    """
    
    def __init__(self, name: str = "AGI_System"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Prevenir duplicación de handlers
        if not self.logger.handlers:
            self._setup_handlers()
        
        # Lock para thread safety
        self._lock = threading.Lock()
        
        # Contexto adicional para logs
        self._context = {}
        
        # Estadísticas de logging
        self._stats = {
            'total_logs': 0,
            'by_level': {
                'DEBUG': 0,
                'INFO': 0,
                'WARNING': 0,
                'ERROR': 0,
                'CRITICAL': 0
            },
            'start_time': datetime.now()
        }
    
    def _setup_handlers(self):
        """Configura los handlers de logging"""
        
        # Handler para consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formateador con colores para consola
        console_formatter = CustomFormatter(use_colors=True, include_thread=True)
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(console_handler)
        
        # Handler para archivo (si el directorio existe o se puede crear)
        try:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            # Archivo principal de logs
            file_handler = logging.handlers.RotatingFileHandler(
                log_dir / "agi_system.log",
                maxBytes=100 * 1024 * 1024,  # 100MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            
            # Formateador sin colores para archivo
            file_formatter = CustomFormatter(use_colors=False, include_thread=True)
            file_handler.setFormatter(file_formatter)
            
            self.logger.addHandler(file_handler)
            
            # Handler separado para errores
            error_handler = logging.handlers.RotatingFileHandler(
                log_dir / "agi_errors.log",
                maxBytes=50 * 1024 * 1024,  # 50MB
                backupCount=3,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(file_formatter)
            
            self.logger.addHandler(error_handler)
            
            # Handler para logs estructurados JSON
            json_handler = logging.handlers.RotatingFileHandler(
                log_dir / "agi_structured.jsonl",
                maxBytes=100 * 1024 * 1024,  # 100MB
                backupCount=3,
                encoding='utf-8'
            )
            json_handler.setLevel(logging.INFO)
            json_handler.setFormatter(self._create_json_formatter())
            
            self.logger.addHandler(json_handler)
            
        except (OSError, PermissionError) as e:
            # Si no se pueden crear archivos de log, continuar solo con consola
            self.logger.warning(f"Could not setup file logging: {e}")
    
    def _create_json_formatter(self):
        """Crea formateador JSON personalizado"""
        
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno,
                    'thread': record.threadName,
                    'process': record.process
                }
                
                # Añadir información de excepción si existe
                if record.exc_info:
                    log_entry['exception'] = self.formatException(record.exc_info)
                
                # Añadir campos personalizados si existen
                if hasattr(record, 'extra_data'):
                    log_entry['extra'] = record.extra_data
                
                return json.dumps(log_entry, ensure_ascii=False)
        
        return JsonFormatter()
    
    def _update_stats(self, level: str):
        """Actualiza estadísticas de logging"""
        with self._lock:
            self._stats['total_logs'] += 1
            if level in self._stats['by_level']:
                self._stats['by_level'][level] += 1
    
    def set_level(self, level: str):
        """
        Establece el nivel de logging
        
        Args:
            level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.setLevel(numeric_level)
        
        # Actualizar handlers de consola
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(numeric_level)
    
    def add_context(self, **kwargs):
        """
        Añade contexto permanente al logger
        
        Args:
            **kwargs: Pares clave-valor de contexto
        """
        with self._lock:
            self._context.update(kwargs)
    
    def remove_context(self, *keys):
        """
        Remueve claves del contexto
        
        Args:
            *keys: Claves a remover
        """
        with self._lock:
            for key in keys:
                self._context.pop(key, None)
    
    def clear_context(self):
        """Limpia todo el contexto"""
        with self._lock:
            self._context.clear()
    
    @contextmanager
    def context(self, **kwargs):
        """
        Context manager para contexto temporal
        
        Args:
            **kwargs: Contexto temporal
        """
        old_context = self._context.copy()
        try:
            self.add_context(**kwargs)
            yield
        finally:
            with self._lock:
                self._context = old_context
    
    def _create_log_record(self, level: str, message: str, extra_data: Optional[Dict] = None):
        """
        Crea un registro de log con contexto y datos extra
        
        Args:
            level: Nivel de log
            message: Mensaje principal
            extra_data: Datos adicionales
        
        Returns:
            Registro de log preparado
        """
        # Combinar contexto actual con datos extra
        combined_extra = self._context.copy()
        if extra_data:
            combined_extra.update(extra_data)
        
        # Crear registro con datos extra
        extra = {'extra_data': combined_extra} if combined_extra else {}
        
        return message, extra
    
    def debug(self, message: str, **kwargs):
        """Log de debug"""
        msg, extra = self._create_log_record('DEBUG', message, kwargs)
        self.logger.debug(msg, extra=extra)
        self._update_stats('DEBUG')
    
    def info(self, message: str, **kwargs):
        """Log de información"""
        msg, extra = self._create_log_record('INFO', message, kwargs)
        self.logger.info(msg, extra=extra)
        self._update_stats('INFO')
    
    def warning(self, message: str, **kwargs):
        """Log de warning"""
        msg, extra = self._create_log_record('WARNING', message, kwargs)
        self.logger.warning(msg, extra=extra)
        self._update_stats('WARNING')
    
    def error(self, message: str, **kwargs):
        """Log de error"""
        msg, extra = self._create_log_record('ERROR', message, kwargs)
        self.logger.error(msg, extra=extra)
        self._update_stats('ERROR')
    
    def critical(self, message: str, **kwargs):
        """Log crítico"""
        msg, extra = self._create_log_record('CRITICAL', message, kwargs)
        self.logger.critical(msg, extra=extra)
        self._update_stats('CRITICAL')
    
    def exception(self, message: str, **kwargs):
        """Log de excepción con traceback"""
        msg, extra = self._create_log_record('ERROR', message, kwargs)
        self.logger.exception(msg, extra=extra)
        self._update_stats('ERROR')
    
    def log(self, level: str, message: str, **kwargs):
        """
        Log genérico con nivel especificado
        
        Args:
            level: Nivel de log
            message: Mensaje
            **kwargs: Datos adicionales
        """
        level = level.upper()
        if hasattr(self, level.lower()):
            getattr(self, level.lower())(message, **kwargs)
        else:
            self.info(f"[{level}] {message}", **kwargs)
    
    def log_performance(self, operation: str, duration: float, **kwargs):
        """
        Log específico para métricas de rendimiento
        
        Args:
            operation: Nombre de la operación
            duration: Duración en segundos
            **kwargs: Métricas adicionales
        """
        self.info(
            f"Performance: {operation} completed in {duration:.4f}s",
            operation=operation,
            duration=duration,
            performance_metrics=kwargs
        )
    
    def log_consciousness_state(self, state: str, metrics: Dict[str, Any]):
        """
        Log específico para estados de conciencia
        
        Args:
            state: Estado de conciencia
            metrics: Métricas del estado
        """
        self.info(
            f"Consciousness state: {state}",
            consciousness_state=state,
            consciousness_metrics=metrics
        )
    
    def log_neurotransmitter_update(self, neurotransmitter: str, old_level: float, 
                                   new_level: float, **kwargs):
        """
        Log específico para actualizaciones de neurotransmisores
        
        Args:
            neurotransmitter: Nombre del neurotransmisor
            old_level: Nivel anterior
            new_level: Nuevo nivel
            **kwargs: Datos adicionales
        """
        change = new_level - old_level
        self.debug(
            f"Neurotransmitter {neurotransmitter}: {old_level:.2f} -> {new_level:.2f} ({change:+.2f})",
            neurotransmitter=neurotransmitter,
            old_level=old_level,
            new_level=new_level,
            change=change,
            **kwargs
        )
    
    def log_quantum_operation(self, operation: str, coherence: float, **kwargs):
        """
        Log específico para operaciones cuánticas
        
        Args:
            operation: Tipo de operación cuántica
            coherence: Nivel de coherencia
            **kwargs: Parámetros cuánticos adicionales
        """
        self.debug(
            f"Quantum operation: {operation} (coherence: {coherence:.3f})",
            quantum_operation=operation,
            coherence=coherence,
            quantum_params=kwargs
        )
    
    def log_multimodal_processing(self, modalities: list, fusion_result: Dict[str, Any]):
        """
        Log específico para procesamiento multimodal
        
        Args:
            modalities: Lista de modalidades procesadas
            fusion_result: Resultado de la fusión
        """
        self.info(
            f"Multimodal processing: {', '.join(modalities)}",
            modalities=modalities,
            fusion_result=fusion_result
        )
    
    def log_system_health(self, component: str, status: str, metrics: Dict[str, Any]):
        """
        Log específico para salud del sistema
        
        Args:
            component: Componente del sistema
            status: Estado (healthy, warning, error)
            metrics: Métricas de salud
        """
        level = 'info' if status == 'healthy' else 'warning' if status == 'warning' else 'error'
        
        getattr(self, level)(
            f"System health: {component} is {status}",
            component=component,
            health_status=status,
            health_metrics=metrics
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del logger
        
        Returns:
            Diccionario con estadísticas
        """
        with self._lock:
            runtime = datetime.now() - self._stats['start_time']
            
            return {
                'total_logs': self._stats['total_logs'],
                'logs_by_level': self._stats['by_level'].copy(),
                'runtime_seconds': runtime.total_seconds(),
                'logs_per_minute': self._stats['total_logs'] / max(runtime.total_seconds() / 60, 1),
                'current_context_keys': list(self._context.keys()),
                'logger_name': self.name
            }
    
    def export_logs(self, output_file: str, level: str = 'INFO', 
                   start_time: Optional[datetime] = None, 
                   end_time: Optional[datetime] = None):
        """
        Exporta logs filtrados a un archivo
        
        Args:
            output_file: Archivo de destino
            level: Nivel mínimo de logs
            start_time: Tiempo de inicio (opcional)
            end_time: Tiempo de fin (opcional)
        """
        # Esta función requeriría acceso a los archivos de log
        # Por simplicidad, creamos un placeholder
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"# Log export from {self.name}\n")
                f.write(f"# Generated at: {datetime.now().isoformat()}\n")
                f.write(f"# Level filter: {level}\n")
                f.write(f"# Start time: {start_time}\n")
                f.write(f"# End time: {end_time}\n\n")
                
                # En implementación real, leeríamos y filtraríamos los archivos de log
                f.write("Log export functionality requires file system access to log files.\n")
                
            self.info(f"Logs exported to {output_file}")
            
        except Exception as e:
            self.error(f"Failed to export logs: {str(e)}")

# Instancia global del logger
_global_logger = None

def get_logger(name: str = "AGI_System") -> StructuredLogger:
    """
    Obtiene una instancia del logger global o crea una nueva
    
    Args:
        name: Nombre del logger
    
    Returns:
        Instancia del logger
    """
    global _global_logger
    
    if _global_logger is None or _global_logger.name != name:
        _global_logger = StructuredLogger(name)
    
    return _global_logger

def setup_logging(level: str = "INFO", enable_file_logging: bool = True):
    """
    Configuración global de logging
    
    Args:
        level: Nivel de logging
        enable_file_logging: Si habilitar logging a archivos
    """
    logger = get_logger()
    logger.set_level(level)
    
    if not enable_file_logging:
        # Remover handlers de archivo
        file_handlers = [h for h in logger.logger.handlers 
                        if isinstance(h, (logging.FileHandler, logging.handlers.RotatingFileHandler))]
        
        for handler in file_handlers:
            logger.logger.removeHandler(handler)
    
    logger.info(f"Logging configured: level={level}, file_logging={enable_file_logging}")

# Función de conveniencia para crear loggers específicos de módulos
def create_module_logger(module_name: str) -> StructuredLogger:
    """
    Crea un logger específico para un módulo
    
    Args:
        module_name: Nombre del módulo
    
    Returns:
        Logger del módulo
    """
    logger = StructuredLogger(f"AGI.{module_name}")
    return logger

# Alias para compatibilidad
Logger = StructuredLogger

# Funciones de conveniencia para logging rápido
def debug(message: str, **kwargs):
    """Debug log rápido"""
    get_logger().debug(message, **kwargs)

def info(message: str, **kwargs):
    """Info log rápido"""
    get_logger().info(message, **kwargs)

def warning(message: str, **kwargs):
    """Warning log rápido"""
    get_logger().warning(message, **kwargs)

def error(message: str, **kwargs):
    """Error log rápido"""
    get_logger().error(message, **kwargs)

def critical(message: str, **kwargs):
    """Critical log rápido"""
    get_logger().critical(message, **kwargs)

def exception(message: str, **kwargs):
    """Exception log rápido"""
    get_logger().exception(message, **kwargs)

def log_performance(operation: str, duration: float, **kwargs):
    """Performance log rápido"""
    get_logger().log_performance(operation, duration, **kwargs)

# Decorador para logging automático de funciones
def log_function_call(level: str = "DEBUG", include_args: bool = False, include_result: bool = False):
    """
    Decorador para logging automático de llamadas a funciones
    
    Args:
        level: Nivel de log
        include_args: Si incluir argumentos en el log
        include_result: Si incluir el resultado en el log
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger()
            func_name = f"{func.__module__}.{func.__name__}"
            
            # Log de entrada
            log_data = {'function': func_name}
            if include_args:
                log_data['args'] = str(args)[:200]  # Limitar longitud
                log_data['kwargs'] = {k: str(v)[:100] for k, v in kwargs.items()}
            
            logger.log(level, f"Calling function: {func_name}", **log_data)
            
            try:
                # Ejecutar función
                start_time = datetime.now()
                result = func(*args, **kwargs)
                duration = (datetime.now() - start_time).total_seconds()
                
                # Log de salida exitosa
                exit_data = {'function': func_name, 'duration': duration}
                if include_result:
                    exit_data['result'] = str(result)[:200]  # Limitar longitud
                
                logger.log(level, f"Function completed: {func_name}", **exit_data)
                
                return result
                
            except Exception as e:
                # Log de error
                logger.error(f"Function failed: {func_name}", 
                           function=func_name, 
                           error=str(e), 
                           exception_type=type(e).__name__)
                raise
        
        return wrapper
    return decorator
