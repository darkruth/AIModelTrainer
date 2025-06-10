import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
import heapq
from collections import defaultdict
from datetime import datetime

from utils.logger import Logger

class ConnectionRelevanceTracker:
    """
    Rastrea la relevancia de conexiones neuronales para el algoritmo Amiloid Agent
    """
    
    def __init__(self, decay_rate: float = 0.95):
        self.relevance_scores = {}
        self.access_counts = defaultdict(int)
        self.last_access = {}
        self.decay_rate = decay_rate
        
    def update_relevance(self, connection_id: str, relevance: float):
        """Actualiza la relevancia de una conexión"""
        current_time = datetime.now()
        
        # Aplicar decaimiento temporal si existe acceso previo
        if connection_id in self.last_access:
            time_diff = (current_time - self.last_access[connection_id]).total_seconds() / 3600  # horas
            decay_factor = self.decay_rate ** time_diff
            old_relevance = self.relevance_scores.get(connection_id, 0.0)
            self.relevance_scores[connection_id] = old_relevance * decay_factor + relevance * (1 - decay_factor)
        else:
            self.relevance_scores[connection_id] = relevance
        
        self.access_counts[connection_id] += 1
        self.last_access[connection_id] = current_time
    
    def get_relevance(self, connection_id: str) -> float:
        """Obtiene la relevancia actual de una conexión"""
        if connection_id not in self.relevance_scores:
            return 0.0
        
        # Aplicar decaimiento temporal
        current_time = datetime.now()
        if connection_id in self.last_access:
            time_diff = (current_time - self.last_access[connection_id]).total_seconds() / 3600
            decay_factor = self.decay_rate ** time_diff
            return self.relevance_scores[connection_id] * decay_factor
        
        return self.relevance_scores[connection_id]
    
    def get_all_relevances(self) -> Dict[str, float]:
        """Obtiene todas las relevancias actuales"""
        return {conn_id: self.get_relevance(conn_id) for conn_id in self.relevance_scores.keys()}

class DataRelevanceAnalyzer:
    """
    Analiza la relevancia de datos para el entrenamiento
    """
    
    def __init__(self):
        self.data_scores = {}
        self.usage_patterns = defaultdict(list)
        
    def evaluate_data_relevance(self, data_sample, context: Dict = None) -> float:
        """
        Evalúa la relevancia de una muestra de datos
        
        Args:
            data_sample: Muestra de datos a evaluar
            context: Contexto adicional (neurotransmisores, estado de conciencia, etc.)
        
        Returns:
            Puntuación de relevancia (0.0 - 1.0)
        """
        
        relevance_factors = []
        
        # Factor 1: Novedad de los datos
        novelty_score = self._calculate_novelty(data_sample)
        relevance_factors.append(novelty_score * 0.3)
        
        # Factor 2: Complejidad de la información
        complexity_score = self._calculate_complexity(data_sample)
        relevance_factors.append(complexity_score * 0.2)
        
        # Factor 3: Coherencia con el contexto
        if context:
            context_score = self._calculate_context_coherence(data_sample, context)
            relevance_factors.append(context_score * 0.3)
        
        # Factor 4: Potencial de aprendizaje
        learning_potential = self._calculate_learning_potential(data_sample)
        relevance_factors.append(learning_potential * 0.2)
        
        # Calcular relevancia total
        total_relevance = sum(relevance_factors)
        
        return min(1.0, max(0.0, total_relevance))
    
    def _calculate_novelty(self, data_sample) -> float:
        """Calcula la novedad de una muestra de datos"""
        
        # Convertir datos a representación hasheable
        if isinstance(data_sample, torch.Tensor):
            data_hash = hash(tuple(data_sample.flatten().tolist()[:100]))  # Muestra para hash
        elif isinstance(data_sample, (list, tuple)):
            data_hash = hash(tuple(str(x) for x in data_sample[:100]))
        elif isinstance(data_sample, str):
            data_hash = hash(data_sample[:500])  # Primeros 500 caracteres
        else:
            data_hash = hash(str(data_sample)[:500])
        
        # Verificar si ya hemos visto datos similares
        if data_hash in self.data_scores:
            # Datos conocidos, baja novedad
            return max(0.1, 1.0 - len(self.usage_patterns[data_hash]) * 0.1)
        else:
            # Datos nuevos, alta novedad
            self.data_scores[data_hash] = 1.0
            return 1.0
    
    def _calculate_complexity(self, data_sample) -> float:
        """Calcula la complejidad de los datos"""
        
        if isinstance(data_sample, torch.Tensor):
            # Para tensores: usar varianza y distribución
            variance = torch.var(data_sample).item()
            entropy = self._calculate_tensor_entropy(data_sample)
            return min(1.0, (variance + entropy) / 2)
        
        elif isinstance(data_sample, str):
            # Para texto: usar diversidad de palabras y longitud
            words = data_sample.split()
            unique_words = len(set(words))
            word_diversity = unique_words / max(len(words), 1)
            length_factor = min(1.0, len(data_sample) / 1000)
            return (word_diversity + length_factor) / 2
        
        elif isinstance(data_sample, (list, tuple)):
            # Para listas: usar variabilidad de elementos
            if len(data_sample) == 0:
                return 0.0
            
            try:
                # Intentar calcular varianza numérica
                numeric_data = [float(x) for x in data_sample if isinstance(x, (int, float))]
                if numeric_data:
                    variance = np.var(numeric_data)
                    return min(1.0, variance / 10)
                else:
                    # Datos no numéricos: usar diversidad
                    unique_elements = len(set(str(x) for x in data_sample))
                    return unique_elements / len(data_sample)
            except:
                return 0.5  # Complejidad media por defecto
        
        else:
            return 0.5  # Complejidad media para tipos desconocidos
    
    def _calculate_tensor_entropy(self, tensor: torch.Tensor) -> float:
        """Calcula entropía aproximada de un tensor"""
        
        # Normalizar y convertir a histograma
        flat_tensor = tensor.flatten()
        hist, _ = np.histogram(flat_tensor.detach().numpy(), bins=50, density=True)
        
        # Eliminar bins vacíos
        hist = hist[hist > 0]
        
        # Calcular entropía
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        return entropy / 10  # Normalizar aproximadamente
    
    def _calculate_context_coherence(self, data_sample, context: Dict) -> float:
        """Calcula coherencia con el contexto actual"""
        
        coherence_score = 0.5  # Base neutral
        
        # Analizar contexto de neurotransmisores
        if 'neurotransmitters' in context:
            nt_levels = context['neurotransmitters']
            
            # Ajustar relevancia basado en estado neuroquímico
            serotonin = nt_levels.get('serotonin', 7.0)
            dopamine = nt_levels.get('dopamine', 50.0)
            
            # Datos complejos son más relevantes con alta dopamina
            if dopamine > 60:
                complexity = self._calculate_complexity(data_sample)
                coherence_score += complexity * 0.3
            
            # Datos emocionales son más relevantes con serotonina variable
            if abs(serotonin - 7.0) > 2.0:  # Desviación del nivel base
                if isinstance(data_sample, str):
                    # Buscar contenido emocional en texto
                    emotional_words = ['feliz', 'triste', 'miedo', 'amor', 'ira', 'alegría']
                    emotional_content = sum(1 for word in emotional_words if word in data_sample.lower())
                    if emotional_content > 0:
                        coherence_score += 0.2
        
        # Analizar contexto de conciencia
        if 'consciousness_state' in context:
            consciousness = context['consciousness_state']
            
            # Estados elevados de conciencia prefieren datos más complejos
            if consciousness in ['deeply_engaged', 'emotionally_resonant']:
                complexity = self._calculate_complexity(data_sample)
                coherence_score += complexity * 0.2
        
        return min(1.0, max(0.0, coherence_score))
    
    def _calculate_learning_potential(self, data_sample) -> float:
        """Calcula el potencial de aprendizaje de los datos"""
        
        # El potencial de aprendizaje se basa en:
        # 1. Diversidad de información
        # 2. Estructura de los datos
        # 3. Riqueza de características
        
        potential_score = 0.0
        
        if isinstance(data_sample, torch.Tensor):
            # Para tensores: analizar distribución y patrones
            shape_complexity = len(data_sample.shape) / 5.0  # Normalizar
            value_range = (torch.max(data_sample) - torch.min(data_sample)).item()
            potential_score = min(1.0, shape_complexity + value_range / 10)
        
        elif isinstance(data_sample, str):
            # Para texto: analizar riqueza lingüística
            words = data_sample.split()
            if words:
                avg_word_length = np.mean([len(word) for word in words])
                sentence_count = data_sample.count('.') + data_sample.count('!') + data_sample.count('?')
                potential_score = min(1.0, (avg_word_length / 10 + sentence_count / 20))
        
        elif isinstance(data_sample, (list, tuple)):
            # Para listas: analizar estructura
            if data_sample:
                type_diversity = len(set(type(x).__name__ for x in data_sample)) / 5.0
                length_factor = min(1.0, len(data_sample) / 100)
                potential_score = min(1.0, type_diversity + length_factor)
        
        return potential_score

class AmiloidAgent:
    """
    Algoritmo Amiloid Agent para optimización y poda de redes neuronales
    
    Este algoritmo simula el proceso de limpieza neural, removiendo conexiones
    y datos irrelevantes para mantener la eficiencia del sistema de conciencia.
    """
    
    def __init__(self, relevance_threshold: float = 0.3, pruning_rate: float = 0.1,
                 evaluation_frequency: int = 10):
        """
        Inicializa el Amiloid Agent
        
        Args:
            relevance_threshold: Umbral mínimo de relevancia para conservar conexiones
            pruning_rate: Tasa de poda por ciclo (0.0 - 1.0)
            evaluation_frequency: Frecuencia de evaluación en épocas
        """
        
        self.relevance_threshold = relevance_threshold
        self.pruning_rate = pruning_rate
        self.evaluation_frequency = evaluation_frequency
        
        # Componentes del agente
        self.connection_tracker = ConnectionRelevanceTracker()
        self.data_analyzer = DataRelevanceAnalyzer()
        
        # Estadísticas de operación
        self.pruning_history = []
        self.connections_pruned = 0
        self.data_samples_removed = 0
        self.efficiency_gains = []
        
        # Estado interno
        self.last_evaluation_epoch = 0
        self.pruned_connections = set()
        
        self.logger = Logger()
        self.logger.log("INFO", "AmiloidAgent initialized")
    
    def evaluate_model_relevance(self, model: nn.Module) -> Dict[str, float]:
        """
        Evalúa la relevancia de todas las conexiones en un modelo
        
        Args:
            model: Modelo PyTorch a evaluar
        
        Returns:
            Diccionario con puntuaciones de relevancia por parámetro
        """
        
        relevance_scores = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Calcular relevancia basada en múltiples factores
                relevance = self._calculate_parameter_relevance(param, name)
                relevance_scores[name] = relevance
                
                # Actualizar tracker
                self.connection_tracker.update_relevance(name, relevance)
        
        return relevance_scores
    
    def _calculate_parameter_relevance(self, param: torch.Tensor, param_name: str) -> float:
        """
        Calcula la relevancia de un parámetro específico
        
        Args:
            param: Tensor de parámetros
            param_name: Nombre del parámetro
        
        Returns:
            Puntuación de relevancia (0.0 - 1.0)
        """
        
        relevance_factors = []
        
        # Factor 1: Magnitud de los pesos
        weight_magnitude = torch.norm(param).item()
        magnitude_factor = min(1.0, weight_magnitude / 10.0)  # Normalizar
        relevance_factors.append(magnitude_factor * 0.3)
        
        # Factor 2: Variabilidad de los pesos
        weight_variance = torch.var(param).item()
        variance_factor = min(1.0, weight_variance)
        relevance_factors.append(variance_factor * 0.2)
        
        # Factor 3: Gradiente histórico (si está disponible)
        if param.grad is not None:
            gradient_magnitude = torch.norm(param.grad).item()
            gradient_factor = min(1.0, gradient_magnitude)
            relevance_factors.append(gradient_factor * 0.3)
        else:
            relevance_factors.append(0.1)  # Factor mínimo si no hay gradiente
        
        # Factor 4: Posición en la red (capas tempranas vs tardías)
        position_factor = self._calculate_position_relevance(param_name)
        relevance_factors.append(position_factor * 0.2)
        
        # Calcular relevancia total
        total_relevance = sum(relevance_factors)
        
        return min(1.0, max(0.0, total_relevance))
    
    def _calculate_position_relevance(self, param_name: str) -> float:
        """
        Calcula factor de relevancia basado en la posición del parámetro en la red
        
        Args:
            param_name: Nombre del parámetro
        
        Returns:
            Factor de relevancia posicional
        """
        
        # Heurísticas basadas en nombres comunes de parámetros
        
        # Capas de embedding y entrada son muy importantes
        if any(keyword in param_name.lower() for keyword in ['embedding', 'input', 'encoder']):
            return 0.9
        
        # Capas de atención son importantes
        if any(keyword in param_name.lower() for keyword in ['attention', 'attn']):
            return 0.8
        
        # Capas de salida son críticas
        if any(keyword in param_name.lower() for keyword in ['output', 'classifier', 'decoder']):
            return 0.9
        
        # Capas ocultas intermedias
        if any(keyword in param_name.lower() for keyword in ['hidden', 'linear', 'conv']):
            return 0.6
        
        # Sesgos suelen ser menos críticos
        if 'bias' in param_name.lower():
            return 0.4
        
        # Por defecto, relevancia media
        return 0.5
    
    def prune_connections(self, model: nn.Module, current_epoch: int = 0) -> Dict[str, Any]:
        """
        Poda conexiones irrelevantes del modelo
        
        Args:
            model: Modelo a podar
            current_epoch: Época actual de entrenamiento
        
        Returns:
            Estadísticas de la poda realizada
        """
        
        # Verificar si es momento de evaluar
        if current_epoch - self.last_evaluation_epoch < self.evaluation_frequency:
            return {'pruning_performed': False, 'reason': 'not_due_for_evaluation'}
        
        self.last_evaluation_epoch = current_epoch
        
        # Evaluar relevancia de conexiones
        relevance_scores = self.evaluate_model_relevance(model)
        
        # Identificar conexiones a podar
        connections_to_prune = []
        for param_name, relevance in relevance_scores.items():
            if relevance < self.relevance_threshold:
                connections_to_prune.append((param_name, relevance))
        
        # Ordenar por relevancia (menos relevantes primero)
        connections_to_prune.sort(key=lambda x: x[1])
        
        # Limitar número de conexiones a podar
        max_to_prune = max(1, int(len(connections_to_prune) * self.pruning_rate))
        connections_to_prune = connections_to_prune[:max_to_prune]
        
        # Realizar poda
        pruned_params = []
        total_params_before = sum(p.numel() for p in model.parameters())
        
        for param_name, relevance in connections_to_prune:
            success = self._prune_parameter(model, param_name, relevance)
            if success:
                pruned_params.append(param_name)
                self.pruned_connections.add(param_name)
                self.connections_pruned += 1
        
        total_params_after = sum(p.numel() for p in model.parameters())
        
        # Calcular estadísticas
        params_removed = total_params_before - total_params_after
        efficiency_gain = params_removed / total_params_before if total_params_before > 0 else 0
        
        self.efficiency_gains.append(efficiency_gain)
        
        # Registrar en historial
        pruning_record = {
            'epoch': current_epoch,
            'connections_pruned': len(pruned_params),
            'total_connections_pruned': self.connections_pruned,
            'params_removed': params_removed,
            'efficiency_gain': efficiency_gain,
            'pruned_parameters': pruned_params
        }
        
        self.pruning_history.append(pruning_record)
        
        self.logger.log("INFO", 
            f"Amiloid Agent pruned {len(pruned_params)} connections, "
            f"efficiency gain: {efficiency_gain:.2%}"
        )
        
        return {
            'pruning_performed': True,
            'connections_pruned': len(pruned_params),
            'parameters_removed': params_removed,
            'efficiency_gain': efficiency_gain,
            'total_efficiency_gain': sum(self.efficiency_gains),
            'pruned_parameter_names': pruned_params
        }
    
    def _prune_parameter(self, model: nn.Module, param_name: str, relevance: float) -> bool:
        """
        Poda un parámetro específico del modelo
        
        Args:
            model: Modelo PyTorch
            param_name: Nombre del parámetro a podar
            relevance: Relevancia del parámetro
        
        Returns:
            True si la poda fue exitosa, False en caso contrario
        """
        
        try:
            # Encontrar el parámetro en el modelo
            param = None
            for name, p in model.named_parameters():
                if name == param_name:
                    param = p
                    break
            
            if param is None:
                return False
            
            # Estrategia de poda basada en relevancia
            if relevance < 0.1:
                # Relevancia muy baja: poda agresiva (90% de conexiones)
                pruning_ratio = 0.9
            elif relevance < 0.2:
                # Relevancia baja: poda moderada (50% de conexiones)
                pruning_ratio = 0.5
            else:
                # Relevancia media-baja: poda conservativa (20% de conexiones)
                pruning_ratio = 0.2
            
            # Crear máscara de poda
            with torch.no_grad():
                # Identificar elementos a podar (los de menor magnitud)
                flat_param = param.flatten()
                num_to_prune = int(len(flat_param) * pruning_ratio)
                
                if num_to_prune > 0:
                    # Encontrar los elementos de menor magnitud
                    _, indices = torch.topk(torch.abs(flat_param), num_to_prune, largest=False)
                    
                    # Crear máscara
                    mask = torch.ones_like(flat_param)
                    mask[indices] = 0
                    
                    # Aplicar máscara
                    flat_param *= mask
                    
                    # Reshape de vuelta a la forma original
                    param.data = flat_param.reshape(param.shape)
            
            return True
            
        except Exception as e:
            self.logger.log("ERROR", f"Failed to prune parameter {param_name}: {str(e)}")
            return False
    
    def evaluate_data_relevance(self, data_batch: List[Any], context: Dict = None) -> List[float]:
        """
        Evalúa la relevancia de un lote de datos
        
        Args:
            data_batch: Lote de muestras de datos
            context: Contexto actual del sistema
        
        Returns:
            Lista de puntuaciones de relevancia para cada muestra
        """
        
        relevance_scores = []
        
        for data_sample in data_batch:
            relevance = self.data_analyzer.evaluate_data_relevance(data_sample, context)
            relevance_scores.append(relevance)
        
        return relevance_scores
    
    def filter_training_data(self, data_batch: List[Any], labels: List[Any] = None, 
                           context: Dict = None) -> Tuple[List[Any], List[Any]]:
        """
        Filtra datos de entrenamiento basado en relevancia
        
        Args:
            data_batch: Lote de datos
            labels: Etiquetas correspondientes (opcional)
            context: Contexto del sistema
        
        Returns:
            Tupla de (datos_filtrados, etiquetas_filtradas)
        """
        
        # Evaluar relevancia
        relevance_scores = self.evaluate_data_relevance(data_batch, context)
        
        # Filtrar datos relevantes
        filtered_data = []
        filtered_labels = []
        
        for i, relevance in enumerate(relevance_scores):
            if relevance >= self.relevance_threshold:
                filtered_data.append(data_batch[i])
                if labels is not None:
                    filtered_labels.append(labels[i])
        
        # Estadísticas
        removed_samples = len(data_batch) - len(filtered_data)
        self.data_samples_removed += removed_samples
        
        if removed_samples > 0:
            self.logger.log("DEBUG", f"AmiloidAgent filtered out {removed_samples} irrelevant data samples")
        
        return filtered_data, filtered_labels if labels is not None else None
    
    def optimize_model_architecture(self, model: nn.Module) -> Dict[str, Any]:
        """
        Optimiza la arquitectura del modelo basado en análisis de relevancia
        
        Args:
            model: Modelo a optimizar
        
        Returns:
            Diccionario con sugerencias de optimización
        """
        
        # Analizar uso de capas
        layer_utilization = self._analyze_layer_utilization(model)
        
        # Identificar cuellos de botella
        bottlenecks = self._identify_bottlenecks(model)
        
        # Generar recomendaciones
        recommendations = self._generate_optimization_recommendations(layer_utilization, bottlenecks)
        
        return {
            'layer_utilization': layer_utilization,
            'bottlenecks': bottlenecks,
            'recommendations': recommendations,
            'current_efficiency': self._calculate_model_efficiency(model)
        }
    
    def _analyze_layer_utilization(self, model: nn.Module) -> Dict[str, float]:
        """Analiza la utilización de cada capa del modelo"""
        
        utilization = {}
        
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.LSTM, nn.MultiheadAttention)):
                # Calcular utilización basada en magnitud de pesos
                total_params = sum(p.numel() for p in module.parameters())
                active_params = 0
                
                for param in module.parameters():
                    if param.requires_grad:
                        # Contar parámetros "activos" (magnitud > umbral)
                        threshold = torch.std(param) * 0.1  # 10% del desvío estándar
                        active_params += torch.sum(torch.abs(param) > threshold).item()
                
                utilization[name] = active_params / total_params if total_params > 0 else 0.0
        
        return utilization
    
    def _identify_bottlenecks(self, model: nn.Module) -> List[str]:
        """Identifica cuellos de botella en el modelo"""
        
        bottlenecks = []
        
        # Analizar dimensionalidades de capas
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                input_dim = module.in_features
                output_dim = module.out_features
                
                # Detectar reducciones drásticas de dimensionalidad
                if input_dim > output_dim * 4:
                    bottlenecks.append(f"{name}: drastic_reduction ({input_dim} -> {output_dim})")
                
                # Detectar expansiones innecesarias
                elif output_dim > input_dim * 4:
                    bottlenecks.append(f"{name}: excessive_expansion ({input_dim} -> {output_dim})")
        
        return bottlenecks
    
    def _generate_optimization_recommendations(self, utilization: Dict[str, float], 
                                             bottlenecks: List[str]) -> List[str]:
        """Genera recomendaciones de optimización"""
        
        recommendations = []
        
        # Recomendaciones basadas en utilización
        for layer_name, util in utilization.items():
            if util < 0.3:
                recommendations.append(f"Consider reducing size of {layer_name} (utilization: {util:.2%})")
            elif util > 0.9:
                recommendations.append(f"Consider expanding {layer_name} (utilization: {util:.2%})")
        
        # Recomendaciones basadas en cuellos de botella
        for bottleneck in bottlenecks:
            recommendations.append(f"Address bottleneck: {bottleneck}")
        
        # Recomendaciones generales
        avg_utilization = np.mean(list(utilization.values())) if utilization else 0.0
        if avg_utilization < 0.5:
            recommendations.append("Overall model utilization is low - consider reducing model size")
        elif avg_utilization > 0.8:
            recommendations.append("Model is highly utilized - consider expanding capacity")
        
        return recommendations
    
    def _calculate_model_efficiency(self, model: nn.Module) -> float:
        """Calcula la eficiencia actual del modelo"""
        
        total_params = sum(p.numel() for p in model.parameters())
        active_params = 0
        
        for param in model.parameters():
            if param.requires_grad:
                # Contar parámetros activos
                threshold = torch.std(param) * 0.1
                active_params += torch.sum(torch.abs(param) > threshold).item()
        
        efficiency = active_params / total_params if total_params > 0 else 0.0
        return efficiency
    
    def get_agent_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas del agente"""
        
        return {
            'total_connections_pruned': self.connections_pruned,
            'total_data_samples_removed': self.data_samples_removed,
            'total_efficiency_gain': sum(self.efficiency_gains),
            'average_efficiency_gain': np.mean(self.efficiency_gains) if self.efficiency_gains else 0.0,
            'pruning_operations': len(self.pruning_history),
            'relevance_threshold': self.relevance_threshold,
            'pruning_rate': self.pruning_rate,
            'last_evaluation_epoch': self.last_evaluation_epoch
        }
    
    def get_pruning_history(self) -> List[Dict[str, Any]]:
        """Obtiene historial de podas"""
        return self.pruning_history.copy()
    
    def reset_agent(self):
        """Reinicia el estado del agente"""
        
        self.pruning_history.clear()
        self.connections_pruned = 0
        self.data_samples_removed = 0
        self.efficiency_gains.clear()
        self.last_evaluation_epoch = 0
        self.pruned_connections.clear()
        
        # Reiniciar componentes
        self.connection_tracker = ConnectionRelevanceTracker()
        self.data_analyzer = DataRelevanceAnalyzer()
        
        self.logger.log("INFO", "AmiloidAgent reset completed")
    
    def adjust_parameters(self, relevance_threshold: float = None, 
                         pruning_rate: float = None, evaluation_frequency: int = None):
        """
        Ajusta los parámetros del agente
        
        Args:
            relevance_threshold: Nuevo umbral de relevancia
            pruning_rate: Nueva tasa de poda
            evaluation_frequency: Nueva frecuencia de evaluación
        """
        
        if relevance_threshold is not None:
            self.relevance_threshold = relevance_threshold
            self.logger.log("INFO", f"Relevance threshold updated to {relevance_threshold}")
        
        if pruning_rate is not None:
            self.pruning_rate = pruning_rate
            self.logger.log("INFO", f"Pruning rate updated to {pruning_rate}")
        
        if evaluation_frequency is not None:
            self.evaluation_frequency = evaluation_frequency
            self.logger.log("INFO", f"Evaluation frequency updated to {evaluation_frequency}")
