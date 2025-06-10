import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Union
import io
from PIL import Image
import base64

from .text_processor import TextProcessor
from .vision_processor import VisionProcessor
from .audio_processor import AudioProcessor
from utils.logger import Logger

class AttentionFusion(nn.Module):
    """
    Módulo de fusión de atención multimodal para combinar información
    de diferentes modalidades (texto, imagen, audio)
    """
    
    def __init__(self, text_dim=768, vision_dim=512, audio_dim=256, hidden_dim=512):
        super(AttentionFusion, self).__init__()
        
        self.text_dim = text_dim
        self.vision_dim = vision_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        
        # Proyecciones para cada modalidad
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.vision_projection = nn.Linear(vision_dim, hidden_dim)
        self.audio_projection = nn.Linear(audio_dim, hidden_dim)
        
        # Atención multimodal
        self.multimodal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Normalización y activación
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh()
        )
        
    def forward(self, text_features=None, vision_features=None, audio_features=None):
        """
        Fusiona características de diferentes modalidades usando atención
        
        Args:
            text_features: Características de texto [batch, text_dim]
            vision_features: Características de visión [batch, vision_dim]
            audio_features: Características de audio [batch, audio_dim]
        
        Returns:
            Características fusionadas [batch, hidden_dim]
        """
        
        available_features = []
        modality_embeddings = []
        
        # Proyectar características disponibles
        if text_features is not None:
            text_proj = self.text_projection(text_features)
            available_features.append(text_proj)
            modality_embeddings.append(text_proj)
        
        if vision_features is not None:
            vision_proj = self.vision_projection(vision_features)
            available_features.append(vision_proj)
            modality_embeddings.append(vision_proj)
        
        if audio_features is not None:
            audio_proj = self.audio_projection(audio_features)
            available_features.append(audio_proj)
            modality_embeddings.append(audio_proj)
        
        if len(available_features) == 0:
            # No hay características disponibles
            batch_size = 1
            return torch.zeros(batch_size, self.hidden_dim)
        
        elif len(available_features) == 1:
            # Solo una modalidad disponible
            return self.layer_norm(available_features[0])
        
        else:
            # Múltiples modalidades - aplicar atención cruzada
            stacked_features = torch.stack(modality_embeddings, dim=1)  # [batch, n_modalities, hidden_dim]
            
            # Atención multimodal
            attended_features, attention_weights = self.multimodal_attention(
                stacked_features, stacked_features, stacked_features
            )
            
            # Fusión final
            flattened = attended_features.flatten(start_dim=1)  # [batch, n_modalities * hidden_dim]
            
            # Pad si es necesario para fusion_layer
            expected_size = self.hidden_dim * 3
            if flattened.size(1) < expected_size:
                padding_size = expected_size - flattened.size(1)
                padding = torch.zeros(flattened.size(0), padding_size, device=flattened.device)
                flattened = torch.cat([flattened, padding], dim=1)
            elif flattened.size(1) > expected_size:
                flattened = flattened[:, :expected_size]
            
            fused = self.fusion_layer(flattened)
            
            return self.layer_norm(fused)

class MultimodalProcessor:
    """
    Procesador multimodal principal que coordina el análisis de texto, imágenes y audio
    Implementa capacidades de AGI para procesamiento integrado de múltiples modalidades
    """
    
    def __init__(self):
        self.logger = Logger()
        
        # Inicializar procesadores específicos
        self.text_processor = TextProcessor()
        self.vision_processor = VisionProcessor()
        self.audio_processor = AudioProcessor()
        
        # Modelo de fusión multimodal
        self.attention_fusion = AttentionFusion()
        
        # Memoria multimodal para contexto
        self.multimodal_memory = {
            'recent_interactions': [],
            'cross_modal_associations': {},
            'semantic_embeddings': {}
        }
        
        # Configuración
        self.max_memory_size = 100
        
        self.logger.log("INFO", "MultimodalProcessor initialized successfully")
    
    def process_text(self, text: str, context: Dict = None) -> Dict:
        """
        Procesa texto usando el sistema de conciencia
        
        Args:
            text: Texto a procesar
            context: Contexto adicional
        
        Returns:
            Diccionario con análisis del texto
        """
        
        try:
            # Procesar con el sistema de texto
            analysis = self.text_processor.analyze_text(text, context)
            
            # Extraer características
            text_features = self.text_processor.extract_features(text)
            
            # Almacenar en memoria multimodal
            self._store_multimodal_memory('text', {
                'content': text,
                'features': text_features,
                'analysis': analysis,
                'context': context
            })
            
            # Buscar asociaciones cross-modales
            cross_modal_context = self._get_cross_modal_context(text, 'text')
            
            result = {
                'modality': 'text',
                'content': text,
                'analysis': analysis,
                'features': text_features.tolist() if isinstance(text_features, torch.Tensor) else text_features,
                'cross_modal_context': cross_modal_context,
                'processing_status': 'success'
            }
            
            self.logger.log("DEBUG", f"Text processed successfully: {len(text)} characters")
            
            return result
            
        except Exception as e:
            self.logger.log("ERROR", f"Text processing failed: {str(e)}")
            return {
                'modality': 'text',
                'content': text,
                'analysis': {'error': str(e)},
                'processing_status': 'error'
            }
    
    def process_image(self, image_file) -> Dict:
        """
        Procesa imagen usando visión artificial con conciencia
        
        Args:
            image_file: Archivo de imagen (Streamlit UploadedFile o similar)
        
        Returns:
            Diccionario con análisis de la imagen
        """
        
        try:
            # Convertir archivo a imagen PIL
            if hasattr(image_file, 'read'):
                image_bytes = image_file.read()
                image = Image.open(io.BytesIO(image_bytes))
            else:
                image = image_file
            
            # Procesar con el sistema de visión
            analysis = self.vision_processor.analyze_image(image)
            
            # Extraer características visuales
            vision_features = self.vision_processor.extract_features(image)
            
            # Generar descripción textual
            description = self.vision_processor.generate_description(image, analysis)
            
            # Análisis emocional de la imagen
            emotional_analysis = self.vision_processor.analyze_emotions(image)
            
            # Almacenar en memoria multimodal
            self._store_multimodal_memory('vision', {
                'description': description,
                'features': vision_features,
                'analysis': analysis,
                'emotional_analysis': emotional_analysis,
                'image_properties': {
                    'size': image.size,
                    'mode': image.mode
                }
            })
            
            # Buscar asociaciones cross-modales
            cross_modal_context = self._get_cross_modal_context(description, 'vision')
            
            result = {
                'modality': 'vision',
                'description': description,
                'analysis': analysis,
                'emotions': emotional_analysis,
                'features': vision_features.tolist() if isinstance(vision_features, torch.Tensor) else vision_features,
                'cross_modal_context': cross_modal_context,
                'image_properties': {
                    'size': image.size,
                    'mode': image.mode,
                    'format': getattr(image, 'format', 'Unknown')
                },
                'processing_status': 'success'
            }
            
            self.logger.log("DEBUG", f"Image processed successfully: {image.size}")
            
            return result
            
        except Exception as e:
            self.logger.log("ERROR", f"Image processing failed: {str(e)}")
            return {
                'modality': 'vision',
                'analysis': {'error': str(e)},
                'processing_status': 'error'
            }
    
    def process_audio(self, audio_file) -> Dict:
        """
        Procesa audio usando redes recurrentes cuánticas
        
        Args:
            audio_file: Archivo de audio
        
        Returns:
            Diccionario con análisis del audio
        """
        
        try:
            # Procesar con el sistema de audio
            analysis = self.audio_processor.analyze_audio(audio_file)
            
            # Extraer características de audio
            audio_features = self.audio_processor.extract_features(audio_file)
            
            # Análisis de sentimientos en audio
            emotional_analysis = self.audio_processor.analyze_emotions(audio_file)
            
            # Transcripción si es posible
            transcription = self.audio_processor.transcribe_audio(audio_file)
            
            # Almacenar en memoria multimodal
            self._store_multimodal_memory('audio', {
                'transcription': transcription,
                'features': audio_features,
                'analysis': analysis,
                'emotional_analysis': emotional_analysis
            })
            
            # Buscar asociaciones cross-modales
            cross_modal_context = self._get_cross_modal_context(transcription, 'audio')
            
            result = {
                'modality': 'audio',
                'transcription': transcription,
                'analysis': analysis,
                'emotions': emotional_analysis,
                'features': audio_features.tolist() if isinstance(audio_features, torch.Tensor) else audio_features,
                'cross_modal_context': cross_modal_context,
                'processing_status': 'success'
            }
            
            self.logger.log("DEBUG", f"Audio processed successfully")
            
            return result
            
        except Exception as e:
            self.logger.log("ERROR", f"Audio processing failed: {str(e)}")
            return {
                'modality': 'audio',
                'analysis': {'error': str(e)},
                'processing_status': 'error'
            }
    
    def process_multimodal(self, text: str = None, image_file=None, audio_file=None) -> Dict:
        """
        Procesa múltiples modalidades de forma integrada
        
        Args:
            text: Texto opcional
            image_file: Imagen opcional
            audio_file: Audio opcional
        
        Returns:
            Análisis integrado de todas las modalidades
        """
        
        results = {
            'modalities_processed': [],
            'individual_results': {},
            'fusion_result': {},
            'cross_modal_insights': [],
            'processing_status': 'success'
        }
        
        features_for_fusion = {}
        
        try:
            # Procesar cada modalidad disponible
            if text:
                text_result = self.process_text(text)
                results['individual_results']['text'] = text_result
                results['modalities_processed'].append('text')
                
                if text_result['processing_status'] == 'success':
                    features_for_fusion['text'] = torch.tensor(text_result['features'])
            
            if image_file:
                image_result = self.process_image(image_file)
                results['individual_results']['vision'] = image_result
                results['modalities_processed'].append('vision')
                
                if image_result['processing_status'] == 'success':
                    features_for_fusion['vision'] = torch.tensor(image_result['features'])
            
            if audio_file:
                audio_result = self.process_audio(audio_file)
                results['individual_results']['audio'] = audio_result
                results['modalities_processed'].append('audio')
                
                if audio_result['processing_status'] == 'success':
                    features_for_fusion['audio'] = torch.tensor(audio_result['features'])
            
            # Fusión multimodal si hay múltiples modalidades
            if len(features_for_fusion) > 1:
                fusion_result = self._perform_multimodal_fusion(features_for_fusion)
                results['fusion_result'] = fusion_result
                
                # Generar insights cross-modales
                cross_modal_insights = self._generate_cross_modal_insights(results['individual_results'])
                results['cross_modal_insights'] = cross_modal_insights
            
            # Análisis holístico
            holistic_analysis = self._perform_holistic_analysis(results)
            results['holistic_analysis'] = holistic_analysis
            
            self.logger.log("INFO", f"Multimodal processing completed: {results['modalities_processed']}")
            
            return results
            
        except Exception as e:
            self.logger.log("ERROR", f"Multimodal processing failed: {str(e)}")
            results['processing_status'] = 'error'
            results['error'] = str(e)
            return results
    
    def _perform_multimodal_fusion(self, features: Dict[str, torch.Tensor]) -> Dict:
        """Realiza fusión de características multimodales"""
        
        try:
            # Preparar características para fusión
            text_features = features.get('text')
            vision_features = features.get('vision')
            audio_features = features.get('audio')
            
            # Asegurar dimensiones correctas (agregar batch dimension si es necesario)
            if text_features is not None and text_features.dim() == 1:
                text_features = text_features.unsqueeze(0)
            if vision_features is not None and vision_features.dim() == 1:
                vision_features = vision_features.unsqueeze(0)
            if audio_features is not None and audio_features.dim() == 1:
                audio_features = audio_features.unsqueeze(0)
            
            # Aplicar fusión de atención
            with torch.no_grad():
                fused_features = self.attention_fusion(text_features, vision_features, audio_features)
            
            # Análisis de la representación fusionada
            fusion_analysis = {
                'fused_feature_size': fused_features.shape,
                'fusion_magnitude': float(torch.norm(fused_features)),
                'feature_distribution': {
                    'mean': float(torch.mean(fused_features)),
                    'std': float(torch.std(fused_features)),
                    'min': float(torch.min(fused_features)),
                    'max': float(torch.max(fused_features))
                },
                'modality_contributions': self._analyze_modality_contributions(features),
                'semantic_coherence': self._calculate_semantic_coherence(fused_features)
            }
            
            return fusion_analysis
            
        except Exception as e:
            self.logger.log("ERROR", f"Multimodal fusion failed: {str(e)}")
            return {'error': str(e)}
    
    def _analyze_modality_contributions(self, features: Dict[str, torch.Tensor]) -> Dict:
        """Analiza la contribución de cada modalidad a la representación final"""
        
        contributions = {}
        total_magnitude = 0
        
        for modality, feature_tensor in features.items():
            magnitude = float(torch.norm(feature_tensor))
            contributions[modality] = magnitude
            total_magnitude += magnitude
        
        # Normalizar contribuciones
        if total_magnitude > 0:
            for modality in contributions:
                contributions[modality] = contributions[modality] / total_magnitude
        
        return contributions
    
    def _calculate_semantic_coherence(self, fused_features: torch.Tensor) -> float:
        """Calcula la coherencia semántica de las características fusionadas"""
        
        # Medida de coherencia basada en la distribución de activaciones
        std = float(torch.std(fused_features))
        mean = float(torch.mean(torch.abs(fused_features)))
        
        # Coherencia alta = distribución equilibrada (no demasiado dispersa ni concentrada)
        if mean > 0:
            coherence = 1.0 / (1.0 + std / mean)
        else:
            coherence = 0.0
        
        return min(1.0, coherence)
    
    def _generate_cross_modal_insights(self, individual_results: Dict) -> List[str]:
        """Genera insights basados en conexiones entre modalidades"""
        
        insights = []
        
        # Análisis texto-imagen
        if 'text' in individual_results and 'vision' in individual_results:
            text_analysis = individual_results['text'].get('analysis', {})
            vision_analysis = individual_results['vision'].get('analysis', {})
            
            # Coherencia emocional
            if 'emotion' in text_analysis and 'emotions' in individual_results['vision']:
                insights.append("Detectada correlación emocional entre texto e imagen")
            
            # Coherencia temática
            text_content = individual_results['text'].get('content', '').lower()
            image_description = individual_results['vision'].get('description', '').lower()
            
            common_themes = self._find_common_themes(text_content, image_description)
            if common_themes:
                insights.append(f"Temas comunes identificados: {', '.join(common_themes)}")
        
        # Análisis texto-audio
        if 'text' in individual_results and 'audio' in individual_results:
            audio_transcription = individual_results['audio'].get('transcription', '').lower()
            text_content = individual_results['text'].get('content', '').lower()
            
            if audio_transcription and text_content:
                similarity = self._calculate_text_similarity(text_content, audio_transcription)
                if similarity > 0.3:
                    insights.append("Alta coherencia entre texto escrito y contenido de audio")
        
        # Análisis imagen-audio
        if 'vision' in individual_results and 'audio' in individual_results:
            vision_emotions = individual_results['vision'].get('emotions', {})
            audio_emotions = individual_results['audio'].get('emotions', {})
            
            if vision_emotions and audio_emotions:
                emotion_correlation = self._calculate_emotion_correlation(vision_emotions, audio_emotions)
                if emotion_correlation > 0.5:
                    insights.append("Fuerte correlación emocional entre imagen y audio")
        
        return insights
    
    def _find_common_themes(self, text1: str, text2: str) -> List[str]:
        """Encuentra temas comunes entre dos textos"""
        
        # Palabras clave temáticas
        theme_keywords = {
            'naturaleza': ['árbol', 'planta', 'flor', 'animal', 'bosque', 'mar', 'montaña'],
            'tecnología': ['computadora', 'robot', 'inteligencia', 'artificial', 'código', 'programa'],
            'emociones': ['feliz', 'triste', 'alegre', 'enojado', 'miedo', 'amor', 'odio'],
            'arte': ['pintura', 'música', 'danza', 'teatro', 'creatividad', 'belleza'],
            'ciencia': ['experimento', 'hipótesis', 'teoría', 'investigación', 'análisis']
        }
        
        common_themes = []
        
        for theme, keywords in theme_keywords.items():
            if (any(keyword in text1 for keyword in keywords) and 
                any(keyword in text2 for keyword in keywords)):
                common_themes.append(theme)
        
        return common_themes
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calcula similitud simple entre dos textos"""
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if len(words1) == 0 or len(words2) == 0:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_emotion_correlation(self, emotions1: Dict, emotions2: Dict) -> float:
        """Calcula correlación entre análisis emocionales"""
        
        if not emotions1 or not emotions2:
            return 0.0
        
        # Encontrar emociones comunes
        common_emotions = set(emotions1.keys()).intersection(set(emotions2.keys()))
        
        if not common_emotions:
            return 0.0
        
        # Calcular correlación de intensidades
        correlations = []
        for emotion in common_emotions:
            val1 = emotions1.get(emotion, 0)
            val2 = emotions2.get(emotion, 0)
            correlations.append(min(val1, val2) / max(val1, val2, 1e-6))
        
        return np.mean(correlations) if correlations else 0.0
    
    def _perform_holistic_analysis(self, results: Dict) -> Dict:
        """Realiza análisis holístico de toda la información multimodal"""
        
        holistic = {
            'overall_coherence': 0.0,
            'dominant_modality': None,
            'emotional_state': 'neutral',
            'thematic_consistency': 0.0,
            'complexity_score': 0.0,
            'consciousness_indicators': []
        }
        
        # Calcular coherencia general
        coherence_scores = []
        modality_strengths = {}
        
        for modality, result in results.get('individual_results', {}).items():
            if result.get('processing_status') == 'success':
                # Fuerza de la modalidad basada en cantidad de información
                if modality == 'text':
                    strength = len(result.get('content', '')) / 1000.0
                elif modality == 'vision':
                    strength = 0.8  # Fuerza fija para imágenes
                elif modality == 'audio':
                    strength = 0.7  # Fuerza fija para audio
                
                modality_strengths[modality] = min(1.0, strength)
                coherence_scores.append(strength)
        
        if coherence_scores:
            holistic['overall_coherence'] = np.mean(coherence_scores)
            holistic['dominant_modality'] = max(modality_strengths.keys(), 
                                              key=lambda k: modality_strengths[k])
        
        # Análisis emocional integrado
        all_emotions = {}
        for result in results.get('individual_results', {}).values():
            emotions = result.get('emotions', {})
            for emotion, intensity in emotions.items():
                if emotion in all_emotions:
                    all_emotions[emotion] = max(all_emotions[emotion], intensity)
                else:
                    all_emotions[emotion] = intensity
        
        if all_emotions:
            dominant_emotion = max(all_emotions.keys(), key=lambda k: all_emotions[k])
            holistic['emotional_state'] = dominant_emotion
        
        # Indicadores de conciencia artificial
        consciousness_indicators = []
        
        if len(results.get('cross_modal_insights', [])) > 0:
            consciousness_indicators.append("cross_modal_integration")
        
        if holistic['overall_coherence'] > 0.7:
            consciousness_indicators.append("high_coherence_processing")
        
        if len(results.get('modalities_processed', [])) > 1:
            consciousness_indicators.append("multimodal_awareness")
        
        holistic['consciousness_indicators'] = consciousness_indicators
        holistic['complexity_score'] = len(consciousness_indicators) / 3.0
        
        return holistic
    
    def _store_multimodal_memory(self, modality: str, data: Dict):
        """Almacena información en memoria multimodal"""
        
        from datetime import datetime
        
        memory_entry = {
            'timestamp': datetime.now().isoformat(),
            'modality': modality,
            'data': data
        }
        
        self.multimodal_memory['recent_interactions'].append(memory_entry)
        
        # Mantener tamaño limitado
        if len(self.multimodal_memory['recent_interactions']) > self.max_memory_size:
            self.multimodal_memory['recent_interactions'] = (
                self.multimodal_memory['recent_interactions'][-self.max_memory_size:]
            )
    
    def _get_cross_modal_context(self, content: str, current_modality: str) -> Dict:
        """Obtiene contexto cross-modal relevante"""
        
        context = {
            'related_memories': [],
            'semantic_associations': {},
            'temporal_context': None
        }
        
        # Buscar memorias relacionadas de otras modalidades
        content_lower = content.lower()
        
        for memory in self.multimodal_memory['recent_interactions'][-10:]:  # Últimas 10
            if memory['modality'] != current_modality:
                # Buscar palabras clave comunes
                if memory['modality'] == 'text':
                    memory_content = memory['data'].get('content', '').lower()
                elif memory['modality'] == 'vision':
                    memory_content = memory['data'].get('description', '').lower()
                elif memory['modality'] == 'audio':
                    memory_content = memory['data'].get('transcription', '').lower()
                else:
                    memory_content = ''
                
                # Calcular similitud
                similarity = self._calculate_text_similarity(content_lower, memory_content)
                
                if similarity > 0.2:  # Umbral de similitud
                    context['related_memories'].append({
                        'modality': memory['modality'],
                        'similarity': similarity,
                        'timestamp': memory['timestamp']
                    })
        
        return context
    
    def get_memory_summary(self) -> Dict:
        """Obtiene resumen de la memoria multimodal"""
        
        summary = {
            'total_interactions': len(self.multimodal_memory['recent_interactions']),
            'modality_distribution': {},
            'recent_activity': [],
            'cross_modal_associations': len(self.multimodal_memory['cross_modal_associations'])
        }
        
        # Distribución por modalidad
        for memory in self.multimodal_memory['recent_interactions']:
            modality = memory['modality']
            if modality in summary['modality_distribution']:
                summary['modality_distribution'][modality] += 1
            else:
                summary['modality_distribution'][modality] = 1
        
        # Actividad reciente (últimas 5 interacciones)
        for memory in self.multimodal_memory['recent_interactions'][-5:]:
            summary['recent_activity'].append({
                'modality': memory['modality'],
                'timestamp': memory['timestamp']
            })
        
        return summary
    
    def reset_memory(self):
        """Reinicia la memoria multimodal"""
        
        self.multimodal_memory = {
            'recent_interactions': [],
            'cross_modal_associations': {},
            'semantic_embeddings': {}
        }
        
        self.logger.log("INFO", "Multimodal memory reset")
    
    def check_health(self) -> bool:
        """Verifica el estado de salud del procesador multimodal"""
        
        try:
            # Verificar componentes
            text_health = self.text_processor.check_health()
            vision_health = self.vision_processor.check_health()
            audio_health = self.audio_processor.check_health()
            
            # Test de fusión
            test_features = {
                'text': torch.randn(768),
                'vision': torch.randn(512)
            }
            _ = self._perform_multimodal_fusion(test_features)
            
            return text_health and vision_health and audio_health
            
        except Exception as e:
            self.logger.log("ERROR", f"Multimodal processor health check failed: {str(e)}")
            return False

