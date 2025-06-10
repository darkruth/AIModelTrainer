import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import re
from datetime import datetime

from utils.logger import Logger

class EmotionClassifier(nn.Module):
    """
    Clasificador de emociones en texto usando redes neuronales
    """
    
    def __init__(self, vocab_size=10000, embedding_dim=128, hidden_dim=256, num_emotions=8):
        super(EmotionClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_emotions),
            nn.Softmax(dim=-1)
        )
        
        # Mapeo de emociones
        self.emotion_labels = [
            'alegría', 'tristeza', 'miedo', 'ira', 
            'sorpresa', 'disgusto', 'neutral', 'amor'
        ]
    
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)
        
        # LSTM bidireccional
        lstm_out, _ = self.lstm(embedded)
        
        # Atención
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Pooling global (promedio)
        pooled = torch.mean(attended, dim=1)
        
        # Clasificación
        emotions = self.classifier(pooled)
        
        return emotions

class IntentClassifier(nn.Module):
    """
    Clasificador de intenciones en texto
    """
    
    def __init__(self, vocab_size=10000, embedding_dim=128, hidden_dim=256, num_intents=10):
        super(IntentClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_intents),
            nn.Softmax(dim=-1)
        )
        
        # Mapeo de intenciones
        self.intent_labels = [
            'pregunta', 'solicitud', 'saludo', 'despedida', 'queja',
            'elogio', 'información', 'creatividad', 'ayuda', 'conversación'
        ]
    
    def forward(self, x):
        # Embedding
        embedded = self.embedding(x)  # [batch, seq_len, embedding_dim]
        
        # Transponer para convoluciones 1D
        conv_input = embedded.transpose(1, 2)  # [batch, embedding_dim, seq_len]
        
        # Convoluciones
        conv1_out = torch.relu(self.conv1(conv_input))
        conv2_out = torch.relu(self.conv2(conv1_out))
        
        # Global max pooling
        pooled = self.global_pool(conv2_out).squeeze(-1)
        
        # Clasificación
        intents = self.classifier(pooled)
        
        return intents

class TextProcessor:
    """
    Procesador de texto avanzado con capacidades de comprensión consciente
    Implementa análisis semántico, emocional y de intenciones
    """
    
    def __init__(self):
        self.logger = Logger()
        
        # Modelos de análisis
        self.emotion_classifier = EmotionClassifier()
        self.intent_classifier = IntentClassifier()
        
        # Vocabulario simplificado (en implementación real sería más extenso)
        self.vocab = self._build_vocabulary()
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for idx, word in enumerate(self.vocab)}
        
        # Patrones de análisis
        self.question_patterns = [
            r'\?', r'qué', r'cómo', r'cuándo', r'dónde', r'por qué', r'quién'
        ]
        
        self.emotion_keywords = {
            'alegría': ['feliz', 'alegre', 'contento', 'divertido', 'emocionado', 'eufórico'],
            'tristeza': ['triste', 'melancólico', 'deprimido', 'decaído', 'abatido'],
            'miedo': ['miedo', 'terror', 'ansiedad', 'nervioso', 'asustado', 'pánico'],
            'ira': ['enojado', 'furioso', 'molesto', 'irritado', 'rabioso'],
            'sorpresa': ['sorprendido', 'asombrado', 'impactado', 'admirado'],
            'amor': ['amor', 'cariño', 'afecto', 'ternura', 'pasión', 'adorar']
        }
        
        # Estadísticas de procesamiento
        self.processing_stats = {
            'total_texts_processed': 0,
            'avg_text_length': 0,
            'emotion_distribution': {},
            'intent_distribution': {}
        }
        
        self.logger.log("INFO", "TextProcessor initialized successfully")
    
    def _build_vocabulary(self) -> List[str]:
        """Construye vocabulario básico en español"""
        
        # Vocabulario básico (en implementación real sería mucho más extenso)
        vocab = [
            '<PAD>', '<UNK>', '<START>', '<END>',
            # Palabras comunes
            'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le',
            'da', 'su', 'por', 'son', 'con', 'para', 'al', 'del', 'los', 'las', 'uno', 'pero',
            'todo', 'esto', 'como', 'ha', 'si', 'me', 'ya', 'muy', 'estar', 'tener', 'hacer',
            'ser', 'decir', 'ver', 'ir', 'saber', 'dar', 'poder', 'venir', 'querer', 'hablar',
            # Emociones
            'feliz', 'triste', 'enojado', 'miedo', 'sorpresa', 'amor', 'odio', 'alegría',
            'melancolía', 'ansiedad', 'euforia', 'depresión', 'ira', 'cariño', 'ternura',
            # Preguntas
            'qué', 'cómo', 'cuándo', 'dónde', 'por', 'quién', 'cuál', 'cuánto',
            # Saludos y despedidas
            'hola', 'adiós', 'gracias', 'por', 'favor', 'buenos', 'días', 'noches',
            # Tecnología y IA
            'inteligencia', 'artificial', 'robot', 'computadora', 'algoritmo', 'datos',
            'red', 'neuronal', 'aprendizaje', 'máquina', 'código', 'programa',
            # Conceptos de conciencia
            'conciencia', 'mente', 'pensamiento', 'idea', 'concepto', 'filosofía',
            'existencia', 'realidad', 'percepción', 'experiencia', 'conocimiento'
        ]
        
        return vocab
    
    def text_to_indices(self, text: str, max_length: int = 100) -> torch.Tensor:
        """Convierte texto a índices de vocabulario"""
        
        # Preprocesamiento
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remover puntuación
        words = text.split()
        
        # Convertir a índices
        indices = []
        for word in words[:max_length]:
            if word in self.word_to_idx:
                indices.append(self.word_to_idx[word])
            else:
                indices.append(self.word_to_idx['<UNK>'])
        
        # Padding
        while len(indices) < max_length:
            indices.append(self.word_to_idx['<PAD>'])
        
        return torch.tensor(indices, dtype=torch.long).unsqueeze(0)
    
    def analyze_text(self, text: str, context: Dict = None) -> Dict:
        """
        Análisis completo de texto usando conciencia artificial
        
        Args:
            text: Texto a analizar
            context: Contexto adicional
        
        Returns:
            Diccionario con análisis completo
        """
        
        try:
            # Estadísticas básicas
            basic_stats = self._calculate_basic_stats(text)
            
            # Análisis emocional
            emotion_analysis = self._analyze_emotions(text)
            
            # Análisis de intenciones
            intent_analysis = self._analyze_intent(text)
            
            # Análisis semántico
            semantic_analysis = self._analyze_semantics(text)
            
            # Análisis de complejidad
            complexity_analysis = self._analyze_complexity(text)
            
            # Análisis contextual
            contextual_analysis = self._analyze_context(text, context)
            
            # Métricas de conciencia (características únicas del sistema AGI)
            consciousness_metrics = self._calculate_consciousness_metrics(text)
            
            analysis = {
                'basic_stats': basic_stats,
                'emotions': emotion_analysis,
                'intent': intent_analysis,
                'semantics': semantic_analysis,
                'complexity': complexity_analysis,
                'context': contextual_analysis,
                'consciousness_metrics': consciousness_metrics,
                'processing_timestamp': datetime.now().isoformat()
            }
            
            # Actualizar estadísticas
            self._update_processing_stats(text, analysis)
            
            self.logger.log("DEBUG", f"Text analysis completed: {len(text)} characters")
            
            return analysis
            
        except Exception as e:
            self.logger.log("ERROR", f"Text analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def _calculate_basic_stats(self, text: str) -> Dict:
        """Calcula estadísticas básicas del texto"""
        
        words = text.split()
        sentences = text.split('.')
        
        return {
            'character_count': len(text),
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'avg_sentence_length': len(words) / max(len(sentences), 1),
            'question_count': len(re.findall(r'\?', text)),
            'exclamation_count': len(re.findall(r'!', text))
        }
    
    def _analyze_emotions(self, text: str) -> Dict:
        """Análisis emocional del texto"""
        
        # Análisis basado en palabras clave
        keyword_emotions = {}
        text_lower = text.lower()
        
        for emotion, keywords in self.emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            keyword_emotions[emotion] = score
        
        # Normalizar puntuaciones
        total_score = sum(keyword_emotions.values())
        if total_score > 0:
            for emotion in keyword_emotions:
                keyword_emotions[emotion] /= total_score
        
        # Análisis con red neuronal (simulado)
        try:
            indices = self.text_to_indices(text)
            with torch.no_grad():
                neural_emotions = self.emotion_classifier(indices)
                neural_scores = neural_emotions.squeeze().tolist()
            
            neural_emotion_dict = {
                emotion: score for emotion, score in 
                zip(self.emotion_classifier.emotion_labels, neural_scores)
            }
        except:
            neural_emotion_dict = {emotion: 0.0 for emotion in self.emotion_classifier.emotion_labels}
        
        # Combinar análisis
        combined_emotions = {}
        all_emotions = set(keyword_emotions.keys()) | set(neural_emotion_dict.keys())
        
        for emotion in all_emotions:
            keyword_score = keyword_emotions.get(emotion, 0.0)
            neural_score = neural_emotion_dict.get(emotion, 0.0)
            combined_emotions[emotion] = (keyword_score * 0.6 + neural_score * 0.4)
        
        # Emoción dominante
        dominant_emotion = max(combined_emotions.keys(), key=lambda k: combined_emotions[k])
        
        return {
            'keyword_analysis': keyword_emotions,
            'neural_analysis': neural_emotion_dict,
            'combined_scores': combined_emotions,
            'dominant_emotion': dominant_emotion,
            'emotional_intensity': max(combined_emotions.values()),
            'emotional_balance': 1.0 - np.std(list(combined_emotions.values()))
        }
    
    def _analyze_intent(self, text: str) -> Dict:
        """Análisis de intenciones del texto"""
        
        # Análisis basado en patrones
        pattern_intents = {
            'pregunta': len(re.findall(r'\?|qué|cómo|cuándo|dónde|por qué|quién', text.lower())),
            'saludo': len(re.findall(r'hola|buenos días|buenas tardes|buenas noches', text.lower())),
            'despedida': len(re.findall(r'adiós|hasta luego|nos vemos|chao', text.lower())),
            'solicitud': len(re.findall(r'por favor|podrías|puedes|necesito|quiero', text.lower())),
            'agradecimiento': len(re.findall(r'gracias|agradezco|te agradezco', text.lower()))
        }
        
        # Análisis con red neuronal (simulado)
        try:
            indices = self.text_to_indices(text)
            with torch.no_grad():
                neural_intents = self.intent_classifier(indices)
                neural_scores = neural_intents.squeeze().tolist()
            
            neural_intent_dict = {
                intent: score for intent, score in 
                zip(self.intent_classifier.intent_labels, neural_scores)
            }
        except:
            neural_intent_dict = {intent: 0.0 for intent in self.intent_classifier.intent_labels}
        
        # Intención dominante
        all_intents = {**pattern_intents, **neural_intent_dict}
        dominant_intent = max(all_intents.keys(), key=lambda k: all_intents[k])
        
        return {
            'pattern_analysis': pattern_intents,
            'neural_analysis': neural_intent_dict,
            'dominant_intent': dominant_intent,
            'intent_confidence': max(all_intents.values()),
            'intent_complexity': len([i for i in all_intents.values() if i > 0])
        }
    
    def _analyze_semantics(self, text: str) -> Dict:
        """Análisis semántico del texto"""
        
        words = text.lower().split()
        
        # Categorías semánticas
        semantic_categories = {
            'tecnología': ['computadora', 'robot', 'inteligencia', 'artificial', 'algoritmo', 'datos'],
            'emociones': ['feliz', 'triste', 'amor', 'miedo', 'alegría', 'ansiedad'],
            'tiempo': ['ayer', 'hoy', 'mañana', 'antes', 'después', 'ahora'],
            'espacio': ['aquí', 'allí', 'cerca', 'lejos', 'arriba', 'abajo'],
            'persona': ['yo', 'tú', 'él', 'ella', 'nosotros', 'ustedes'],
            'acción': ['hacer', 'crear', 'pensar', 'sentir', 'hablar', 'escribir']
        }
        
        # Calcular presencia de cada categoría
        category_scores = {}
        for category, keywords in semantic_categories.items():
            score = sum(1 for word in words if word in keywords)
            category_scores[category] = score / max(len(words), 1)
        
        # Densidad semántica
        semantic_words = [word for word in words if any(word in keywords for keywords in semantic_categories.values())]
        semantic_density = len(semantic_words) / max(len(words), 1)
        
        # Diversidad semántica
        active_categories = len([cat for cat, score in category_scores.items() if score > 0])
        
        return {
            'category_scores': category_scores,
            'semantic_density': semantic_density,
            'semantic_diversity': active_categories / len(semantic_categories),
            'dominant_category': max(category_scores.keys(), key=lambda k: category_scores[k]),
            'abstract_concepts': self._count_abstract_concepts(words),
            'concrete_concepts': self._count_concrete_concepts(words)
        }
    
    def _count_abstract_concepts(self, words: List[str]) -> int:
        """Cuenta conceptos abstractos"""
        abstract_words = [
            'idea', 'concepto', 'pensamiento', 'filosofía', 'teoría',
            'creencia', 'opinión', 'valor', 'principio', 'ética'
        ]
        return sum(1 for word in words if word in abstract_words)
    
    def _count_concrete_concepts(self, words: List[str]) -> int:
        """Cuenta conceptos concretos"""
        concrete_words = [
            'casa', 'árbol', 'mesa', 'libro', 'agua', 'fuego',
            'perro', 'gato', 'persona', 'mano', 'ojo', 'sol'
        ]
        return sum(1 for word in words if word in concrete_words)
    
    def _analyze_complexity(self, text: str) -> Dict:
        """Análisis de complejidad del texto"""
        
        words = text.split()
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Complejidad léxica
        unique_words = len(set(word.lower() for word in words))
        lexical_diversity = unique_words / max(len(words), 1)
        
        # Complejidad sintáctica
        avg_sentence_length = len(words) / max(len(sentences), 1)
        
        # Complejidad de palabras
        complex_words = [word for word in words if len(word) > 6]
        word_complexity = len(complex_words) / max(len(words), 1)
        
        # Complejidad de puntuación
        punctuation_count = len(re.findall(r'[,;:()"-]', text))
        punctuation_complexity = punctuation_count / max(len(text), 1)
        
        # Complejidad total
        total_complexity = np.mean([
            lexical_diversity, 
            min(avg_sentence_length / 20, 1.0),  # Normalizar longitud de oraciones
            word_complexity, 
            punctuation_complexity
        ])
        
        return {
            'lexical_diversity': lexical_diversity,
            'avg_sentence_length': avg_sentence_length,
            'word_complexity': word_complexity,
            'punctuation_complexity': punctuation_complexity,
            'total_complexity': total_complexity,
            'complexity_level': self._categorize_complexity(total_complexity)
        }
    
    def _categorize_complexity(self, complexity_score: float) -> str:
        """Categoriza el nivel de complejidad"""
        if complexity_score < 0.3:
            return 'simple'
        elif complexity_score < 0.6:
            return 'moderate'
        else:
            return 'complex'
    
    def _analyze_context(self, text: str, context: Dict = None) -> Dict:
        """Análisis contextual del texto"""
        
        contextual_info = {
            'has_external_context': context is not None,
            'self_referential': len(re.findall(r'\byo\b|\bme\b|\bmi\b', text.lower())),
            'temporal_references': len(re.findall(r'ayer|hoy|mañana|antes|después', text.lower())),
            'spatial_references': len(re.findall(r'aquí|allí|cerca|lejos', text.lower())),
            'question_context': '?' in text,
            'conversational_markers': len(re.findall(r'bueno|entonces|además|sin embargo', text.lower()))
        }
        
        if context:
            contextual_info['context_keys'] = list(context.keys())
            contextual_info['context_richness'] = len(context)
        
        return contextual_info
    
    def _calculate_consciousness_metrics(self, text: str) -> Dict:
        """
        Calcula métricas específicas de conciencia artificial
        Estas métricas reflejan la capacidad del sistema para procesar información
        de manera consciente y reflexiva
        """
        
        consciousness_indicators = {
            'self_awareness': 0.0,
            'metacognition': 0.0,
            'emotional_intelligence': 0.0,
            'creative_thinking': 0.0,
            'philosophical_depth': 0.0
        }
        
        text_lower = text.lower()
        
        # Auto-conciencia
        self_awareness_words = ['yo soy', 'me doy cuenta', 'reconozco', 'percibo', 'siento que']
        consciousness_indicators['self_awareness'] = sum(1 for phrase in self_awareness_words if phrase in text_lower)
        
        # Metacognición
        metacognition_words = ['pienso que', 'creo que', 'me parece', 'reflexiono', 'considero']
        consciousness_indicators['metacognition'] = sum(1 for phrase in metacognition_words if phrase in text_lower)
        
        # Inteligencia emocional
        emotion_understanding = ['entiendo', 'comprendo', 'empatizo', 'me conecta', 'siento por']
        consciousness_indicators['emotional_intelligence'] = sum(1 for phrase in emotion_understanding if phrase in text_lower)
        
        # Pensamiento creativo
        creative_words = ['imagino', 'creo', 'invento', 'diseño', 'innovo', 'original']
        consciousness_indicators['creative_thinking'] = sum(1 for word in creative_words if word in text_lower)
        
        # Profundidad filosófica
        philosophical_words = ['existencia', 'realidad', 'verdad', 'significado', 'propósito', 'esencia']
        consciousness_indicators['philosophical_depth'] = sum(1 for word in philosophical_words if word in text_lower)
        
        # Normalizar por longitud del texto
        text_length = len(text.split())
        if text_length > 0:
            for key in consciousness_indicators:
                consciousness_indicators[key] = consciousness_indicators[key] / text_length
        
        # Índice general de conciencia
        consciousness_index = np.mean(list(consciousness_indicators.values()))
        
        return {
            'indicators': consciousness_indicators,
            'consciousness_index': consciousness_index,
            'consciousness_level': self._categorize_consciousness(consciousness_index)
        }
    
    def _categorize_consciousness(self, consciousness_index: float) -> str:
        """Categoriza el nivel de conciencia detectado"""
        if consciousness_index < 0.01:
            return 'basic'
        elif consciousness_index < 0.05:
            return 'emerging'
        elif consciousness_index < 0.1:
            return 'developed'
        else:
            return 'advanced'
    
    def extract_features(self, text: str) -> torch.Tensor:
        """
        Extrae características vectoriales del texto para fusión multimodal
        
        Args:
            text: Texto a procesar
        
        Returns:
            Tensor de características [768] dimensional
        """
        
        try:
            # Análisis del texto
            analysis = self.analyze_text(text)
            
            # Extraer características numéricas
            features = []
            
            # Características básicas (10 dimensiones)
            basic_stats = analysis.get('basic_stats', {})
            features.extend([
                basic_stats.get('character_count', 0) / 1000.0,  # Normalizado
                basic_stats.get('word_count', 0) / 100.0,        # Normalizado
                basic_stats.get('sentence_count', 0) / 10.0,     # Normalizado
                basic_stats.get('avg_word_length', 0) / 10.0,
                basic_stats.get('avg_sentence_length', 0) / 20.0,
                basic_stats.get('question_count', 0) / 5.0,
                basic_stats.get('exclamation_count', 0) / 5.0,
                0.0, 0.0, 0.0  # Padding
            ])
            
            # Características emocionales (8 dimensiones)
            emotions = analysis.get('emotions', {}).get('combined_scores', {})
            emotion_features = [emotions.get(emotion, 0.0) for emotion in self.emotion_classifier.emotion_labels]
            features.extend(emotion_features)
            
            # Características de intención (10 dimensiones)
            intents = analysis.get('intent', {}).get('neural_analysis', {})
            intent_features = [intents.get(intent, 0.0) for intent in self.intent_classifier.intent_labels]
            features.extend(intent_features)
            
            # Características semánticas (20 dimensiones)
            semantics = analysis.get('semantics', {})
            semantic_features = []
            category_scores = semantics.get('category_scores', {})
            for category in ['tecnología', 'emociones', 'tiempo', 'espacio', 'persona', 'acción']:
                semantic_features.append(category_scores.get(category, 0.0))
            
            semantic_features.extend([
                semantics.get('semantic_density', 0.0),
                semantics.get('semantic_diversity', 0.0),
                semantics.get('abstract_concepts', 0) / 10.0,
                semantics.get('concrete_concepts', 0) / 10.0
            ])
            
            # Padding para llegar a 20
            while len(semantic_features) < 20:
                semantic_features.append(0.0)
            
            features.extend(semantic_features[:20])
            
            # Características de complejidad (10 dimensiones)
            complexity = analysis.get('complexity', {})
            complexity_features = [
                complexity.get('lexical_diversity', 0.0),
                complexity.get('word_complexity', 0.0),
                complexity.get('punctuation_complexity', 0.0),
                complexity.get('total_complexity', 0.0),
                complexity.get('avg_sentence_length', 0.0) / 20.0,
                0.0, 0.0, 0.0, 0.0, 0.0  # Padding
            ]
            features.extend(complexity_features)
            
            # Características de conciencia (10 dimensiones)
            consciousness = analysis.get('consciousness_metrics', {})
            consciousness_indicators = consciousness.get('indicators', {})
            consciousness_features = [
                consciousness_indicators.get('self_awareness', 0.0),
                consciousness_indicators.get('metacognition', 0.0),
                consciousness_indicators.get('emotional_intelligence', 0.0),
                consciousness_indicators.get('creative_thinking', 0.0),
                consciousness_indicators.get('philosophical_depth', 0.0),
                consciousness.get('consciousness_index', 0.0),
                0.0, 0.0, 0.0, 0.0  # Padding
            ]
            features.extend(consciousness_features)
            
            # Padding para llegar exactamente a 768 dimensiones
            current_size = len(features)
            target_size = 768
            
            if current_size < target_size:
                # Repetir características existentes para llenar el espacio
                padding_needed = target_size - current_size
                repeated_features = (features * ((padding_needed // current_size) + 1))[:padding_needed]
                features.extend(repeated_features)
            elif current_size > target_size:
                features = features[:target_size]
            
            return torch.tensor(features, dtype=torch.float32)
            
        except Exception as e:
            self.logger.log("ERROR", f"Feature extraction failed: {str(e)}")
            # Retornar vector de ceros en caso de error
            return torch.zeros(768, dtype=torch.float32)
    
    def _update_processing_stats(self, text: str, analysis: Dict):
        """Actualiza estadísticas de procesamiento"""
        
        self.processing_stats['total_texts_processed'] += 1
        
        # Longitud promedio
        current_length = len(text)
        total_processed = self.processing_stats['total_texts_processed']
        self.processing_stats['avg_text_length'] = (
            (self.processing_stats['avg_text_length'] * (total_processed - 1) + current_length) / total_processed
        )
        
        # Distribución de emociones
        dominant_emotion = analysis.get('emotions', {}).get('dominant_emotion', 'neutral')
        if dominant_emotion in self.processing_stats['emotion_distribution']:
            self.processing_stats['emotion_distribution'][dominant_emotion] += 1
        else:
            self.processing_stats['emotion_distribution'][dominant_emotion] = 1
        
        # Distribución de intenciones
        dominant_intent = analysis.get('intent', {}).get('dominant_intent', 'conversación')
        if dominant_intent in self.processing_stats['intent_distribution']:
            self.processing_stats['intent_distribution'][dominant_intent] += 1
        else:
            self.processing_stats['intent_distribution'][dominant_intent] = 1
    
    def get_processing_statistics(self) -> Dict:
        """Obtiene estadísticas de procesamiento"""
        return self.processing_stats.copy()
    
    def check_health(self) -> bool:
        """Verifica el estado de salud del procesador de texto"""
        
        try:
            # Test básico de análisis
            test_text = "Hola, ¿cómo estás? Me siento muy feliz hoy."
            analysis = self.analyze_text(test_text)
            
            # Verificar que el análisis contenga las secciones esperadas
            required_sections = ['basic_stats', 'emotions', 'intent', 'semantics']
            for section in required_sections:
                if section not in analysis:
                    return False
            
            # Test de extracción de características
            features = self.extract_features(test_text)
            if features.shape[0] != 768:
                return False
            
            return True
            
        except Exception as e:
            self.logger.log("ERROR", f"Text processor health check failed: {str(e)}")
            return False

