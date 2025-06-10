import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import cv2
import io

from utils.logger import Logger

class VisionEncoder(nn.Module):
    """
    Encoder de visión basado en CNN para extracción de características
    """
    
    def __init__(self, output_dim=512):
        super(VisionEncoder, self).__init__()
        
        # Arquitectura CNN simplificada inspirada en ResNet
        self.features = nn.Sequential(
            # Bloque 1
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Bloque 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Bloque 3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Bloque 4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        encoded = self.classifier(features)
        return encoded

class EmotionDetector(nn.Module):
    """
    Detector de emociones en imágenes
    """
    
    def __init__(self, num_emotions=7):
        super(EmotionDetector, self).__init__()
        
        # Arquitectura específica para detección emocional
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        self.emotion_classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_emotions),
            nn.Softmax(dim=1)
        )
        
        self.emotion_labels = [
            'felicidad', 'tristeza', 'sorpresa', 'miedo', 
            'ira', 'disgusto', 'neutral'
        ]
    
    def forward(self, x):
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)
        emotions = self.emotion_classifier(features)
        return emotions

class ObjectDetector:
    """
    Detector de objetos simplificado usando características básicas
    """
    
    def __init__(self):
        self.object_categories = [
            'persona', 'animal', 'vehículo', 'edificio', 'naturaleza',
            'objeto', 'comida', 'tecnología', 'arte', 'texto'
        ]
    
    def detect_objects(self, image_array: np.ndarray) -> Dict[str, float]:
        """
        Detecta objetos en la imagen usando análisis de características básicas
        
        Args:
            image_array: Array numpy de la imagen
        
        Returns:
            Diccionario con probabilidades de cada categoría
        """
        
        # Análisis de color
        color_analysis = self._analyze_colors(image_array)
        
        # Análisis de textura
        texture_analysis = self._analyze_texture(image_array)
        
        # Análisis de formas
        shape_analysis = self._analyze_shapes(image_array)
        
        # Heurísticas simples para clasificación
        object_probabilities = {}
        
        # Persona (basado en tonos de piel y formas)
        skin_tones = color_analysis.get('skin_tone_ratio', 0)
        vertical_lines = shape_analysis.get('vertical_lines', 0)
        object_probabilities['persona'] = min(0.9, skin_tones * 2 + vertical_lines * 0.5)
        
        # Naturaleza (basado en verdes y texturas orgánicas)
        green_ratio = color_analysis.get('green_ratio', 0)
        organic_texture = texture_analysis.get('organic_score', 0)
        object_probabilities['naturaleza'] = min(0.9, green_ratio * 1.5 + organic_texture)
        
        # Tecnología (basado en líneas rectas y colores metálicos)
        geometric_score = shape_analysis.get('geometric_score', 0)
        metallic_colors = color_analysis.get('metallic_ratio', 0)
        object_probabilities['tecnología'] = min(0.9, geometric_score + metallic_colors)
        
        # Llenar el resto con valores heurísticos
        for category in self.object_categories:
            if category not in object_probabilities:
                # Valor aleatorio pequeño para categorías no analizadas
                object_probabilities[category] = np.random.uniform(0.0, 0.3)
        
        return object_probabilities
    
    def _analyze_colors(self, image_array: np.ndarray) -> Dict[str, float]:
        """Analiza distribución de colores en la imagen"""
        
        # Convertir a HSV para mejor análisis de color
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        
        # Análisis de tonos de piel (heurística simple)
        skin_mask = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 255, 255]))
        skin_ratio = np.sum(skin_mask > 0) / (image_array.shape[0] * image_array.shape[1])
        
        # Análisis de verde (naturaleza)
        green_mask = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))
        green_ratio = np.sum(green_mask > 0) / (image_array.shape[0] * image_array.shape[1])
        
        # Análisis de colores metálicos (grises)
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        metallic_mask = (gray > 100) & (gray < 200)
        metallic_ratio = np.sum(metallic_mask) / (image_array.shape[0] * image_array.shape[1])
        
        return {
            'skin_tone_ratio': skin_ratio,
            'green_ratio': green_ratio,
            'metallic_ratio': metallic_ratio
        }
    
    def _analyze_texture(self, image_array: np.ndarray) -> Dict[str, float]:
        """Analiza texturas en la imagen"""
        
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Calcular gradientes para textura
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Varianza de gradientes como medida de textura
        texture_variance = np.var(np.sqrt(grad_x**2 + grad_y**2))
        
        # Puntuación orgánica basada en suavidad/rugosidad
        organic_score = min(1.0, texture_variance / 1000)
        
        return {
            'texture_variance': texture_variance,
            'organic_score': organic_score
        }
    
    def _analyze_shapes(self, image_array: np.ndarray) -> Dict[str, float]:
        """Analiza formas y líneas en la imagen"""
        
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Detectar líneas usando transformada de Hough
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        vertical_lines = 0
        horizontal_lines = 0
        
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                # Líneas verticales (theta cerca de 0 o π)
                if abs(theta) < 0.2 or abs(theta - np.pi) < 0.2:
                    vertical_lines += 1
                # Líneas horizontales (theta cerca de π/2)
                elif abs(theta - np.pi/2) < 0.2:
                    horizontal_lines += 1
        
        # Puntuación geométrica basada en líneas rectas
        total_lines = vertical_lines + horizontal_lines
        geometric_score = min(1.0, total_lines / 20)
        
        return {
            'vertical_lines': min(1.0, vertical_lines / 10),
            'horizontal_lines': min(1.0, horizontal_lines / 10),
            'geometric_score': geometric_score
        }

class VisionProcessor:
    """
    Procesador de visión principal con capacidades de análisis consciente de imágenes
    Implementa análisis de contenido, emociones y descripción automática
    """
    
    def __init__(self):
        self.logger = Logger()
        
        # Modelos de visión
        self.encoder = VisionEncoder()
        self.emotion_detector = EmotionDetector()
        self.object_detector = ObjectDetector()
        
        # Transformaciones de imagen
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Configuración
        self.supported_formats = ['RGB', 'RGBA', 'L']
        
        # Estadísticas de procesamiento
        self.processing_stats = {
            'images_processed': 0,
            'avg_image_size': (0, 0),
            'emotion_distribution': {},
            'object_distribution': {}
        }
        
        self.logger.log("INFO", "VisionProcessor initialized successfully")
    
    def analyze_image(self, image: Image.Image) -> Dict:
        """
        Análisis completo de imagen usando conciencia artificial
        
        Args:
            image: Imagen PIL a analizar
        
        Returns:
            Diccionario con análisis completo
        """
        
        try:
            # Preprocesamiento
            processed_image = self._preprocess_image(image)
            image_array = np.array(image)
            
            # Análisis básico de la imagen
            basic_analysis = self._analyze_basic_properties(image)
            
            # Análisis de contenido/objetos
            object_analysis = self.object_detector.detect_objects(image_array)
            
            # Análisis de composición
            composition_analysis = self._analyze_composition(image_array)
            
            # Análisis de calidad
            quality_analysis = self._analyze_quality(image_array)
            
            # Análisis de estilo artístico
            style_analysis = self._analyze_artistic_style(image_array)
            
            # Análisis de contexto visual
            context_analysis = self._analyze_visual_context(image_array)
            
            analysis = {
                'basic_properties': basic_analysis,
                'objects': object_analysis,
                'composition': composition_analysis,
                'quality': quality_analysis,
                'artistic_style': style_analysis,
                'visual_context': context_analysis,
                'processing_timestamp': self._get_timestamp()
            }
            
            # Actualizar estadísticas
            self._update_processing_stats(image, analysis)
            
            self.logger.log("DEBUG", f"Image analysis completed: {image.size}")
            
            return analysis
            
        except Exception as e:
            self.logger.log("ERROR", f"Image analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def analyze_emotions(self, image: Image.Image) -> Dict[str, float]:
        """
        Analiza emociones presentes en la imagen
        
        Args:
            image: Imagen PIL a analizar
        
        Returns:
            Diccionario con puntuaciones emocionales
        """
        
        try:
            # Preprocesar imagen
            processed_image = self._preprocess_image(image)
            
            # Análisis con red neuronal
            with torch.no_grad():
                emotion_scores = self.emotion_detector(processed_image.unsqueeze(0))
                emotion_scores = emotion_scores.squeeze().tolist()
            
            # Crear diccionario de emociones
            emotions = {
                emotion: score for emotion, score in 
                zip(self.emotion_detector.emotion_labels, emotion_scores)
            }
            
            # Análisis adicional basado en características visuales
            image_array = np.array(image)
            visual_emotions = self._analyze_visual_emotions(image_array)
            
            # Combinar análisis neuronal y visual
            combined_emotions = {}
            for emotion in emotions:
                neural_score = emotions[emotion]
                visual_score = visual_emotions.get(emotion, 0.0)
                combined_emotions[emotion] = (neural_score * 0.7 + visual_score * 0.3)
            
            # Añadir métricas adicionales
            dominant_emotion = max(combined_emotions.keys(), key=lambda k: combined_emotions[k])
            emotional_intensity = max(combined_emotions.values())
            
            return {
                'neural_emotions': emotions,
                'visual_emotions': visual_emotions,
                'combined_emotions': combined_emotions,
                'dominant_emotion': dominant_emotion,
                'emotional_intensity': emotional_intensity
            }
            
        except Exception as e:
            self.logger.log("ERROR", f"Emotion analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def generate_description(self, image: Image.Image, analysis: Dict = None) -> str:
        """
        Genera descripción textual de la imagen
        
        Args:
            image: Imagen PIL a describir
            analysis: Análisis previo (opcional)
        
        Returns:
            Descripción textual de la imagen
        """
        
        try:
            if analysis is None:
                analysis = self.analyze_image(image)
            
            description_parts = []
            
            # Descripción básica
            basic = analysis.get('basic_properties', {})
            size_desc = self._describe_size(basic.get('size', (0, 0)))
            mode_desc = self._describe_mode(basic.get('mode', 'RGB'))
            
            description_parts.append(f"Esta es una imagen {size_desc} en formato {mode_desc}.")
            
            # Descripción de objetos
            objects = analysis.get('objects', {})
            dominant_objects = sorted(objects.items(), key=lambda x: x[1], reverse=True)[:3]
            
            if dominant_objects and dominant_objects[0][1] > 0.3:
                object_desc = "La imagen contiene principalmente "
                object_names = [obj[0] for obj in dominant_objects if obj[1] > 0.3]
                if len(object_names) == 1:
                    object_desc += f"elementos de {object_names[0]}."
                elif len(object_names) == 2:
                    object_desc += f"{object_names[0]} y {object_names[1]}."
                else:
                    object_desc += f"{', '.join(object_names[:-1])} y {object_names[-1]}."
                
                description_parts.append(object_desc)
            
            # Descripción de composición
            composition = analysis.get('composition', {})
            if composition.get('symmetry_score', 0) > 0.7:
                description_parts.append("La composición es altamente simétrica.")
            elif composition.get('balance_score', 0) > 0.6:
                description_parts.append("La composición está bien balanceada.")
            
            # Descripción de colores
            color_desc = self._describe_colors(analysis)
            if color_desc:
                description_parts.append(color_desc)
            
            # Descripción de estilo
            style = analysis.get('artistic_style', {})
            style_desc = self._describe_style(style)
            if style_desc:
                description_parts.append(style_desc)
            
            # Descripción de calidad
            quality = analysis.get('quality', {})
            if quality.get('sharpness_score', 0) > 0.8:
                description_parts.append("La imagen es muy nítida y clara.")
            elif quality.get('noise_level', 1) > 0.7:
                description_parts.append("La imagen presenta algo de ruido visual.")
            
            # Unir todas las partes
            full_description = " ".join(description_parts)
            
            # Si la descripción es muy corta, añadir descripción genérica
            if len(full_description) < 50:
                full_description = "Esta imagen contiene contenido visual diverso con elementos variados y una composición interesante."
            
            return full_description
            
        except Exception as e:
            self.logger.log("ERROR", f"Description generation failed: {str(e)}")
            return "No se pudo generar una descripción de esta imagen."
    
    def extract_features(self, image: Image.Image) -> torch.Tensor:
        """
        Extrae características vectoriales de la imagen para fusión multimodal
        
        Args:
            image: Imagen PIL a procesar
        
        Returns:
            Tensor de características [512] dimensional
        """
        
        try:
            # Preprocesar imagen
            processed_image = self._preprocess_image(image)
            
            # Extraer características con el encoder
            with torch.no_grad():
                features = self.encoder(processed_image.unsqueeze(0))
                features = features.squeeze()
            
            return features
            
        except Exception as e:
            self.logger.log("ERROR", f"Feature extraction failed: {str(e)}")
            # Retornar vector de ceros en caso de error
            return torch.zeros(512, dtype=torch.float32)
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocesa imagen para análisis"""
        
        # Convertir a RGB si es necesario
        if image.mode != 'RGB':
            if image.mode == 'RGBA':
                # Crear fondo blanco para transparencias
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            else:
                image = image.convert('RGB')
        
        # Aplicar transformaciones
        return self.transform(image)
    
    def _analyze_basic_properties(self, image: Image.Image) -> Dict:
        """Analiza propiedades básicas de la imagen"""
        
        return {
            'size': image.size,
            'mode': image.mode,
            'format': getattr(image, 'format', 'Unknown'),
            'aspect_ratio': image.size[0] / max(image.size[1], 1),
            'total_pixels': image.size[0] * image.size[1]
        }
    
    def _analyze_composition(self, image_array: np.ndarray) -> Dict:
        """Analiza composición visual de la imagen"""
        
        # Convertir a escala de grises para análisis
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if len(image_array.shape) == 3 else image_array
        
        # Análisis de simetría
        h, w = gray.shape
        left_half = gray[:, :w//2]
        right_half = np.fliplr(gray[:, w//2:])
        
        # Asegurar que ambas mitades tengan el mismo tamaño
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        # Calcular similitud entre mitades
        symmetry_score = 1.0 - (np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255.0)
        
        # Análisis de balance (distribución de masa visual)
        moments = cv2.moments(gray)
        if moments['m00'] != 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            center_x, center_y = w // 2, h // 2
            balance_score = 1.0 - (abs(cx - center_x) + abs(cy - center_y)) / (w + h)
        else:
            balance_score = 0.5
        
        # Análisis de regla de tercios
        third_x, third_y = w // 3, h // 3
        thirds_points = [
            (third_x, third_y), (2 * third_x, third_y),
            (third_x, 2 * third_y), (2 * third_x, 2 * third_y)
        ]
        
        # Calcular intensidad en puntos de tercios
        thirds_intensity = np.mean([gray[y, x] for x, y in thirds_points if x < w and y < h])
        rule_of_thirds_score = thirds_intensity / 255.0
        
        return {
            'symmetry_score': max(0, min(1, symmetry_score)),
            'balance_score': max(0, min(1, balance_score)),
            'rule_of_thirds_score': rule_of_thirds_score,
            'visual_center': (cx if 'cx' in locals() else w//2, cy if 'cy' in locals() else h//2)
        }
    
    def _analyze_quality(self, image_array: np.ndarray) -> Dict:
        """Analiza calidad técnica de la imagen"""
        
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if len(image_array.shape) == 3 else image_array
        
        # Análisis de nitidez usando Laplaciano
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness_score = laplacian.var() / 10000  # Normalizar
        sharpness_score = min(1.0, sharpness_score)
        
        # Análisis de ruido
        noise_level = np.std(gray) / 255.0
        
        # Análisis de contraste
        contrast_score = (np.max(gray) - np.min(gray)) / 255.0
        
        # Análisis de exposición
        mean_brightness = np.mean(gray) / 255.0
        exposure_score = 1.0 - abs(mean_brightness - 0.5) * 2  # Óptimo en 0.5
        
        return {
            'sharpness_score': sharpness_score,
            'noise_level': noise_level,
            'contrast_score': contrast_score,
            'exposure_score': exposure_score,
            'overall_quality': np.mean([sharpness_score, 1-noise_level, contrast_score, exposure_score])
        }
    
    def _analyze_artistic_style(self, image_array: np.ndarray) -> Dict:
        """Analiza estilo artístico de la imagen"""
        
        # Análisis de textura para determinar estilo
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if len(image_array.shape) == 3 else image_array
        
        # Detectar bordes para estilo
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Análisis de color
        if len(image_array.shape) == 3:
            color_variance = np.var(image_array, axis=(0, 1))
            color_richness = np.mean(color_variance) / 10000
        else:
            color_richness = 0.0
        
        # Clasificación de estilo (heurística)
        style_scores = {
            'fotografía': 0.5,  # Base
            'pintura': edge_density * 2 + color_richness,
            'dibujo': edge_density * 3,
            'abstracto': color_richness * 2,
            'realista': 1.0 - edge_density
        }
        
        # Normalizar puntuaciones
        max_score = max(style_scores.values())
        if max_score > 0:
            for style in style_scores:
                style_scores[style] = min(1.0, style_scores[style] / max_score)
        
        dominant_style = max(style_scores.keys(), key=lambda k: style_scores[k])
        
        return {
            'style_scores': style_scores,
            'dominant_style': dominant_style,
            'edge_density': edge_density,
            'color_richness': color_richness
        }
    
    def _analyze_visual_context(self, image_array: np.ndarray) -> Dict:
        """Analiza contexto visual de la imagen"""
        
        h, w = image_array.shape[:2]
        
        # Análisis de iluminación
        if len(image_array.shape) == 3:
            brightness = np.mean(cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY))
        else:
            brightness = np.mean(image_array)
        
        lighting_condition = 'normal'
        if brightness < 85:
            lighting_condition = 'oscuro'
        elif brightness > 170:
            lighting_condition = 'brillante'
        
        # Análisis de profundidad (heurística basada en bordes)
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY) if len(image_array.shape) == 3 else image_array
        blur_kernel = cv2.GaussianBlur(gray, (15, 15), 0)
        depth_map = cv2.absdiff(gray, blur_kernel)
        depth_variance = np.var(depth_map)
        
        depth_perception = 'bajo'
        if depth_variance > 1000:
            depth_perception = 'alto'
        elif depth_variance > 500:
            depth_perception = 'medio'
        
        # Análisis de movimiento (basado en desenfoques direccionales)
        motion_blur = self._detect_motion_blur(gray)
        
        return {
            'lighting_condition': lighting_condition,
            'brightness_level': brightness / 255.0,
            'depth_perception': depth_perception,
            'depth_variance': depth_variance,
            'motion_blur_detected': motion_blur,
            'image_orientation': 'landscape' if w > h else 'portrait' if h > w else 'square'
        }
    
    def _detect_motion_blur(self, gray: np.ndarray) -> bool:
        """Detecta desenfoque de movimiento"""
        
        # Usar FFT para detectar patrones direccionales
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Buscar líneas en el espectro que indiquen movimiento
        edges = cv2.Canny((magnitude_spectrum * 255 / np.max(magnitude_spectrum)).astype(np.uint8), 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=30)
        
        return lines is not None and len(lines) > 5
    
    def _analyze_visual_emotions(self, image_array: np.ndarray) -> Dict[str, float]:
        """Analiza emociones basado en características visuales"""
        
        # Análisis de color para emociones
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        
        # Colores cálidos vs fríos
        warm_mask = (hsv[:, :, 0] <= 60) | (hsv[:, :, 0] >= 300)  # Rojos, naranjas, amarillos
        cold_mask = (hsv[:, :, 0] >= 180) & (hsv[:, :, 0] <= 240)  # Azules
        
        warm_ratio = np.sum(warm_mask) / warm_mask.size
        cold_ratio = np.sum(cold_mask) / cold_mask.size
        
        # Análisis de saturación y brillo
        saturation_mean = np.mean(hsv[:, :, 1]) / 255.0
        brightness_mean = np.mean(hsv[:, :, 2]) / 255.0
        
        # Mapear a emociones
        emotions = {
            'felicidad': warm_ratio * saturation_mean * brightness_mean,
            'tristeza': cold_ratio * (1 - brightness_mean),
            'sorpresa': saturation_mean * brightness_mean,
            'miedo': (1 - brightness_mean) * (1 - saturation_mean),
            'ira': warm_ratio * saturation_mean * (1 - brightness_mean),
            'disgusto': (1 - saturation_mean) * (1 - brightness_mean),
            'neutral': 1 - max(warm_ratio, cold_ratio)
        }
        
        # Normalizar
        max_emotion = max(emotions.values())
        if max_emotion > 0:
            emotions = {k: v / max_emotion for k, v in emotions.items()}
        
        return emotions
    
    def _describe_size(self, size: Tuple[int, int]) -> str:
        """Describe el tamaño de la imagen"""
        w, h = size
        total_pixels = w * h
        
        if total_pixels < 100000:  # < 0.1 MP
            return "pequeña"
        elif total_pixels < 2000000:  # < 2 MP
            return "mediana"
        else:
            return "grande"
    
    def _describe_mode(self, mode: str) -> str:
        """Describe el modo de color"""
        mode_descriptions = {
            'RGB': 'color',
            'RGBA': 'color con transparencia',
            'L': 'escala de grises',
            'P': 'paleta de colores'
        }
        return mode_descriptions.get(mode, mode.lower())
    
    def _describe_colors(self, analysis: Dict) -> str:
        """Describe los colores predominantes"""
        
        # Esta función sería más compleja en una implementación real
        # Por ahora, retorna descripciones genéricas basadas en el análisis
        
        quality = analysis.get('quality', {})
        brightness = quality.get('exposure_score', 0.5)
        
        if brightness > 0.7:
            return "Los colores son brillantes y vibrantes."
        elif brightness < 0.3:
            return "Los tonos son oscuros y profundos."
        else:
            return "La paleta de colores es equilibrada."
    
    def _describe_style(self, style_analysis: Dict) -> str:
        """Describe el estilo artístico"""
        
        dominant_style = style_analysis.get('dominant_style', 'fotografía')
        edge_density = style_analysis.get('edge_density', 0)
        
        style_descriptions = {
            'fotografía': "Tiene características de fotografía digital.",
            'pintura': "Presenta cualidades pictóricas y artísticas.",
            'dibujo': "Muestra características de dibujo o sketch.",
            'abstracto': "Tiene elementos abstractos y conceptuales.",
            'realista': "Presenta un estilo realista y detallado."
        }
        
        base_desc = style_descriptions.get(dominant_style, "Tiene un estilo visual distintivo.")
        
        if edge_density > 0.1:
            base_desc += " Con líneas y contornos bien definidos."
        
        return base_desc
    
    def _get_timestamp(self) -> str:
        """Obtiene timestamp actual"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _update_processing_stats(self, image: Image.Image, analysis: Dict):
        """Actualiza estadísticas de procesamiento"""
        
        self.processing_stats['images_processed'] += 1
        
        # Tamaño promedio
        current_size = image.size
        total_processed = self.processing_stats['images_processed']
        
        if total_processed == 1:
            self.processing_stats['avg_image_size'] = current_size
        else:
            avg_w = (self.processing_stats['avg_image_size'][0] * (total_processed - 1) + current_size[0]) / total_processed
            avg_h = (self.processing_stats['avg_image_size'][1] * (total_processed - 1) + current_size[1]) / total_processed
            self.processing_stats['avg_image_size'] = (int(avg_w), int(avg_h))
        
        # Distribución de objetos
        objects = analysis.get('objects', {})
        dominant_object = max(objects.keys(), key=lambda k: objects[k]) if objects else 'unknown'
        
        if dominant_object in self.processing_stats['object_distribution']:
            self.processing_stats['object_distribution'][dominant_object] += 1
        else:
            self.processing_stats['object_distribution'][dominant_object] = 1
    
    def get_processing_statistics(self) -> Dict:
        """Obtiene estadísticas de procesamiento"""
        return self.processing_stats.copy()
    
    def check_health(self) -> bool:
        """Verifica el estado de salud del procesador de visión"""
        
        try:
            # Crear imagen de prueba
            test_image = Image.new('RGB', (224, 224), color='red')
            
            # Test de análisis
            analysis = self.analyze_image(test_image)
            if 'error' in analysis:
                return False
            
            # Test de extracción de características
            features = self.extract_features(test_image)
            if features.shape[0] != 512:
                return False
            
            # Test de análisis emocional
            emotions = self.analyze_emotions(test_image)
            if 'error' in emotions:
                return False
            
            return True
            
        except Exception as e:
            self.logger.log("ERROR", f"Vision processor health check failed: {str(e)}")
            return False

