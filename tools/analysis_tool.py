"""
Herramienta de análisis de datos para el agente ReAct
"""

import json
import re
from typing import Dict, List, Any

def execute(data_input: str) -> str:
    """
    Analiza datos estructurados y no estructurados
    """
    
    try:
        # Intentar parsear como JSON
        try:
            data = json.loads(data_input)
            return analyze_structured_data(data)
        except json.JSONDecodeError:
            pass
        
        # Análisis de texto libre
        return analyze_text_data(data_input)
        
    except Exception as e:
        return f"Error en análisis: {str(e)}"

def analyze_structured_data(data: Any) -> str:
    """Analiza datos estructurados (JSON, dict, list)"""
    
    analysis = []
    
    if isinstance(data, dict):
        analysis.append(f"Estructura: Diccionario con {len(data)} claves")
        analysis.append(f"Claves: {list(data.keys())}")
        
        # Análisis de tipos de valores
        value_types = {}
        for key, value in data.items():
            value_type = type(value).__name__
            value_types[value_type] = value_types.get(value_type, 0) + 1
        
        analysis.append(f"Tipos de datos: {value_types}")
        
    elif isinstance(data, list):
        analysis.append(f"Estructura: Lista con {len(data)} elementos")
        
        if data:
            first_type = type(data[0]).__name__
            analysis.append(f"Tipo del primer elemento: {first_type}")
            
            # Verificar homogeneidad
            is_homogeneous = all(type(item).__name__ == first_type for item in data)
            analysis.append(f"Lista homogénea: {'Sí' if is_homogeneous else 'No'}")
    
    else:
        analysis.append(f"Tipo de dato: {type(data).__name__}")
        analysis.append(f"Valor: {str(data)[:100]}...")
    
    return "Análisis de datos estructurados:\n" + "\n".join(analysis)

def analyze_text_data(text: str) -> str:
    """Analiza datos de texto libre"""
    
    analysis = []
    
    # Estadísticas básicas
    word_count = len(text.split())
    char_count = len(text)
    line_count = len(text.split('\n'))
    
    analysis.append(f"Palabras: {word_count}")
    analysis.append(f"Caracteres: {char_count}")
    analysis.append(f"Líneas: {line_count}")
    
    # Análisis de patrones
    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    numbers = re.findall(r'\b\d+\.?\d*\b', text)
    
    if emails:
        analysis.append(f"Emails encontrados: {len(emails)}")
    if urls:
        analysis.append(f"URLs encontradas: {len(urls)}")
    if numbers:
        analysis.append(f"Números encontrados: {len(numbers)}")
    
    # Palabras más frecuentes
    words = re.findall(r'\b\w+\b', text.lower())
    word_freq = {}
    for word in words:
        if len(word) > 3:  # Solo palabras de más de 3 caracteres
            word_freq[word] = word_freq.get(word, 0) + 1
    
    if word_freq:
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        analysis.append(f"Palabras más frecuentes: {dict(top_words)}")
    
    return "Análisis de texto:\n" + "\n".join(analysis)