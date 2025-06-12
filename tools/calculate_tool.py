"""
Herramienta de cálculo matemático para el agente ReAct
"""

import math
import numpy as np
from typing import Union

def execute(expression: str) -> str:
    """
    Ejecuta cálculos matemáticos seguros
    """
    
    # Diccionario de funciones matemáticas permitidas
    safe_dict = {
        '__builtins__': {},
        'abs': abs,
        'round': round,
        'min': min,
        'max': max,
        'sum': sum,
        'pow': pow,
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'exp': math.exp,
        'pi': math.pi,
        'e': math.e,
        'factorial': math.factorial,
        'gcd': math.gcd,
        'ceil': math.ceil,
        'floor': math.floor
    }
    
    try:
        # Verificar caracteres permitidos
        allowed_chars = set('0123456789+-*/.()abcdefghijklmnopqrstuvwxyz_ABCDEFGHIJKLMNOPQRSTUVWXYZ ')
        if not all(c in allowed_chars for c in expression):
            return f"Error: Expresión contiene caracteres no permitidos"
        
        # Evaluar expresión
        result = eval(expression, safe_dict)
        
        # Formatear resultado
        if isinstance(result, float):
            if result.is_integer():
                return f"Resultado: {int(result)}"
            else:
                return f"Resultado: {result:.6f}"
        else:
            return f"Resultado: {result}"
            
    except ZeroDivisionError:
        return "Error: División por cero"
    except OverflowError:
        return "Error: Número demasiado grande"
    except ValueError as e:
        return f"Error: Valor inválido - {str(e)}"
    except Exception as e:
        return f"Error en cálculo: {str(e)}"