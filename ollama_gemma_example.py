"""
Ejemplo minimo de uso de Ollama con el modelo Gemma.

Requisitos previos:
    1. Tener Ollama instalado holy corriendo (https://ollama.com)
    2. Descargar el modelo:  ollama pull gemma3
    3. Instalar el cliente python:  pip install ollama
"""

import sys

import ollama

sys.stdout.reconfigure(encoding="utf-8")  # evita errores al imprimir tildes/emojis en la consola de Windows

MODEL = "gemma4:12b"  # cambia el tag segun lo que tengas descargado (ej: gemma3:1b, gemma2:9b)


def chat_simple(prompt: str) -> str:
    """Manda un mensaje a Ollama y devuelve la respuesta completa de una sola vez."""
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]


def chat_streaming(prompt: str) -> None:
    """Igual que chat_simple, pero imprime la respuesta a medida que llega (token a token)."""
    stream = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,  # el modelo devuelve un iterador de pedacitos en vez de un string entero
    )
    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)  # sin salto de linea entre chunks
    print()


if __name__ == "__main__":
    # Uso: python ollama_gemma_example.py "tu pregunta"  (o sin argumentos -> "hola mundo")
    pregunta = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "hola mundo"

    print(f"--- Respuesta simple ({pregunta!r}) ---")
    print(chat_simple(pregunta))

    #print("\n--- Respuesta en streaming ---")
    #chat_streaming(pregunta)
