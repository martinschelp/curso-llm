"""
Orquestacion de agentes con LangGraph (patron supervisor / router).

Idea: en vez de UN solo agente que hace todo, tenemos VARIOS agentes
especializados y un "supervisor" que, segun la pregunta, decide a quien
derivarla. LangGraph modela esto como un GRAFO de estados:

        ┌──────────────┐
        │  supervisor  │   decide la ruta segun la pregunta
        └──────┬───────┘
   ┌───────────┼────────────┐
   ▼           ▼            ▼
┌────────┐ ┌──────────┐ ┌────────────┐
│matematico│ │traductor │ │ explicador │   (agentes especializados)
└────┬───┘ └────┬─────┘ └─────┬──────┘
     └──────────┴─────────────┘
                ▼
              END

Cada nodo del grafo es una funcion que recibe el estado (un diccionario
compartido) y devuelve los campos que actualiza. Las aristas condicionales
(add_conditional_edges) hacen el ruteo dinamico.

Componentes:
    - LLM local (Ollama, gemma3) para el supervisor y los agentes de texto.
    - Un agente 'matematico' que ademas usa una mini-herramienta (eval seguro),
      para mostrar que cada agente puede tener sus propias capacidades.

Requisitos:
    - Ollama corriendo con el modelo de chat:
          ollama pull gemma3
    - pip install -r requirements-orquestacion.txt

Uso:
    python multiagente_langgraph.py "Traducir al ingles: hola mundo"
    python multiagente_langgraph.py "Cuanto es 12 * (3 + 4)?"
    python multiagente_langgraph.py "Que es una base de datos vectorial?"
    python multiagente_langgraph.py            # modo interactivo
"""
from __future__ import annotations

import re
import sys
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CHAT_MODEL = "gemma4:12b" # cambia el tag segun lo que tengas descargado (ej: gemma3:1b, gemma2:9b)

# Un unico LLM local, reutilizado por el supervisor y los agentes de texto.
llm = ChatOllama(model=CHAT_MODEL, temperature=0.0)


# --- Estado compartido del grafo ---
# Todos los nodos leen y escriben sobre este diccionario tipado.
class Estado(TypedDict):
    pregunta: str    # la pregunta original del usuario
    ruta: str        # a que agente derivo el supervisor (matematico|traductor|explicador)
    respuesta: str   # la respuesta final que produce el agente elegido


RUTAS_VALIDAS = {"matematico", "traductor", "explicador"}


# --- Nodo supervisor: clasifica y decide la ruta ---
def supervisor(estado: Estado) -> dict:
    """Le pide al LLM que clasifique la pregunta en una de las rutas validas."""
    system = SystemMessage(content=(
        "Sos un supervisor que derriva tareas a agentes especializados. "
        "Segun la pregunta del usuario, respondé con UNA sola palabra, sin nada mas:\n"
        "- 'matematico' si es un calculo o problema aritmetico.\n"
        "- 'traductor' si pide traducir un texto a otro idioma.\n"
        "- 'explicador' para cualquier otra pregunta conceptual o general.\n"
        "Respondé SOLO la palabra clave."
    ))
    msg = llm.invoke([system, HumanMessage(content=estado["pregunta"])])
    etiqueta = msg.content.strip().lower()

    # Parseo tolerante: buscamos alguna ruta valida dentro de la respuesta.
    ruta = next((r for r in RUTAS_VALIDAS if r in etiqueta), "explicador")
    print(f"[supervisor] pregunta derivada a -> {ruta}")
    return {"ruta": ruta}


# --- Mini-herramienta del agente matematico ---
def _eval_seguro(expresion: str) -> str | None:
    """Evalua una expresion aritmetica simple, o None si no es evaluable."""
    m = re.search(r"[0-9+\-*/(). ]{3,}", expresion)
    if not m:
        return None
    expr = m.group(0).strip()
    if not set(expr) <= set("0123456789+-*/(). "):
        return None
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))  # noqa: S307 (sandbox minimo)
    except Exception:  # noqa: BLE001
        return None


# --- Agentes especializados (cada uno es un nodo del grafo) ---
def matematico(estado: Estado) -> dict:
    """Resuelve el calculo con una herramienta (eval seguro); si no puede, usa el LLM."""
    print("[agente:matematico] resolviendo...")
    resultado = _eval_seguro(estado["pregunta"])
    if resultado is not None:
        return {"respuesta": f"El resultado del calculo es: {resultado}"}
    # Fallback: no se detecto una expresion evaluable, que lo intente el LLM.
    system = SystemMessage(content="Sos un agente que resuelve problemas matematicos. "
                                   "Explicá el paso a paso y dá el resultado final.")
    msg = llm.invoke([system, HumanMessage(content=estado["pregunta"])])
    return {"respuesta": msg.content}


def traductor(estado: Estado) -> dict:
    """Traduce el texto pedido."""
    print("[agente:traductor] traduciendo...")
    system = SystemMessage(content="Sos un agente traductor. Traducí lo que pida el usuario. "
                                   "Devolvé SOLO la traduccion, sin explicaciones.")
    msg = llm.invoke([system, HumanMessage(content=estado["pregunta"])])
    return {"respuesta": msg.content}


def explicador(estado: Estado) -> dict:
    """Responde preguntas conceptuales o generales."""
    print("[agente:explicador] explicando...")
    system = SystemMessage(content="Sos un agente que explica conceptos de forma clara y breve, "
                                   "en español. Respondé de manera didactica.")
    msg = llm.invoke([system, HumanMessage(content=estado["pregunta"])])
    return {"respuesta": msg.content}


def _elegir_ruta(estado: Estado) -> str:
    """Funcion de ruteo: devuelve el nombre del nodo al que ir (lo puso el supervisor)."""
    return estado["ruta"]


def construir_grafo():
    """Arma el grafo de orquestacion: supervisor -> (agente elegido) -> END."""
    grafo = StateGraph(Estado)

    grafo.add_node("supervisor", supervisor)
    grafo.add_node("matematico", matematico)
    grafo.add_node("traductor", traductor)
    grafo.add_node("explicador", explicador)

    # El flujo arranca en el supervisor.
    grafo.add_edge(START, "supervisor")

    # Arista CONDICIONAL: segun lo que decida _elegir_ruta, va a un agente u otro.
    grafo.add_conditional_edges(
        "supervisor",
        _elegir_ruta,
        {
            "matematico": "matematico",
            "traductor": "traductor",
            "explicador": "explicador",
        },
    )

    # Cada agente, al terminar, cierra el grafo.
    for agente in ("matematico", "traductor", "explicador"):
        grafo.add_edge(agente, END)

    return grafo.compile()


def responder(app, pregunta: str) -> None:
    print(f"\n=== Pregunta: {pregunta} ===")
    estado_final = app.invoke({"pregunta": pregunta, "ruta": "", "respuesta": ""})
    print("\n--- Respuesta final ---")
    print(estado_final["respuesta"])


def main() -> None:
    app = construir_grafo()
    argv = sys.argv[1:]
    if argv:
        responder(app, " ".join(argv))
    else:
        print("Escribí una pregunta (o 'salir' para terminar):")
        while True:
            try:
                pregunta = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if pregunta.lower() in {"salir", "exit", "quit"}:
                break
            if pregunta:
                responder(app, pregunta)


if __name__ == "__main__":
    main()
