"""
Patron Generador-Critico con LangGraph (reflexion / self-critique loop).

Idea: en vez de quedarse con la primera respuesta del LLM, un segundo
agente ("critico") la revisa y decide si aprobarla o pedir una revision.
Si la rechaza, el "generador" vuelve a escribir usando esa critica como
feedback. Se repite hasta que el critico aprueba, o hasta un tope de
intentos (para no quedar en loop infinito). A diferencia del supervisor
de multiagente_langgraph.py (un solo salto), este grafo tiene un CICLO.

        ┌────────────┐
  ───►  │ generador  │   escribe (o reescribe con la critica anterior)
        └─────┬──────┘
              ▼
        ┌────────────┐
        │  critico   │   evalua el borrador
        └─────┬──────┘
       aprobado, o sin intentos │  rechazado y quedan intentos
              ▼                 │
             END  ◄─────────────┘  (vuelve a "generador")

Componentes:
    - LLM local (Ollama, gemma3) para generador y critico, cada uno con
      su propio system prompt.
    - MAX_INTENTOS como salvavidas: si el critico nunca aprueba, el grafo
      corta igual y devuelve el ultimo borrador (no queda dando vueltas).

Requisitos:
    - Ollama corriendo con el modelo de chat:
          ollama pull gemma3
    - pip install -r requirements-orquestacion.txt   (mismas deps que
      multiagente_langgraph.py, no hace falta nada nuevo)

Uso:
    python generador_critico_langgraph.py "Escribi un slogan para una cafeteria de barrio"
    python generador_critico_langgraph.py            # modo interactivo
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
MAX_INTENTOS = 3  # salvavidas: corta el ciclo aunque el critico no apruebe

# Un unico LLM local, reutilizado por el generador y el critico.
llm = ChatOllama(model=CHAT_MODEL, temperature=0.3)


# --- Estado compartido del grafo ---
class Estado(TypedDict):
    tarea: str        # el pedido original del usuario (no cambia)
    borrador: str      # ultima version escrita por el generador
    criticas: str        # feedback del critico ("" si aprobo o aun no corrio)
    intentos: int          # cuantas veces escribio el generador
    aprobado: bool            # decision del critico en la ultima vuelta


# --- Nodo generador: escribe o reescribe segun la critica ---
def generador(estado: Estado) -> dict:
    """Escribe un borrador; si hay criticas de una vuelta anterior, revisa con ese feedback."""
    intento = estado["intentos"] + 1
    if estado["criticas"]:
        print(f"[generador] intento {intento}: revisando segun la critica...")
        system = SystemMessage(content=(
            "Sos un redactor. Ya escribiste un borrador y un critico te dio feedback. "
            "Reescribi el texto teniendo en cuenta ESA critica puntual. Devolve SOLO el "
            "texto nuevo, sin explicaciones ni comillas."
        ))
        pedido = (
            f"Tarea original: {estado['tarea']}\n\n"
            f"Borrador anterior:\n{estado['borrador']}\n\n"
            f"Critica a resolver:\n{estado['criticas']}"
        )
    else:
        print(f"[generador] intento {intento}: primer borrador...")
        system = SystemMessage(content=(
            "Sos un redactor. Resolve el pedido del usuario. Devolve SOLO el texto "
            "pedido, sin explicaciones ni comillas."
        ))
        pedido = estado["tarea"]

    msg = llm.invoke([system, HumanMessage(content=pedido)])
    return {"borrador": msg.content.strip(), "intentos": intento}


# --- Nodo critico: aprueba o rechaza con una critica concreta ---
def critico(estado: Estado) -> dict:
    """Evalua el borrador contra la tarea original y decide aprobar o rechazar."""
    print("[critico] evaluando el borrador...")
    system = SystemMessage(content=(
        "Sos un critico exigente pero justo. Evaluas si un texto cumple con lo que se "
        "pidio. Respondé en DOS lineas EXACTAS, sin nada mas:\n"
        "APROBADO: si|no\n"
        "CRITICA: (si es 'no', una sola frase concreta de que hay que mejorar; "
        "si es 'si', dejala vacia)"
    ))
    pedido = f"Tarea: {estado['tarea']}\n\nTexto a evaluar:\n{estado['borrador']}"
    msg = llm.invoke([system, HumanMessage(content=pedido)])
    texto = msg.content.strip()

    m_aprobado = re.search(r"APROBADO:\s*(si|sí|no)", texto, re.IGNORECASE)
    aprobado = bool(m_aprobado and m_aprobado.group(1).lower().startswith("s"))
    m_critica = re.search(r"CRITICA:\s*(.*)", texto, re.IGNORECASE)
    critica = m_critica.group(1).strip() if m_critica else ""

    print(f"[critico] aprobado={aprobado}" + (f" | critica: {critica}" if critica else ""))
    return {"aprobado": aprobado, "criticas": "" if aprobado else critica}


def _siguiente_paso(estado: Estado) -> str:
    """Ruteo tras el critico: aprobado (o sin intentos) -> fin; si no, otra vuelta al generador."""
    if estado["aprobado"] or estado["intentos"] >= MAX_INTENTOS:
        return "fin"
    return "revisar"


def construir_grafo():
    """Arma el grafo: generador -> critico -> (fin | vuelve a generador)."""
    grafo = StateGraph(Estado)

    grafo.add_node("generador", generador)
    grafo.add_node("critico", critico)

    grafo.add_edge(START, "generador")
    grafo.add_edge("generador", "critico")

    # Arista CONDICIONAL: el critico decide si el ciclo termina o vuelve a girar.
    grafo.add_conditional_edges(
        "critico",
        _siguiente_paso,
        {"revisar": "generador", "fin": END},
    )

    return grafo.compile()


def responder(app, tarea: str) -> None:
    print(f"\n=== Tarea: {tarea} ===")
    estado_final = app.invoke({
        "tarea": tarea, "borrador": "", "criticas": "", "intentos": 0, "aprobado": False,
    })
    print("\n--- Resultado final ---")
    print(f"Intentos: {estado_final['intentos']} | Aprobado por el critico: {estado_final['aprobado']}")
    print(estado_final["borrador"])


def main() -> None:
    app = construir_grafo()
    argv = sys.argv[1:]
    if argv:
        responder(app, " ".join(argv))
    else:
        print("Escribí una tarea de redaccion (o 'salir' para terminar):")
        while True:
            try:
                tarea = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if tarea.lower() in {"salir", "exit", "quit"}:
                break
            if tarea:
                responder(app, tarea)


if __name__ == "__main__":
    main()
