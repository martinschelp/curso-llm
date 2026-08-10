"""
RAG simple 100% local: Ollama (embeddings + generacion) + ChromaDB (vector store).

Requisitos previos:
    ollama pull nomic-embed-text   # modelo de embeddings
    ollama pull gemma3             # modelo generador
    pip install ollama chromadb

Flujo:
    1. Indexar: cada documento se convierte en un vector (embedding) y se guarda en Chroma.
    2. Recuperar: la pregunta del usuario tambien se convierte en vector y se buscan
       los documentos mas parecidos (similitud coseno) en Chroma.
    3. Generar: se arma un prompt con la pregunta + los documentos recuperados como
       contexto, y se lo pasamos a Gemma para que responda basandose en ese contexto.
"""

import sys

import chromadb
import ollama

sys.stdout.reconfigure(encoding="utf-8")

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "gemma4:12b" # cambia el tag segun lo que tengas descargado (ej: gemma3:1b, gemma2:9b)
COLLECTION_NAME = "clase_rag_demo"
TOP_K = 2

# --- Corpus de ejemplo (documentos "de la clase") ---
DOCUMENTOS = [
    {
        "id": "doc1",
        "text": (
            "RAG (Retrieval-Augmented Generation) es una tecnica que combina la "
            "busqueda de informacion con la generacion de texto: primero se recuperan "
            "documentos relevantes de una base de conocimiento, y luego un modelo de "
            "lenguaje genera la respuesta usando esos documentos como contexto."
        ),
    },
    {
        "id": "doc2",
        "text": (
            "Una base de datos vectorial almacena embeddings (vectores numericos que "
            "representan el significado de un texto) y permite buscar los vectores mas "
            "similares a una consulta usando metricas como la distancia coseno o euclidiana."
        ),
    },
    {
        "id": "doc3",
        "text": (
            "Ollama es una herramienta que permite correr modelos de lenguaje grandes "
            "de forma local, sin depender de una API en la nube. Soporta modelos como "
            "Gemma, Llama y modelos de embeddings como nomic-embed-text."
        ),
    },
    {
        "id": "doc4",
        "text": (
            "ChromaDB es una base de datos vectorial embebida, pensada para prototipos "
            "y aplicaciones RAG: se puede usar en memoria o persistir en disco, sin "
            "necesidad de levantar un servidor aparte."
        ),
    },
    {
        "id": "doc5",
        "text": (
            "El pipeline de un sistema RAG tipico tiene tres etapas: ingesta y chunking "
            "de documentos, indexado en una base vectorial, y en tiempo de consulta, "
            "recuperacion de los chunks mas relevantes seguida de generacion de la respuesta."
        ),
    },
]


def build_index() -> chromadb.Collection:
    """Crea (o recrea) la coleccion de Chroma y la llena con los embeddings de DOCUMENTOS."""
    client = chromadb.PersistentClient(path="./chroma_db")  # persiste en disco, no solo en memoria
    # Si ya existe de una corrida anterior, la recreamos para que quede limpia.
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME)

    for doc in DOCUMENTOS:
        # Cada documento se convierte en un vector numerico (embedding) via Ollama...
        embedding = ollama.embeddings(model=EMBED_MODEL, prompt=doc["text"])["embedding"]
        # ...y se guarda en Chroma junto con su texto original y un id unico.
        collection.add(
            ids=[doc["id"]],
            embeddings=[embedding],
            documents=[doc["text"]],
        )
    return collection


def retrieve(collection: chromadb.Collection, pregunta: str, k: int = TOP_K) -> list[str]:
    """Devuelve los k documentos mas parecidos (por embedding) a la pregunta."""
    query_embedding = ollama.embeddings(model=EMBED_MODEL, prompt=pregunta)["embedding"]
    resultados = collection.query(query_embeddings=[query_embedding], n_results=k)
    return resultados["documents"][0]  # [0] porque query() soporta multiples consultas a la vez


def generar_respuesta(pregunta: str, contexto: list[str]) -> str:
    """Arma un prompt con el contexto recuperado y le pide a Gemma que responda solo con eso."""
    contexto_str = "\n\n".join(f"- {c}" for c in contexto)
    prompt = f"""Respondé la pregunta usando SOLO la informacion del contexto. Si el contexto no alcanza, decilo.

Contexto:
{contexto_str}

Pregunta: {pregunta}

Respuesta:"""

    response = ollama.chat(model=CHAT_MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


def rag(collection: chromadb.Collection, pregunta: str) -> None:
    """Orquesta el flujo completo: recuperar contexto y generar la respuesta, con logs por consola."""
    print(f"\n=== Pregunta: {pregunta} ===")
    contexto = retrieve(collection, pregunta)
    print("\n--- Documentos recuperados ---")
    for c in contexto:
        print(f"  * {c[:90]}...")
    respuesta = generar_respuesta(pregunta, contexto)
    print("\n--- Respuesta del modelo ---")
    print(respuesta)


if __name__ == "__main__":
    print("Indexando documentos en ChromaDB...")
    collection = build_index()  # se reindexa siempre al arrancar, es rapido con este corpus chico
    print(f"Listo: {collection.count()} documentos indexados.\n")

    if len(sys.argv) > 1:
        # Uso: python rag_simple.py "¿pregunta libre?"
        rag(collection, " ".join(sys.argv[1:]))
    else:
        # Modo interactivo para probar en clase con preguntas de los alumnos.
        print("Escribí una pregunta (o 'salir' para terminar):")
        while True:
            pregunta = input("\n> ").strip()
            if pregunta.lower() in {"salir", "exit", "quit"}:
                break
            if pregunta:
                rag(collection, pregunta)
