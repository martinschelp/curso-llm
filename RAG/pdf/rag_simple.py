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

import os
import sys
import chromadb
import ollama
from pypdf import PdfReader  # lector de PDFs (pip install pypdf)
    
sys.stdout.reconfigure(encoding="utf-8")

EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "gemma4:12b" # cambia el tag segun lo que tengas descargado (ej: gemma3:1b, gemma2:9b)
TOP_K = 2
BASE_DIR = os.path.dirname(__file__)
DOCS_DIR = os.path.join(BASE_DIR, "docs")
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_pdf")
COLLECTION_NAME = "pdf_docs"
CHUNK_SIZE = 800      # caracteres por chunk
CHUNK_OVERLAP = 150   # solapamiento entre chunks consecutivos


def chunk_text(texto: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Parte un texto en fragmentos de `size` caracteres, solapados `overlap` caracteres
    (asi una idea que cae justo en el borde no queda partida al medio)."""
    texto = " ".join(texto.split())  # normaliza espacios y saltos de linea
    if not texto:
        return []
    chunks = []
    inicio = 0
    while inicio < len(texto):
        chunks.append(texto[inicio:inicio + size])
        inicio += size - overlap  # avanza menos que "size": de ahi viene el solapamiento
    return chunks


def extraer_chunks_pdf(ruta: str) -> list[dict]:
    """Extrae el texto de un PDF y devuelve una lista de chunks con metadata."""
    reader = PdfReader(ruta)
    nombre = os.path.basename(ruta)
    items = []
    for n_pagina, page in enumerate(reader.pages, start=1):
        texto = page.extract_text() or ""
        for i, ch in enumerate(chunk_text(texto)):
            items.append({
                "id": f"{nombre}::p{n_pagina}::c{i}",
                "text": ch,
                "metadata": {"source": nombre, "page": n_pagina},
            })
    return items

def build_index() -> chromadb.Collection:
    """Crea (o recrea) la coleccion de Chroma y la llena con los embeddings de los PDFs de docs/."""
    client = chromadb.PersistentClient(path=PERSIST_DIR)  # persiste en disco, no solo en memoria
    # Si ya existe de una corrida anterior, la recreamos para que quede limpia.
    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(COLLECTION_NAME)

    # extraer_chunks_pdf trabaja sobre UN archivo, asi que recorremos la carpeta docs/.
    pdfs = [f for f in sorted(os.listdir(DOCS_DIR)) if f.lower().endswith(".pdf")]
    if not pdfs:
        print(f"[aviso] No hay PDFs en {DOCS_DIR}. Poné tus archivos ahi y volvé a correr.")

    for nombre in pdfs:
        items = extraer_chunks_pdf(os.path.join(DOCS_DIR, nombre))
        if not items:
            print(f"  ! {nombre}: sin texto extraible (¿PDF escaneado?)")
            continue
        for item in items:
            # Cada chunk se convierte en un vector numerico (embedding) via Ollama...
            embedding = ollama.embeddings(model=EMBED_MODEL, prompt=item["text"])["embedding"]
            # ...y se guarda en Chroma junto con su texto original, su id unico y de donde salio.
            collection.add(
                ids=[item["id"]],
                embeddings=[embedding],
                documents=[item["text"]],
                metadatas=[item["metadata"]],
            )
        print(f"  + {nombre}: {len(items)} chunks indexados")
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
