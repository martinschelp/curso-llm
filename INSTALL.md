# Instalación rápida del entorno

Pasos mínimos para dejar el entorno listo. Los detalles y el resto de las demos
están en [`README.md`](README.md).

## Windows (PowerShell)

```powershell
# 1. Entorno (una sola vez)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Dependencias
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Si PowerShell bloquea la activación con un error de *execution policy*, corré una vez:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

## macOS / Linux (bash o zsh)

```bash
# 1. Entorno (una sola vez)
python3 -m venv .venv
source .venv/bin/activate

# 2. Dependencias
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Diferencias con Windows a tener en cuenta:

- El comando es **`python3`**, no `python`. En macOS `python` a secas no existe.
  Una vez activado el venv, adentro sí podés usar `python`.
- La activación es `source .venv/bin/activate` (no hay `.ps1`).
- Los lanzadores `.ps1` del repo (`iniciar_gemma_demo.ps1`) son de PowerShell y no
  corren acá; ejecutá los `.py` directamente.
- Las rutas de los ejemplos van con `/` en vez de `\`:
  ```bash
  python RAG/rag_simple.py "¿Que es RAG?"
  ```

### Versión de Python

Las versiones de `requirements.txt` están fijadas y probadas con **Python 3.11**.
Funcionan también en **3.12**. Evitá 3.13+ por ahora: varios pins (`torch`, `gensim`,
`numba`) todavía no tienen ruedas para esas versiones y la instalación falla.

Si tu `python3` por defecto es más nuevo, apuntá explícitamente al intérprete al crear
el venv:

```bash
python3.12 -m venv .venv
```

### Apple Silicon (M1/M2/M3/M4)

`torch` usa **MPS**, el backend de GPU de Apple, así que no estás limitado a CPU.
Para verificarlo:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

## Kernel para los notebooks

Con el venv activado, registrá el kernel una vez para poder elegirlo en VS Code o
Jupyter:

```bash
python -m ipykernel install --user --name curso-llm --display-name "curso-llm"
```
