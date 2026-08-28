# rag-sanitizer

Escáner de **corpus RAG antes de la ingestión**. Detecta documentos *poison*
(corpus poisoning) y los marca o pone en cuarentena **antes** de que entren en la
pipeline RAG (`chunk` + `embed` + `index`). Determinista, barato y *black-box*
sobre el corpus: no necesita el generador ni un LLM grande en inferencia.

> Parte del ecosistema de defensa de agentes de `amurlaniakea` (patrón
> keybound → topowatch → weightwatch, aplicado a la cadena de suministro de
> **conocimiento**).

## Por qué existe (el gap)

El *corpus poisoning* es el ataque de menor coste y mayor impacto a RAG: el
adversario cola unos pocos documentos en la base de conocimiento y el modelo los
trata como autoritativos. Las defensas existentes operan **después de recuperar**
o **en inferencia**:

- **RAGSentinel** (arXiv 2608.23965): filtra documentos ya recuperados, consenso geométrico, requiere *honest-majority*.
- **Trustworthy RAG** (arXiv 2608.21095): *Evaluation Agent* en inferencia (Llama 3.3 70B), 91 % acc / 100 % prec; *"entity swaps remain hard to detect"*.
- **CEG-RAG**: defensa post-retrieval por cross-encoder.
- **Vis-Poison** (arXiv 2608.20756): ataque multimodal (40–65 % ASR), frente sin defensa empaquetada.

GitHub (total_count por sub-nicho estrecho, 2026-08-28): el tronco amplio
*"RAG poisoning defense"* da 522 repos, pero los de **pre-ingestión** están casi
vacíos — **`RAG corpus sanitizer` = 1**, **`RAG ingest sanitizer` = 3**,
**`RAG backdoor detection` = 1**. Los de post-hoc (detector 9, trust agent 27)
tienen competencia. Ese es el hueco que `rag-sanitizer` cubre.

## Qué detecta (v0.1)

| Señal | Método (v0.1) | Veredicto |
|-------|---------------|-----------|
| Semantic mimicry | outlier de embedding respecto al perfil de confianza (`k·σ`) | POISON |
| Entity-swap | entidades (años, importes, ORG) ausentes en el perfil | POISON |
| Visual poison (multimodal) | heurístico: imagen declarada faltante / caption desajustado | SUSPECT |

## Instalación

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
```

Embeddings reales (v0.2): `pip install -e ".[real-embeddings]"`.

## Uso

```bash
# Corpus en JSONL con campo "trust": "clean" para el perfil de confianza
rag-sanitizer scan fixtures/corpus.jsonl --out report.json

# O contra un directorio de .txt/.md (modo no supervisado: usa el centroide mayoritario)
rag-sanitizer scan ./mi-corpus --out report.md
```

El reporte incluye `corpus_sha256` (reproducibilidad) y, por documento, el
veredicto (`CLEAN` / `SUSPECT` / `POISON`) con sus razones legibles.

## Honestidad (lo que v0.1 NO es)

- **No** es defensa post-retrieval (eso lo hacen RAGSentinel / Trustworthy RAG).
- **No** reescribe ni repara documentos; solo marca / quarantina.
- Embeddings de la *fast suite* son deterministas y Dummy (no semánticos): validan
  la **lógica** del escáner, no la calidad del embedding (feature 002).
- El detector multimodal v0.1 es heurístico; la detección por embeddings CLIP es
  feature 002.
- Umbrales por defecto calibrados en fixture sintético; necesitan validación contra
  corpus reales del usuario (riesgo de falsos positivos en corpus heterogéneo).

Ver `KNOWN_ISSUES` en la spec de la bóveda y `RESEARCH.md` para las fuentes.

## Licencia

AGPL-3.0-or-later. Texto oficial en [`LICENSE`](./LICENSE).
Autor: Pedro Sordo Martínez — amurlaniakea@gmail.com.
