# RESEARCH — fuentes del gap (rag-sanitizer)

Documento de trazabilidad de la investigación que justifica el nicho. Todos los
papers citados fueron verificados por `arxiv.org/abs/<ID>` (HTTP 200, abstract
leído) el 2026-08-28. El conteo por sub-nicho de arXiv NO se pudo obtener por API
(export.arxiv.org devolvió 429 persistentes desde WSL); los números de GitHub SÍ
son de `gh api` autenticado (total_count real).

## Método

Barrido profundo de sub-nichos de RAG poisoning (patrón find-niche: gap = papers
con datos ÷ repos de solución empaquetada). Queries estrechas, no "moda".

## GitHub — sub-nichos estrechos (total_count, gh api autenticado, 2026-08-28)

| Sub-nicho | Repos |
|-----------|------:|
| RAG corpus sanitizer | 1 |
| RAG ingest sanitizer | 3 |
| RAG backdoor detection | 1 |
| multimodal RAG poisoning | 6 |
| RAG poisoning detector (post-hoc) | 9 |
| retrieval poisoning detection | 9 |
| RAG trust evaluation agent | 27 |
| document poisoning RAG | 23 |
| *(ref. tronco amplio 2026-08-27)* RAG poisoning defense | 522 |

**Lectura:** el tronco amplio está cubierto, pero la defensa de **pre-ingestión**
(corpus/ingest sanitizer) tiene 1–3 repos. Hueco limpio.

## Papers verificados (arxiv.org/abs, abstract leído)

- **2608.23965 — RAGSentinel: Certifiable Geometric Consensus for Robust RAG.**
  Defensa POST-retrieval: surrogate encoder mide hidden-state shifts condicionados a
  la query, quita direcciones de tema compartido, filtra poison como outliers
  geométricos de un consenso mayoritario robusto. Requiere *honest-majority* +
  separación a nivel de representación. Training-free, label-free.
- **2608.21095 — Trustworthy RAG: An Evaluation Agent for Detecting Misinformation
  and Knowledge Poisoning.** Middleware en INFERENCIA (NLI + 5-signal poison
  detector + Trust Index T = 0.4F + 0.35C + 0.25(1−P)). TruthfulQA con Llama 3.3
  70B: 91 % accuracy, 100 % precision, 100 % recall en instruction injection.
  *"entity swaps (in-place edits) remain hard to detect"*. Costoso (LLM 70B).
- **2608.20756 — Vis-Poison: Poisoning Visual Knowledge in Multimodal RAG.** ATAQUE:
  la imagen envenenada es el payload (sin tocar caption/metadata). ASR 40.16 %–
  65.40 % contra knowledge bases multimodales de 30k entradas, black-box. Frente
  multimodal sin defensa empaquetada.
- **2606.12469 — Influence Factors on RAG Poisoning.** Análisis (no herramienta): la
  vulnerabilidad surge de la interacción retrieval + generation + kb-config.
- **2605.05632 — Architecture Matters: Comparing RAG Systems under Knowledge Base
  Poisoning.** Análisis comparativo de arquitecturas.
- **CEG-RAG** (moradi26a, proc. mlresearch v318): defensa POST-retrieval por
  cross-encoder.
- **corpus-poisoning** (princeton-nlp, EMNLP 2023): ataque contra corpus de
  retrieval denso; su defensa correspondiente es escasa.

## Veredicto

TODAS las defensas existentes operan POST-retrieval o EN INFERENCIA. El corpus
poisoning (inyección ANTES de la ingestión) tiene papers que lo cuantifican pero la
defensa de PRE-ingestión empaquetada tiene 1–3 repos. `rag-sanitizer` cubre ese
tramo: escáner determinista y barato que marca el corpus antes de `chunk + embed`.

## Limitaciones de la evidencia

- Conteo de papers arXiv 2026 por sub-nicho: NO disponible (API rate-limited). Los
  papers arriba están confirmados individualmente, no por volumen.
- La efectividad contra embeddings reales (MiniLM/OpenAI/CLIP) no está medida en
  v0.1 (fast suite usa embeddings dummy deterministas); es feature 002.
