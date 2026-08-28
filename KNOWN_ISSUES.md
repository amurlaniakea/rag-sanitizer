# KNOWN_ISSUES — rag-sanitizer v0.1

Lista de limitaciones conocidas y aceptadas. No son bloqueos de release; son el
territorio donde v0.2 (embeddings reales CLIP/MiniLM + NER) debe entrar.

- **KI-1** (aceptado): v0.1 usa embeddings dummy deterministas en la fast suite; no
  mide calidad contra embeddings reales de MiniLM/OpenAI. La lógica del escáner sí
  está validada.
- **KI-2** (aceptado): el detector multimodal v0.1 es heurístico (imagen faltante /
  caption mismatch por solapamiento de tokens >=0.5). La detección por embeddings
  CLIP es feature 002.
- **KI-3** (aceptado): el detector de entity-swap usa patrones, no NER; los umbrales
  por defecto están calibrados en fixture sintético.
- **KI-4** (aceptado): umbrales por defecto (k=3.0 mimicry, 0.5 caption overlap)
  pueden dar falsos positivos en corpus heterogéneo real; requieren validación del
  usuario.
- **KI-5** (aceptado): el escáner marca/quarantina, no repara ni reescribe documentos.
- **KI-6** (aceptado): el conteo de papers arXiv 2026 por sub-nicho no se obtuvo por
  API (rate-limit); los papers citados están confirmados individualmente por abs.
- **KI-7** (trade-off del fix entity-swap, post-auditoría Claude): `_is_real_org`
  exige 2+ palabras capitalizadas o sigla ALL-CAPS. Ya NO detecta el swap de un
  nombre propio de UNA sola palabra en mayúscula-minúscula (p.ej. `Madrid` →
  `Beijing`). Confirmado por la auditoría: `profile="Our headquarters is in Madrid,
  Spain."` + `doc="Our headquarters is in Beijing, Spain."` → `flagged=False`. Se
  aceptó el trade-off (eliminó falsos positivos masivos de "The/This/Our") y se
  deja para NER real en v0.2. No invalida el fix de los bugs 1 y 2.
