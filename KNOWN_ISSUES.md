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

## Known issues (v0.2)

- **KI-8 (supply-chain de modelo, B615 bandit): RESUELTO en v0.2.** `ClipEmbedder`
  ahora carga CLIP con `revision` pinneada por defecto
  (`DEFAULT_CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"`, commit de
  referencia de `openai/clip-vit-base-patch32` según HF API). Bandit ya no marca B615
  (verificado: `bandit` reporta 0 issues en `clip_embedder.py`). El usuario puede
  override con `model_name="org/model@<sha>` si requiere otro commit. No se usa `# nosec`;
  la mitigación real es el pin, no silenciar el linter.

- **KI-9 (MiniLM real no separa gibberish fluido, hallado por la batería `real_embeddings`):**
  Con `SentenceTransformerEmbedder` (MiniLM, 384-d) y un perfil de confianza de 12 docs
  financieros, el detector de mimicry **no** marca como outlier texto *gibberish fluido*
  de longitud similar al perfil: medido `distance=0.981` vs `threshold(k=3.0)=1.475`
  (clean "Madrid" en `0.889`). Incluso un doc de dominio totalmente ajeno (medicina/
  cocina) cae en `0.97–1.00`, por debajo de `1.475`; solo con `k=2.0` el doc de cocina
  se marca (`1.004 > 0.983`), pero el gibberish sigue sin marcarse. Causa: MiniLM coloca
  prosa fluida (sea real o inventada) cerca del centro del cluster de prosa del perfil.
  **Limitación real del embedder para ese vector de ataque concreto**, no un bug de
  `detect_mimicry`. Mitigación: (a) no fiarse de mimicry para detectar gibberish; (b)
  calibrar `k` contra el corpus real del usuario (el default 3.0 es muy holgado para
  MiniLM); (c) combinar con el detector entity-swap (KI-7). No se fuerza el test a verde:
  `test_real_embedder_does_not_separate_fluent_gibberish` fija el comportamiento honesto.

- **Perfil de confianza pequeño (guard `MIN_PROFILE_SIZE=10`):** `detect_mimicry` exige
  `n >= 10` documentos de confianza para que el umbral `k*sigma` sea estadísticamente
  significativo (con n=3, sigma es ruido). Si `n < 10`, el resultado lleva
  `low_confidence=True` y `note="profile too small for reliable k-sigma threshold
  (n=X < 10)"`; el scanner lo propaga a `reasons` para que el veredicto nunca sea silencioso.
  No es un fallback que oculta el problema: es señalización explícita de baja confianza.
