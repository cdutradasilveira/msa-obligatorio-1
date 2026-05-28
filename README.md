# Obligatorio 1 — Sistemas Multiagente

Implementamos cuatro algoritmos de aprendizaje en juegos (Fictitious Play, Regret Matching, Independent Q-Learning y JAL-AM) y los probamos en varios ambientes (Matching Pennies, RPS, Blotto, Foraging, y también BoS, Chicken, Cournot y ThreePlayers como opcionales).

**Integrantes:** Bruno Dinello (192158) y Carlos Dutra da Silveira (342909)

## Cómo correrlo

1. Instalar `uv` desde [Astral](https://docs.astral.sh/uv/getting-started/installation/).
2. Crear el entorno y bajar las dependencias:
    ```bash
    uv sync
    ```
3. Correr todos los experimentos y generar las figuras (van a parar a `figures/`):
    ```bash
    uv run python experiments.py
    ```

## Qué hay en el repo

- **`base/`**: clases base (`SimultaneousGame` y `Agent`).
- **`agents/`**: los cuatro algoritmos más el RandomAgent.
- **`games/`**: los ocho ambientes.
- **`runner.py`**: motor que hace jugar a cualquier combinación de agentes en cualquier juego.
- **`experiments.py`**: define y corre todos los experimentos.
- **`notebooks/`**: notebooks para probar a mano.
