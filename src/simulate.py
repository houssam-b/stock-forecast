"""Génération de données de ventes et de stock simulées.

Pour chaque produit et chaque jour, on simule :
- la demande des clients (tendance, saisonnalité, promotions, bruit) ;
- les ventes réelles (on ne peut pas vendre plus que le stock) ;
- le stock et les réapprovisionnements (commande quand le stock est bas).

Usage : python -m src.simulate
"""

from pathlib import Path

import numpy as np
import pandas as pd

N_PRODUCTS = 20
START_DATE = "2023-01-01"
N_DAYS = 730  # 2 ans
SEED = 42
OUTPUT_PATH = Path("data/sales.csv")

# Effet du jour de la semaine (lundi = 0 ... dimanche = 6)
WEEKDAY_EFFECT = np.array([0.85, 0.90, 0.95, 1.00, 1.15, 1.30, 0.85])


def make_product_params(rng: np.random.Generator, product_id: int) -> dict:
    """Tire au hasard les caractéristiques d'un produit."""
    base = rng.uniform(5, 50)  # ventes moyennes par jour
    lead_time_mean = rng.integers(3, 8)  # délai de livraison moyen (jours)
    return {
        "product_id": product_id,
        "base": base,
        "trend": rng.uniform(-0.15, 0.30),  # évolution par an (-15 % à +30 %)
        "season_amp": rng.uniform(0.0, 0.35),  # force de la saison annuelle
        "season_phase": rng.uniform(0, 2 * np.pi),  # mois du pic de saison
        "promo_prob": rng.uniform(0.02, 0.08),  # chance qu'une promo commence
        "promo_uplift": rng.uniform(1.3, 2.0),  # multiplicateur pendant la promo
        "lead_time_mean": lead_time_mean,
        # Seuil de commande : un facteur < 1 donne souvent des ruptures
        "reorder_point": base * lead_time_mean * rng.uniform(0.7, 1.4),
        "order_qty": base * rng.uniform(10, 20),  # quantité commandée
    }


def make_promo_calendar(rng: np.random.Generator, p: dict) -> np.ndarray:
    """Crée les jours de promo : chaque promo dure entre 3 et 7 jours."""
    promo = np.zeros(N_DAYS, dtype=int)
    day = 0
    while day < N_DAYS:
        if rng.random() < p["promo_prob"]:
            length = rng.integers(3, 8)
            promo[day : day + length] = 1
            day += length + 7  # au moins une semaine entre deux promos
        else:
            day += 1
    return promo


def simulate_product(rng: np.random.Generator, p: dict, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Simule les ventes et le stock d'un produit, jour par jour."""
    t = np.arange(N_DAYS) / 365  # temps en années
    promo = make_promo_calendar(rng, p)

    # Demande moyenne de chaque jour
    expected = (
        p["base"]
        * (1 + p["trend"] * t)
        * WEEKDAY_EFFECT[dates.day_of_week]
        * (1 + p["season_amp"] * np.sin(2 * np.pi * dates.day_of_year / 365 + p["season_phase"]))
        * np.where(promo == 1, p["promo_uplift"], 1.0)
    )
    # Demande réelle = moyenne + hasard (loi de Poisson)
    demand = rng.poisson(np.maximum(expected, 0))

    stock = p["order_qty"]  # stock au départ
    pending = {}  # livraisons en attente : {jour d'arrivée: quantité}
    rows = []

    for day in range(N_DAYS):
        # 1. Réception d'une livraison
        received = pending.pop(day, 0)
        stock += received
        stock_start = stock

        # 2. Ventes : on ne peut pas vendre plus que le stock
        sales = min(demand[day], stock)
        stock -= sales

        # 3. Commande si le stock est bas et qu'aucune livraison n'est attendue
        if stock <= p["reorder_point"] and not pending:
            lead_time = max(1, p["lead_time_mean"] + rng.integers(-1, 3))
            pending[day + lead_time] = p["order_qty"]

        rows.append(
            {
                "date": dates[day],
                "product_id": p["product_id"],
                "promo": promo[day],
                "stock_start": round(stock_start),
                "received": round(received),
                "sales": int(sales),
                "stock_end": round(stock),
            }
        )

    return pd.DataFrame(rows)


def simulate(n_products: int = N_PRODUCTS, seed: int = SEED) -> pd.DataFrame:
    """Simule tous les produits et renvoie un seul tableau."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(START_DATE, periods=N_DAYS, freq="D")
    frames = [simulate_product(rng, make_product_params(rng, pid), dates) for pid in range(1, n_products + 1)]
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    df = simulate()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    stockout_rate = (df["stock_end"] == 0).mean()
    print(f"Données enregistrées dans {OUTPUT_PATH}")
    print(f"{len(df)} lignes, {df['product_id'].nunique()} produits, du {df['date'].min():%Y-%m-%d} au {df['date'].max():%Y-%m-%d}")
    print(f"Jours en rupture de stock : {stockout_rate:.1%}")


if __name__ == "__main__":
    main()
