"""Exploration des données simulées : graphiques et statistiques simples.

Vérifie que les données contiennent bien ce qu'on attend :
saisonnalité hebdomadaire et annuelle, effet des promotions, ruptures de stock.

Usage : python -m src.explore
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # enregistre les graphiques sans ouvrir de fenêtre
import matplotlib.pyplot as plt
import pandas as pd

DATA_PATH = Path("data/sales.csv")
FIG_DIR = Path("reports/figures")
EXAMPLE_PRODUCT = 1
WEEKDAYS = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    df["stockout"] = (df["stock_end"] == 0).astype(int)
    return df


def plot_product_history(df: pd.DataFrame, product_id: int) -> None:
    """Ventes et stock d'un produit dans le temps, avec promos et ruptures."""
    d = df[df["product_id"] == product_id]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax1.plot(d["date"], d["sales"], lw=0.8, color="tab:blue", label="Ventes")
    promo = d[d["promo"] == 1]
    ax1.scatter(promo["date"], promo["sales"], s=8, color="tab:orange", label="Jour de promo", zorder=3)
    ax1.set_ylabel("Ventes / jour")
    ax1.set_title(f"Produit {product_id} : ventes")
    ax1.legend(loc="upper left")

    ax2.plot(d["date"], d["stock_end"], lw=0.8, color="tab:green", label="Stock (soir)")
    out = d[d["stockout"] == 1]
    ax2.scatter(out["date"], out["stock_end"], s=10, color="tab:red", label="Rupture", zorder=3)
    ax2.set_ylabel("Stock")
    ax2.set_title(f"Produit {product_id} : stock")
    ax2.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "1_historique_produit.png", dpi=110)
    plt.close(fig)


def plot_weekday_effect(df: pd.DataFrame) -> pd.Series:
    """Ventes moyennes par jour de la semaine (hors jours de rupture et de promo)."""
    d = df[(df["stockout"] == 0) & (df["promo"] == 0)].copy()
    # On divise par la moyenne du produit pour comparer des produits de tailles différentes
    d["relative_sales"] = d["sales"] / d.groupby("product_id")["sales"].transform("mean")
    effect = d.groupby(d["date"].dt.day_of_week)["relative_sales"].mean()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(WEEKDAYS, effect.values, color="tab:blue")
    ax.axhline(1, color="grey", ls="--", lw=1)
    ax.set_ylabel("Ventes relatives (1 = moyenne)")
    ax.set_title("Effet du jour de la semaine")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "2_jour_semaine.png", dpi=110)
    plt.close(fig)
    return effect


def plot_yearly_season(df: pd.DataFrame, product_ids: list[int]) -> None:
    """Ventes moyennes par mois pour quelques produits."""
    d = df[(df["product_id"].isin(product_ids)) & (df["stockout"] == 0)]
    monthly = d.groupby(["product_id", d["date"].dt.to_period("M")])["sales"].mean().unstack(0)

    fig, ax = plt.subplots(figsize=(12, 4))
    for pid in product_ids:
        ax.plot(monthly.index.to_timestamp(), monthly[pid], marker="o", ms=3, label=f"Produit {pid}")
    ax.set_ylabel("Ventes moyennes / jour")
    ax.set_title("Saisonnalité annuelle (moyenne par mois)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "3_saison_annuelle.png", dpi=110)
    plt.close(fig)


def plot_stockout_rate(df: pd.DataFrame) -> pd.Series:
    """Pourcentage de jours en rupture pour chaque produit."""
    rate = df.groupby("product_id")["stockout"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(rate.index.astype(str), rate.values * 100, color="tab:red")
    ax.axhline(df["stockout"].mean() * 100, color="grey", ls="--", lw=1, label="Moyenne")
    ax.set_xlabel("Produit")
    ax.set_ylabel("% de jours en rupture")
    ax.set_title("Taux de rupture par produit")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "4_taux_rupture.png", dpi=110)
    plt.close(fig)
    return rate


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    print("=== Aperçu ===")
    print(df.head(), "\n")
    print("=== Statistiques ===")
    print(df[["sales", "stock_end", "promo", "stockout"]].describe().round(2), "\n")
    print("Valeurs manquantes :", int(df.isna().sum().sum()))

    no_out = df[df["stockout"] == 0]
    promo_mean = no_out.loc[no_out["promo"] == 1, "sales"].mean()
    normal_mean = no_out.loc[no_out["promo"] == 0, "sales"].mean()
    print(f"Ventes moyennes : {normal_mean:.1f} sans promo, {promo_mean:.1f} avec promo")

    plot_product_history(df, EXAMPLE_PRODUCT)
    effect = plot_weekday_effect(df)
    print("Effet jour de la semaine :", dict(zip(WEEKDAYS, effect.round(2))))
    plot_yearly_season(df, [1, 2, 3])
    rate = plot_stockout_rate(df)
    print(f"Taux de rupture : de {rate.min():.1%} à {rate.max():.1%} selon le produit")
    print(f"\nGraphiques enregistrés dans {FIG_DIR}/")


if __name__ == "__main__":
    main()
