import pandas as pd
import matplotlib.pyplot as plt

TARGET_FILE = "elo_full_evolution_1997_2024.csv"
STRATEGIES = ["BuyAndHold", "FadeExtremes_63d", "MeanReversion_20d"]


def main():
    df = pd.read_csv(TARGET_FILE)
    # Parse timezone-aware then drop tz to avoid comparison issues
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_localize(None)

    df = df[df["strategy"].isin(STRATEGIES)].copy()
    if df.empty:
        print("No matching strategies found.")
        return

    # Sort for clean plotting
    df.sort_values(["strategy", "date"], inplace=True)

    plt.figure(figsize=(14, 7))
    for name in STRATEGIES:
        sub = df[df["strategy"] == name]
        if sub.empty:
            continue
        plt.plot(sub["date"], sub["mu"], label=name, linewidth=2)

    plt.title("ELO (mu) Evolution 1997-2024")
    plt.xlabel("Date")
    plt.ylabel("ELO (mu)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("elo_trio_evolution.png", dpi=300, bbox_inches="tight")
    print("Saved elo_trio_evolution.png")

    # Last year plot (last 365 calendar days from max date)
    max_date = df["date"].max()
    cutoff = max_date - pd.Timedelta(days=365)
    dfl = df[df["date"] >= cutoff].copy()
    if not dfl.empty:
        plt.figure(figsize=(14, 7))
        for name in STRATEGIES:
            sub = dfl[dfl["strategy"] == name]
            if sub.empty:
                continue
            plt.plot(sub["date"], sub["mu"], label=name, linewidth=2)

        plt.title(f"ELO (mu) Evolution - Last Year ({cutoff.date()} to {max_date.date()})")
        plt.xlabel("Date")
        plt.ylabel("ELO (mu)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("elo_trio_evolution_last_year.png", dpi=300, bbox_inches="tight")
        print("Saved elo_trio_evolution_last_year.png")


if __name__ == "__main__":
    main()


