from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "feasibility_graphs"
POWER_MW = 300.0
HP_POWER_MW = POWER_MW / 3
HP_EFFICIENCY = 0.60
DC_HEAT_TEMP_C = 25.0
DH_SALE_PRICE_EUR_MWH = 116.06 - 36.0
VAT = 0.255
HP_ENERGY_TAX_EUR_MWH = 1.35
HP_COST_EUR_MW = 1.5e6
INFRASTRUCTURE_CAPEX_EUR = 10e6
MAINTENANCE_EUR_YEAR = 1e6
BASE_PRICE_EUR_MONTH = 568.92 * (1 + VAT)
POWER_TARIFF_EUR_MW_MONTH = 1.45e3 * (1 + VAT)
DAY_FEE_EUR_MWH = 0.021 * 1e3 * (1 + VAT)
NIGHT_FEE_EUR_MWH = 0.0093 * 1e3 * (1 + VAT)
INTERVAL_HOURS = 0.25

BG = "#101923"
PANEL = "#17232f"
TEXT = "#eef4f7"
MUTED = "#a9bac5"
TEAL = "#46d6c5"
BLUE = "#6da9ff"
GOLD = "#ffca72"
CORAL = "#ff7d73"
PURPLE = "#b69cff"
GRID = "#334452"


def source_file(pattern: str) -> Path:
    matches = list(ROOT.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one file matching {pattern!r} in {ROOT}; "
            f"found {len(matches)}."
        )
    return matches[0]


def load_data() -> pd.DataFrame:
    spot = pd.read_csv(source_file("spot_price.csv"))
    heat = pd.read_csv(source_file("helen_open_dh_data_2025.csv"), sep=";")
    weather = pd.read_csv(
        source_file("Daily average air temperature*.csv"), encoding="cp1252"
    )
    if len(spot.columns) < 2 or len(heat.columns) < 2 or len(weather.columns) < 6:
        raise ValueError("An input data file is missing required columns.")

    spot = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                spot.iloc[:, 0], format="%d/%m/%Y %H.%M.%S", errors="raise"
            ),
            "spot_price_eur_mwh": pd.to_numeric(spot.iloc[:, 1], errors="raise")
            * 10,
        }
    ).sort_values("datetime")
    heat = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                heat.iloc[:, 0], format="%d.%m.%Y %H:%M", errors="raise"
            ),
            "dh_demand_mw": pd.to_numeric(
                heat.iloc[:, 1].astype(str).str.replace(",", ".", regex=False),
                errors="raise",
            ),
        }
    ).sort_values("datetime")
    weather = pd.DataFrame(
        {
            "date": pd.to_datetime(
                weather.iloc[:, 1].astype(str)
                + "-"
                + weather.iloc[:, 2].astype(str)
                + "-"
                + weather.iloc[:, 3].astype(str),
                errors="raise",
            ),
            "outdoor_temp_c": pd.to_numeric(weather.iloc[:, 5], errors="raise"),
        }
    )
    if spot.empty or heat.empty or weather.empty:
        raise ValueError("An input data file contains no records.")
    if spot["datetime"].dt.year.nunique() != 1:
        raise ValueError("Spot-price data must cover a single calendar year.")

    data = spot.merge(heat, on="datetime", how="left")
    data["dh_demand_mw"] = data["dh_demand_mw"].ffill()
    data["outdoor_temp_c"] = data["datetime"].dt.normalize().map(
        weather.set_index("date")["outdoor_temp_c"]
    )
    if data[["dh_demand_mw", "outdoor_temp_c"]].isna().any().any():
        raise ValueError("Heat demand or weather data could not be aligned to prices.")
    if (data["dh_demand_mw"] < 0).any() or heat["dh_demand_mw"].max() <= 0:
        raise ValueError("District-heat demand must be non-negative with a positive peak.")

    data["modeled_heat_mw"] = (
        POWER_MW * data["dh_demand_mw"] / heat["dh_demand_mw"].max()
    )
    data["dh_supply_temp_c"] = np.clip(
        75 + (5 - data["outdoor_temp_c"]) * (45 / 35), 75, 120
    )
    hot_k = data["dh_supply_temp_c"] + 273.15
    cold_k = DC_HEAT_TEMP_C + 273.15
    data["cop"] = HP_EFFICIENCY * hot_k / (hot_k - cold_k)
    data["hp_power_mw"] = data["modeled_heat_mw"] / data["cop"]

    electricity_mwh = data["hp_power_mw"] * INTERVAL_HOURS
    data["revenue_eur"] = (
        data["modeled_heat_mw"]
        * INTERVAL_HOURS
        * DH_SALE_PRICE_EUR_MWH
        * (1 - VAT)
    )
    data["electricity_cost_eur"] = (
        electricity_mwh * data["spot_price_eur_mwh"]
    )
    data["energy_tax_eur"] = (
        electricity_mwh * HP_ENERGY_TAX_EUR_MWH * (1 + VAT)
    )
    distribution_fee = np.where(
        data["datetime"].dt.hour.between(7, 21),
        DAY_FEE_EUR_MWH,
        NIGHT_FEE_EUR_MWH,
    )
    data["distribution_cost_eur"] = electricity_mwh * distribution_fee
    return data


def economics(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    monthly_peak_mw = data.groupby(data["datetime"].dt.to_period("M"))[
        "hp_power_mw"
    ].max()
    annual_power_charge = (
        monthly_peak_mw.nlargest(2).mean() * POWER_TARIFF_EUR_MW_MONTH * 12
    )
    annual_base_charge = BASE_PRICE_EUR_MONTH * 12
    annual_capex = INFRASTRUCTURE_CAPEX_EUR + HP_POWER_MW * HP_COST_EUR_MW

    monthly = data.set_index("datetime").resample("MS").agg(
        revenue_eur=("revenue_eur", "sum"),
        electricity_cost_eur=("electricity_cost_eur", "sum"),
        energy_tax_eur=("energy_tax_eur", "sum"),
        distribution_cost_eur=("distribution_cost_eur", "sum"),
        heat_mwh=("modeled_heat_mw", lambda s: s.sum() * INTERVAL_HOURS),
        hp_electricity_mwh=("hp_power_mw", lambda s: s.sum() * INTERVAL_HOURS),
    )
    monthly["base_charge_eur"] = annual_base_charge / len(monthly)
    monthly["power_charge_eur"] = annual_power_charge / len(monthly)
    monthly["maintenance_eur"] = MAINTENANCE_EUR_YEAR / len(monthly)
    monthly["operating_cost_eur"] = monthly[
        [
            "electricity_cost_eur",
            "energy_tax_eur",
            "distribution_cost_eur",
            "base_charge_eur",
            "power_charge_eur",
            "maintenance_eur",
        ]
    ].sum(axis=1)
    monthly["operating_margin_eur"] = (
        monthly["revenue_eur"] - monthly["operating_cost_eur"]
    )
    totals = {
        "annual_revenue_eur": float(monthly["revenue_eur"].sum()),
        "electricity_cost_eur": float(monthly["electricity_cost_eur"].sum()),
        "energy_tax_eur": float(monthly["energy_tax_eur"].sum()),
        "distribution_cost_eur": float(monthly["distribution_cost_eur"].sum()),
        "base_charge_eur": float(annual_base_charge),
        "power_charge_eur": float(annual_power_charge),
        "maintenance_eur": float(MAINTENANCE_EUR_YEAR),
        "annual_operating_cost_eur": float(monthly["operating_cost_eur"].sum()),
        "annual_operating_margin_eur": float(monthly["operating_margin_eur"].sum()),
        "annual_heat_mwh": float(monthly["heat_mwh"].sum()),
        "annual_hp_electricity_mwh": float(monthly["hp_electricity_mwh"].sum()),
        "capex_eur": float(annual_capex),
        "simple_payback_years": (
            float(annual_capex / monthly["operating_margin_eur"].sum())
            if monthly["operating_margin_eur"].sum() > 0
            else float("inf")
        ),
        "five_year_net_eur": float(
            -annual_capex + 5 * monthly["operating_margin_eur"].sum()
        ),
        "peak_heat_mw": float(data["modeled_heat_mw"].max()),
        "peak_hp_power_mw": float(data["hp_power_mw"].max()),
        "mean_cop": float(data["cop"].mean()),
        "minimum_cop": float(data["cop"].min()),
        "maximum_cop": float(data["cop"].max()),
    }
    return monthly, totals


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "text.color": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 10,
        }
    )


def finish_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUTPUT / name, dpi=180, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def heat_match_chart(data: pd.DataFrame) -> None:
    monthly = data.set_index("datetime").resample("MS").agg(
        dh_demand_mw=("dh_demand_mw", "mean"),
        modeled_heat_mw=("modeled_heat_mw", "mean"),
        outdoor_temp_c=("outdoor_temp_c", "mean"),
    )
    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = np.arange(len(monthly))
    width = 0.36
    ax.bar(
        x - width / 2,
        monthly["dh_demand_mw"],
        width,
        label="Helen DH demand (monthly mean)",
        color=BLUE,
        alpha=0.88,
    )
    ax.bar(
        x + width / 2,
        monthly["modeled_heat_mw"],
        width,
        label="Modeled recovered heat (monthly mean)",
        color=TEAL,
        alpha=0.92,
    )
    ax.set_ylabel("Thermal power (MW)")
    ax.set_xticks(x, monthly.index.strftime("%b"))
    ax.grid(axis="y", alpha=0.45)
    ax.set_axisbelow(True)
    ax2 = ax.twinx()
    ax2.plot(
        x,
        monthly["outdoor_temp_c"],
        color=GOLD,
        marker="o",
        linewidth=2.2,
        label="Lappeenranta mean outdoor temperature",
    )
    ax2.set_ylabel("Outdoor temperature (°C)", color=GOLD)
    ax2.tick_params(axis="y", colors=GOLD)
    ax2.spines["right"].set_color(GOLD)
    ax.set_title("A winter-peaking heat sink—and a demand-shaped recovery scenario")
    ax.text(
        0.01,
        0.98,
        "Demand profile is scaled to a 300 MW model peak; it is not measured data-centre output.",
        transform=ax.transAxes,
        va="top",
        color=MUTED,
        fontsize=9,
    )
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper right",
        frameon=False,
        fontsize=8.5,
    )
    fig.text(
        0.01,
        0.01,
        "Source inputs: Helen Helsinki DH demand and Lappeenranta weather, 2025. Demand is shown as average MW.",
        color=MUTED,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    finish_figure(fig, "01_heat_match.png")


def heat_pump_chart(data: pd.DataFrame) -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5.8))
    sample = data.iloc[::4]
    points = ax.scatter(
        sample["outdoor_temp_c"],
        sample["cop"],
        c=sample["modeled_heat_mw"],
        cmap="viridis",
        s=11,
        alpha=0.62,
        linewidths=0,
    )
    ax.set_xlabel("Outdoor temperature (°C)")
    ax.set_ylabel("Heat-pump COP")
    ax.set_title("Modeled COP across the year")
    ax.grid(alpha=0.38)
    cbar = fig.colorbar(points, ax=ax, pad=0.02)
    cbar.set_label("Modeled heat delivery (MW)")
    cbar.ax.yaxis.set_tick_params(color=MUTED)
    plt.setp(cbar.ax.get_yticklabels(), color=MUTED)

    monthly = data.set_index("datetime").resample("MS").agg(
        mean_cop=("cop", "mean"),
        hp_peak_mw=("hp_power_mw", "max"),
    )
    x = np.arange(len(monthly))
    ax2.plot(x, monthly["mean_cop"], color=TEAL, marker="o", linewidth=2.5)
    ax2.set_ylabel("Monthly mean COP", color=TEAL)
    ax2.tick_params(axis="y", colors=TEAL)
    ax2.set_xticks(x, monthly.index.strftime("%b"))
    ax2.grid(axis="y", alpha=0.38)
    ax2.set_title("Seasonality and heat-pump sizing check")
    ax2b = ax2.twinx()
    ax2b.plot(x, monthly["hp_peak_mw"], color=CORAL, marker="s", linewidth=1.8)
    ax2b.axhline(HP_POWER_MW, color=CORAL, linestyle="--", alpha=0.75)
    ax2b.set_ylabel("Monthly peak electric input (MW)", color=CORAL)
    ax2b.tick_params(axis="y", colors=CORAL)
    ax2.text(
        0.02,
        0.03,
        f"Assumed heat-pump rating: {HP_POWER_MW:.0f} MW",
        transform=ax2.transAxes,
        color=MUTED,
        fontsize=9,
    )
    fig.suptitle(
        "Heat-pump performance is promising in the model; temperature lift drives COP",
        fontsize=14,
        fontweight="bold",
        color=TEXT,
        y=1.02,
    )
    fig.text(
        0.01,
        -0.02,
        "COP uses the notebook's 60%-of-Carnot assumption, 25°C source heat, and modeled 75–120°C DH supply curve.",
        color=MUTED,
        fontsize=8,
    )
    fig.tight_layout()
    finish_figure(fig, "02_heat_pump_performance.png")


def economics_chart(monthly: pd.DataFrame, totals: dict[str, float]) -> None:
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(14, 6.5), gridspec_kw={"width_ratios": [1.15, 1]}
    )
    x = np.arange(len(monthly))
    ax.bar(
        x,
        monthly["revenue_eur"] / 1e6,
        color=TEAL,
        label="Heat-sale revenue",
        width=0.68,
    )
    bottom = np.zeros(len(monthly))
    cost_columns = [
        ("electricity_cost_eur", "Spot electricity", BLUE),
        ("energy_tax_eur", "Energy tax", PURPLE),
        ("distribution_cost_eur", "Grid distribution", GOLD),
        ("power_charge_eur", "Power tariff", CORAL),
        ("base_charge_eur", "Base fee", "#81929d"),
        ("maintenance_eur", "Maintenance", "#d99ab4"),
    ]
    for column, label, color in cost_columns:
        values = monthly[column].to_numpy() / 1e6
        ax.bar(x, values, bottom=bottom, color=color, label=label, width=0.68)
        bottom += values
    ax.plot(
        x,
        monthly["operating_margin_eur"] / 1e6,
        color=TEXT,
        marker="o",
        markersize=3,
        linewidth=1.6,
        label="Monthly operating margin",
    )
    ax.set_xticks(x, monthly.index.strftime("%b"))
    ax.set_ylabel("€ million per month")
    ax.set_title("Monthly revenue, costs & margin")
    ax.grid(axis="y", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=7.6, ncol=2)

    labels = [
        "Revenue",
        "Electricity",
        "Energy tax",
        "Distribution",
        "Fixed charges",
        "Maintenance",
        "Operating\nmargin",
    ]
    values = [
        totals["annual_revenue_eur"],
        -totals["electricity_cost_eur"],
        -totals["energy_tax_eur"],
        -totals["distribution_cost_eur"],
        -(totals["base_charge_eur"] + totals["power_charge_eur"]),
        -totals["maintenance_eur"],
    ]
    running = 0.0
    for index, value in enumerate(values):
        ax2.bar(
            index,
            value / 1e6,
            bottom=running / 1e6 if value < 0 else 0,
            color=TEAL if value > 0 else [BLUE, PURPLE, GOLD, CORAL, "#d99ab4"][index - 1],
            width=0.7,
        )
        running += value
        ax2.plot(
            [index + 0.35, index + 0.65],
            [running / 1e6, running / 1e6],
            color=MUTED,
            linewidth=1,
        )
    ax2.bar(
        len(values),
        running / 1e6,
        color=TEAL if running > 0 else CORAL,
        width=0.7,
    )
    for index, value in enumerate(values + [running]):
        y = (
            (sum(values[: index + 1]) if index < len(values) - 1 else running)
            if index < len(values) - 1
            else running
        )
        if index == 0:
            y = value
        ax2.text(
            index,
            (y / 1e6) + (1.0 if y >= 0 else -2.0),
            f"{value / 1e6:+.1f}" if index not in (0, len(values)) else f"{value / 1e6:.1f}",
            ha="center",
            va="bottom" if y >= 0 else "top",
            fontsize=8,
            color=TEXT,
        )
    ax2.axhline(0, color=MUTED, linewidth=0.8)
    ax2.set_xticks(np.arange(len(labels)), labels, rotation=18, ha="right")
    ax2.set_ylabel("€ million per year")
    ax2.set_title("Annual base-case operating waterfall")
    ax2.grid(axis="y", alpha=0.35)
    ax2.set_axisbelow(True)
    fig.suptitle(
        "Base-case economics: attractive margins depend on the modeled sale price and load match",
        fontsize=14,
        fontweight="bold",
        color=TEXT,
        y=1.02,
    )
    fig.text(
        0.01,
        -0.04,
        "Assumptions from Simulation.ipynb. Energy tax is applied per MWh of HP electricity; "
        f"power tariff uses the two highest monthly HP peaks. "
        f"CAPEX €{totals['capex_eur'] / 1e6:.0f}m; simple payback "
        f"{totals['simple_payback_years']:.1f} years.",
        color=MUTED,
        fontsize=8,
    )
    fig.tight_layout()
    finish_figure(fig, "03_base_case_economics.png")


def sensitivity_chart(data: pd.DataFrame, totals: dict[str, float]) -> None:
    sale_prices = np.arange(20, 121, 10)
    electricity_multipliers = np.arange(0.5, 1.51, 0.1)
    baseline_sale_price = DH_SALE_PRICE_EUR_MWH * (1 - VAT)
    baseline_variable_electricity = totals["electricity_cost_eur"]
    fixed_and_other_costs = (
        totals["energy_tax_eur"]
        + totals["distribution_cost_eur"]
        + totals["base_charge_eur"]
        + totals["power_charge_eur"]
        + totals["maintenance_eur"]
    )
    heat_mwh = totals["annual_heat_mwh"]
    project_value = np.array(
        [
            [
                (
                    5
                    * (
                        heat_mwh * price
                        - baseline_variable_electricity * multiplier
                        - fixed_and_other_costs
                    )
                    - totals["capex_eur"]
                )
                / 1e6
                for price in sale_prices
            ]
            for multiplier in electricity_multipliers
        ]
    )
    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    norm = TwoSlopeNorm(
        vmin=float(project_value.min()),
        vcenter=0,
        vmax=float(project_value.max()),
    )
    image = ax.imshow(
        project_value,
        aspect="auto",
        origin="lower",
        cmap="RdYlGn",
        norm=norm,
    )
    ax.set_xticks(np.arange(len(sale_prices)), [f"€{p}" for p in sale_prices])
    ax.set_yticks(
        np.arange(len(electricity_multipliers)),
        [f"{m:.1f}×" for m in electricity_multipliers],
    )
    ax.set_xlabel("Effective heat-sale price (€/MWh)")
    ax.set_ylabel("Spot-electricity price multiplier")
    ax.set_title("Five-year project value after modeled CAPEX (€m)")
    ax.grid(False)
    baseline_x = int(np.argmin(np.abs(sale_prices - baseline_sale_price)))
    baseline_y = int(np.argmin(np.abs(electricity_multipliers - 1.0)))
    ax.scatter(
        baseline_x,
        baseline_y,
        s=140,
        facecolors="none",
        edgecolors="#101923",
        linewidths=2.2,
        label=f"Base case ≈ €{baseline_sale_price:.0f}/MWh",
    )
    ax.legend(loc="upper left", frameon=True, facecolor=PANEL, edgecolor=GRID)
    colorbar = fig.colorbar(image, ax=ax, pad=0.025)
    colorbar.set_label("Five-year net value (€m)")
    colorbar.ax.yaxis.set_tick_params(color=MUTED)
    plt.setp(colorbar.ax.get_yticklabels(), color=MUTED)
    fig.text(
        0.01,
        0.01,
        "Sale price is the notebook's €80.06/MWh less its 25.5% adjustment. "
        "Other costs and modeled annual heat volume are held constant.",
        color=MUTED,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    finish_figure(fig, "04_five_year_sensitivity.png")


def duration_chart(data: pd.DataFrame) -> None:
    duration = data["modeled_heat_mw"].sort_values(ascending=False).to_numpy()
    hours = np.arange(1, len(duration) + 1) * INTERVAL_HOURS
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.plot(hours, duration, color=TEAL, linewidth=2.6)
    ax.fill_between(hours, duration, color=TEAL, alpha=0.16)
    ax.axhline(POWER_MW, color=GOLD, linestyle="--", linewidth=1.7)
    ax.set_xlim(0, hours[-1])
    ax.set_xlabel("Hours per year at or above this modeled heat output")
    ax.set_ylabel("Modeled heat delivery (MW)")
    ax.set_title("Annual heat-output duration curve")
    ax.grid(alpha=0.4)
    ax.text(
        0.99,
        0.95,
        f"Peak: {duration[0]:.0f} MW\n"
        f"Annual modeled heat: {duration.sum() * INTERVAL_HOURS / 1e6:.2f} TWh",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=MUTED,
    )
    fig.text(
        0.01,
        0.01,
        "Modeled heat is scaled directly from Helen's 2025 DH demand profile; it is not a measured AI data-centre waste-heat curve.",
        color=MUTED,
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    finish_figure(fig, "05_heat_duration_curve.png")


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    apply_style()
    data = load_data()
    monthly, totals = economics(data)

    heat_match_chart(data)
    heat_pump_chart(data)
    economics_chart(monthly, totals)
    sensitivity_chart(data, totals)
    duration_chart(data)

    assumptions = {
        "year": int(data["datetime"].dt.year.iloc[0]),
        "heat_sale_price_before_adjustment_eur_mwh": DH_SALE_PRICE_EUR_MWH,
        "heat_sale_price_after_notebook_adjustment_eur_mwh": DH_SALE_PRICE_EUR_MWH
        * (1 - VAT),
        "modeled_data_centre_heat_peak_mw": POWER_MW,
        "heat_pump_electric_rating_mw": HP_POWER_MW,
        "heat_pump_efficiency_fraction_of_carnot": HP_EFFICIENCY,
        **totals,
    }
    pd.Series(assumptions, name="value").to_csv(
        OUTPUT / "model_summary.csv", float_format="%.4f"
    )
    print(f"Charts and model summary saved to: {OUTPUT}")
    print(f"Annual operating margin: €{totals['annual_operating_margin_eur']:,.0f}")
    print(f"Simple payback: {totals['simple_payback_years']:.2f} years")
    print(f"Peak HP input: {totals['peak_hp_power_mw']:.1f} MW of {HP_POWER_MW:.0f} MW")
    print(
        "Note: the DH demand profile is used as the modelled heat-recovery profile; "
        "weather and demand datasets are from different Finnish cities."
    )


if __name__ == "__main__":
    main()
