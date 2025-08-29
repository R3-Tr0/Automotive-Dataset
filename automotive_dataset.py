# synthetic_automotive_sales_v4.py
# Synthetic automotive sales dataset generator v4
# - Prices bounded to [0, 20_000_000]
# - Data skewed toward ₱1,000,000 (stronger for gasoline sedans & middle-income)
# - Majority gasoline bias retained
# - Regression-targetable adjustments kept

import numpy as np
import pandas as pd
import uuid
import random
from datetime import datetime, timedelta
import os

def generate_synthetic_sales(n_rows=1000,
                             start_date="2023-01-01",
                             end_date="2025-07-31",
                             seed=42,
                             out_path=None,
                             regression_targets=None):
    # reproducible RNG
    random.seed(seed)
    np.random.seed(seed)

    # default regression targets strengths
    if regression_targets is None:
        regression_targets = {
            "power_hp": "high",
            "safety_rating": "moderate",
            "reliability_score": "moderate",
            "trim": "moderate",
            "test_drives": "high",
            "leads": "moderate",
            "promo_flag": "high",
            "days_on_lot": "moderate",
            "trade_in": "low",
            "income_bracket": "moderate",
            "customer_age": "low",
            "customer_region": "negligible",
            "customer_gender": "negligible",
            "fuel_price_php_l": "low",
            "interest_rate": "low",
            "used_car_index": "low",
            "sales_channel": "negligible",
            "source_system": "negligible",
            "transmission": "low",
            "drivetrain": "low",
            "body_type": "low",
            "quantity": "low"
        }

    if out_path is None:
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            script_dir = os.getcwd()
        out_path = os.path.join(script_dir, "synthetic_automotive_sales_v4.csv")

    # mapping of regression strength to numeric weight
    strength_weight = {"high": 0.18, "moderate": 0.09, "low": 0.03, "negligible": 0.005}

    # =========================================================
    # Model specs as before (kept mostly unchanged)
    # =========================================================
    model_specs = {
        "Aurion": {
            "A1":    {"body":"Hatchback", "power":"Gasoline",  "engine_kW_or_L_range":(1.0,1.4), "fuel_l_per_100km":(5.8,6.8),
                      "transmissions":["Manual","CVT","Automatic"], "drivetrains":["FWD"], "base_msrp":520000, "safety":4, "reliability":4.2},
            "A1 Sport":{"body":"Hatchback","power":"Gasoline","engine_kW_or_L_range":(1.4,1.8), "fuel_l_per_100km":(6.5,8.0),
                        "transmissions":["CVT","Automatic"], "drivetrains":["FWD"], "base_msrp":650000, "safety":4, "reliability":4.0},
            "A2":    {"body":"Sedan",    "power":"Hybrid",   "engine_kW_or_L_range":(1.5,2.0), "fuel_l_per_100km":(3.8,5.0),
                      "transmissions":["Automatic"], "drivetrains":["FWD"], "base_msrp":950000, "safety":5, "reliability":4.1},
            "A3 LX":{"body":"SUV",      "power":"Gasoline", "engine_kW_or_L_range":(1.8,2.4), "fuel_l_per_100km":(7.5,10.5),
                      "transmissions":["Automatic"], "drivetrains":["FWD","AWD"], "base_msrp":1350000, "safety":5, "reliability":4.0},
        },
        "Vexel": {
            "VX-compact":{"body":"Sedan","power":"Gasoline","engine_kW_or_L_range":(1.0,1.2),"fuel_l_per_100km":(5.5,6.5),
                          "transmissions":["Manual","CVT"], "drivetrains":["FWD"], "base_msrp":480000, "safety":3, "reliability":3.9},
            "VX-Plus":   {"body":"Sedan","power":"Diesel","engine_kW_or_L_range":(1.5,2.0),"fuel_l_per_100km":(5.0,6.5),
                          "transmissions":["Manual","Automatic"], "drivetrains":["FWD","RWD"], "base_msrp":720000, "safety":4, "reliability":4.0},
            "VX-Pro":    {"body":"Truck","power":"Diesel","engine_kW_or_L_range":(2.4,3.5),"fuel_l_per_100km":(8.0,12.0),
                          "transmissions":["Manual","Automatic"], "drivetrains":["RWD","AWD"], "base_msrp":1100000, "safety":4, "reliability":4.1},
        },
        "Solara": {
            "Sol 100":       {"body":"Coupe", "power":"Gasoline","engine_kW_or_L_range":(1.6,2.5),"fuel_l_per_100km":(7.0,10.0),
                              "transmissions":["Manual","Automatic"], "drivetrains":["RWD","FWD"], "base_msrp":820000, "safety":4, "reliability":4.0},
            "Sol 200 Touring":{"body":"SUV",  "power":"Hybrid", "engine_kW_or_L_range":(2.0,2.5),"fuel_l_per_100km":(4.0,6.0),
                              "transmissions":["Automatic"], "drivetrains":["FWD","AWD"], "base_msrp":1550000, "safety":5, "reliability":4.2},
            "Sol EV":        {"body":"Hatchback","power":"Electric","engine_kW_or_L_range":(45,85),"fuel_l_per_100km":(12,18),
                              "transmissions":["Automatic"], "drivetrains":["FWD"], "base_msrp":1800000, "safety":5, "reliability":4.3},
        },
        "Kestrel": {
            "K-Urban": {"body":"Hatchback","power":"Gasoline","engine_kW_or_L_range":(1.0,1.4),"fuel_l_per_100km":(5.5,7.0),
                        "transmissions":["Manual","CVT"], "drivetrains":["FWD"], "base_msrp":510000, "safety":3, "reliability":3.8},
            "K-SUV":   {"body":"SUV","power":"Diesel","engine_kW_or_L_range":(2.0,3.0),"fuel_l_per_100km":(7.5,10.0),
                        "transmissions":["Automatic"], "drivetrains":["AWD","FWD"], "base_msrp":1400000, "safety":5, "reliability":4.0},
            "K-Coupe": {"body":"Coupe","power":"Gasoline","engine_kW_or_L_range":(1.8,3.2),"fuel_l_per_100km":(8.0,12.0),
                        "transmissions":["Manual","Automatic"], "drivetrains":["RWD"], "base_msrp":1250000, "safety":4, "reliability":3.9},
        },
        "Orion": {
            "O-Ranger":{"body":"Truck","power":"Diesel","engine_kW_or_L_range":(2.5,4.2),"fuel_l_per_100km":(9.0,14.0),
                        "transmissions":["Manual","Automatic"], "drivetrains":["RWD","AWD"], "base_msrp":1300000, "safety":4, "reliability":4.1},
            "O-Luxe":  {"body":"Sedan","power":"Hybrid","engine_kW_or_L_range":(1.6,2.2),"fuel_l_per_100km":(3.5,5.5),
                        "transmissions":["Automatic"], "drivetrains":["FWD"], "base_msrp":1250000, "safety":5, "reliability":4.2},
            "O-Fleet": {"body":"Sedan","power":"Gasoline","engine_kW_or_L_range":(1.2,1.6),"fuel_l_per_100km":(6.0,7.5),
                        "transmissions":["CVT","Automatic"], "drivetrains":["FWD"], "base_msrp":600000, "safety":3, "reliability":3.9},
        }
    }

    # static enums
    channels = ["Dealer Walk-in", "Online Lead", "Marketplace", "Fleet/Government", "Referral"]
    regions = ["NCR", "Luzon_North", "Luzon_South", "Visayas", "Mindanao"]
    genders = ["Male", "Female", "Other"]
    income_brackets = ["<20k", "20-40k", "40-70k", "70-120k", ">120k"]
    income_weights = [0.05, 0.40, 0.35, 0.15, 0.05]  # skewed toward lower-middle & middle

    # gasoline bias
    power_weights = {"Gasoline": 7.0, "Hybrid": 2.0, "Diesel": 1.5, "Electric": 1.0}

    # build model catalog and adjusted base MSRPs (nudged as before)
    model_catalog = []
    for brand, models in model_specs.items():
        for model_name, spec in models.items():
            power = spec["power"]
            body = spec["body"]
            raw_base = float(spec.get("base_msrp", 800000))
            adj_base = raw_base
            if power == "Gasoline":
                if body == "Sedan":
                    adj_base = max(700_000, int(np.random.normal(1_020_000, 120_000)))
                elif body in ("Hatchback", "Coupe"):
                    adj_base = max(600_000, int(raw_base * np.random.uniform(0.95, 1.25)))
                elif body in ("SUV", "Truck"):
                    adj_base = max(1_100_000, int(raw_base * np.random.uniform(1.0, 1.4)))
            elif power == "Hybrid":
                adj_base = int(max(raw_base * 1.05, np.random.normal(1_200_000, 150_000)))
            elif power == "Electric":
                adj_base = int(max(raw_base * 1.1, np.random.normal(1_800_000, 300_000)))
            elif power == "Diesel":
                adj_base = int(raw_base * np.random.uniform(0.95, 1.2))
            adj_base = int(max(200_000, adj_base))
            model_catalog.append({
                "brand": brand,
                "model": model_name,
                "spec": spec,
                "adj_base_msrp": adj_base,
                "power": power,
                "body": body,
                "weight": power_weights.get(power, 1.0)
            })

    weights = np.array([m["weight"] for m in model_catalog], dtype=float)
    weights = weights / weights.sum()

    # dealers / customers
    n_dealers = 120
    n_customers = max(2000, int(n_rows * 0.5))
    dealers = [f"DLR{str(i).zfill(4)}" for i in range(1, n_dealers+1)]
    customers = [f"CUST{str(i).zfill(6)}" for i in range(1, n_customers+1)]

    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    days = (end - start).days + 1
    date_pool = [start + timedelta(days=i) for i in range(days)]

    # macro series (unchanged)
    macro_df = []
    base_interest = 6.0
    base_fuel = 60.0
    base_unemp = 5.5
    base_used_index = 100.0
    for d in date_pool:
        month_frac = (d - start).days / max(1, days)
        interest = base_interest + 0.5*np.sin(2*np.pi*(d.timetuple().tm_yday)/365) + 0.3*(month_frac-0.5)
        fuel = base_fuel + 6*np.sin(2*np.pi*(d.timetuple().tm_yday)/365) + 3*(0.5-month_frac)
        unemp = base_unemp + 0.4*np.cos(2*np.pi*(d.timetuple().tm_yday)/365) + 0.2*(month_frac-0.5)
        used_idx = base_used_index + 10*np.sin(2*np.pi*(d.timetuple().tm_yday)/180) - 5*(month_frac-0.5)
        macro_df.append({
            "date": d.date(),
            "interest_rate": round(interest, 2),
            "fuel_price_php_l": round(max(fuel, 30), 2),
            "unemployment": round(max(unemp,1), 2),
            "used_car_index": round(max(used_idx, 50), 2)
        })
    macro_df = pd.DataFrame(macro_df)
    macro_dict = macro_df.set_index('date').to_dict('index')

    # Price bounds and target center for skew
    PRICE_MIN = 0.0
    PRICE_MAX = 20_000_000.0
    TARGET_CENTER = 1_000_000.0  # center of skew

    rows = []
    for i in range(n_rows):
        tx_id = str(uuid.uuid4())
        txn_date = random.choice(date_pool)
        txn_date_only = txn_date.date()
        dealer = random.choice(dealers)
        customer = random.choice(customers)

        sales_channel = random.choices(channels, weights=[0.45,0.18,0.18,0.08,0.11], k=1)[0]

        quantity = 1
        if random.random() < 0.005:
            quantity = random.randint(2, 20)

        chosen_idx = np.random.choice(len(model_catalog), p=weights)
        entry = model_catalog[int(chosen_idx)]
        brand = entry["brand"]
        model = entry["model"]
        spec = entry["spec"]
        power = entry["power"]
        body = entry["body"]
        base_msrp = float(entry["adj_base_msrp"])

        # model year & age
        model_year = txn_date.year - np.random.choice([0,0,0,1,1,2,3,4,5],
                                                     p=[0.4,0.2,0.1,0.1,0.07,0.05,0.04,0.03,0.01])
        age = max(0, txn_date.year - model_year)

        # power_hp and fuel_economy
        es_min, es_max = spec["engine_kW_or_L_range"]
        if power == "Electric":
            kw = np.random.uniform(es_min, es_max)
            perf_mult = np.random.normal(1.0, 0.03)
            power_hp = round(kw * 1.341 * perf_mult, 1)
            fuel_economy = round(np.random.uniform(spec["fuel_l_per_100km"][0], spec["fuel_l_per_100km"][1]), 2)
        else:
            liters = np.random.uniform(es_min, es_max)
            hp_per_liter = np.random.uniform(65, 105)
            power_hp = round(liters * hp_per_liter * np.random.normal(1.0, 0.04), 1)
            fuel_economy = round(np.random.uniform(spec["fuel_l_per_100km"][0], spec["fuel_l_per_100km"][1]), 2)

        transmission = random.choices(spec["transmissions"], k=1)[0]
        drivetrain = random.choice(spec["drivetrains"])

        # Trim and year multipliers
        trims = ["Base", "Mid", "Premium", "Limited"]
        trim = random.choices(trims, weights=[0.45,0.30,0.20,0.05])[0]
        trim_mult = {"Base":0.95,"Mid":1.05,"Premium":1.2,"Limited":1.4}[trim]
        year_mult = max(0.6, 1.0 - 0.06 * age)

        baseline_unit = ((es_min + es_max) / 2.0 if power != "Electric" else (((es_min + es_max) / 2.0) * 1.341))
        perf_score = (power_hp / max(1, baseline_unit))
        perf_score = float(perf_score)

        channel_uplift = {"Dealer Walk-in":1.00, "Online Lead":1.02, "Marketplace":1.03, "Fleet/Government":0.96, "Referral":0.99}[sales_channel]

        # initial MSRP built from adj_base_msrp and features
        msrp = base_msrp * (1 + 0.25 * (perf_score - 1)) * trim_mult * year_mult
        msrp *= (1 + 0.03 * (spec["safety"] - 3) + 0.02 * (spec["reliability"] - 3.8))
        macro = macro_dict[txn_date_only]
        msrp *= (1 + 0.001 * (macro["used_car_index"] - 100))
        msrp = round(msrp * np.random.uniform(0.98, 1.06) * channel_uplift, 2)

        # income influence
        income = random.choices(income_brackets, weights=income_weights, k=1)[0]
        income_mult = {"<20k":0.85, "20-40k":0.95, "40-70k":1.00, "70-120k":1.08, ">120k":1.15}[income]
        msrp *= income_mult

        # nudge gasoline cars toward 7-digit center (kept as before)
        if power == "Gasoline":
            if msrp < 800_000:
                msrp *= np.random.uniform(1.08, 1.25)
            msrp = round(msrp * np.random.uniform(0.98, 1.06), 2)

        if body in ("Coupe",):
            msrp = round(msrp * np.random.uniform(1.05, 1.25), 2)
        if power == "Electric":
            msrp = round(msrp * np.random.uniform(1.02, 1.20), 2)

        # Inventory & discount logic (unchanged)
        days_on_lot = int(np.random.exponential(scale=20))
        month_end = txn_date.day > 24
        inventory_pressure = min(1.0, days_on_lot / 90)
        fleet_discount = 0.0
        if quantity > 1 or sales_channel == "Fleet/Government":
            fleet_discount = np.random.uniform(0.02, 0.12)
        promo_flag = int(random.random() < (0.18 if sales_channel != "Fleet/Government" else 0.25))
        promo_depth = np.random.uniform(0.01, 0.12) if promo_flag else 0.0
        dealer_neg = np.random.normal(loc=0.0, scale=0.02)

        # demand signals (unchanged)
        base_leads = 3 + 0.8 * (trim_mult - 1) + 0.002 * (power_hp) - 0.000001 * msrp
        channel_lead_mod = {"Dealer Walk-in":1.0, "Online Lead":1.3, "Marketplace":1.2, "Fleet/Government":2.0, "Referral":1.1}[sales_channel]
        leads = max(1, int(np.random.poisson(max(0.5, base_leads * channel_lead_mod))))
        gender_effect = {"Male":0.05, "Female":-0.02, "Other":0.0}
        cust_age = int(np.clip(np.random.normal(37, 11), 18, 75))
        gender = random.choice(genders)
        customer_location = random.choice(regions)

        region_test_mod = {"NCR":1.15, "Luzon_North":1.0, "Luzon_South":0.95, "Visayas":0.9, "Mindanao":0.85}[customer_location]
        age_test_mod = max(0.6, 1.2 - (cust_age - 25)/100.0)
        test_drive_prob = min(0.95, 0.40 + 0.06*(trim_mult-1) + 0.001*(power_hp/100) + gender_effect.get(gender, 0.0))
        test_drives = min(leads, int(np.random.binomial(leads, max(0.05, test_drive_prob * region_test_mod * age_test_mod))))

        # discount fraction (unchanged)
        base_disc = 0.02 + 0.08*inventory_pressure + (0.03 if month_end else 0.0) + promo_depth + dealer_neg + fleet_discount
        demand_modifier = max(0.0, 1.0 - 0.02 * test_drives)
        fuel_price = macro["fuel_price_php_l"]
        if power == "Electric":
            fuel_modifier = max(0.7, 1.0 - 0.008*(fuel_price - 60))
        else:
            fuel_modifier = 1.0 + 0.004*(fuel_price - 60)
        channel_disc_mod = {"Dealer Walk-in":1.0, "Online Lead":0.95, "Marketplace":0.9, "Fleet/Government":0.85, "Referral":0.98}[sales_channel]
        discount_frac = max(0, min(0.45, base_disc * demand_modifier * fuel_modifier * channel_disc_mod))

        # base listed price before regression adjustment
        listed_price = round(msrp * np.random.uniform(0.99, 1.03), 2)

        # Financing, trade-in, reliability, safety (unchanged)
        tax_pct = np.random.choice([0.12, 0.12, 0.12, 0.10, 0.14])

        region_reg_mult = {"NCR":1.08, "Luzon_North":1.00, "Luzon_South":0.98, "Visayas":0.99, "Mindanao":0.97}[customer_location]
        registration_fee = round(20000 * (1 + np.random.uniform(-0.15, 0.25)) * region_reg_mult, 2)

        # Financing
        interest_rate = macro["interest_rate"]
        base_finance_prob = 0.55 - 0.02 * (interest_rate - 6.0)
        age_finance_mod = max(0.25, 1.0 - (cust_age - 30)/100.0)
        income_fin_mod = {"<20k":1.05, "20-40k":1.0, "40-70k":0.95, "70-120k":0.9, ">120k":0.8}[income]
        financed = (random.random() < (base_finance_prob * age_finance_mod * income_fin_mod)) if False else (random.random() < (base_finance_prob * age_finance_mod * income_fin_mod))  # keep logic consistent
        # The above line intentionally preserves variable names used earlier (age_fin_mod) and won't raise in normal run.
        if financed:
            months = random.choices([12,24,36,48,60,72,84], weights=[5,10,20,25,25,10,5], k=1)[0]
            rate = max(2.5, np.random.normal(loc=interest_rate + 3.5, scale=1.0))
            income_idx = ["<20k","20-40k","40-70k","70-120k",">120k"].index(income)
            loc = 0.20 + (cust_age - 30) / 400.0 + (income_idx - 2) * 0.02
            down_payment_frac = float(np.random.normal(loc=loc, scale=0.05))
            down_payment_frac = float(np.clip(down_payment_frac, 0.05, 0.6))
            down_payment = round((listed_price * (1 - discount_frac) * quantity) * down_payment_frac, 2)
        else:
            months = 0
            rate = 0.0
            down_payment = round((listed_price * (1 - discount_frac) * quantity) * np.random.uniform(0.0, 0.25), 2)

        trade_in = int(random.random() < (0.18 + max(0.0, (cust_age-40)/400.0)))
        trade_in_value = 0.0
        if trade_in:
            trade_in_value = round(msrp * np.random.uniform(0.08, 0.45) * (macro["used_car_index"]/100.0), 2)

        reliability_score = round(np.clip(np.random.normal(spec["reliability"] - 0.08*age/2, 0.2), 1.5, 5.0), 2)
        safety_rating = spec["safety"]

        # occasional missing values
        if random.random() < 0.002:
            listed_price = np.nan
        if random.random() < 0.002:
            customer_location = None

        # ---------- Regression-targetable adjustment ----------
        def safe_div(a, b):
            return (a / b) if b != 0 else 0.0

        feats = {}
        feats["power_hp"] = safe_div((power_hp - baseline_unit), max(1.0, baseline_unit))
        feats["safety_rating"] = (safety_rating - 3) / 2.0
        feats["reliability_score"] = safe_div((reliability_score - 3.8), 1.2)
        trim_numeric = {"Base":0, "Mid":1, "Premium":2, "Limited":3}[trim]
        feats["trim"] = safe_div((trim_numeric - 1.0), 1.5)
        feats["days_on_lot"] = -inventory_pressure
        feats["promo_flag"] = -float(promo_flag)
        feats["promo_depth"] = -promo_depth
        feats["leads"] = safe_div((leads - 3.0), 6.0)
        feats["test_drives"] = safe_div((test_drives - 1.0), max(1.0, leads))
        feats["trade_in"] = float(trade_in) * 0.5
        feats["trade_in_value"] = safe_div(trade_in_value, max(1.0, msrp))
        feats["customer_age"] = safe_div((cust_age - 37.0), 25.0)
        income_ord = {"<20k":0, "20-40k":1, "40-70k":2, "70-120k":3, ">120k":4}[income]
        feats["income_bracket"] = safe_div((income_ord - 2.0), 2.0)
        feats["customer_region"] = {"NCR":0.5,"Luzon_North":0.0,"Luzon_South":-0.1,"Visayas":-0.2,"Mindanao":-0.3}.get(customer_location or "Luzon_North", 0.0)
        feats["customer_gender"] = {"Male":0.05, "Female":-0.02, "Other":0.0}[gender]
        feats["fuel_price_php_l"] = safe_div((fuel_price - 60.0), 30.0)
        feats["interest_rate"] = safe_div((interest_rate - 6.0), 4.0)
        feats["used_car_index"] = safe_div((macro["used_car_index"] - 100.0), 40.0)
        feats["sales_channel"] = {"Dealer Walk-in":0.0, "Online Lead":0.02, "Marketplace":-0.02, "Fleet/Government":-0.04, "Referral":0.01}[sales_channel]
        feats["source_system"] = {"DMS":0.0, "Marketplace":0.01, "CRM":0.0, "ManualEntry":-0.02}["DMS"]
        feats["transmission"] = {"Manual":-0.01, "CVT":0.0, "Automatic":0.02}.get(transmission, 0.0)
        feats["drivetrain"] = {"FWD":0.0, "RWD":0.01, "AWD":0.02}.get(drivetrain, 0.0)
        feats["body_type"] = {"Hatchback":0.0, "Sedan":0.01, "SUV":0.03, "Truck":-0.01, "Coupe":0.02}.get(body, 0.0)
        feats["quantity"] = safe_div((quantity - 1.0), 5.0)

        regression_adjustment = 0.0
        for var, strength in regression_targets.items():
            w = strength_weight.get(strength, 0.0)
            val = feats.get(var, None)
            if val is None:
                val = safe_div((hash((tx_id, var)) % 100 - 50), 100.0)
            val = max(-2.0, min(2.0, float(val)))
            regression_adjustment += w * val

        noise = np.random.normal(loc=0.0, scale=0.01)

        # apply regression adjustment to listed_price (if not missing)
        if not pd.isna(listed_price):
            listed_price = round(msrp * (1.0 + regression_adjustment + noise) * np.random.uniform(0.995, 1.005) * channel_uplift, 2)
        else:
            listed_price = np.nan

        # ---------------------------
        # NEW: Pull-to-1M skew + enforce bounds [0, 20M]
        # - stronger pull for gasoline sedans and middle-income buyers
        # - then clamp strictly between PRICE_MIN and PRICE_MAX
        # ---------------------------
        if not pd.isna(listed_price):
            # base pull strength (by power & body)
            if power == "Gasoline" and body == "Sedan":
                base_pull = 0.35
            elif power == "Gasoline":
                base_pull = 0.18
            elif power == "Hybrid":
                base_pull = 0.10
            elif power == "Diesel":
                base_pull = 0.08
            else:  # Electric & others
                base_pull = 0.03

            # increase pull if income is middle-class (skew toward 1M)
            income_pull_map = {"<20k":0.12, "20-40k":0.22, "40-70k":0.35, "70-120k":0.15, ">120k":0.06}
            income_boost = income_pull_map.get(income, 0.15)

            # final pull strength capped to avoid overpowering base logic
            pull_strength = min(0.85, base_pull + 0.5 * income_boost)

            # sample a modest target around TARGET_CENTER to avoid deterministic center
            target_sample = float(np.random.normal(TARGET_CENTER, 200_000))
            # combine using weighted average (keeps distribution shape but pulls mass toward 1M)
            listed_price = float(round(listed_price * (1.0 - pull_strength) + target_sample * pull_strength, 2))

            # enforce hard bounds
            listed_price = float(np.clip(listed_price, PRICE_MIN, PRICE_MAX))
        # ---------------------------

        # recompute sale/total/taxes using clipped listed_price
        sale_price = round(listed_price * (1 - discount_frac), 2) if not pd.isna(listed_price) else np.nan
        # ensure sale_price within bounds as well
        if not pd.isna(sale_price):
            sale_price = float(np.clip(sale_price, PRICE_MIN, PRICE_MAX))
        total_price = round(sale_price * quantity, 2) if not pd.isna(sale_price) else np.nan
        if not pd.isna(total_price):
            total_price = float(np.clip(total_price, PRICE_MIN, PRICE_MAX))

        taxes = round(total_price * tax_pct, 2) if not pd.isna(total_price) else np.nan

        # Compose row
        row = {
            "transaction_id": tx_id,
            "date": txn_date_only,
            "dealer_id": dealer,
            "customer_id": customer,
            "sales_channel": sales_channel,
            "quantity": quantity,
            "brand": brand,
            "model": model,
            "trim": trim,
            "model_year": model_year,
            "body_type": body,
            "powertrain": power,
            "power_hp": power_hp,
            "transmission": transmission,
            "drivetrain": drivetrain,
            "listed_price": listed_price,
            "sale_price": sale_price,
            "total_price": total_price,
            "listed_minus_sale": round(listed_price - sale_price, 2) if (not pd.isna(listed_price) and not pd.isna(sale_price)) else np.nan,
            "price_gap_pct": round(((listed_price - sale_price)/listed_price) if (not pd.isna(listed_price) and listed_price>0 and not pd.isna(sale_price)) else 0, 4),
            "taxes": taxes,
            "registration_fee": registration_fee,
            "financed": int(bool(financed)),
            "finance_months": months,
            "finance_rate_pct": round(rate,2),
            "down_payment": down_payment,
            "days_on_lot": days_on_lot,
            "promo_flag": int(bool(promo_flag)),
            "promo_depth": round(promo_depth,3),
            "leads": leads,
            "test_drives": test_drives,
            "trade_in": int(bool(trade_in)),
            "trade_in_value": trade_in_value,
            "customer_age": cust_age,
            "customer_gender": gender,
            "income_bracket": income,
            "customer_region": customer_location,
            "reliability_score": reliability_score,
            "safety_rating": safety_rating,
            "interest_rate": interest_rate,
            "fuel_price_php_l": fuel_price,
            "unemployment": macro["unemployment"],
            "used_car_index": macro["used_car_index"],
            "fuel_economy": fuel_economy,
            "perf_score": round(perf_score, 3)
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Order columns (unchanged)
    cols = ["transaction_id","date","dealer_id","customer_id","sales_channel","brand","model","trim","model_year","body_type","powertrain",
            "power_hp","transmission","drivetrain","quantity","listed_price","sale_price","total_price","listed_minus_sale",
            "price_gap_pct","taxes","registration_fee","financed","finance_months","finance_rate_pct","down_payment",
            "days_on_lot","promo_flag","promo_depth","leads","test_drives","trade_in","trade_in_value","customer_age",
            "customer_gender","income_bracket","customer_region","reliability_score","safety_rating","interest_rate",
            "fuel_price_php_l","unemployment","used_car_index","fuel_economy","perf_score"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]

    # Source system metadata
    df["source_system"] = np.random.choice(["DMS","Marketplace","CRM","ManualEntry"], size=len(df), p=[0.5,0.25,0.18,0.07])
    df.loc[(df["source_system"]=="ManualEntry") & (np.random.rand(len(df)) < 0.03), "listed_price"] = np.nan

    df["last_updated"] = pd.Timestamp(datetime.now())
    df["missing_price_flag"] = df["listed_price"].isna().astype(int)

    # Ensure output directory exists
    out_dir = os.path.dirname(out_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Save CSV
    df.to_csv(out_path, index=False)

    return df, out_path

# ----------------------------
if __name__ == "__main__":
    n = 1000
    custom_targets = {
        "power_hp": "high",
        "safety_rating": "moderate",
        "reliability_score": "moderate",
        "trim": "moderate",
        "promo_flag": "high",
        "days_on_lot": "moderate",
        "test_drives": "high",
        "leads": "moderate",
        "income_bracket": "moderate",
    }
    df, saved_path = generate_synthetic_sales(
        n_rows=n,
        start_date="2023-01-01",
        end_date="2025-08-15",
        seed=2025,
        regression_targets=custom_targets
    )

    sample = df.sample(10, random_state=1).reset_index(drop=True)
    print("Sample 10 rows:")
    print(sample.to_string(index=False))

    num_stats = df.select_dtypes(include=[np.number]).describe().transpose().round(3)
    print("\nNumeric summary (top rows):")
    print(num_stats.head().to_string())

    print(f"\nSaved CSV to: {saved_path}")
