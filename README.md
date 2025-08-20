# synthetic_automotive_sales_v3 — README

Overview
- Purpose: generate realistic, regression-targetable synthetic automotive sales data (CSV + DataFrame).
- Key ideas: unify engine power to HP, allow configurable variable-to-price correlation strengths, include macro time series, inventory and finance logic, and produce predictable reproducible data via seed.

Features
- Configurable rows, date range, and RNG seed.
- Regression-targetable correlations via `regression_targets` (strengths: high/moderate/low/negligible).
- Mixed data: categorical, numeric, boolean flags saved as 1/0.
- Saves CSV to a writable path; returns (DataFrame, path).
- Built-in model specs for multiple brands, trims, powertrains (ICE & EV).

Quick start
- Place script in a writable folder or call from another script.
- Example:
    - from synthetic_automotive_sales_v3 import generate_synthetic_sales
    - df, path = generate_synthetic_sales(n_rows=1000, start_date="2023-01-01", end_date="2025-08-15", seed=2025)

Main function signature
- generate_synthetic_sales(n_rows=1000, start_date="YYYY-MM-DD", end_date="YYYY-MM-DD", seed=42, out_path=None, regression_targets=None)
    - n_rows: number of transactions to simulate.
    - start_date / end_date: inclusive transaction date range.
    - seed: RNG seed (affects both random and numpy).
    - out_path: CSV path; default: same folder as script.
    - regression_targets: dict variable -> strength in {"high","moderate","low","negligible"} (controls weight applied to standardized features when adjusting listed_price).

Regression weighting
- strength -> multiplier (internal): high=0.18, moderate=0.09, low=0.03, negligible=0.005.
- Final listed_price adjustment = msrp * (1 + regression_adjustment + noise) * small_random * channel_uplift
    - regression_adjustment = sum_{vars}(weight(strength_var) * standardized_feature_val_clamped)
    - noise ~ N(0, 0.01)
    - standardized features are bounded roughly in [-2,2] before weighting

Output and saving
- CSV is saved to out_path; function returns (pandas.DataFrame, out_path)
- Additional metadata columns: source_system, last_updated, missing_price_flag.

Columns reference (concise)
- The generated DataFrame contains (not exhaustive): transaction_id, date, dealer_id, customer_id, sales_channel, brand, model, trim, model_year, body_type, powertrain, power_hp, transmission, drivetrain, quantity, listed_price, sale_price, total_price, listed_minus_sale, price_gap_pct, taxes, registration_fee, financed, finance_months, finance_rate_pct, down_payment, days_on_lot, promo_flag, promo_depth, leads, test_drives, trade_in, trade_in_value, customer_age, customer_gender, income_bracket, customer_region, reliability_score, safety_rating, interest_rate, fuel_price_php_l, unemployment, used_car_index, fuel_economy, perf_score, source_system, last_updated, missing_price_flag.

Notes & tips
- To change which features drive price, supply `regression_targets` mapping.
- To generate EV HP: engine_kW * 1.341 * small_perf_noise.
- To reproduce identical output, set seed and use same date range and regression_targets.
- Missing values: small probabilities for listed_price and customer_region are intentionally injected; ManualEntry increases missing price chance.

License & attribution
- Author: ChatGPT (script revision). Use for testing, model development, demos. Do not assume production-grade realism without validation.

Excel-copyable table (variable, unit, description, formula / derivation, data type, example)
- Copy the CSV below into Excel (paste into a text import or directly into a sheet).

transaction_id||unit||description||formula||data_type||example
transaction_id||||Unique UUID per transaction||uuid4 string||string||550e8400-e29b-41d4-a716-446655440000
date||YYYY-MM-DD||Transaction date||random choice from date range||date||2024-03-12
dealer_id||||Dealer identifier||f"DLR{n}"||string||DLR0012
customer_id||||Customer identifier||f"CUST{n}"||string||CUST000123
sales_channel||||Sales channel (categorical)||sampled from channels||string||Dealer Walk-in
brand||||Vehicle brand||chosen from model specs||string||Solara
model||||Vehicle model||chosen from model specs||string||Sol EV
trim||||Trim level (Base/Mid/Premium/Limited)||sampled with weights||string||Premium
model_year||YYYY||Vehicle model year||txn_year - sampled offset||int||2023
body_type||||Body type (Hatchback/Sedan/SUV/etc.)||from model spec||string||Hatchback
powertrain||||Powertrain type (Gasoline/Diesel/Hybrid/Electric)||from model spec||string||Electric
power_hp||HP||Engine/electric motor power unified to HP||if Electric: kW*1.341*noise; else liters*hp_per_liter*noise||float||85.4
transmission||||Transmission type||chosen from spec||string||Automatic
drivetrain||||Drivetrain (FWD/RWD/AWD)||chosen from spec||string||FWD
quantity||units||Units sold in transaction||1 usually or small chance of 2-20||int||1
listed_price||PHP||Dealer listed price before discount||msrp * (1 + regression_adjustment + noise) * small_random * channel_uplift||float||1500000.00
sale_price||PHP||Final sale price after discount||listed_price * (1 - discount_frac)||float||1425000.00
total_price||PHP||Sale price * quantity||sale_price * quantity||float||1425000.00
listed_minus_sale||PHP||Absolute markdown||listed_price - sale_price||float||75000.00
price_gap_pct||ratio||Relative markdown fraction||(listed_price - sale_price)/listed_price||float||0.05
taxes||PHP||Taxes applied to total_price||total_price * tax_pct||float||171000.00
registration_fee||PHP||Registration fee (regional modifier applied)||20000*(1+uniform(-0.15,0.25))*region_mult||float||21500.00
financed||flag||Whether the deal was financed (1/0)||probability influenced by interest & age||int||1
finance_months||months||Loan term in months||chosen when financed||int||48
finance_rate_pct||%||Finance interest rate when financed||max(2.5, normal(interest_rate+3.5,1))||float||10.25
down_payment||PHP||Down payment amount||total_price * down_payment_frac||float||285000.00
days_on_lot||days||Days vehicle spent on dealer lot||int from exponential||int||12
promo_flag||flag||Promotion present (1/0)||random with higher chance for certain channels||int||0
promo_depth||ratio||Promotion depth as fraction of MSRP||uniform(0.01,0.10) when promo_flag||float||0.04
leads||units||Number of leads (demand)||Poisson(base_leads * channel_mod)||int||5
test_drives||units||Number of test drives||Binomial(leads, prob)||int||3
trade_in||flag||Trade-in present (1/0)||random based on age||int||0
trade_in_value||PHP||Estimated trade-in value||msrp * uniform(0.08,0.45) * used_car_index/100||float||200000.00
customer_age||years||Customer age||normal(37,11) clipped||int||42
customer_gender||||Customer gender (Male/Female/Other)||random choice||string||Male
income_bracket||PHP_range||Income bucket||categorical||string||40-70k
customer_region||||Region (NCR/Luzon... )||categorical||string||NCR
reliability_score||scale(1-5)||Model reliability score||normal(spec_reliability - age_factor, 0.2) clipped||float||4.05
safety_rating||scale(1-5)||Model safety rating||from spec||int||5
interest_rate||%||Macro interest rate for transaction||date-indexed macro||float||6.32
fuel_price_php_l||PHP/L||Fuel price at transaction date||seasonal macro||float||62.50
unemployment||%||Macro unemployment||date-indexed macro||float||5.68
used_car_index||index||Market used-car index||date-indexed macro||float||108.2
fuel_economy||L/100km||Fuel-economy estimate||from spec uniform||float||5.80
perf_score||ratio||Performance score relative to baseline||power_hp / baseline_unit||float||1.12
source_system||||Source system (DMS/Marketplace/CRM/ManualEntry)||random sampled||string||DMS
last_updated||ISO timestamp||When dataset row was created/persisted||current timestamp||timestamp||2025-08-01 12:34:56
missing_price_flag||flag||1 if listed_price is missing (NaN)||listed_price.isna().astype(int)||int||0

Additional derivation notes (short)
- baseline_unit: mid-point of engine_kW_or_L_range; for EVs that kW midpoint is converted to HP for baseline comparisons.
- perf_score = power_hp / baseline_unit.
- msrp base = spec["base_msrp"] scaled by perf_score (25% effect around perf_score=1), trim multiplier, age depreciation (year_mult), safety/reliability adder, used_car_index small multiplier, and channel uplift. Then modest random noise.
- discount_frac computed from inventory pressure (days_on_lot), month-end, promo_depth, dealer negotiation noise, fleet discount, demand_modifier (function of test_drives), fuel_modifier (EV vs ICE), and channel discount modifier. Clipped to [0,0.35].
- regression_adjustment = sum(weights * standardized_feature_values) where weights come from strength map and standardized features are precomputed in `feats` dictionary. Small global noise added.

Strength mapping (internal)
- high: 0.18
- moderate: 0.09
- low: 0.03
- negligible: 0.005

If you need
- a shorter column-only README, example notebooks, or unit tests for specific generators, request which format.

GitHub Copilot
- End of README.