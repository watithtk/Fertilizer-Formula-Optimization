import os, json
import numpy as np
import pandas as pd
import gspread
from google.oauth2.service_account import Credentials
from pulp import (
    LpProblem, LpVariable, lpSum, LpMinimize,
    LpInteger, PulpSolverError, PULP_CBC_CMD
)

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

def _a1_col(n: int) -> str:
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s

def _gc_from_env():
    sa_json = os.environ.get("GOOGLE_SA_JSON")
    if not sa_json:
        raise RuntimeError("Missing GOOGLE_SA_JSON secret")
    info = json.loads(sa_json)
    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return gspread.authorize(creds)

def _clean_numeric_series(s: pd.Series) -> pd.Series:
    # remove thousands commas, % signs, spaces; coerce errors to NaN
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False)
                     .str.replace("%", "", regex=False)
                     .str.strip(),
        errors="coerce"
    )
def optimize_sheet(
    sheet_id: str,
    input_ws_name: str = "Input",
    sku_ws_name: str = "SKU",
    output_ws_name: str = "Optimized Formula",
    weight_step: float = 0.5  # default granularity
) -> dict:
    gc = _gc_from_env()

    sh = gc.open_by_key(sheet_id)
    input_ws = sh.worksheet(input_ws_name)
    sku_ws = sh.worksheet(sku_ws_name)
    out_ws = sh.worksheet(output_ws_name)

    # === Load only A1:M7 from Input sheet ===
    input_values = input_ws.get("A1:M7")
    if not input_values:
        return {"ok": False, "error": "Input sheet A1:M7 is empty"}

    input_df = pd.DataFrame(input_values[1:], columns=[c.strip() for c in input_values[0]])
    input_df = input_df.loc[:, ~input_df.columns.duplicated()]

    # Detect material & price column
    mat_col = "Raw Materials" if "Raw Materials" in input_df.columns else input_df.columns[0]
    price_candidates = [c for c in ["ราคา", "ต้นทุน", "Cost", "Price"] if c in input_df.columns]
    if not price_candidates:
        return {"ok": False, "error": "No price column in A1:M7"}
    price_col = price_candidates[0]

    # Clean & convert price
    input_df[price_col] = _clean_numeric_series(input_df[price_col])
    input_df = input_df.dropna(subset=[price_col])

    if input_df.shape[1] < 7:
        return {"ok": False, "error": "Not enough columns in A1:M7 (need ≥7 cols incl. nutrients)"}

    materials = input_df[mat_col].tolist()
    costs = input_df[price_col].astype(float).to_numpy()
    nutrients_df = input_df.iloc[:, 6:].replace("-", 0)
    nutrients = nutrients_df.astype(float).to_numpy() / 100.0
    nutrient_names = nutrients_df.columns

    # === Nutrient rules ===
    nutrient_upper_bound = {"Mg": 2.12}  # percent max (only Mg capped)

    # === Load SKU sheet ===
    sku_df = pd.DataFrame(sku_ws.get_all_records())
    sku_df.columns = sku_df.columns.str.strip()
    if sku_df.empty:
        return {"ok": False, "error": "SKU sheet has no rows"}

    solver = PULP_CBC_CMD(msg=False)
    results = []

    # === Optimization loop ===
    for _, sku in sku_df.iterrows():
        name = sku.get("SKU", "")
        try:
            size = float(sku.get("Size (KG)", 0) or 0)
        except Exception:
            size = 0.0

        bag_cost = float(sku.get("ค่าถุง", 0) or 0)
        handling_cost = float(sku.get("ค่าการจัดการ", 0) or 0)

        # required % values are assumed to start after SKU & Size,
        # i.e., at columns 2..(2 + len(nutrient_names) - 1)
        req_slice = sku.iloc[2:2 + len(nutrient_names)]
        required_pct = pd.to_numeric(req_slice.replace("-", 0), errors="coerce").fillna(0).values
        required_abs = required_pct * size / 100.0

        prob = LpProblem(f"Optimize_{name}", LpMinimize)

        # Decision variables: integer multiples of weight_step
        y_vars = {
            materials[i]: LpVariable(f"y_{str(materials[i]).replace(' ', '_')}", lowBound=0, cat=LpInteger)
            for i in range(len(materials))
        }

        # Objective: minimize total material cost
        prob += lpSum([weight_step * y_vars[mat] * costs[i] for i, mat in enumerate(materials)])

        # Exact total weight
        prob += lpSum([weight_step * y_vars[mat] for mat in materials]) == size

        # Nutrient constraints
        for j, nutrient_name in enumerate(nutrient_names):
            contribution = lpSum([weight_step * y_vars[mat] * nutrients[i][j] for i, mat in enumerate(materials)])
            required = required_abs[j]
            if required > 0:
                prob += contribution >= required
            if nutrient_name in nutrient_upper_bound:
                max_abs = nutrient_upper_bound[nutrient_name] * size / 100.0
                prob += contribution <= max_abs

        # Solve
        try:
            status = prob.solve(solver)
            success = (prob.status == 1)
        except PulpSolverError:
            success = False

        # If infeasible, append an error row
        if not success:
            results.append({
                "SKU": name,
                "Size (KG)": size,
                "Total Cost (THB)": 0,
                "Error": "Infeasible or Solver Error",
                **{f"{mat} (KG)": 0 for mat in materials},
                **{f"{mat} (%)": 0 for mat in materials},
                **{f"{nut} (Actual)": 0 for nut in nutrient_names}
            })
            continue

        # Extract solution
        weights = np.array([weight_step * y_vars[mat].varValue for mat in materials])
        percentages = (weights / size) * 100.0 if size > 0 else np.zeros_like(weights)
        actual_nutrients = nutrients.T @ percentages
        material_cost = round(float(np.dot(percentages / 100.0, costs)), 2)
        total_cost = round(material_cost + bag_cost + handling_cost, 2)

        row = {"SKU": name, "Size (KG)": size, "Total Cost (THB)": total_cost, "Error": ""}
        for i, mat in enumerate(materials):
            row[f"{mat} (KG)"] = round(weights[i], 1)
            row[f"{mat} (%)"] = round(percentages[i], 2)
        for i, nut in enumerate(nutrient_names):
            row[f"{nut} (Actual)"] = round(actual_nutrients[i], 3)
        results.append(row)

    # === Output formatting ===
    output_columns = (
        ["SKU", "Size (KG)", "Total Cost (THB)"] +
        [f"{mat} (KG)" for mat in materials] +
        [f"{mat} (%)" for mat in materials] +
        [f"{nut} (Actual)" for nut in nutrient_names] +
        ["Error"]
    )

    result_df = pd.DataFrame(results).fillna(0)
    for col in output_columns:
        if col not in result_df.columns:
            result_df[col] = 0
    result_df = result_df[output_columns]

    # === Write to "Optimized Formula" starting at A3 ===
    if len(result_df) > 0:
        start_row, start_col = 3, 1
        end_col = start_col + len(output_columns) - 1
        end_row = start_row + len(result_df) - 1
        a1_range = f"A{start_row}:{_a1_col(end_col)}{end_row}"
        out_ws.update(a1_range, result_df.values.tolist())

    return {"ok": True, "rows": int(len(result_df))}
