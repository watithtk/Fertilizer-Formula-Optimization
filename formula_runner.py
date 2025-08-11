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

def optimize_sheet(sheet_id: str,
                   input_ws_name: str = "Input",
                   sku_ws_name: str = "SKU",
                   output_ws_name: str = "Optimized Formula") -> dict:
    gc = _gc_from_env()

    sh = gc.open_by_key(sheet_id)
    input_ws = sh.worksheet(input_ws_name)
    sku_ws = sh.worksheet(sku_ws_name)
    out_ws = sh.worksheet(output_ws_name)

    raw_df = pd.DataFrame(input_ws.get_all_records())
    raw_df.columns = raw_df.columns.str.strip()
    raw_df = raw_df.dropna(subset=["ราคา"])

    materials = raw_df["Raw Materials"].tolist()
    costs = raw_df["ราคา"].astype(float).to_numpy()
    nutrients = raw_df.iloc[:, 6:].replace("-", 0).astype(float).to_numpy() / 100.0
    nutrient_names = raw_df.columns[6:]
    nutrient_upper_bound = {"Mg": 2.12}  # percent max

    sku_df = pd.DataFrame(sku_ws.get_all_records())
    sku_df.columns = sku_df.columns.str.strip()

    solver = PULP_CBC_CMD(msg=False)
    results = []

    for _, sku in sku_df.iterrows():
        name = sku["SKU"]
        size = float(sku["Size (KG)"])
        bag_cost = float(sku.get("ค่าถุง", 0) or 0)
        handling_cost = float(sku.get("ค่าการจัดการ", 0) or 0)

        req_pct = sku.iloc[2:2 + len(nutrient_names)].replace("-", 0).astype(float).values
        req_abs = req_pct * size / 100.0

        prob = LpProblem(f"Optimize_{name}", LpMinimize)
        y_vars = {
            materials[i]: LpVariable(f"y_{materials[i].replace(' ', '_')}", lowBound=0, cat=LpInteger)
            for i in range(len(materials))
        }

        prob += lpSum([0.5 * y_vars[mat] * costs[i] for i, mat in enumerate(materials)])  # minimize cost
        prob += lpSum([0.5 * y_vars[mat] for mat in materials]) == size                  # exact weight

        for j, nutrient_name in enumerate(nutrient_names):
            contribution = lpSum([0.5 * y_vars[mat] * nutrients[i][j] for i, mat in enumerate(materials)])
            required = req_abs[j]
            if required > 0:
                prob += contribution >= required
            if nutrient_name in nutrient_upper_bound:
                max_abs = nutrient_upper_bound[nutrient_name] * size / 100.0
                prob += contribution <= max_abs

        try:
            status = prob.solve(solver)
            success = (prob.status == 1)
        except PulpSolverError:
            success = False

        if not success:
            results.append({
                "SKU": name,
                "Size (KG)": size,
                "Total Cost (THB)": 0,
                "Error": "Infeasible or Solver Error",
                **{f"{mat} (KG)": 0 for mat in materials},
                **{f"{mat} (%)": 0 for mat in materials},
                **{f"{nut} (Actual)" : 0 for nut in nutrient_names}
            })
            continue

        weights = np.array([0.5 * y_vars[mat].varValue for mat in materials])
        percentages = (weights / size) * 100.0
        actual_nutrients = nutrients.T @ percentages
        material_cost = round(np.dot(percentages / 100.0, costs), 2)
        total_cost = round(material_cost + bag_cost + handling_cost, 2)

        row = {"SKU": name, "Size (KG)": size, "Total Cost (THB)": total_cost, "Error": ""}
        for i, mat in enumerate(materials):
            row[f"{mat} (KG)"] = round(weights[i], 1)
            row[f"{mat} (%)"] = round(percentages[i], 2)
        for i, nut in enumerate(nutrient_names):
            row[f"{nut} (Actual)"] = round(actual_nutrients[i], 3)
        results.append(row)

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

    # A3: dynamic range
    start_row, start_col = 3, 1
    end_col = start_col + len(output_columns) - 1
    end_row = start_row + len(result_df) - 1
    a1_range = f"A{start_row}:{_a1_col(end_col)}{end_row}"

    if len(result_df) > 0:
        out_ws.update(a1_range, result_df.values.tolist())

    return {"ok": True, "rows": len(result_df)}
