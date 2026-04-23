from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder=".")
CORS(app)

_report      = None
_sensitivity = None
_optimal     = None

def set_data(final_report, sensitivity_df, optimal_df):
    global _report, _sensitivity, _optimal
    _report      = final_report
    _sensitivity = sensitivity_df
    _optimal     = optimal_df

@app.route("/api/results")
def get_results():
    day_type = request.args.get("day_type", None)
    data = _report[[
        "item", "discount_hour", "day_type", "profit_gain", "is_profitable",
        "sold_rate", "waste_rate", "baseline_profit", "final_profit"
    ]].copy()
    if day_type:
        data = data[data["day_type"] == day_type]
    data["is_profitable"] = data["is_profitable"].astype(bool)
    return jsonify(data.to_dict(orient="records"))

@app.route("/api/sensitivity")
def get_sensitivity():
    day_type = request.args.get("day_type", None)
    data = _sensitivity[[
        "item", "discount_hour", "day_type", "discount_rate",
        "profit_gain", "is_profitable"
    ]].copy()
    if day_type:
        data = data[data["day_type"] == day_type]
    data["is_profitable"] = data["is_profitable"].astype(bool)
    return jsonify(data.to_dict(orient="records"))

@app.route("/api/optimal")
def get_optimal():
    day_type = request.args.get("day_type", None)
    data = _optimal.copy()
    if day_type:
        data = data[data["day_type"] == day_type]
    return jsonify(data.to_dict(orient="records"))

@app.route("/")
def index():
    return send_from_directory(".", "index.html")