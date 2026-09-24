import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from medical_engine import analyze_medical_text
from medical_event_engine_v2 import build_medical_event
from medical_timeline import build_timeline
from intent_classifier import predict_intent


app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return send_from_directory("frontend", "index.html")
# ============================================================
# INTENT RESULT NORMALIZER
# ============================================================

def get_intent_result(text):

    result = predict_intent(text)

    # If classifier returns a dictionary
    if isinstance(result, dict):

        name = (
            result.get("intent")
            or result.get("name")
            or result.get("label")
            or "unknown"
        )

        confidence = result.get(
            "confidence",
            0
        )

        return {
            "name": name,
            "confidence": float(confidence)
        }


    # If classifier returns:
    # ("medical_event", 0.97)
    if isinstance(result, (tuple, list)):

        if len(result) >= 2:

            return {
                "name": str(result[0]),
                "confidence": float(result[1])
            }

        if len(result) == 1:

            return {
                "name": str(result[0]),
                "confidence": 0
            }


    # If classifier returns only a string
    if isinstance(result, str):

        return {
            "name": result,
            "confidence": 0
        }


    return {
        "name": "unknown",
        "confidence": 0
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "service": "Medical SLM",
        "version": "1.0"
    })


# ============================================================
# MEDICAL ANALYSIS
# ============================================================

@app.route("/analyze", methods=["POST"])
def analyze():

    data = request.get_json(silent=True) or {}

    text = data.get(
        "text",
        ""
    ).strip()


    if not text:

        return jsonify({
            "error": "Text is required"
        }), 400


    try:

        # ----------------------------------------------------
        # 1. Core medical analysis
        # ----------------------------------------------------

        analysis = analyze_medical_text(text)


        # ----------------------------------------------------
        # 2. Medical event understanding
        # ----------------------------------------------------

        event = build_medical_event(text)


        # ----------------------------------------------------
        # 3. Intent classification
        # ----------------------------------------------------

        intent = get_intent_result(text)


        # ----------------------------------------------------
        # 4. Timeline
        # ----------------------------------------------------

        timeline = build_timeline(text)


        # ----------------------------------------------------
        # 5. Unified response
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "input": text,

            "corrected_text":
                analysis.get(
                    "corrected_text",
                    text
                ),

            "intent": intent,

            "subject":
                event.get(
                    "subject"
                ),

            "medical_entities":
                event.get(
                    "medical_entities",
                    []
                ),

            "relationships":
                event.get(
                    "relationships",
                    []
                ),

            "medical_information":
                analysis.get(
                    "medical_information",
                    []
                ),

            "time":
                event.get(
                    "time"
                ),

            "timeline":
                timeline

        })


    except Exception as error:

        print(
            "\nMEDICAL SLM ERROR:",
            error
        )

        return jsonify({

            "success": False,

            "error": str(error)

        }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )