"""
CropSense AI — Flask USSD Backend
===================================
Receives POST requests from Africa's Talking USSD gateway.
Manages session state via the 'text' field (cumulative user inputs).
Responds with CON (continue session) or END (close session) strings.

Deploy free on Render.com or Railway.app.
Africa's Talking will POST to: https://your-app.onrender.com/ussd

Session flow:
  Step 0 : Language select
  Step 1 : Main menu
  Step 2 : District entry
  Step 3 : Crop select
  Step 4 : Area entry
  Step 5 : Purpose select
  Step 6 : Planting date entry
  Step 7 : Advisory output (END)

Parallel flows for menu options 2, 3, 4, 5.
Option 5 is new: "What should I plant?" — ranks all crops for the
farmer's district instead of requiring them to already know which
crop to ask about.

CHANGES FROM V1:
  - District input is now validated (unknown district -> re-prompt,
    not a silent fallback to Mbarara buried three steps later).
  - Crop selection re-prompts on invalid input instead of dead-ending.
  - Purpose selection re-prompts on invalid input instead of silently
    defaulting.
  - Planting date re-prompts on unparseable input instead of silently
    substituting today's date (which previously could produce a
    confidently wrong advisory with no indication anything was off).
  - Advisory output now includes: yield confidence range, an
    explainable factor breakdown, disease/pest risk flags, and a
    data-source honesty note (live vs fallback weather/soil data).
"""

from flask import Flask, request, Response, jsonify, render_template
from datetime import datetime, timedelta
import os
import requests
from crop_engine import (
    get_coords, fetch_weather, fetch_soil, is_known_district,
    predict_yield, score_suitability, rank_crops, assess_risks,
    get_growth_stage, get_weather_summary, merge_sensor_reading,
    forward_geocode, reverse_geocode, generate_checkin_advisory,
    CROP_PARAMS, CROP_MENU_MAP, DISTRICT_COORDS,
)
from languages import t

app = Flask(__name__)

# ── ESP32 SENSOR STORE ─────────────────────────────────────────────────────────
# In-memory store of the latest ground-sensor reading per phone number.
# NOTE: this resets on every server restart/redeploy, and won't work
# across multiple server instances (e.g. Render's free tier can spin up
# fresh dynos). Fine for a competition demo; a real deployment needs a
# small persistent store (SQLite/Postgres/Redis) keyed the same way.
DEVICE_READINGS = {}
SENSOR_READING_MAX_AGE_HOURS = 72  # a reading older than this is treated
                                     # as stale and ignored (soil conditions
                                     # can genuinely shift after rain, etc.)

# ── FARM PROFILE STORE (periodic check-in subscriptions) ───────────────────────
# In-memory store of a farmer's registered plot, keyed by phone number, so
# they can dial back in anytime ("Option 6: My plot status") for a LIVE
# re-check against current weather — not a cached prediction. Same
# in-memory caveat as DEVICE_READINGS above: fine for a demo, needs a real
# database (SQLite/Postgres) for production so profiles survive restarts.
FARM_PROFILES = {}


@app.route("/sensor", methods=["POST"])
def sensor_ingest():
    """
    Ingestion endpoint for ESP32 field devices. A farmer's ESP32 (soil pH
    probe + soil moisture + nitrogen sensor + DS18B20 temp probe, wired to
    an ESP32 dev board) POSTs a JSON reading here whenever it has a fix.

    Expected JSON body:
        {
          "phone": "+256700000000",   // links reading to the farmer's USSD sessions
          "ph": 6.1,                   // optional, from analog pH probe
          "nitrogen": 1.4,             // optional, g/kg, from NPK sensor module
          "moisture_pct": 38.2,        // optional, from capacitive soil moisture sensor
          "temperature_c": 24.5,       // optional, from DS18B20 probe
          "device_id": "esp32-001"     // optional, for multi-device farms
        }
    Only 'phone' is required — send whatever the attached sensors support.
    Returns 400 on malformed input rather than silently accepting garbage,
    since a bad soil reading could skew a yield prediction the farmer
    relies on.
    """
    data = request.get_json(silent=True)
    if not data or "phone" not in data:
        return jsonify({"error": "JSON body with 'phone' is required"}), 400

    phone = str(data["phone"]).strip()
    if not phone:
        return jsonify({"error": "'phone' cannot be empty"}), 400

    reading = {}
    for field in ("ph", "nitrogen", "moisture_pct", "temperature_c"):
        val = data.get(field)
        if val is None:
            continue
        try:
            reading[field] = float(val)
        except (TypeError, ValueError):
            return jsonify({"error": f"'{field}' must be numeric"}), 400

    # Basic plausibility bounds — reject physically impossible sensor
    # noise rather than feeding it into a yield prediction.
    bounds = {"ph": (3.0, 10.0), "nitrogen": (0.0, 10.0),
              "moisture_pct": (0.0, 100.0), "temperature_c": (-5.0, 55.0)}
    for field, (lo, hi) in bounds.items():
        if field in reading and not (lo <= reading[field] <= hi):
            return jsonify({"error": f"'{field}' out of plausible range ({lo}-{hi})"}), 400

    if not reading:
        return jsonify({"error": "no recognised sensor fields in body"}), 400

    DEVICE_READINGS[phone] = {
        **reading,
        "device_id": data.get("device_id", "unknown"),
        "received_at": datetime.utcnow(),
    }
    return jsonify({"status": "ok", "stored_fields": list(reading.keys())}), 200


def get_recent_sensor_reading(phone: str):
    """Return the farmer's latest sensor reading if it exists and isn't
    stale, else None (falls back to API/default soil data)."""
    entry = DEVICE_READINGS.get(phone)
    if not entry:
        return None
    age = datetime.utcnow() - entry["received_at"]
    if age > timedelta(hours=SENSOR_READING_MAX_AGE_HOURS):
        return None
    return entry


# ── SESSION PARSER ────────────────────────────────────────────────────────────
def parse_steps(text: str) -> list:
    """Split Africa's Talking cumulative text into individual steps."""
    return [s.strip() for s in text.split("*") if s.strip()] if text else []


def get_lang(steps: list) -> str:
    """Derive language code from step 0 selection."""
    if not steps:
        return "en"
    return {"1": "en", "2": "lg", "3": "rk"}.get(steps[0], "en")


def parse_date_safe(date_raw: str):
    """
    Parse a dd/mm/yyyy planting date. Returns (date_or_none, ok_flag).
    '0' means "planted today". Anything unparseable returns ok=False
    so the caller can re-prompt instead of silently guessing.
    """
    date_raw = date_raw.strip()
    if date_raw == "0":
        return datetime.today(), True
    try:
        parsed = datetime.strptime(date_raw, "%d/%m/%Y")
        if parsed > datetime.today():
            return None, False  # future planting dates aren't supported yet
        return parsed, True
    except ValueError:
        return None, False


# ── MAIN USSD ROUTE ───────────────────────────────────────────────────────────
@app.route("/ussd", methods=["POST"])
def ussd():
    session_id   = request.form.get("sessionId", "")
    phone        = request.form.get("phoneNumber", "")
    text         = request.form.get("text", "")

    steps = parse_steps(text)
    lang  = get_lang(steps)
    depth = len(steps)

    # ── STEP 0: Language selection ──
    if depth == 0:
        response = t("en", "welcome")  # always show language menu in English first

    # ── STEP 1: Main menu ──
    elif depth == 1:
        response = t(lang, "menu_main")

    # ── BRANCH: Option 1 — Full crop advisory ──────────────────────────────
    # Location is now collected hierarchically (district -> county ->
    # subcounty -> village) instead of district alone, so soil/weather
    # data can be fetched for the farmer's actual plot rather than the
    # district's centroid. Each of county/subcounty/village can be
    # skipped with "0" — skipping narrows precision but never blocks
    # the flow, since USSD typing is slow and not every farmer knows
    # their subcounty/village by its official name.
    elif depth == 2 and steps[1] == "1":
        response = t(lang, "ask_district")

    elif depth == 3 and steps[1] == "1":
        if not is_known_district(steps[2]):
            response = t(lang, "error_district") + t(lang, "ask_district")
        else:
            response = t(lang, "ask_county")

    elif depth == 4 and steps[1] == "1":
        response = t(lang, "ask_subcounty")

    elif depth == 5 and steps[1] == "1":
        response = t(lang, "ask_village")

    elif depth == 6 and steps[1] == "1":
        response = t(lang, "ask_crop")

    elif depth == 7 and steps[1] == "1":
        if steps[6] not in CROP_MENU_MAP:
            response = t(lang, "error_input") + t(lang, "ask_crop")
        else:
            response = t(lang, "ask_area")

    elif depth == 8 and steps[1] == "1":
        try:
            if float(steps[7]) <= 0:
                raise ValueError
            response = t(lang, "ask_purpose")
        except ValueError:
            response = t(lang, "error_input") + t(lang, "ask_area")

    elif depth == 9 and steps[1] == "1":
        if steps[8] not in ("1", "2", "3"):
            response = t(lang, "error_input") + t(lang, "ask_purpose")
        else:
            response = t(lang, "ask_planting_date")

    elif depth == 10 and steps[1] == "1":
        # — All inputs collected — run the engine —
        district   = steps[2]
        county     = "" if steps[3] == "0" else steps[3]
        subcounty  = "" if steps[4] == "0" else steps[4]
        village    = "" if steps[5] == "0" else steps[5]
        crop_key   = CROP_MENU_MAP.get(steps[6], "maize")
        purpose_map = {"1": "food", "2": "feed", "3": "both"}
        purpose    = purpose_map.get(steps[8], "food")

        try:
            area_acres = float(steps[7])
        except ValueError:
            area_acres = 1.0

        planting_date, date_ok = parse_date_safe(steps[9])
        if not date_ok:
            response = t(lang, "error_date") + t(lang, "ask_planting_date")
        else:
            # Geocode the farmer's full hierarchical location. Falls back
            # to the district centroid automatically (see forward_geocode)
            # if no network, or if village/subcounty/county aren't found —
            # this never blocks the flow, it only affects precision.
            geo = forward_geocode(district, county, subcounty, village)
            lat, lon = geo["lat"], geo["lon"]

            weather  = fetch_weather(lat, lon)
            soil     = fetch_soil(lat, lon)

            # If this farmer's phone has a recent ESP32 ground-sensor
            # reading, prefer it over the regional API/fallback estimate
            # for the fields it covers (pH, nitrogen, moisture) — a
            # direct in-field reading beats a ~250m-resolution API value.
            sensor = get_recent_sensor_reading(phone)
            used_sensor = False
            if sensor:
                soil = merge_sensor_reading(soil, sensor)
                used_sensor = True

            result   = predict_yield(crop_key, weather, soil,
                                     area_acres, planting_date)
            suit     = score_suitability(crop_key, weather, soil)
            stage    = get_growth_stage(crop_key, planting_date)
            risks    = assess_risks(crop_key, weather, planting_date)
            crop_p   = CROP_PARAMS[crop_key]

            # Harvest tip based on purpose
            if purpose == "feed":
                harvest_tip = crop_p["feed_harvest_tip"]
            else:
                harvest_tip = crop_p["food_harvest_tip"]

            # Build advisory response
            response = t(lang, "advisory_header")
            response += t(lang, "suitability_line",
                          score=suit, crop=crop_p["display"])
            response += t(lang, "yield_line",
                          yield_kg=f"{result['yield_per_acre']:,.0f}",
                          low=f"{result['yield_range'][0]:,.0f}",
                          high=f"{result['yield_range'][1]:,.0f}",
                          conf=result["confidence_pct"])

            # Explainability: show the top limiting factor, not the full
            # breakdown (USSD screens are ~160 chars — keep it scannable)
            limiting = [(k, v) for k, v in result["explanation"].items()
                        if v["impact"] == "limiting"]
            if limiting:
                limiting.sort(key=lambda kv: kv[1]["share_of_penalty_pct"],
                               reverse=True)
                top_factor_name, top_factor_info = limiting[0]
                response += t(lang, "limiting_factor_line",
                              factor=top_factor_name,
                              pct=top_factor_info["share_of_penalty_pct"])

            response += t(lang, "harvest_line",
                          days=result["days_to_harvest"])
            response += t(lang, "action_line",  action=stage["advice"][:60])

            # Risk flag (top 1 for USSD brevity)
            if risks:
                response += t(lang, "risk_line",
                              name=risks[0]["name"], pct=risks[0]["risk_pct"])

            response += t(lang, "tip_line",     tip=harvest_tip[:60])

            # Honesty notes, in priority order: sensor > geocode precision > data fallback
            if used_sensor:
                response += t(lang, "sensor_note")
            elif geo["precision"] == "district":
                response += t(lang, "district_precision_note")
            elif not result["data_sources_live"]["weather"] or \
                 not result["data_sources_live"]["soil"]:
                response += t(lang, "fallback_note")

            # Continue the session (CON, not END) to offer ongoing
            # check-in monitoring for this plot — answered at depth 11.
            response += t(lang, "subscribe_prompt")

    elif depth == 11 and steps[1] == "1":
        if steps[10] == "1":
            district   = steps[2]
            county     = "" if steps[3] == "0" else steps[3]
            subcounty  = "" if steps[4] == "0" else steps[4]
            village    = "" if steps[5] == "0" else steps[5]
            crop_key   = CROP_MENU_MAP.get(steps[6], "maize")
            try:
                area_acres = float(steps[7])
            except ValueError:
                area_acres = 1.0
            planting_date, _ = parse_date_safe(steps[9])

            FARM_PROFILES[phone] = {
                "district": district, "county": county, "subcounty": subcounty,
                "village": village, "crop_key": crop_key, "area_acres": area_acres,
                "planting_date": planting_date, "registered_at": datetime.utcnow(),
            }
            response = t(lang, "subscribe_confirmed")
        else:
            response = t(lang, "subscribe_declined")

    # ── BRANCH: Option 6 — Check my plot status (live re-check) ────────────
    elif depth == 2 and steps[1] == "6":
        profile = FARM_PROFILES.get(phone)
        if not profile:
            response = t(lang, "no_profile")
        else:
            geo = forward_geocode(profile["district"], profile["county"],
                                   profile["subcounty"], profile["village"])
            weather = fetch_weather(geo["lat"], geo["lon"])
            checkin = generate_checkin_advisory(profile["crop_key"], weather,
                                                 profile["planting_date"])
            crop_p = CROP_PARAMS[profile["crop_key"]]

            response = t(lang, "checkin_header", crop=crop_p["display"])
            response += t(lang, "checkin_stage", stage=checkin["stage"],
                          days=checkin["days_to_harvest"])
            if checkin["urgent"]:
                for msg in checkin["urgent"][:2]:
                    response += t(lang, "checkin_urgent", msg=msg[:100])
            else:
                response += t(lang, "checkin_ok")
            if checkin["risks"]:
                response += t(lang, "risk_line", name=checkin["risks"][0]["name"],
                              pct=checkin["risks"][0]["risk_pct"])

    # ── BRANCH: Option 2 — Check crop status ──────────────────────────────
    elif depth == 2 and steps[1] == "2":
        response = t(lang, "status_prompt")

    elif depth == 3 and steps[1] == "2":
        raw = steps[2].strip()
        parts = raw.rsplit(" ", 1)
        if len(parts) != 2:
            response = t(lang, "error_input")
        else:
            crop_name_raw, date_raw = parts
            # Match crop name to key; no silent maize default — re-prompt
            # instead, since mislabeling a farmer's actual crop is worse
            # than asking again.
            crop_key = next(
                (k for k, v in CROP_PARAMS.items()
                 if k in crop_name_raw.lower()), None
            )
            planting_date, date_ok = parse_date_safe(date_raw)

            if crop_key is None or not date_ok:
                response = t(lang, "error_input")
            else:
                stage = get_growth_stage(crop_key, planting_date)
                response = t(lang, "status_result",
                             crop=CROP_PARAMS[crop_key]["display"],
                             stage=stage["stage"],
                             days=stage["days"],
                             advice=stage["advice"][:70])

    # ── BRANCH: Option 3 — Weekly weather forecast ─────────────────────────
    elif depth == 2 and steps[1] == "3":
        response = t(lang, "ask_district")

    elif depth == 3 and steps[1] == "3":
        district = steps[2]
        if not is_known_district(district):
            response = t(lang, "error_district")
        else:
            lat, lon = get_coords(district)
            weather  = fetch_weather(lat, lon)
            summary  = get_weather_summary(weather, num_days=5)

            response = t(lang, "weather_header", district=district.title())
            for day_num, desc, rain in summary:
                response += t(lang, "weather_line",
                              d=day_num, desc=desc, rain=rain)

    # ── BRANCH: Option 4 — Extension worker contact ────────────────────────
    elif depth == 2 and steps[1] == "4":
        response = t(lang, "extension_msg")

    # ── BRANCH: Option 5 — What should I plant? (multi-crop ranking) ───────
    elif depth == 2 and steps[1] == "5":
        response = t(lang, "ask_district")

    elif depth == 3 and steps[1] == "5":
        district = steps[2]
        if not is_known_district(district):
            response = t(lang, "error_district")
        else:
            lat, lon = get_coords(district)
            weather  = fetch_weather(lat, lon)
            soil     = fetch_soil(lat, lon)
            top_crops = rank_crops(weather, soil, top_n=3)

            response = t(lang, "rank_header", district=district.title())
            for rank_pos, c in enumerate(top_crops, start=1):
                response += t(lang, "rank_line",
                              pos=rank_pos, crop=c["display"],
                              score=c["suitability"])

    # ── FALLBACK ────────────────────────────────────────────────────────────
    else:
        response = t(lang, "error_input")

    return Response(response, mimetype="text/plain")


# ── WEB UI (smartphone users) ────────────────────────────────────────────────
# The USSD flow above serves feature-phone farmers over Africa's Talking.
# These routes serve the same underlying engine to smartphone users via a
# normal web page + JSON API, so nothing agronomic is duplicated — the
# math, thresholds, and risk rules all live in crop_engine.py exactly once.

@app.route("/", methods=["GET"])
def web_home():
    """Serve the mobile-friendly web form."""
    return render_template("index.html")


@app.route("/api/meta", methods=["GET"])
def api_meta():
    """Crop + district lists for populating the web form's dropdowns,
    sourced from the engine so the UI can never drift out of sync with
    what the backend actually supports."""
    crops = [{"key": k, "display": v["display"]} for k, v in CROP_PARAMS.items()]
    districts = sorted(DISTRICT_COORDS.keys())
    return jsonify({"crops": crops, "districts": districts})


@app.route("/api/reverse-geocode", methods=["POST"])
def api_reverse_geocode():
    """
    Resolve browser GPS coordinates to a human-readable place name, so the
    web UI can show the farmer what location was detected before they
    submit the advisory request. Body: { lat, lon }.
    """
    data = request.get_json(silent=True) or {}
    try:
        lat = float(data.get("lat"))
        lon = float(data.get("lon"))
    except (TypeError, ValueError):
        return jsonify({"error": "lat and lon must be numbers"}), 400

    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return jsonify({"error": "lat/lon out of range"}), 400

    result = reverse_geocode(lat, lon)
    return jsonify(result)


@app.route("/api/advisory", methods=["POST"])
def api_advisory():
    """
    Full crop advisory for a smartphone user, as JSON.

    Location is resolved in priority order:
      1. lat/lon — from the browser's own GPS (most precise, if given)
      2. district + optional county/subcounty/village — geocoded to a
         specific point via forward_geocode(), falling back to the
         district centroid if not found
      3. district alone — district centroid (least precise)

    Soil readings can come from two places, taking the most trustworthy:
      - Manual entry in this same request (sensor_ph / sensor_moisture /
        sensor_temp) — a farmer typing in what their own probe just read
      - A phone number with a recent ESP32 auto-POSTed reading (see
        /sensor) — used if no manual reading was given in this request

    Body: { district, county?, subcounty?, village?, lat?, lon?,
             crop_key, area_acres, purpose, planting_date, phone?,
             sensor_ph?, sensor_moisture?, sensor_temp? }
    planting_date: "YYYY-MM-DD" or omitted/"" for "planted today".
    """
    data = request.get_json(silent=True) or {}

    crop_key = str(data.get("crop_key", "")).strip()
    purpose  = str(data.get("purpose", "food")).strip()
    phone    = str(data.get("phone", "")).strip()

    if crop_key not in CROP_PARAMS:
        return jsonify({"error": "Unknown or missing crop_key"}), 400
    if purpose not in ("food", "feed", "both"):
        purpose = "food"

    try:
        area_acres = float(data.get("area_acres", 1.0))
        if area_acres <= 0:
            raise ValueError
    except (TypeError, ValueError):
        return jsonify({"error": "area_acres must be a positive number"}), 400

    date_raw = str(data.get("planting_date", "")).strip()
    if not date_raw:
        planting_date = datetime.today()
    else:
        try:
            planting_date = datetime.strptime(date_raw, "%Y-%m-%d")
            if planting_date > datetime.today():
                return jsonify({"error": "planting_date cannot be in the future"}), 400
        except ValueError:
            return jsonify({"error": "planting_date must be YYYY-MM-DD"}), 400

    # ── Resolve location ──
    lat_raw, lon_raw = data.get("lat"), data.get("lon")
    location_label = ""
    geocode_precision = "gps"

    if lat_raw is not None and lon_raw is not None:
        try:
            lat, lon = float(lat_raw), float(lon_raw)
            if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError
        except (TypeError, ValueError):
            return jsonify({"error": "lat/lon out of range"}), 400
        geo_name = reverse_geocode(lat, lon)
        location_label = geo_name["place_name"]
    else:
        district  = str(data.get("district", "")).strip()
        county    = str(data.get("county", "")).strip()
        subcounty = str(data.get("subcounty", "")).strip()
        village   = str(data.get("village", "")).strip()
        if not district or not is_known_district(district):
            return jsonify({"error": "Unknown or missing district (or provide lat/lon)"}), 400
        geo = forward_geocode(district, county, subcounty, village)
        lat, lon = geo["lat"], geo["lon"]
        geocode_precision = geo["precision"]
        location_label = geo["matched_name"] if geo["precision"] != "district" else district.title()

    weather  = fetch_weather(lat, lon)
    soil     = fetch_soil(lat, lon)

    # ── Resolve soil sensor reading: manual entry beats phone-linked ESP32 ──
    used_sensor = False
    sensor_source = None
    manual_sensor = {}
    for field, key in (("sensor_ph", "ph"), ("sensor_moisture", "moisture_pct"),
                        ("sensor_temperature", "temperature_c")):
        val = data.get(field)
        if val is not None and str(val).strip() != "":
            try:
                manual_sensor[key] = float(val)
            except (TypeError, ValueError):
                return jsonify({"error": f"'{field}' must be numeric"}), 400

    if manual_sensor:
        soil = merge_sensor_reading(soil, manual_sensor)
        used_sensor = True
        sensor_source = "manual"
    elif phone:
        sensor = get_recent_sensor_reading(phone)
        if sensor:
            soil = merge_sensor_reading(soil, sensor)
            used_sensor = True
            sensor_source = "esp32"

    result = predict_yield(crop_key, weather, soil, area_acres, planting_date)
    suit   = score_suitability(crop_key, weather, soil)
    stage  = get_growth_stage(crop_key, planting_date)
    risks  = assess_risks(crop_key, weather, planting_date, top_n=3)
    crop_p = CROP_PARAMS[crop_key]
    harvest_tip = crop_p["feed_harvest_tip"] if purpose == "feed" else crop_p["food_harvest_tip"]

    return jsonify({
        "crop": crop_p["display"],
        "location_label": location_label,
        "location_precision": geocode_precision,
        "suitability_pct": suit,
        "yield_per_acre": result["yield_per_acre"],
        "yield_range": result["yield_range"],
        "total_yield": result["total_yield"],
        "confidence_pct": result["confidence_pct"],
        "explanation": result["explanation"],
        "days_to_harvest": result["days_to_harvest"],
        "growth_stage": stage["stage"],
        "stage_advice": stage["advice"],
        "risks": risks,
        "harvest_tip": harvest_tip,
        "used_sensor_reading": used_sensor,
        "sensor_source": sensor_source,
        "data_sources_live": result["data_sources_live"],
    })


@app.route("/api/rank", methods=["POST"])
def api_rank():
    """'What should I plant?' for a smartphone user, as JSON.
    Body: { district }"""
    data = request.get_json(silent=True) or {}
    district = str(data.get("district", "")).strip()
    if not district or not is_known_district(district):
        return jsonify({"error": "Unknown or missing district"}), 400

    lat, lon = get_coords(district)
    weather  = fetch_weather(lat, lon)
    soil     = fetch_soil(lat, lon)
    top_crops = rank_crops(weather, soil, top_n=5)
    return jsonify({"district": district.title(), "ranking": top_crops})


@app.route("/api/subscribe", methods=["POST"])
def api_subscribe():
    """
    Register a plot for ongoing check-in monitoring (web UI equivalent of
    the USSD subscribe flow). Body matches the location/crop fields used
    in /api/advisory, plus a required phone number as the profile key.
    """
    data = request.get_json(silent=True) or {}
    phone = str(data.get("phone", "")).strip()
    if not phone:
        return jsonify({"error": "phone is required to subscribe"}), 400

    district = str(data.get("district", "")).strip()
    crop_key = str(data.get("crop_key", "")).strip()
    if not district or not is_known_district(district):
        return jsonify({"error": "Unknown or missing district"}), 400
    if crop_key not in CROP_PARAMS:
        return jsonify({"error": "Unknown or missing crop_key"}), 400

    try:
        area_acres = float(data.get("area_acres", 1.0))
    except (TypeError, ValueError):
        area_acres = 1.0

    date_raw = str(data.get("planting_date", "")).strip()
    planting_date = datetime.today() if not date_raw else datetime.strptime(date_raw, "%Y-%m-%d")

    FARM_PROFILES[phone] = {
        "district": district,
        "county": str(data.get("county", "")).strip(),
        "subcounty": str(data.get("subcounty", "")).strip(),
        "village": str(data.get("village", "")).strip(),
        "crop_key": crop_key,
        "area_acres": area_acres,
        "planting_date": planting_date,
        "registered_at": datetime.utcnow(),
    }
    return jsonify({"status": "subscribed", "phone": phone})


@app.route("/api/plot-status", methods=["POST"])
def api_plot_status():
    """Live re-check of a subscribed plot, for the web UI. Body: { phone }"""
    data = request.get_json(silent=True) or {}
    phone = str(data.get("phone", "")).strip()
    profile = FARM_PROFILES.get(phone)
    if not profile:
        return jsonify({"error": "No registered plot for that phone number. "
                                  "Subscribe via /api/subscribe first."}), 404

    geo = forward_geocode(profile["district"], profile["county"],
                           profile["subcounty"], profile["village"])
    weather = fetch_weather(geo["lat"], geo["lon"])
    checkin = generate_checkin_advisory(profile["crop_key"], weather,
                                         profile["planting_date"])
    crop_p = CROP_PARAMS[profile["crop_key"]]

    return jsonify({
        "crop": crop_p["display"],
        "stage": checkin["stage"],
        "days_to_harvest": checkin["days_to_harvest"],
        "urgent": checkin["urgent"],
        "risks": checkin["risks"],
    })


@app.route("/cron/checkin", methods=["POST", "GET"])
def cron_checkin():
    """
    Scheduled entry point for PUSH alerts (SMS), as opposed to the pull-based
    /api/plot-status and USSD option 6 above. This does nothing on its own —
    Render's free tier has no built-in scheduler, so this route needs to be
    triggered periodically (e.g. daily) by a free external cron service such
    as cron-job.org pointed at this URL.

    Sending real SMS requires an Africa's Talking account with SMS enabled,
    plus AT_USERNAME and AT_API_KEY set as environment variables on Render
    (Settings -> Environment). Without those set, this route still runs the
    full check for every subscribed profile and returns what WOULD be sent,
    which is enough to demo the logic without live SMS credentials.
    """
    at_username = os.environ.get("AT_USERNAME")
    at_api_key = os.environ.get("AT_API_KEY")
    sms_configured = bool(at_username and at_api_key)

    results = []
    for phone, profile in FARM_PROFILES.items():
        try:
            geo = forward_geocode(profile["district"], profile["county"],
                                   profile["subcounty"], profile["village"])
            weather = fetch_weather(geo["lat"], geo["lon"])
            checkin = generate_checkin_advisory(profile["crop_key"], weather,
                                                 profile["planting_date"])
            crop_p = CROP_PARAMS[profile["crop_key"]]

            if not checkin["urgent"]:
                results.append({"phone": phone, "sent": False, "reason": "no urgent items"})
                continue

            message = f"CropSense: {crop_p['display']} update — " + " ".join(checkin["urgent"][:1])

            if sms_configured:
                try:
                    resp = requests.post(
                        "https://api.africastalking.com/version1/messaging",
                        data={"username": at_username, "to": phone, "message": message},
                        headers={"apiKey": at_api_key, "Accept": "application/json"},
                        timeout=10,
                    )
                    results.append({"phone": phone, "sent": resp.ok, "message": message})
                except Exception as e:
                    results.append({"phone": phone, "sent": False, "error": str(e)})
            else:
                results.append({"phone": phone, "sent": False,
                                 "reason": "AT_USERNAME/AT_API_KEY not configured — message not sent",
                                 "would_send": message})
        except Exception as e:
            results.append({"phone": phone, "sent": False, "error": str(e)})

    return jsonify({"sms_configured": sms_configured, "checked": len(FARM_PROFILES),
                     "results": results})


# ── RUN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Local testing: run with `python app.py`
    # Then simulate USSD with curl:
    # curl -X POST http://localhost:5000/ussd \
    #   -d "sessionId=test001&phoneNumber=%2B256700000000&text=1*1*Mbarara*1*2*1*01%2F06%2F2026"
    #
    # Try the new "what should I plant" flow:
    # curl -X POST http://localhost:5000/ussd \
    #   -d "sessionId=test002&phoneNumber=%2B256700000000&text=1*5*Mbarara"
    app.run(debug=True, port=5000)
