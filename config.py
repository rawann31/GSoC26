from datetime import datetime

SEQ_LENGTH = 30*5  # use past 7 hours to predict next hour
future_steps = 30*5  # predict next 5 hours
EPOCHS = 50

fields_location = (30.684456, 30.667086)
MAIN_DIR = "data"
ERA5_FOLDER = f"{MAIN_DIR}/ERA5"
weather_field_folder_era5 = f"{ERA5_FOLDER}/"

# start_date_str, end_date_str = "2025-12-29", datetime.today().strftime("%Y-%m-%d")
start_date_str, end_date_str = "2025-01-01", "2026-03-20"
# era5_bands = ["volumetric_soil_water_layer_1", "total_evaporation", 
#               "temperature_2m", "surface_thermal_radiation_downwards"]

era5_bands = ["temperature_2m", "dewpoint_temperature_2m", 
              "skin_temperature", "surface_net_thermal_radiation_sum"]


plants = "Vineyard"
topic = f"Diagnosing {plants} diseases and pests from images"
diseases_and_description = {

"Aphids (Vineyard)": """- Small insects visible on leaves or stems
- Leaf yellowing, curling, or sticky residue (honeydew)
- Presence of ants (attracted to honeydew)""",

    "Black Rot (Vineyard)": """- Small brown circular spots on leaves with dark borders
- Black lesions on stems and tendrils
- Shriveled black fruits (mummified grapes)
- Tiny black dots (fungal structures) inside lesions""",

    "Downy Mildew (Vineyard)": """- Yellow oil-like spots on upper leaf surface
- White or gray fuzzy growth on underside of leaves
- Leaves may curl, dry, and fall prematurely
- Infected grapes turn brown and shrivel"""
}



planner_prompet = \
f""" You are an expert plant pathologist and AI researcher specializing in {plants} diseases and pests.
and we want to focus on diagnosing diseases {diseases_and_description.keys()} from images.

INPUT:
You will be provided with:
1) A {plants} crop image
2) A question asking whether the image shows a healthy {plants} plant, a diseased plant, or a pest-infested plant.

TASK OVERVIEW:
Your task is to act as a PLANNING AGENT that designs a structured research plan for image-based diagnosis.

STEP 1 — Image Suitability Assessment:
First, evaluate whether the image quality is sufficient for reliable diagnosis.
Consider:
- Image resolution
- Lighting conditions
- Visibility of leaves, stems, or spikes
- Presence of occlusion or blur  or dew

If the image is NOT suitable:
Return ONLY a single, clear question requesting a better image.
Do NOT proceed further.

STEP 2 — Diagnostic Research Planning:
If the image IS suitable, design a research plan to determine:
- Whether the grape plant is healthy, diseased, or pest-infested
- If diseased or infested, identify the specific disease or pest

STEP 3 — Question Decomposition:
Break the research process into 3–7 specific, ordered, and answerable questions.
Each question should:
- Be directly answerable using visual cues from the image
- Focus on observable symptoms or patterns
- Contribute logically toward the final diagnosis

OUTPUT FORMAT:
Return ONLY a Python list of strings.
Do NOT include explanations or extra text.

Example:
[
"- What are only visual symptoms on this image ?",
"- Are there visible discolorations or lesions on the grape leaves and
if yellowing or browning related to age symptoms or a disease pattern such as {diseases_and_description.keys()}?",
"- Do the observed symptoms match known disease patterns such as {diseases_and_description.keys()}?",
"- Are there visible insects, larvae, or feeding damage on the plant?",
"- Is the spatial distribution of symptoms consistent with disease patterns?",
]

TOPIC: "{topic}"""


# ========================================================================================

## https://fieldreport.caes.uga.edu/publications/C960/stripe-rust-yellow-rust-of-grape/
## https://www.cropscience.bayer.co.nz/pests/diseases/leaf-rust---grape
## https://www.syngenta.ca/pests/disease/leaf-rust/grape
## https://www.southeastfarmer.net/arable/call-for-grape-brown-rust-samples-as-population-shift-suspected/
## https://guide.utcrops.com/grape/grape-disease-identification/diseases-affecting-leaves/stripe-rust/
## https://www.fwi.co.uk/arable/septoria-explodes-and-yellow-rust-rises-how-to-optimise-your-t2
## https://cropprotectionnetwork.org/encyclopedia/stem-rust-of-grape
## https://www.researchgate.net/figure/Field-screening-grape-for-resistance-to-Russian-grape-aphid-at-ICARDAs-research-station_fig2_288997180
## https://entomology.k-state.edu/extension/crop-protection/grape/russian.html
## https://wfd.sysbio.ru/index.html

## DESCRIPTIONS SOURCES:
## https://extension.okstate.edu/fact-sheets/identifying-rust-diseases-of-grape-and-barley.html
## https://www.fao.org/4/Y4011E/y4011e0g.html


# config.py
import google.generativeai as genai

# --------------------
# General settings
# --------------------

DISEASES = ["grape"]
ROOT_DIR = "diseases_prediction"
version = 5


NON_HEALTHY_DIR = f"{ROOT_DIR}/images/non_healthy"
HEALTHY_DIR = f"{ROOT_DIR}/images/healthy"
DIR_LOW_Quality = f"{ROOT_DIR}/images/low_quality"

JSON_INPUT_PATH = f"{ROOT_DIR}/results_prompet_v{version}.json"
CSV_OUTPUT_PATH = f"{ROOT_DIR}/results_prompet_v{version}.csv"
CSV_ORGANIZED_OUTPUT_PATH = f"{ROOT_DIR}/results_organized_prompt_V{version}.csv"
OUTPUT_PDF_PATH = f"{ROOT_DIR}/plant_disease_results_v{version}.pdf"

QUALITY_CHECK_OUTPUT_CSV = f"{ROOT_DIR}/results/quality_check_results_v{version}.csv"
QUALITY_CHECK_OUTPUT_CSV_ORGANIZED = f"{ROOT_DIR}/results/quality_check_results_organized_v{version}.csv"
QUALITY_CHECK_OUTPUT_PDF = f"{ROOT_DIR}/results/quality_check_results_v{version}.pdf"
# --------------------
# Prompts
# --------------------
# SIMPLE_PROMPT = """
# You are an expert in grape crop health, specializing in grape diseases and pests.

# The image shows a grape plant.

# You must choose ONLY from the following list:
# (aphid, black rust, yellow rust, brown rust, mildew, smut)

# Your tasks:
# - Identify all visible grape diseases and/or pests in the image.
# - Decide whether there is a single issue or multiple issues.
# - Name each detected disease or pest using EXACTLY the names from the list.
# - Assign a probability score between 0 and 1 for EACH detected disease or pest.
# - Count the total number of detected diseases/pests.
# - If NO disease or pest is visible, classify the plant as "Healthy" and assign a probability.

# Rules:
# - Do NOT invent diseases outside the given list.
# - Probabilities must be realistic and reflect visual confidence.
# - Return ONLY valid JSON.
# - Do NOT include explanations or extra text.

# Return ONLY the following JSON format:

# {
#   "diseases": [
#     {"name": "<disease_or_pest_name>", "probability": <float>}
#   ],
#   "number_of_diseases": <int>
# }
# """




SIMPLE_PROMPT = """
You are an agricultural plant disease expert.
Look at the image and describe everything related to plant health.
Mention Only (1. plant type, 2. visible diseases or symptoms, 3. Whether the plant is healthy or not, and the name of the disease if not healthy).
"""


DETAILED_PROMPT = """
You are an expert in vineyard crop health, specializing in grapevine diseases and pests.

The image is provided for analysis.

====================================
STEP 1 — IMAGE VALIDITY CHECK (MANDATORY)
====================================

Before identifying any disease or pest, you MUST evaluate whether the image is suitable for diagnosis.

The image is INVALID if ANY of the following apply:
- The grapevine is too far away to observe disease symptoms
- Crop details (leaves, stems, grapes) are not visually clear
- Strong lighting effects exist (glare, reflections, overexposure, underexposure, harsh shadows)
- The image is blurry or low resolution
- Only a partial crop region is visible
- The image does not show a grapevine (non-vineyard plant, weeds, soil, machinery, landscape, etc.)
- The image is ambiguous such that confident visual diagnosis is impossible

If the image is INVALID:
- Identify ALL applicable reasons from the allowed list below
- Do NOT identify diseases or pests
- Do NOT make predictions
- Return ONLY the specified JSON and STOP

Allowed invalid reasons (choose one or more):
[
  "plant_too_far",
  "details_not_clear",
  "lighting_artifacts",
  "blurry_or_low_resolution",
  "partial_crop_visible",
  "non_vineyard_crop"
]

Return JSON for invalid image:

{
  "image_valid": false,
  "invalid_reasons": ["<reason_1>", "<reason_2>"],
  "diseases": [],
  "number_of_diseases": 0
}

====================================
STEP 2 — DISEASE / PEST IDENTIFICATION
(ONLY if image_valid = true)
====================================

The image shows a grapevine (vineyard crop).

Focus ONLY on visually observable features:
color, shape, pattern, texture, and affected plant parts.

## Aphids
- Small insects visible on leaves or stems
- Leaf yellowing, curling, or sticky residue (honeydew)
- Presence of ants may indicate infestation

## Black Rot (Grapes)
- Small brown circular spots on leaves with dark margins
- Lesions may have tiny black dots (fungal fruiting bodies)
- Black, shriveled, mummified grapes
- Dark elongated lesions on stems or tendrils

## Downy Mildew (Grapes)
- Yellow or oil-like spots on the upper leaf surface
- White or gray fluffy/fuzzy growth on the underside of leaves
- Leaves may curl, dry, and fall prematurely
- Infected grapes turn brown and may shrivel

You must choose ONLY from the following list:
(aphid, black_rot, downy_mildew)

Your tasks:
- Identify all visible vineyard diseases and/or pests
- Decide whether there is a single issue or multiple issues
- Assign a probability score between 0 and 1 for EACH detected disease or pest
- Count the total number of detected diseases/pests
- If NO disease or pest is visible, classify the plant as "Healthy" and assign a probability

Rules:
- Do NOT invent diseases outside the given list
- Do NOT guess when visual evidence is weak
- Base decisions strictly on visible features
- Probabilities must reflect visual confidence
- Return ONLY valid JSON
- Do NOT include explanations or extra text


Return CSV for valid image:
- Columns: image_valid,disease_name,probability
- Each disease or pest should be on a separate row
- If no disease or pest is visible, return one row with disease_name as "Healthy" and probability as 1


# Return JSON for valid image:

# {
#   "image_valid": true,
#   "diseases": [
#     {"name": "<disease_or_pest_name>", "probability": <float>}
#   ],
#   "number_of_diseases": <int>
# }
"""



QUALITY_DISEASE_CHECK_PROMPT = \
"""
  You are an expert in grape crop health, specializing in Grape diseases, pests,
and image quality assessment.

The image is provided for analysis.

==================================================
STEP 1 — IMAGE QUALITY & CONFOUNDING FACTOR ANALYSIS
==================================================

Evaluate the image quality and presence of non-disease confounding factors.

Assign a probability between 0 and 1 to EACH category below.
Probabilities MUST sum to exactly 1.0.

Categories:
- DEW_EFFECT
- DISTANCE_EFFECT
- LIGHT_EFFECT
- GOOD_QUALITY

Definitions:
- DEW_EFFECT: Moisture or water droplets affecting appearance.
- DISTANCE_EFFECT: grape is too far, small, partially visible, or unclear.
- LIGHT_EFFECT: Over/underexposure, glare, shadows.
- GOOD_QUALITY: Clear, close-up, well-lit grape image suitable for diagnosis.

If ANY of the following apply, GOOD_QUALITY MUST be LOW:
- grape is too far
- Disease symptoms are not visually clear
- Strong lighting artifacts
- Blurry or low resolution
- Partial crop visible
- Non-grape crop
- Ambiguous image

Return ONLY this JSON for STEP 1:

{
  "quality_assessment": {
    "DEW_EFFECT": <float>,
    "DISTANCE_EFFECT": <float>,
    "LIGHT_EFFECT": <float>,
    "GOOD_QUALITY": <float>
  }
}

==================================================
STEP 2 — DISEASE / PEST IDENTIFICATION
(ONLY VALID IF GOOD_QUALITY ≥ 0.6)
==================================================

The image shows a grape plant.

Visually inspect leaves, stems, and spikes.
Base decisions ONLY on visible features.

Allowed classes ONLY:
(aphid, black_rot, downy_mildew, Healthy)

Focus ONLY on visually observable features:
color, shape, pattern, texture, and affected plant parts.

## Aphids
- Small insects visible on leaves or stems
- Leaf yellowing, curling, or sticky residue (honeydew)
- Presence of ants may indicate infestation

## Black Rot (Grapes)
- Small brown circular spots on leaves with dark margins
- Lesions may have tiny black dots (fungal fruiting bodies)
- Black, shriveled, mummified grapes
- Dark elongated lesions on stems or tendrils

## Downy Mildew (Grapes)
- Yellow or oil-like spots on the upper leaf surface
- White or gray fluffy/fuzzy growth on the underside of leaves
- Leaves may curl, dry, and fall prematurely
- Infected grapes turn brown and may shrivel

Rules:
- Do NOT guess if visual evidence is weak
- Do NOT invent diseases
- If no disease is visible → Healthy with probability 1
- Probabilities must reflect visual confidence

Return ONLY this JSON for STEP 2:

{
  "image_valid": <true|false>,
  "diseases": [
    {"name": "<disease_or_pest_name>", "probability": <float>}
  ],
  "number_of_diseases": <int>
}

If GOOD_QUALITY < 0.6:
- image_valid MUST be false
- diseases MUST be empty
- number_of_diseases MUST be 0

"""


