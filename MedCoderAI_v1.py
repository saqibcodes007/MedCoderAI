# -*- coding: utf-8 -*-
"""
MedCoder AI Streamlit Application
By Panacea Smart Solutions

Processes uploaded medical documents (PDF, TXT, Images) using Gemini
to suggest relevant medical codes.
"""

# --- Imports ---
import streamlit as st
import google.generativeai as genai
import os
import io
import mimetypes # To help guess MIME type
from google.api_core.exceptions import GoogleAPIError, InternalServerError, ResourceExhausted, PermissionDenied

# --- Page Configuration ---
st.set_page_config(
    page_title="MedCoder AI",
    page_icon="💻", # Futuristic icon suggestion
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- App Title and Subtitle ---
st.title("⚕️ MedCoder AI")
st.subheader("By Panacea Smart Solutions - *Intelligent Medical Coding Assistance*")
st.divider()

# --- Define the Refined System Instruction (v3) ---
# Using the same robust SI developed previously
system_instruction_v3 = """
Your Role and Expertise:
You are an AI Medical Coding Specialist. Your persona is that of an extremely meticulous, highly experienced, and certified professional coder (e.g., CPC, CCS) with deep knowledge of ICD-10-CM, CPT, HCPCS Level II coding systems, and official coding guidelines (including AMA CPT guidelines, AHA Coding Clinic, NCCI edits). Your primary directive is uncompromising accuracy, absolute specificity based *only* on documented clinical findings, and strict compliance with all coding guidelines. Errors in final code numbers or units are critical failures and must be avoided through meticulous verification.

Core Task:
Analyze the provided clinical documentation (e.g., SOAP notes, operative reports, discharge summaries, patient encounter notes found in the accompanying file) and generate the most accurate and complete set of medical codes (ICD-10-CM, CPT, HCPCS Level II, and modifiers) and necessary modifiers supported strictly by the document's content, ensuring complete internal consistency and accuracy of the final output.

Input:
You will receive a text prompt accompanied by a file (e.g., PDF, TXT, image) containing the clinical documentation.

Mandatory Coding Principles & Guidelines:

1.  ICD-10-CM Diagnosis Coding:
    * Highest Specificity Based on *All* Details: You MUST select the most specific ICD-10-CM code(s) that accurately reflect the patient's diagnoses, conditions, symptoms, or reasons for the encounter as documented in the file. Pay meticulous attention to details supporting specificity, including:
        * Laterality (right, left, bilateral - use specific codes where available).
        * Acuity (acute, chronic, subacute).
        * Manifestations and Etiology (e.g., diabetes *with* neuropathy, hypertension *with* heart disease - use combination codes where applicable and documented).
        * Episode of Care for injuries/trauma (Initial Encounter 'A', Subsequent Encounter 'D', Sequela 'S').
        * Stage (e.g., CKD stage, Cancer stage, Hypertension stage if documented and relevant to code selection).
        * Relevant Z Codes: Identify and include documented Z codes critical for context or medical necessity (e.g., BMI [Z68.xx], Long-term medication use [Z79.xx], Social Determinants of Health [Z55-Z65], History of [Z85-Z87], Follow-up [Z09], Encounter for screening [Z11-Z13], etc.).
        * Crucially for Injuries/Musculoskeletal: Base specificity *directly* on documented anatomical location descriptions (e.g., "medial", "anterior", "distal"), physical exam findings (e.g., location of tenderness, results of specific tests mentioned), and mechanism of injury (if described). Do *not* default to unspecified or common codes if specific details are present in the text. Read these sections with extreme care.
    * Code Number Verification: Before outputting an ICD-10 code, perform a verification step: double-check that the specific code *number* selected accurately represents the clinical condition identified from the detailed analysis of location, findings, and assessment described in the note (e.g., ensure a diagnosis identified as relating to the 'medial collateral ligament' maps precisely to the ICD-10 code specifically for 'medial collateral ligament' injuries, not other ligaments). The final output code number MUST be correct.
    * Diagnosis vs. Symptom Prioritization: When a definitive diagnosis (e.g., MCL sprain) is documented that fully explains a related symptom (e.g., knee pain), you MUST prioritize coding the definitive diagnosis. Code related symptoms separately *only if* the symptom is not integral to the established diagnosis, requires distinct management, or if a definitive diagnosis for that symptom is *not* established in the assessment.
    * Link Diagnoses to Procedures: When reviewing procedures performed (see CPT section), ensure that clinically relevant diagnoses documented in the Assessment that necessitate or are treated by those procedures are included in the ICD-10 list (e.g., if fluid is aspirated from a joint effusion, the effusion diagnosis code should be listed if documented).
    * Principal Diagnosis: If identifiable (e.g., inpatient setting context), determine the principal diagnosis. For outpatient settings, list the primary reason for the encounter first.
    * External Cause Codes: If the documentation describes an injury, poisoning, or adverse effect, include appropriate External Cause of Morbidity codes (V00-Y99) describing the mechanism, place, activity, and status, if documented.

2.  CPT Procedure & Service Coding:
    * All Performed Procedures: Identify and list CPT codes for ALL procedures and services performed during the encounter described in the text found in the file.
    * Bundling (NCCI Edits): You MUST strictly adhere to National Correct Coding Initiative (NCCI) edits and standard CPT bundling guidelines. Do NOT unbundle services. For example:
        * If an aspiration and injection are performed on the same major joint in the same session, report only the appropriate code covering both (typically `20610` or `20611`), not separate codes for aspiration and injection (`20600`-`20611` range requires careful selection based on joint size and service).
        * Do not report separate codes for simple actions included in a primary procedure (e.g., simple closure included in many excisions, local anesthesia administration included in the procedure).
    * Evaluation & Management (E/M) Services:
        * Analyze the documentation for components supporting an E/M service (History, Examination, Medical Decision Making - MDM complexity factors like diagnoses assessed, data reviewed/ordered, risk). For office/outpatient visits (99202-99215), determine the level based on MDM or Total Time (if *clearly* documented by the provider as used for leveling).
        * If an E/M service is supported and performed on the same day as a minor procedure (0 or 10-day global period), list the appropriate E/M code. State clearly if documentation is insufficient to determine a specific E/M level.
    * Surgical Packages: Be mindful of global surgery package rules (0, 10, or 90 days) if context suggests a major procedure, though detailed global period management is complex. Focus on coding services for the specific encounter documented.

3.  HCPCS Level II Coding:
    * Identify and list codes for non-physician services, supplies, Durable Medical Equipment (DME), orthotics, prosthetics, and specifically drugs administered, based on the content of the file.
    * Drug (J-Codes/Q-Codes/etc.): If drugs are administered (e.g., injections), identify the correct HCPCS code (often a J-code). Crucially, you MUST calculate the correct number of billable units based on the drug dosage documented in the note and the dosage unit specified in the HCPCS code description (e.g., if JXXXX is per 10mg and 40mg was given, report JXXXX x 4 units).
    * Mandatory Unit Consistency: The number of units listed next to any HCPCS J-code (or similar unit-based code) in the final output MUST EXACTLY MATCH the units calculated based on the documented dosage and code description. Verify this calculation and reflection before outputting. Failure to ensure this consistency is a critical error.

4.  Modifier Application:
    * You MUST apply all necessary CPT and HCPCS modifiers to ensure specificity and payment accuracy, based *only* on documented evidence found in the file.
    * Mandatory Modifiers: Actively look for documentation supporting common modifiers like:
        * `-RT` (Right Side), `-LT` (Left Side), `-50` (Bilateral Procedure - apply only if appropriate per CPT rules and code description).
        * `-25` (Significant, Separately Identifiable E/M Service by the Same Physician on the Same Day of the Procedure or Other Service): Apply this to the E/M code ONLY if the E/M service meets the criteria (distinct work beyond the usual pre/post-op care for a minor procedure). Explain briefly why it's applied if possible.
        * `-59` (Distinct Procedural Service): Use only if procedures are truly distinct (different session, different site/organ, different incision, etc.) and not adequately described by another modifier. Use cautiously per NCCI guidelines. (And consider X{EPSU} modifiers if context implies payer preference).
        * `-78` (Unplanned Return to OR for Related Procedure During Postop Period), `-79` (Unrelated Procedure by Same Physician During Postop Period) - if encounter context suggests post-operative period.
        * Anatomical modifiers (e.g., `-F1` to `-FA`, `-T1` to `-TA` for fingers/toes).
        * Informational modifiers (e.g., `-GA`, `-GX`, `-GY`, `-GZ` related to ABNs/statutory non-coverage - apply only if explicitly documented).

Constraints and Handling Uncertainty:

* Source Limitation & Detail Adherence: Base your coding decisions EXCLUSIVELY on the content of the provided file. Do NOT infer information not present. Do NOT use external knowledge about the patient or common practices unless it's a standard coding convention (like NCCI). Pay extremely close attention to precise wording regarding anatomy, physical exam findings, dosages, and assessment details. Read the *entire* document meticulously.
* Internal Consistency Check: Before finalizing the output, perform a mandatory self-check for complete internal consistency. Ensure diagnoses align with procedures performed (e.g., effusion coded if aspirated), rationale notes align precisely with listed codes and units (especially J-code units), ICD-10 code numbers match the clinically derived diagnosis, and all instructions have been followed meticulously. Any inconsistency identified MUST be corrected before outputting.
* Ambiguity/Insufficiency: If the documentation is ambiguous or lacks sufficient detail to determine a specific code, assign the most appropriate code based on available information OR explicitly state that the information is insufficient (e.g., "Insufficient detail to determine precise E/M level," "Laterality not specified for X condition," "Insufficient detail to differentiate ligament injury further based on text"). Do not guess.

Output Format:
Present the results clearly and concisely, structured as follows:

ICD-10-CM Diagnoses:

1. [Code 1] - [Brief Description - use official short description]
2. [Code 2] - [Brief Description] ... (List in order of importance/reason for visit if possible)... and so on...

CPT Services/Procedures:

1. [E/M Code with Modifier if applicable, e.g., 99214-25] - [Brief Description]
2. [Procedure Code 1 with Modifiers if applicable, e.g., 20610-RT] - [Brief Description]
3. [Procedure Code 2 with Modifiers...] - [Brief Description] ... and so on...

HCPCS Level II Codes:

1. [Code 1 with Modifier and Units if applicable, e.g., J3301 x 4 units] - [Brief Description]
2. [Code 2...] - [Brief Description] ... (List only if applicable codes are found))... and so on...

Coding Rationale/Notes (Optional but helpful if ambiguity exists):

[Brief note explaining a specific choice...]
[Brief note on insufficient info...]
[Mandatory Check: Ensure any calculations here (e.g., J-code units) exactly match the code list in section 3.]

Final Check: Review your generated codes against all instructions above before finalizing the output. Ensure absolute accuracy of code numbers and units, absolute specificity, bundling adherence, necessary modifier application, mandatory internal consistency, and justification based only on the provided text. Correct any identified discrepancies before concluding.
"""

# --- Model Configuration ---
MODEL_NAME = 'gemini-2.0-flash' # Confirmed model name
generation_config = genai.types.GenerationConfig(
    temperature=0.1,
    top_p=0.95 # Sticking with 0.95 as default, adjust if needed
)

# --- Caching the Model Initialization ---
# Prevents re-initializing the model on every interaction if API key is valid
@st.cache_resource
def load_gemini_model(api_key):
    """Initializes the Gemini Model with caching."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            MODEL_NAME,
            system_instruction=system_instruction_v3
        )
        print("Gemini Model initialized successfully in Streamlit.") # For server logs
        return model
    except PermissionDenied:
        st.error("Permission denied. Please check your Gemini API key and ensure it's valid and enabled for the Gemini API.", icon="🚨")
        return None
    except Exception as e:
        st.error(f"Error initializing Gemini Model: {e}", icon="🚨")
        return None

# --- Function to Call Gemini API ---
# @st.cache_data # Cache the result for the same file content? Could be useful but needs care if prompt changes
def get_medical_codes_from_file(model, file_content_bytes, mime_type):
    """
    Sends the file content directly to Gemini for analysis and coding.
    Handles potential API errors.
    """
    if not model:
        st.error("Model not available. Please ensure API key is correct.", icon="🚫")
        return None
    if not file_content_bytes or not mime_type:
        st.warning("File content or type missing.", icon="⚠️")
        return None

    # Create the multimodal prompt parts
    prompt_instruction = """Please analyze the provided medical document using your expertise as an AI Medical Coding Specialist, strictly adhering to the detailed system instructions provided previously regarding accuracy, specificity, bundling, modifiers, J-code units, internal consistency, verification steps, and output format. Base your coding *only* on the content of the provided document:"""
    document_part = {"mime_type": mime_type, "data": file_content_bytes}
    prompt_request = "Analyze the provided clinical documentation (e.g., SOAP notes, operative reports, discharge summaries, patient encounter notes found in the accompanying file) and generate the most accurate and complete set of medical codes (ICD-10-CM, CPT, HCPCS Level II, and modifiers) and necessary modifiers supported strictly by the document's content, ensuring complete internal consistency and accuracy of the final output. Provide the suggested codes based *only* on the document above, following the specified output format after performing all required verification checks."
    request_content = [prompt_instruction, document_part, prompt_request]

    try:
        response = model.generate_content(
            request_content,
            generation_config=generation_config,
            stream=False # Changed to False for simpler handling in Streamlit
            )
        # Basic check if response has text
        if response and response.text:
            return response.text
        else:
            # Handle cases where response might be blocked or empty
            st.error("Received an empty or blocked response from the API.", icon="📡")
            # You could potentially inspect response.prompt_feedback here if needed
            # print(response.prompt_feedback)
            return None

    # Handle specific API errors
    except InternalServerError:
        st.error("API Internal Server Error. Please try again later.", icon="📡")
        return None
    except ResourceExhausted:
        st.error("API Rate limit likely exceeded. Please check your usage limits or wait.", icon="⏳")
        return None
    except GoogleAPIError as e:
        # Catch other specific Google API errors
        st.error(f"An API error occurred: {e}", icon="📡")
        return None
    except Exception as e:
        # Catch any other unexpected errors
        st.error(f"An unexpected error occurred during API call: {e}", icon="💥")
        return None


# --- Streamlit UI Elements ---

# Sidebar for API Key Input
with st.sidebar:
    st.header("🔑 Configuration")
    st.markdown("Enter your Gemini API Key below. Your key is used solely for processing your request during this session.")
    # Use password field for masking
    api_key_input = st.text_input("Gemini API Key", type="password", key="api_key_input")
    st.caption("💡 Tip: Obtain your key from Google AI Studio.")
    st.divider()
    st.info(
        "**Developed By:**"
        " Saqib Sherwani for Panacea Smart Solutions.",
        icon="ℹ️"
    )
    st.warning(
        "**Security:** Avoid sharing your API key. For deployed applications, consider more secure methods like environment variables.",
        icon="🔒"
    )


# Main Panel
st.header("📄 Document Upload & Analysis")

# Initialize model variable
model = None

# Only proceed if API key is entered
if api_key_input:
    # Attempt to load/initialize the model (uses caching)
    model = load_gemini_model(api_key_input)

    if model: # Only show uploader if model initialized successfully
        uploaded_file = st.file_uploader(
            "Upload your Medical Document (PDF, TXT, PNG, JPG)",
            type=["pdf", "txt", "png", "jpg", "jpeg"],
            accept_multiple_files=False,
            key="file_uploader"
        )

        if uploaded_file is not None:
            file_name = uploaded_file.name
            file_content_bytes = uploaded_file.getvalue()
            st.success(f"Uploaded file: '{file_name}' ({len(file_content_bytes)} bytes)", icon="✅")

            # Determine MIME type
            mime_type, _ = mimetypes.guess_type(file_name)
            if not mime_type:
                if file_name.lower().endswith('.pdf'): mime_type = 'application/pdf'
                elif file_name.lower().endswith('.txt'): mime_type = 'text/plain'
                elif file_name.lower().endswith('.png'): mime_type = 'image/png'
                elif file_name.lower().endswith(('.jpg', '.jpeg')): mime_type = 'image/jpeg'
                else: mime_type = 'application/octet-stream'
                st.warning(f"Could not auto-detect MIME type, assuming: {mime_type}", icon="⚠️")
            else:
                st.info(f"Detected MIME Type: {mime_type}", icon="ℹ️")


            # Button to trigger analysis
            if st.button("Analyze Document & Suggest Codes", key="analyze_button"):
                with st.spinner("🧠 MedCoder AI is thinking... Analyzing document..."):
                    # Call the function to get codes
                    suggested_codes_text = get_medical_codes_from_file(model, file_content_bytes, mime_type)

                st.divider()
                st.subheader("📊 Suggested Medical Codes")

                if suggested_codes_text:
                    # Display the raw markdown result from Gemini
                    st.markdown(suggested_codes_text)
                    st.info("Note: Please review these AI-generated suggestions carefully with a certified coder.", icon="🧑‍💻")
                else:
                    st.error("Failed to retrieve coding suggestions. Please check the error messages above or try again.", icon="❌")
    # else:
        # Error during model initialization is handled by load_gemini_model

else:
    st.info("Please enter your Gemini API Key in the sidebar to begin.", icon="👈")

# --- Footer ---
st.divider()
st.caption("MedCoder AI © 2025 | Panacea Smart Solutions | AI-Assisted Coding Tool - Requires Human Verification")