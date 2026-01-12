import streamlit as st
import cv2
import easyocr
import numpy as np
import re
from PIL import Image

# =====================================================
# BRANDING
# =====================================================
APP_TITLE = "Indian ANPR System"
COMPANY = "EquinoxSphere"

# =====================================================
# STATE & RTO DATA
# =====================================================
STATE_CODE_MAP = {
    "RJ": "Rajasthan", "DL": "Delhi", "MH": "Maharashtra",
    "UP": "Uttar Pradesh", "HR": "Haryana", "GJ": "Gujarat",
    "PB": "Punjab", "TN": "Tamil Nadu", "KA": "Karnataka"
}

RJ_RTO_MAP = {
    "01": "Ajmer", "14": "Jaipur North", "15": "Jaipur South",
    "16": "Jaisalmer", "18": "Jhunjhunu", "19": "Jodhpur",
    "20": "Kota", "27": "Udaipur"
}

# =====================================================
# OCR LOADER
# =====================================================
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# =====================================================
# IMAGE AUTO ROTATION (CRITICAL)
# =====================================================
def auto_rotate(img):
    h, w = img.shape[:2]
    if h > w:
        return img
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

# =====================================================
# TEXT NORMALIZATION + PLATE EXTRACTION
# =====================================================
def extract_plate(text):
    text = text.upper()
    text = text.replace(" ", "").replace("-", "")
    text = text.replace("IND", "")  # remove IND logo text

    rep = {
        'O': '0', 'Q': '0',
        'I': '1', 'L': '1',
        'Z': '2', 'S': '5', 'B': '8'
    }
    text = ''.join(rep.get(c, c) for c in text)

    match = re.search(r'[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{3,4}', text)
    return match.group() if match else None

def get_state_district(plate):
    state_code = plate[:2]
    rto = plate[2:4]
    state = STATE_CODE_MAP.get(state_code, "Unknown State")
    district = RJ_RTO_MAP.get(rto, "Unknown District") if state_code == "RJ" else "N/A"
    return state, district

# =====================================================
# PLATE DETECTION (OPENCV)
# =====================================================
def detect_plate_boxes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 2.0 < (w / float(h)) < 6.5 and w > 80 and h > 25:
            boxes.append((x, y, w, h))
    return boxes

# =====================================================
# STREAMLIT UI
# =====================================================
st.set_page_config(page_title=APP_TITLE, page_icon="🚗", layout="centered")

st.markdown(f"""
<h1 style="text-align:center;">🚗 {APP_TITLE}</h1>
<p style="text-align:center;color:gray;">
AI-powered Automatic Number Plate Recognition<br>
<b>Built by {COMPANY}</b>
</p>
<hr>
""", unsafe_allow_html=True)

mode = st.radio(
    "Choose input method",
    ["📸 Take Photo", "🖼️ Upload Photo"],
    horizontal=True
)

image = None

if mode == "📸 Take Photo":
    cam = st.camera_input("Capture vehicle number plate")
    if cam:
        image = cv2.cvtColor(np.array(Image.open(cam)), cv2.COLOR_RGB2BGR)
        image = auto_rotate(image)

else:
    up = st.file_uploader("Upload vehicle image", type=["jpg", "jpeg", "png"])
    if up:
        image = cv2.cvtColor(np.array(Image.open(up)), cv2.COLOR_RGB2BGR)
        image = auto_rotate(image)

# =====================================================
# RUN ANPR
# =====================================================
if image is not None and st.button("🔍 Run ANPR Detection", use_container_width=True):

    output = image.copy()
    results = set()

    # ---- Try plate detection first ----
    boxes = detect_plate_boxes(image)

    for (x, y, w, h) in boxes:
        pad = 10
        plate_img = image[
            max(0, y - pad):min(image.shape[0], y + h + pad),
            max(0, x - pad):min(image.shape[1], x + w + pad)
        ]

        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        texts = (
            reader.readtext(plate_img, detail=0) +
            reader.readtext(gray, detail=0) +
            reader.readtext(th, detail=0)
        )

        for t in texts:
            plate = extract_plate(t)
            if plate:
                state, district = get_state_district(plate)
                results.add((plate, state, district))
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    output,
                    f"{plate} | {state} | {district}",
                    (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2
                )

    # ---- FALLBACK: FULL IMAGE OCR ----
    if not results:
        full_texts = reader.readtext(image, detail=0)
        for t in full_texts:
            plate = extract_plate(t)
            if plate:
                state, district = get_state_district(plate)
                results.add((plate, state, district))

    st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB),
             caption="ANPR Result", use_container_width=True)

    if results:
        st.success("Vehicle detected successfully")
        for p, s, d in results:
            st.markdown(f"""
            **🚗 Vehicle Number:** `{p}`  
            **📍 State:** {s}  
            **🏙️ District / RTO:** {d}
            ---
            """)
    else:
        st.warning("No valid Indian number plate detected.")

st.markdown(f"""
<hr>
<p style="text-align:center;color:gray;font-size:13px;">
© {COMPANY} – AI Vision Solutions
</p>
""", unsafe_allow_html=True)
