import streamlit as st
import cv2
import easyocr
import numpy as np
import re
from PIL import Image

# ===================== BRANDING =====================
APP_TITLE = "Indian ANPR System"
COMPANY = "EquinoxSphere"

# ===================== STATE / RTO DATA =====================
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

# ===================== OCR LOADER =====================
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# ===================== OCR HELPERS =====================
def normalize_text(text: str) -> str:
    text = text.upper().replace(" ", "").replace("-", "")
    rep = {'O':'0','Q':'0','I':'1','L':'1','Z':'2','S':'5','B':'8'}
    return ''.join(rep.get(c, c) for c in text)

def extract_plate(text: str):
    text = normalize_text(text)
    m = re.search(r'[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{3,4}', text)
    return m.group() if m else None

def get_state_district(plate: str):
    state_code = plate[:2]
    rto_code = plate[2:4]
    state = STATE_CODE_MAP.get(state_code, "Unknown State")
    district = RJ_RTO_MAP.get(rto_code, "Unknown District") if state_code=="RJ" else "N/A"
    return state, district

# ===================== PLATE DETECTION =====================
def detect_plate_boxes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if 2.0 < (w / float(h)) < 6.5 and w > 80 and h > 25:
            boxes.append((x,y,w,h))
    return boxes

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="CAR",
    layout="centered"
)

# ===================== HEADER =====================
st.markdown(
    f"""
    <h1 style='text-align:center;'> {APP_TITLE}</h1>
    <p style='text-align:center; color:gray;'>
        AI-powered Automatic Number Plate Recognition<br>
        <b>Built by {COMPANY}</b>
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ===================== INPUT SECTION =====================
st.subheader("Input Method")

mode = st.radio(
    "Select how you want to provide the vehicle image:",
    ["Take Photo", "Upload Photo"],
    horizontal=True
)

image = None

if mode == "Take Photo":
    cam = st.camera_input("Capture a clear image of the vehicle number plate")
    if cam:
        image = cv2.cvtColor(np.array(Image.open(cam)), cv2.COLOR_RGB2BGR)

else:
    up = st.file_uploader(
        "Upload a vehicle image (JPG / PNG)",
        type=["jpg", "jpeg", "png"]
    )
    if up:
        image = cv2.cvtColor(np.array(Image.open(up)), cv2.COLOR_RGB2BGR)

# ===================== ACTION =====================
st.markdown(" Detection")

if image is not None:
    detect_btn = st.button("Run ANPR Detection", use_container_width=True)

    if detect_btn:
        output = image.copy()
        results = set()

        boxes = detect_plate_boxes(image)

        for (x,y,w,h) in boxes:
            pad = 10
            plate_img = image[
                max(0,y-pad):min(image.shape[0],y+h+pad),
                max(0,x-pad):min(image.shape[1],x+w+pad)
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

                    cv2.rectangle(output,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.putText(
                        output,
                        f"{plate} | {state} | {district}",
                        (x, max(20,y-10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,(0,255,0),2
                    )

        # ===================== RESULT =====================
        st.markdown(" Result")
        st.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), use_container_width=True)

        if results:
            st.success("Vehicle details detected successfully")
            for p,s,d in results:
                st.markdown(f"""
                ** Vehicle Number:** `{p}`  
                ** State:** {s}  
                ** District / RTO:** {d}
                ---
                """)
        else:
            st.warning("No valid Indian number plate detected in the image.")

else:
    st.info("Please provide an image to start detection.")

# ===================== FOOTER =====================
st.markdown(
    f"""
    <hr>
    <p style='text-align:center; color:gray; font-size:13px;'>
        © {COMPANY} · AI Vision Solutions<br>
        Indian Automatic Number Plate Recognition System
    </p>
    """,
    unsafe_allow_html=True
)
