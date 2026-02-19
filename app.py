import cv2
import numpy as np
import streamlit as st
from PIL import Image

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Signature Verification (ORB + OpenCV)",
    page_icon="🖊️",
    layout="wide"
)

# -------------------------
# Styling
# -------------------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

      .stButton button { width: 100%; border-radius: 12px; }

      .card {
        padding: 16px;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.12);
        background: rgba(255,255,255,0.03);
      }

      .big { font-size: 1.12rem; font-weight: 700; margin: 0; }
      .muted { opacity: 0.78; font-size: 0.92rem; margin-top: 2px; }

      .pill {
        display:inline-flex;
        align-items:center;
        gap:8px;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.14);
        background: rgba(255,255,255,0.06);
        margin-right: 8px;
        font-size: 0.92rem;
      }

      .row-gap { margin-top: 10px; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🖊️ Signature Verification System")
st.caption("Python • OpenCV • ORB • BFMatcher • RANSAC • Streamlit UI")

# -------------------------
# Helpers
# -------------------------
def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def safe_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img

def auto_crop_binary(bin_img: np.ndarray, pad: int = 12) -> np.ndarray:
    """
    bin_img: 0 background, 255 strokes
    Crops around strokes with padding.
    """
    ys, xs = np.where(bin_img > 0)
    if len(xs) == 0 or len(ys) == 0:
        return bin_img

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(bin_img.shape[1] - 1, x2 + pad)
    y2 = min(bin_img.shape[0] - 1, y2 + pad)

    return bin_img[y1:y2 + 1, x1:x2 + 1]

def center_on_canvas(bin_img: np.ndarray, size=(420, 900)) -> np.ndarray:
    """
    Places cropped signature in the center of a fixed canvas.
    """
    H, W = size
    canvas = np.zeros((H, W), dtype=np.uint8)

    h, w = bin_img.shape[:2]
    if h == 0 or w == 0:
        return canvas

    # Scale to fit while keeping aspect ratio
    scale = min((W - 20) / max(1, w), (H - 20) / max(1, h))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(bin_img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Center
    y0 = (H - new_h) // 2
    x0 = (W - new_w) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas

def skeletonize(bin_img: np.ndarray) -> np.ndarray:
    """
    Morphological skeletonization on binary strokes (255=ink).
    Helps reduce pen thickness variance.
    """
    img = (bin_img > 0).astype(np.uint8) * 255
    skel = np.zeros(img.shape, np.uint8)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        if cv2.countNonZero(img) == 0:
            done = True

    return skel

def preprocess_signature(img_bgr: np.ndarray, use_skeleton: bool = True):
    """
    Robust signature preprocessing:
      - grayscale + contrast normalization (CLAHE)
      - denoise
      - adaptive threshold (handles uneven lighting)
      - morphological cleanup
      - crop strokes
      - normalize on fixed canvas
      - optional skeletonization
    Returns:
      processed_bin (255 strokes)
      debug dict with steps
    """
    dbg = {}

    gray = to_gray(img_bgr)
    dbg["gray"] = gray

    # Contrast normalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(gray)
    dbg["clahe"] = norm

    # Denoise while preserving edges
    den = cv2.fastNlMeansDenoising(norm, None, h=12, templateWindowSize=7, searchWindowSize=21)
    dbg["denoise"] = den

    # Adaptive threshold (signature strokes dark -> invert so strokes become white)
    th = cv2.adaptiveThreshold(
        den, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41, 9
    )
    dbg["th_adapt"] = th

    # Remove tiny noise
    kernel = np.ones((2, 2), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    dbg["morph"] = th

    cropped = auto_crop_binary(th, pad=14)
    dbg["cropped"] = cropped

    normalized = center_on_canvas(cropped, size=(420, 900))
    dbg["normalized"] = normalized

    if use_skeleton:
        sk = skeletonize(normalized)
        dbg["skeleton"] = sk
        return sk, dbg

    return normalized, dbg

def match_orb_ransac(bin1: np.ndarray, bin2: np.ndarray,
                     nfeatures: int = 2000,
                     ratio: float = 0.75,
                     top_k_viz: int = 80,
                     ransac_thresh: float = 3.0):
    """
    Improved matching:
      - ORB on normalized binary/skeleton images
      - KNN + Lowe ratio
      - symmetry check (mutual matches)
      - RANSAC homography to validate geometry
    Score uses inlier ratio + match strength.
    """
    orb = cv2.ORB_create(
        nfeatures=nfeatures,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        fastThreshold=8
    )

    k1, d1 = orb.detectAndCompute(bin1, None)
    k2, d2 = orb.detectAndCompute(bin2, None)

    if d1 is None or d2 is None or len(k1) < 12 or len(k2) < 12:
        return {
            "ok": False,
            "reason": "Not enough keypoints detected. Use clearer images or adjust preprocessing.",
            "k1": k1, "k2": k2, "good": [], "inliers": 0, "score": 0.0, "viz": None
        }

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Forward KNN
    knn12 = bf.knnMatch(d1, d2, k=2)
    good12 = []
    for m, n in knn12:
        if m.distance < ratio * n.distance:
            good12.append(m)

    # Backward KNN (symmetry check)
    knn21 = bf.knnMatch(d2, d1, k=2)
    good21 = []
    for m, n in knn21:
        if m.distance < ratio * n.distance:
            good21.append(m)

    # Mutual matches only (more stable)
    back_map = {(m.trainIdx, m.queryIdx): m for m in good21}
    mutual = []
    for m in good12:
        if (m.queryIdx, m.trainIdx) in back_map:
            mutual.append(m)

    if len(mutual) < 10:
        return {
            "ok": False,
            "reason": "Too few reliable matches after symmetry check. Try increasing ORB features or improving image quality.",
            "k1": k1, "k2": k2, "good": mutual, "inliers": 0, "score": 0.0, "viz": None
        }

    # RANSAC homography
    src_pts = np.float32([k1[m.queryIdx].pt for m in mutual]).reshape(-1, 1, 2)
    dst_pts = np.float32([k2[m.trainIdx].pt for m in mutual]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    if mask is None:
        inliers = 0
        inlier_ratio = 0.0
    else:
        inliers = int(mask.sum())
        inlier_ratio = inliers / max(1, len(mutual))

    # Distance quality (lower is better)
    mutual_sorted = sorted(mutual, key=lambda x: x.distance)
    show = mutual_sorted[:max(10, min(top_k_viz, len(mutual_sorted)))]

    # Score: mostly inlier_ratio, slightly match volume normalized
    # This is more stable than (good / keypoints).
    volume = min(1.0, len(mutual) / 120.0)  # saturate after ~120 mutual matches
    score = (0.78 * inlier_ratio + 0.22 * volume) * 100.0
    score = float(np.clip(score, 0.0, 100.0))

    viz = cv2.drawMatches(
        bin1, k1,
        bin2, k2,
        show, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return {
        "ok": True,
        "reason": None,
        "k1": k1,
        "k2": k2,
        "good": mutual,
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "H": H,
        "score": score,
        "viz": viz,
        "shown": show
    }

# -------------------------
# Sidebar Controls
# -------------------------
with st.sidebar:
    st.subheader("⚙️ Settings")

    nfeatures = st.slider("ORB features", 500, 5000, 2200, 100)
    ratio = st.slider("Lowe ratio test", 0.55, 0.90, 0.75, 0.01)
    top_k = st.slider("Show top matches", 20, 200, 90, 5)

    st.markdown("---")
    use_skeleton = st.toggle("Use skeletonization (recommended)", value=True)
    ransac_thresh = st.slider("RANSAC reprojection threshold", 1.0, 8.0, 3.0, 0.5)

    st.markdown("---")
    decision_threshold = st.slider("Decision threshold (score %)", 1, 100, 35, 1)

    st.markdown("---")
    st.markdown("**Notes**")
    st.markdown("- Best results: clean scans or good contrast photos.")
    st.markdown("- Threshold depends on your dataset. Tune with real samples.")

# -------------------------
# Upload UI
# -------------------------
colA, colB = st.columns([1, 1], gap="large")

with colA:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="big">Upload Reference Signature</div><div class="muted">Known genuine signature</div>', unsafe_allow_html=True)
    ref_file = st.file_uploader("Reference image", type=["png", "jpg", "jpeg"], key="ref")
    st.markdown("</div>", unsafe_allow_html=True)

with colB:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="big">Upload Test Signature</div><div class="muted">Signature to verify</div>', unsafe_allow_html=True)
    test_file = st.file_uploader("Test image", type=["png", "jpg", "jpeg"], key="test")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# Previews
# -------------------------
st.markdown('<div class="row-gap"></div>', unsafe_allow_html=True)
p1, p2 = st.columns(2, gap="large")

ref_bin = test_bin = None
ref_dbg = test_dbg = None
ref_pil = test_pil = None

if ref_file:
    ref_pil = Image.open(ref_file)
    ref_bgr = pil_to_cv2(ref_pil)
    ref_bin, ref_dbg = preprocess_signature(ref_bgr, use_skeleton=use_skeleton)

if test_file:
    test_pil = Image.open(test_file)
    test_bgr = pil_to_cv2(test_pil)
    test_bin, test_dbg = preprocess_signature(test_bgr, use_skeleton=use_skeleton)

with p1:
    st.subheader("Reference Preview")
    if ref_pil is not None:
        st.image(ref_pil)
        with st.expander("Preprocessing output"):
            st.caption("Processed (normalized)")
            st.image(ref_dbg["normalized"], clamp=True)
            if use_skeleton:
                st.caption("Skeleton")
                st.image(ref_dbg["skeleton"], clamp=True)
    else:
        st.info("Upload a reference signature to preview.")

with p2:
    st.subheader("Test Preview")
    if test_pil is not None:
        st.image(test_pil)
        with st.expander("Preprocessing output"):
            st.caption("Processed (normalized)")
            st.image(test_dbg["normalized"], clamp=True)
            if use_skeleton:
                st.caption("Skeleton")
                st.image(test_dbg["skeleton"], clamp=True)
    else:
        st.info("Upload a test signature to preview.")

st.markdown("---")

# -------------------------
# Action
# -------------------------
run = st.button("✅ Verify Signature", type="primary", disabled=not (ref_bin is not None and test_bin is not None))

if run:
    with st.spinner("Extracting ORB features, matching, and validating geometry..."):
        res = match_orb_ransac(
            safe_uint8(ref_bin),
            safe_uint8(test_bin),
            nfeatures=nfeatures,
            ratio=ratio,
            top_k_viz=top_k,
            ransac_thresh=ransac_thresh
        )

    if not res["ok"]:
        st.error(res["reason"])
    else:
        score = res["score"]
        verdict = "MATCH ✅" if score >= decision_threshold else "NOT A MATCH ❌"

        # Results row
        r1, r2, r3, r4 = st.columns([1.2, 1.2, 1.6, 2.0], gap="large")

        with r1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<span class="pill">Score</span> <span class="big">{score:.2f}%</span>', unsafe_allow_html=True)
            st.progress(min(1.0, score / 100.0))
            st.markdown("</div>", unsafe_allow_html=True)

        with r2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<span class="pill">Verdict</span> <span class="big">{verdict}</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="muted">Threshold: {decision_threshold}%</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with r3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                f"""
                <span class="pill">Keypoints</span> Ref: <b>{len(res["k1"])}</b> &nbsp;&nbsp; Test: <b>{len(res["k2"])}</b><br>
                <span class="pill">Mutual matches</span> <b>{len(res["good"])}</b>
                """,
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with r4:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                f"""
                <span class="pill">RANSAC inliers</span> <b>{res["inliers"]}</b><br>
                <span class="pill">Inlier ratio</span> <b>{res["inlier_ratio"]*100:.1f}%</b><br>
                <div class="muted">RANSAC threshold: {ransac_thresh:.1f}px</div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("🔍 Match Visualization")
        st.image(bgr_to_rgb(res["viz"]))

        with st.expander("How the score is computed"):
            st.write(
                "Score is primarily based on RANSAC inlier ratio (geometry-consistent matches), "
                "with a smaller contribution from mutual match volume. This is much more stable than "
                "raw match count or match/keypoints alone."
            )

# -------------------------
# Accuracy reality check (important)
# -------------------------
with st.expander("About accuracy (important)"):
    st.write(
        "If you need ~99% accuracy across real-world variations, classical ORB matching usually won’t be enough. "
        "You typically need a trained model (Siamese/Triplet network) on a signature dataset and proper evaluation. "
        "This app is a strong classical baseline, but results depend heavily on image quality and user variance."
    )