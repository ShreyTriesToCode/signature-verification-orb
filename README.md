# 🖊️ Signature Verification System (Streamlit + OpenCV ORB)

A clean, interactive **signature verification** web app built with **Streamlit** and **OpenCV**.  
Upload a **reference (genuine) signature** and a **test signature**, then verify using **ORB feature extraction + BFMatcher** with tunable parameters and match visualization.

---

## ✨ Features

- **Two-image verification**: Reference vs Test signature
- **Robust preprocessing**: grayscale → blur → Otsu threshold → morphology cleanup
- **ORB keypoints + descriptors** for signature texture/shape matching
- **BFMatcher + Lowe’s ratio test** to filter good matches
- **Score + verdict** (threshold-based)
- **Match visualization** showing top matched keypoints
- **Modern UI** (cards, sidebar controls, previews)

---

## 🧠 How it works (High-level)

1. **Preprocessing**
   - Convert to grayscale  
   - Apply blur (noise reduction)  
   - Otsu threshold (binary separation)  
   - Morphological opening (remove small artifacts)

2. **Feature Extraction**
   - ORB detects keypoints and computes binary descriptors.

3. **Matching**
   - BFMatcher finds candidate matches.
   - Lowe’s ratio test removes ambiguous matches.

4. **Scoring**
   - Score is computed using good matches normalized by detected keypoints.
   - Final verdict is based on a configurable threshold.

---


