## 1️⃣ `mean` — Average Intensity
### 🔍 What it is:
The **weighted average** of all intensity values, where weights are the histogram counts.

### 💡 Intuition:
> "If I randomly pick a pixel from this channel, what intensity do I expect?"

### 🍎 Why it matters for disease detection:
| Disease | Typical Mean Shift |
|---------|-------------------|
| **Healthy leaf** | Balanced green/red means |
| **Apple scab** | Darker spots → lower mean in Lightness channel |
| **Rust** | Orange/brown lesions → higher mean in Red, lower in Green |
| **Black rot** | Very dark necrotic areas → low mean across channels |

---

## 2️⃣ `std` — Standard Deviation (Contrast)
### 🔍 What it is:
Measures how **spread out** the pixel intensities are around the mean. High std = high contrast.


### 💡 Intuition:
> "How much do pixel values vary? Is the image flat/grayscale-ish, or does it have strong lights AND darks?"

### 🍎 Why it matters:
| Scenario | std Value | Interpretation |
|----------|-----------|---------------|
| Uniform healthy skin | Low (~15-30) | Smooth color distribution |
| Spotty disease | High (~50-80) | Mix of healthy tissue + dark lesions + bright reflections |
| Overexposed photo | Very high | Technical artifact, not biological |

---

## 3️⃣ `peak_pos` — Bin with Highest Frequency
### 🔍 What it is:
The **intensity value** (0–255) that appears most often in the channel.

### 💡 Intuition:
> "What's the *most common* color/brightness in this channel?"

### 🍎 Why it matters:
Diseases often introduce **new dominant colors**:
- Healthy apple skin → peak near green (~80-120 in Green channel)
- Scab lesions → peak shifts darker (~30-60 in Lightness)
- Rust → peak appears in orange range (~150-180 in Red, ~30-50 in Blue)

⚠️ **Note**: `peak_pos` is the *intensity value*, not the count. It answers "what color is dominant?", not "how dominant is it?" (that's `peak_val`).

---

## 4️⃣ `peak_val` — Normalized Height of the Peak
### 🔍 What it is:
The **proportion of pixels** that have the `peak_pos` intensity (after normalizing the histogram to sum to 1).

### 💡 Intuition:
> "How *concentrated* is the dominant color? Is it a sharp spike or a broad hill?"

### 🍎 Why it matters:
| `peak_val` | Interpretation | Disease Relevance |
|------------|----------------|-------------------|
| **High (0.1–0.4)** | One color dominates strongly | Healthy, uniform tissue |
| **Medium (0.03–0.1)** | Moderate concentration | Early/mild disease |
| **Low (<0.03)** | Colors are very mixed | Advanced disease, multiple symptom types |


🔑 **Key insight**: `peak_pos` tells you *what* is dominant; `peak_val` tells you *how confidently* it's dominant.

---

## 5️⃣ `entropy` — Texture / Randomness of Distribution
### 🔍 What it is:
A measure from information theory: **how unpredictable** the pixel intensities are. High entropy = more randomness/complexity.

### 🍎 Why it matters for disease:
| Condition | Entropy | Why |
|-----------|---------|-----|
| **Healthy, uniform skin** | Low (~4–6 bits) | Predictable color distribution |
| **Early disease (small spots)** | Medium (~6–7 bits) | Mostly healthy + few anomalies |
| **Advanced disease** | High (~7–8 bits) | Complex mix: healthy tissue, necrosis, mold, discoloration, shadows |