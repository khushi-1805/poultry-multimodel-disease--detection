# 🐥 Multimodal Poultry Disease Detection System  
**Computer Vision · Audio Analysis · Posture Keypoints**

This project implements a **research-grade multimodal framework** for detecting poultry diseases by integrating **three complementary sensing modalities**:

- **Image-based detection** (YOLOv10 + Xception)  
- **Posture analysis** (keypoint detection + MLP classifier)  
- **Audio signal classification** (CRNN + Bi-GRU)

The goal is to build a **robust, real-time health monitoring pipeline** that overcomes the limitations of single-modality detection used in traditional poultry farms.

---

## 🚀 Research Motivation  
Poultry farms often rely on manual inspection, which is inconsistent and unable to detect early symptoms.  
Diseases can manifest as:

- Abnormal posture  
- Changes in appearance  
- Distressed vocal patterns  

A **multimodal AI approach** captures all three, dramatically improving detection accuracy and reducing false positives.

---

## 🔬 System Architecture

### **1️⃣ Computer Vision Module (Images)**
- Used **YOLOv10** for bird detection & localization  
- Extracted visual features with **Xception CNN**  
- Focused on changes in:  
  - feather texture  
  - swelling  
  - discoloration  
  - drooping wings  
- Outputs a probability score for visual disease indicators  

---

### **2️⃣ Posture Analysis Module (Keypoints)**
- Used **pose estimation** to extract keypoints:  
  - head, beak, neck, wings, legs  
- Designed a feature representation capturing:  
  - symmetry  
  - limb angle deviations  
  - abnormal bending  
- Trained an **MLP classifier** on keypoint vectors to detect posture deviation patterns linked to diseases  
- Helps detect early-stage symptoms not visible in images  

---

### **3️⃣ Audio Classification Module (Vocal Patterns)**
- Collected raw poultry audio in farm-like environments  
- Converted audio to **mel-spectrograms**  
- Used a **CRNN (Convolutional Recurrent Neural Network)**  
  + **Bi-GRU** layers to capture:  
  - distress calls  
  - coughing  
  - irregular breathing  
- Provides another independent signal for disease detection  

---

## 🤝 **4️⃣ Multimodal Fusion (Combined Model)**
The three independent modules are merged into a **unified prediction layer** to improve:

- Accuracy  
- Reliability  
- Noise robustness  
- Early detection capability  

Fusion approach (planned):  
- Weighted averaging of modality scores  
- Late-stage feature concatenation  
- Evaluation of ablation settings (single vs multi-modality)

---

## 📊 Evaluation Strategy
- Precision / Recall / F1-Score for each modality  
- Comparison of **single-modality vs multimodal performance**  
- Noise-robustness testing for audio  
- Real-farm simulation (planned)

---

## 🛠 Tech Stack  
- **Python**  
- **TensorFlow / Keras**  
- **YOLOv10**  
- **Xception CNN**  
- **MediaPipe / Pose estimation**  
- **CRNN + Bi-GRU**  
- **Librosa**  
- **NumPy, Pandas, Matplotlib**

---

## 📁 Project Structure
```
poultry-disease-detection/
│── images/                 # CV dataset
│── audio/                  # raw audio recordings
│── spectrograms/           # processed mel-spectrograms
│── posture/                # keypoint extraction outputs
│── models/                 # saved model weights
│── notebooks/              # research experiments
└── src/                    # multimodal pipeline code
```

---

## 🧪 Current Progress
- ✔ Image dataset processed & YOLOv10 detection pipeline working  
- ✔ Keypoint extraction for posture completed  
- ✔ Audio → spectrogram conversion done  
- ✔ Baseline models trained for each modality  
- ⏳ Multimodal fusion layer being developed  
- ⏳ Real-farm simulation tests planned  

---

## 🎯 Research Goal  
To develop a **low-cost, scalable**, and **real-time** poultry health monitoring system for farms, enabling:

- Early disease detection  
- Reduced mortality  
- Automated monitoring  
- Improved accuracy over single-modality systems  

---

## 📬 Contact  
**LinkedIn:** www.linkedin.com/in/khushi-kalinge-a250212aa  
**Email:** khushikalinge20@gmail.com
