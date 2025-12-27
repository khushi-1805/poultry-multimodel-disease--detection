import os
import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from PIL import Image
import pickle

# ------------------------------
# FILE PATHS
# ------------------------------
AUDIO_MODEL_PATH = "audio_model.h5"
AUDIO_ENCODER_PATH = "label_encoder_audio.pkl"

DISEASE_MODEL_PATH = "xception_disease.h5"
# Use a class order list instead of old pickle wrapper
DISEASE_CLASSES = ['Bumblefoot', 'Fowlpox', 'Healthy', 'Unlabeled', 
                   'cocci', 'coryza', 'crd', 'ncd', 'salmo']

POSTURE_MODEL_PATH = "posture_mlp_adv.h5"
POSTURE_SCALER_PATH = "label_encoder_posture.pkl"

# ------------------------------
# LOAD MODELS
# ------------------------------
audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH)
disease_model = tf.keras.models.load_model(DISEASE_MODEL_PATH)

with open(AUDIO_ENCODER_PATH, "rb") as f:
    audio_encoder = pickle.load(f)

# ------------------------------
# AUDIO FEATURE EXTRACTION
# ------------------------------
def extract_audio_features(audio_path, max_pad_len=130):
    y, sr = librosa.load(audio_path, sr=22050)
    
    # Mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64)
    
    # Pad/truncate MFCC and Mel to max_pad_len frames
    mel_db = librosa.util.fix_length(mel_db, size=max_pad_len, axis=1)
    mfcc = librosa.util.fix_length(mfcc, size=max_pad_len, axis=1)
    
    # Stack as 2 channels
    combined = np.stack([mel_db, mfcc], axis=-1)  # shape: (64, 130, 2)
    
    # Add batch dimension
    combined = np.expand_dims(combined, axis=0)  # shape: (1, 64, 130, 2)
    
    return combined




# ------------------------------
# IMAGE PREPROCESSING
# ------------------------------
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((299, 299))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ------------------------------
# STREAMLIT UI
# ------------------------------
st.title("🐔 Poultry AI Health System")
st.subheader("Analyze Chicken Health via Audio + Image")

option = st.sidebar.radio(
    "Choose Feature",
    ["Audio Health Analysis", "Disease Detection", "Posture Analysis"]
)

# -----------------------------------------------------------
# 🎤 AUDIO CLASSIFICATION
# -----------------------------------------------------------
if option == "Audio Health Analysis":
    st.header("🔊 Audio-based Health Detection")
    audio_file = st.file_uploader("Upload Chicken Audio", type=["wav", "mp3"])
    AUDIO_CLASSES = ['healthy', 'unhealthy', 'unlabeled']

    if audio_file:
        st.audio(audio_file)
        features = extract_audio_features(audio_file)
        preds = audio_model.predict(features)[0]
        idx = np.argmax(preds)
        label = AUDIO_CLASSES[idx]

        st.success(f"### 🐣 Prediction: **{label}**")
        st.write("#### Class Probabilities:")
        for cls, prob in zip(AUDIO_CLASSES, preds):
            st.write(f"- **{cls}**: {prob:.2f}")

# -----------------------------------------------------------
# 🦠 DISEASE DETECTION
# -----------------------------------------------------------
# -----------------------------------------------------------
# 🦠 DISEASE DETECTION
# -----------------------------------------------------------
elif option == "Disease Detection":
    st.header("🩺 Disease Detection (Image Input)")
    uploaded_image = st.file_uploader("Upload Poultry Image", type=["jpg", "jpeg", "png"])

    # Disease info mapping
    DISEASE_INFO = {
    'Bumblefoot (Plantar Pododermatitis)': {
        'Cause': 'Bacterial infection (usually Staphylococcus) of foot pads.',
        'Symptoms': 'Swollen, pus-filled abscess under foot; hard scab; limping; reluctance to walk. Heavy breeds/males prone.',
        'Treatment': {
            'Early': 'Foot soak (warm water + Epsom salts), remove scab, flush with antiseptic, apply antibiotic ointment, bandage.',
            'Severe': 'Systemic antibiotics (vet-prescribed), surgical debridement if deep infection.'
        },
        'Home Remedies': 'Foot soaks, topical honey/antibiotic ointment, soft bedding.',
        'Prevention': 'Clean/dry coops, cushioned litter, low smooth perches, trim nails, good nutrition.',
        'Contagiousness': 'Not bird-to-bird contagious; pus is infectious.',
        'Quarantine': 'Isolate severely affected birds; intensive care pen; disinfect equipment.'
    },
    'Fowlpox': {
        'Cause': 'Viral; two forms – Dry (cutaneous) and Wet (diphtheritic).',
        'Symptoms': 'Dry: Wart-like scabs on unfeathered skin (comb, wattles, eyelids, legs). Wet: Plaques in mouth/throat/trachea, difficulty swallowing/breathing. Other: Swollen eyes, reduced appetite, weight loss, drop in egg production.',
        'Treatment': 'Supportive care, clean lesions, wound dressings, nutrition/fluids. Broad-spectrum antibiotics for secondary infections. Vitamin A sometimes used.',
        'Home Remedies': 'Dilute iodine, ointments, probiotics or apple cider vinegar (supportive, anecdotal).',
        'Prevention': 'Vaccination (wing-web stab at 12–16 weeks), mosquito control, biosecurity.',
        'Contagiousness': 'Contagious within flock; transmitted by insects and contact with infected birds/scabs.',
        'Quarantine': 'Isolate sick birds, vaccinate healthy birds, disinfect coops, waterers, and feeders.'
    },
    'cocci': {
        'Cause': 'Protozoan (Eimeria) infection of intestines.',
        'Symptoms': 'Diarrhea (bloody), dehydration, anemia, weight loss, lethargy, ruffled feathers. Severe outbreaks can be fatal.',
        'Treatment': 'Medications: Amprolium (Corid), Toltrazuril; antibiotics to prevent secondary infections. Supportive care: Fluids, electrolytes, warmth, easy-to-eat feed.',
        'Home Remedies': 'Probiotics, fermented feed, herbs (supportive only).',
        'Prevention': 'Dry/clean litter, avoid overcrowding, anticoccidial coccidiostats in feed, live vaccines for chicks.',
        'Contagiousness': 'Highly contagious; spreads via oocysts in droppings.',
        'Quarantine': 'Isolate sick birds, disinfect housing/feeders/waterers, rotate/replace litter, quarantine new birds.'
    },
    'coryza': {
        'Cause': 'Bacterial (Avibacterium paragallinarum), acute upper respiratory disease.',
        'Symptoms': 'Swollen puffy face, foul nasal/eye discharge, sneezing, rattling sounds, reduced egg production. Mortality 20–50%.',
        'Treatment': 'Antibiotics (erythromycin, oxytetracycline, tylosin) under vet guidance; supportive care.',
        'Home Remedies': 'Improve ventilation, reduce dust/ammonia, vitamins/probiotics (supportive only).',
        'Prevention': 'Vaccination at 8–16 weeks with booster, quarantine new birds, strict sanitation, minimize stress.',
        'Contagiousness': 'Highly contagious via direct contact/airborne droplets; recovered birds may be carriers.',
        'Quarantine': 'Isolate affected birds, treat flock, cull/separate carriers, disinfect premises.'
    },
    'crd': {
        'Cause': 'Bacterial (Mycoplasma gallisepticum), upper respiratory infection.',
        'Symptoms': 'Coughing, sneezing, nasal discharge, tracheal rales, conjunctivitis, gasping. Mild in adults, severe in chicks.',
        'Treatment': 'Antibiotics (tylosin, doxycycline, tiamulin, lincomycin/spectinomycin) under vet supervision. Supportive care. Birds usually remain carriers.',
        'Home Remedies': 'Immune-supportive supplements, probiotics (supportive only).',
        'Prevention': 'Biosecurity, certified MG-free chicks, quarantine new/returning birds, vaccination with live MG strains.',
        'Contagiousness': 'Highly contagious horizontally (contact, aerosols, fomites) and vertically (egg transmission).',
        'Quarantine': 'Strict isolation, ideally depopulate infected flock, disinfect premises, all-in/all-out management.'
    },
    'ncd': {
        'Cause': 'Viral (Avian paramyxovirus), severe strain can be fatal.',
        'Symptoms': 'Respiratory distress, green diarrhea, depression, nervous signs (trembling, twisted neck), sudden death, collapse in egg production.',
        'Treatment': 'No cure; supportive care only. Antibiotics prevent secondary infection. Culling often necessary.',
        'Prevention': 'Vaccination (live LaSota/B1, inactivated boosters), biosecurity, monitor flock, avoid wild birds.',
        'Contagiousness': 'Extremely contagious; spreads via aerosols, feces, contaminated equipment.',
        'Quarantine': 'Notifiable disease; quarantine, culling, strict disinfection before repopulation.'
    },
    'salmo': {
        'Cause': 'Bacterial (Salmonella spp.), including Pullorum disease (chicks), Fowl typhoid (adults), Paratyphoid.',
        'Symptoms': 'Diarrhea, pasty droppings, poor growth, decreased egg production, high mortality in young chicks. Adults often subclinical carriers.',
        'Treatment': 'Broad-spectrum antibiotics under vet guidance, supportive care (fluids, warmth, vitamins); antibiotics rarely eliminate carrier state.',
        'Home Remedies': 'Fermented feed, probiotics, herbal additives (supportive, preventive only).',
        'Prevention': 'Biosecurity, NPIP-certified chicks, sanitation, rodent/insect control, vaccination (some types), competitive exclusion programs.',
        'Contagiousness': 'Highly contagious via fecal-oral route; vertical transmission through eggs possible; humans at risk.',
        'Quarantine': 'Isolate/cull infected birds, disinfect housing/feeders/waterers, monitor with fecal cultures, strict hygiene.'
    }
}


    if uploaded_image:
        img = Image.open(uploaded_image)
        st.image(img, width=300, caption="Uploaded Image")

        img_processed = preprocess_image(img)
        preds = disease_model.predict(img_processed)[0]

        idx = np.argmax(preds)
        label = DISEASE_CLASSES[idx]

        st.success(f"### 🦠 Detected Disease: **{label}**")
        # st.write("#### Model Probabilities:")
        # for cls, prob in zip(DISEASE_CLASSES, preds):
        #     st.write(f"- **{cls}**: {prob:.2f}")

        # Show precautions, medicines, and contagious info
        info = DISEASE_INFO.get(label, None)
        if info:
            st.subheader("💊 Disease Information")
            st.markdown(f"**Cause:** {info.get('Cause', 'N/A')}")
            st.markdown(f"**Symptoms:** {info.get('Symptoms', 'N/A')}")
            
            treatment = info.get('Treatment', {})
            if isinstance(treatment, dict):
                st.markdown("**Treatment:**")
                for key, val in treatment.items():
                    st.markdown(f"- **{key}:** {val}")
            else:
                st.markdown(f"**Treatment:** {treatment}")
            
            st.markdown(f"**Home Remedies:** {info.get('Home Remedies', 'N/A')}")
            st.markdown(f"**Prevention:** {info.get('Prevention', 'N/A')}")
            st.markdown(f"**Contagiousness:** {info.get('Contagiousness', 'N/A')}")
            st.markdown(f"**Quarantine:** {info.get('Quarantine', 'N/A')}")


# -----------------------------------------------------------
# 🐓 POSTURE ANALYSIS
# -----------------------------------------------------------
elif option == "Posture Analysis":
    st.header("🐓 Posture Analysis")

    if not os.path.exists(POSTURE_MODEL_PATH) or not os.path.exists(POSTURE_SCALER_PATH):
        st.warning("⚠ Posture model or scaler missing. Ensure posture_mlp_adv.h5 and posture_scaler.pkl are in the folder.")
    else:
        st.success("Posture model & scaler loaded successfully ✔")
        posture_model = tf.keras.models.load_model(POSTURE_MODEL_PATH)
        with open(POSTURE_SCALER_PATH, "rb") as f:
            posture_scaler = pickle.load(f)
       

        def preprocess_posture_image(img):
            img = img.convert("RGB")
            img = img.resize((128, 128))
            img_array = np.array(img) / 255.0
            img_flat = img_array.flatten()

            # Take only first 16 pixels (matches your trained model input)
            img_flat = img_flat[:16]

            # Expand dims to make it (1, 16) for the model
            img_flat = np.expand_dims(img_flat, axis=0)
            return img_flat



        posture_classes = ["healthy", "sick"]
        uploaded_posture_image = st.file_uploader(
            "📤 Upload a chicken posture image", type=["jpg", "jpeg", "png"]
        )

        if uploaded_posture_image:
            img = Image.open(uploaded_posture_image)
            st.image(img, caption="Uploaded Image", width=300)

            img_processed = preprocess_posture_image(img)
            pred = posture_model.predict(img_processed)[0]
            pred_class = np.argmax(pred)
            pred_label = posture_classes[pred_class]

            st.subheader("🩺 Posture Prediction")
            st.success(f"### Result: **{pred_label.upper()}**")
            st.write("#### Confidence")
            for cls_name, prob in zip(posture_classes, pred):
                st.write(f"- **{cls_name}**: {prob:.4f}")
            st.info("✔ Posture detection completed successfully.")
