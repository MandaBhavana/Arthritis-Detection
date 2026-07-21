import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image




MODEL_PATH = "models/best_finetuned_densenet.keras"

CLASS_NAMES = [
    "Grade 0 - Normal",
    "Grade 1 - Doubtful",
    "Grade 2 - Mild",
    "Grade 3 - Moderate",
    "Grade 4 - Severe",
]

GRADE_DESCRIPTIONS = [
    "No clear signs of osteoarthritis are visible.",
    "Possible early or doubtful signs of osteoarthritis.",
    "Mild osteoarthritis features may be present.",
    "Moderate osteoarthritis features may be present.",
    "Severe osteoarthritis features may be present.",
]



st.set_page_config(
    page_title="Arthritis Detection",
    page_icon="🦴",
    layout="centered",
)




st.markdown(
    """
    <style>
    .block-container {
        max-width: 900px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    .main-title {
        font-size: 3rem;
        font-weight: 750;
        line-height: 1.15;
        margin-bottom: 0.5rem;
        color: lightblue;
    }

    .subtitle {
        font-size: 1.08rem;
        color: #667085;
        margin-bottom: 1.5rem;
    }

    .info-card {
        background-color: #f7f9fc;
        border: 1px solid #e4e9f0;
        border-radius: 14px;
        padding: 1.2rem 1.3rem;
        margin-bottom: 1.2rem;
    }

    .info-title {
        font-size: 1rem;
        font-weight: 700;
        color: gray;
        margin-bottom: 0.35rem;
    }

    .info-text {
        color: #667085;
        font-size: 0.95rem;
        line-height: 1.55;
    }

    div[data-testid="stFileUploader"] {
        background-color: #fafbfc;
        border: 1px dashed #aeb8c6;
        border-radius: 14px;
        padding: 0.8rem;
    }

    div[data-testid="stImage"] img {
        border-radius: 14px;
    }


    .result-card {
        background-color: #f7f9fc;
        border: 1px solid #e4e9f0;
        border-radius: 14px;
        padding: 1.2rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }

    .result-label {
        color: #667085;
        font-size: 0.9rem;
        margin-bottom: 0.25rem;
    }

    .result-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #252936;
        margin-bottom: 0.5rem;
    }

    .result-description {
        color: #667085;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #252936;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }

    .footer {
        text-align: center;
        color: #8a919d;
        font-size: 0.85rem;
        margin-top: 3rem;
    }

    .stButton > button {
        width: 100%;
        border-radius: 10px;
        padding: 0.7rem 1rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)




@st.cache_resource
def load_saved_model():
    return tf.keras.models.load_model(MODEL_PATH)


try:
    model = load_saved_model()
except Exception as error:
    st.error("The trained model could not be loaded.")
    st.code(str(error))
    st.stop()


IMG_HEIGHT = model.input_shape[1]
IMG_WIDTH = model.input_shape[2]



st.markdown(
    '<div class="main-title">🦴 Arthritis Detection</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        AI-assisted osteoarthritis severity grading from knee X-ray images.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-card">
        <div class="info-title">How it works</div>
        <div class="info-text">
            Upload a clear knee X-ray image in PNG or JPG format.
            The model will estimate an osteoarthritis severity grade
            between Grade 0 and Grade 4.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "Educational prototype only. This application is not a medical diagnostic tool."
)




uploaded_file = st.file_uploader(
    "Upload a knee X-ray image",
    type=["png", "jpg", "jpeg"],
    help="Use a clear knee X-ray image in PNG, JPG, or JPEG format.",
)



if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("The uploaded file could not be opened as an image.")
        st.stop()

    st.markdown(
        '<div class="section-title">Image preview</div>',
        unsafe_allow_html=True,
    )

    image_column, details_column = st.columns(
        [1.15, 0.85],
        gap="large",
    )

    with image_column:
        st.image(
            image,
            caption="Uploaded knee X-ray",
            use_container_width=True,
        )

    with details_column:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="info-title">Image details</div>
                <div class="info-text">
                    File: {uploaded_file.name}<br>
                    Original size: {image.width} × {image.height} pixels<br>
                    Model input: {IMG_WIDTH} × {IMG_HEIGHT} pixels
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        analyze_button = st.button(
            "Analyze X-ray",
            type="primary",
            use_container_width=True,
        )

    if analyze_button:

        with st.spinner("Analyzing image..."):

            resized_image = image.resize(
                (IMG_WIDTH, IMG_HEIGHT)
            )

            image_array = np.asarray(
                resized_image,
                dtype=np.float32,
            )

            image_array = np.expand_dims(
                image_array,
                axis=0,
            )

            predictions = model.predict(
                image_array,
                verbose=0,
            )[0]

        predicted_index = int(np.argmax(predictions))
        confidence = float(
            predictions[predicted_index] * 100
        )

        st.markdown(
            '<div class="section-title">Analysis result</div>',
            unsafe_allow_html=True,
        )

        result_column, confidence_column = st.columns(
            2,
            gap="medium",
        )

        with result_column:
            st.metric(
                label="Predicted severity",
                value=CLASS_NAMES[predicted_index],
            )

        with confidence_column:
            st.metric(
                label="Model confidence",
                value=f"{confidence:.2f}%",
            )

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">Model interpretation</div>
                <div class="result-value">
                    {CLASS_NAMES[predicted_index]}
                </div>
                <div class="result-description">
                    {GRADE_DESCRIPTIONS[predicted_index]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-title">Probability by grade</div>',
            unsafe_allow_html=True,
        )

        for index, probability in enumerate(predictions):
            percentage = float(probability * 100)

            label_column, percentage_column = st.columns(
                [4, 1]
            )

            with label_column:
                st.write(CLASS_NAMES[index])

            with percentage_column:
                st.write(f"**{percentage:.2f}%**")

            st.progress(
                min(
                    max(
                        int(round(percentage)),
                        0,
                    ),
                    100,
                )
            )

        st.info(
            "The confidence score shows how strongly the model preferred "
            "one grade. It does not guarantee that the prediction is correct."
        )

        st.caption(
            "The model achieved approximately 44.75% test accuracy. "
            "Results may be incorrect and must not be used for diagnosis."
        )

else:
    st.caption(
        "Upload a knee X-ray image to begin the analysis."
    )



st.markdown(
    """
    <div class="footer">
        Built with TensorFlow, DenseNet121 and Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)