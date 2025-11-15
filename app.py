# app.py
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

st.set_page_config(page_title="RBC Classifier", layout="centered")
st.title("🧬 Malaria RBC Classifier")
st.write("Upload an RBC image (microscope) — the app predicts Parasitized / Uninfected and shows Grad-CAM.")

# ---- Load model ----
MODEL_PATH = "final_rbc_classifier.h5"
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at {MODEL_PATH}. Put the .h5 file in the same folder as app.py.")
    st.stop()

# load model
model = tf.keras.models.load_model(MODEL_PATH)

# warm up the model so its input/output tensors are defined
try:
    _ = model.predict(np.zeros((1, 128, 128, 3)))
except Exception:
    # try single-channel fallback or different size if needed
    try:
        _ = model.predict(np.zeros((1,) + model.input_shape[1:]))
    except Exception:
        pass

st.success("✅ Model loaded successfully.")

# Detect model output type (sigmoid single-output or softmax multi-output)
output_shape = model.output_shape
if len(output_shape) == 2 and output_shape[-1] == 1:
    MODEL_TYPE = "sigmoid"
    class_names = ["Uninfected", "Parasitized"]  # textual mapping used for display
else:
    MODEL_TYPE = "softmax"
    class_names = ["Uninfected", "Parasitized"]  # adjust if your training used reversed order

st.write(f"Model type detected: **{MODEL_TYPE}**")

# ---- Helper: find last Conv2D layer ----
def find_last_conv_layer(m):
    for layer in reversed(m.layers):
        # if it's a nested Model/Sequential, drill down
        if isinstance(layer, tf.keras.Model):
            name = find_last_conv_layer(layer)
            if name:
                return name
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

# ---- Robust Grad-CAM that uses model.input/model.output ----
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    img_array: preprocessed batch (1, H, W, C)
    model: keras model
    last_conv_layer_name: string name of conv layer
    returns: heatmap (Hconv x Wconv)
    """
    # ensure model has been called (warm-up done above)
    # build a model that maps the original model input to the chosen conv layer outputs + predictions
    try:
        last_conv_layer = model.get_layer(last_conv_layer_name)
    except Exception as e:
        raise ValueError(f"Could not find layer named '{last_conv_layer_name}' in the model.") from e

    # Build grad model using model.input (singular) and model.output (singular)
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        if pred_index is None:
            if MODEL_TYPE == "sigmoid":
                # for sigmoid single-output, take the single channel
                class_channel = predictions[:, 0]
            else:
                pred_index = tf.argmax(predictions[0])
                class_channel = predictions[:, pred_index]
        else:
            if MODEL_TYPE == "sigmoid":
                class_channel = predictions[:, 0]
            else:
                class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        # fallback: if gradients are None, return zero heatmap
        h = conv_outputs.shape[1]
        w = conv_outputs.shape[2]
        return np.zeros((h, w))

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]  # (Hc, Wc, channels)

    # Weight channels by pooled grads
    heatmap = tf.matmul(conv_outputs, pooled_grads[..., tf.newaxis])
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return np.zeros_like(heatmap.numpy())
    heatmap = heatmap / max_val
    return heatmap.numpy()

# ---- UI: upload & predict ----
uploaded_file = st.file_uploader("Upload an RBC image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded image", use_container_width=False, width=300)

    # Preprocess (must match training preprocessing exactly)
    img_resized = img.resize((128, 128))
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    input_tensor = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(input_tensor)

    if MODEL_TYPE == "sigmoid":
        score = float(preds[0][0])
        # If your model was trained with labels reversed, flip this mapping; otherwise this is standard:
        pred_label = "Uninfected" if score > 0.5 else "Parasitized"

        confidence = (score if score > 0.5 else 1 - score) * 100.0
    else:
        idx = int(np.argmax(preds[0]))
        pred_label = class_names[idx]
        confidence = float(np.max(preds[0]) * 100.0)

    st.markdown(f"### 🧾 **Prediction:** {pred_label}")
    st.markdown(f"**Confidence:** {confidence:.2f}%")

    # Grad-CAM
    last_conv = find_last_conv_layer(model)
    if last_conv is None:
        st.warning("⚠️ No Conv2D layer found — cannot compute Grad-CAM.")
    else:
        try:
            heatmap = make_gradcam_heatmap(input_tensor, model, last_conv)
            heatmap = cv2.resize(heatmap, (128, 128))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(np.array(img_resized), 0.6, heatmap, 0.4, 0)
            st.subheader("🔥 Grad-CAM visualization")
            st.image(overlay, width=300)
        except Exception as e:
            st.error(f"Grad-CAM error: {e}")
