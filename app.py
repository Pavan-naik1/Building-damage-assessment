
import time
import os
import uuid
import sqlite3
import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash
)

from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "results",
    "CNN_Clean_Training",
    "best_cnn_clean.keras"
)

DB_PATH = os.path.join(BASE_DIR, "users.db")

UPLOAD_FOLDER = os.path.join(
    BASE_DIR, "static", "uploads"
)

HEATMAP_FOLDER = os.path.join(
    BASE_DIR, "static", "heatmaps"
)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

IMG_SIZE = 128


# --------------------------------------------------
# FLASK INITIALIZATION
# --------------------------------------------------

app = Flask(__name__)

app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "change-this-secret-key"
)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)


# --------------------------------------------------
# DATABASE
# --------------------------------------------------

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

def load_damage_model():
    """Load the evaluated CNN checkpoint used for the reported test metrics."""

    if not os.path.exists(MODEL_PATH):
        app.logger.error("CNN model not found: %s", MODEL_PATH)
        return None

    try:
        loaded_model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        app.logger.info(
            "CNN model loaded successfully from %s",
            MODEL_PATH
        )

        return loaded_model

    except Exception:
        app.logger.exception(
            "Failed to load CNN model from %s",
            MODEL_PATH
        )
        return None


init_db()
model = load_damage_model()


# --------------------------------------------------
# IMAGE VALIDATION AND PREPROCESSING
# --------------------------------------------------

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def prepare_image(image_path):
    """
    Match CNN evaluation preprocessing:
    RGB image -> 128x128 using nearest-neighbor interpolation
    -> float32 / 255.
    """

    image = Image.open(image_path).convert("RGB")

    original = image.copy()

    resized = image.resize(
        (IMG_SIZE, IMG_SIZE),
        resample=Image.Resampling.NEAREST
    )

    arr = np.asarray(resized, dtype=np.float32) / 255.0

    batch = np.expand_dims(arr, axis=0)

    return original, resized, batch


# --------------------------------------------------
# DAMAGE PREDICTION
# --------------------------------------------------

def predict_damage(batch):

    start_time = time.perf_counter()

    probability_no_damage = float(
        model.predict(batch, verbose=0)[0][0]
    )

    inference_ms = (
        time.perf_counter() - start_time
    ) * 1000.0

    print(
        f"[TIMING] Model inference: {inference_ms:.2f} ms"
    )

    if probability_no_damage >= 0.5:

        prediction = "No Damage"
        confidence = probability_no_damage * 100.0

    else:

        prediction = "Damage"
        confidence = (1.0 - probability_no_damage) * 100.0

    return prediction, confidence, probability_no_damage


# --------------------------------------------------
# HEATMAP NORMALIZATION
# --------------------------------------------------

def normalize_map(arr):

    arr = np.asarray(arr, dtype=np.float32)

    arr = arr - arr.min()

    max_value = arr.max()

    if max_value > 0:
        arr = arr / max_value

    return arr


# --------------------------------------------------
# EXPLANATION SCORE
# --------------------------------------------------

def get_explain_score(prob_no_damage):
    """
    The model's sigmoid output represents the No Damage class.
    Explain whichever class the model predicts.
    """

    return tf.where(
        prob_no_damage >= 0.5,
        prob_no_damage,
        1.0 - prob_no_damage
    )


# --------------------------------------------------
# SALIENCY MAP
# --------------------------------------------------

def create_saliency_map(batch, output_path):
    """
    Calculate the gradient of the predicted class score
    with respect to the uploaded input image.
    """

    x = tf.convert_to_tensor(
        batch,
        dtype=tf.float32
    )

    with tf.GradientTape() as tape:

        tape.watch(x)

        prob_no_damage = model(
            x,
            training=False
        )[:, 0]

        score = get_explain_score(prob_no_damage)

    grads = tape.gradient(score, x)

    if grads is None:
        raise RuntimeError(
            "Could not calculate input gradients for saliency map."
        )

    saliency = tf.reduce_max(
        tf.abs(grads),
        axis=-1
    )[0]

    saliency = normalize_map(saliency.numpy())

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(saliency, cmap="jet")
    ax.axis("off")

    fig.tight_layout(pad=0)

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close(fig)


# --------------------------------------------------
# GRAD-CAM OVERLAY
# --------------------------------------------------

def create_gradcam_overlay(batch, resized_image, output_path):
    """
    Generate genuine layer-based Grad-CAM using the final
    convolutional layer of the CNN.

    A direct layer-by-layer forward pass is used to preserve
    the gradient connection between the convolutional feature
    map and the model prediction.

    Input-gradient saliency is used only as a fallback if
    layer-based Grad-CAM cannot be calculated.
    """

    x = tf.convert_to_tensor(
        batch,
        dtype=tf.float32
    )

    try:

        # Find the final convolutional layer.

        last_conv = next(
            layer for layer in reversed(model.layers)
            if isinstance(layer, tf.keras.layers.Conv2D)
        )

        # Direct forward pass through the CNN.
        # This preserves the computational gradient path.

        with tf.GradientTape() as tape:

            z = x

            conv_outputs = None

            for layer in model.layers:

                z = layer(z, training=False)

                if layer.name == last_conv.name:
                    conv_outputs = z

            model_output = z

            prob_no_damage = model_output[:, 0]

            score = get_explain_score(prob_no_damage)

        # Calculate gradients with respect to
        # the final convolutional feature map.

        grads = tape.gradient(
            score,
            conv_outputs
        )

        if grads is None:
            raise RuntimeError(
                "Grad-CAM gradients unavailable for the final convolutional layer."
            )

        # Calculate channel importance weights.

        channel_weights = tf.reduce_mean(
            grads,
            axis=(1, 2),
            keepdims=True
        )

        # Weighted combination of feature maps.

        cam = tf.reduce_sum(
            channel_weights * conv_outputs,
            axis=-1
        )

        # Apply ReLU.

        cam = tf.nn.relu(cam)[0]

        cam = normalize_map(cam.numpy())

        app.logger.info(
            "Genuine layer-based Grad-CAM generated successfully."
        )

    except Exception as exc:

        app.logger.warning(
            "Grad-CAM failed; using input-gradient fallback: %s",
            exc
        )

        # Input-gradient fallback.
        # This is not genuine Grad-CAM.

        with tf.GradientTape() as tape:

            tape.watch(x)

            prob_no_damage = model(
                x,
                training=False
            )[:, 0]

            score = get_explain_score(prob_no_damage)

        grads = tape.gradient(score, x)

        if grads is None:
            raise RuntimeError(
                "Input gradients are also unavailable."
            )

        cam = tf.reduce_mean(
            tf.abs(grads),
            axis=-1
        )[0].numpy()

        cam = normalize_map(cam)

    # Resize heatmap to the original model input dimensions.

    cam_tensor = tf.convert_to_tensor(
        cam[None, ..., None],
        dtype=tf.float32
    )

    cam_tensor = tf.image.resize(
        cam_tensor,
        (IMG_SIZE, IMG_SIZE),
        method="bilinear"
    )

    cam = normalize_map(
        cam_tensor[0, ..., 0].numpy()
    )

    # Overlay heatmap on the resized satellite image.

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.imshow(resized_image)

    ax.imshow(
        cam,
        cmap="jet",
        alpha=0.5
    )

    ax.axis("off")

    fig.tight_layout(pad=0)

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close(fig)


# --------------------------------------------------
# LOGIN REQUIRED DECORATOR
# --------------------------------------------------

def login_required(view_func):

    from functools import wraps

    @wraps(view_func)
    def wrapper(*args, **kwargs):

        if "user_id" not in session:

            flash(
                "Please login to access the prediction page.",
                "warning"
            )

            return redirect(url_for("login"))

        return view_func(*args, **kwargs)

    return wrapper


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.route("/")
def home():

    return render_template("home.html")


# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------

@app.route("/about")
def about():

    return render_template("about.html")


# --------------------------------------------------
# USER REGISTRATION
# --------------------------------------------------

@app.route("/register", methods=["GET", "POST"])
def register():

    if "user_id" in session:
        return redirect(url_for("home"))

    if request.method == "POST":

        name = request.form.get("name", "").strip()

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        confirm_password = request.form.get(
            "confirm_password",
            ""
        )

        if not name or not email or not password:

            flash(
                "All fields are required.",
                "danger"
            )

            return redirect(url_for("register"))

        if password != confirm_password:

            flash(
                "Passwords do not match.",
                "danger"
            )

            return redirect(url_for("register"))

        if len(password) < 6:

            flash(
                "Password must contain at least 6 characters.",
                "danger"
            )

            return redirect(url_for("register"))

        password_hash = generate_password_hash(password)

        try:

            with sqlite3.connect(DB_PATH) as conn:

                conn.execute(
                    """
                    INSERT INTO users (name, email, password)
                    VALUES (?, ?, ?)
                    """,
                    (name, email, password_hash)
                )

                conn.commit()

            flash(
                "Registration successful. Please login.",
                "success"
            )

            return redirect(url_for("login"))

        except sqlite3.IntegrityError:

            flash(
                "An account with this email already exists.",
                "danger"
            )

    return render_template("register.html")


# --------------------------------------------------
# USER LOGIN
# --------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():

    if "user_id" in session:
        return redirect(url_for("home"))

    if request.method == "POST":

        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        with sqlite3.connect(DB_PATH) as conn:

            conn.row_factory = sqlite3.Row

            user = conn.execute(
                "SELECT * FROM users WHERE email = ?",
                (email,)
            ).fetchone()

        if user and check_password_hash(
            user["password"],
            password
        ):

            session["user_id"] = user["id"]
            session["user_name"] = user["name"]

            flash(
                "Login successful.",
                "success"
            )

            return redirect(url_for("home"))

        flash(
            "Invalid email or password.",
            "danger"
        )

    return render_template("login.html")


# --------------------------------------------------
# PREDICTION PAGE
# --------------------------------------------------

@app.route("/prediction", methods=["GET", "POST"])
@login_required
def prediction():

    result = None

    if request.method == "POST":

        if model is None:

            flash(
                "CNN model file not found or could not be loaded. "
                "Expected: results/CNN_Clean_Training/best_cnn_clean.keras.",
                "danger"
            )

            return render_template(
                "prediction.html",
                result=None
            )

        if "image" not in request.files:

            flash(
                "Please choose an image.",
                "danger"
            )

            return redirect(request.url)

        file = request.files["image"]

        if file.filename == "":

            flash(
                "Please choose an image.",
                "danger"
            )

            return redirect(request.url)

        if not allowed_file(file.filename):

            flash(
                "Allowed formats: PNG, JPG, JPEG, WEBP.",
                "danger"
            )

            return redirect(request.url)

        ext = secure_filename(
            file.filename
        ).rsplit(".", 1)[1].lower()

        unique_name = f"{uuid.uuid4().hex}.{ext}"

        upload_path = os.path.join(
            UPLOAD_FOLDER,
            unique_name
        )

        file.save(upload_path)

        try:

            _, resized, batch = prepare_image(
                upload_path
            )

            label, confidence, probability_no_damage = predict_damage(
                batch
            )

            # Generate saliency map.

            saliency_name = (
                f"{uuid.uuid4().hex}_saliency.png"
            )

            saliency_path = os.path.join(
                HEATMAP_FOLDER,
                saliency_name
            )

            create_saliency_map(
                batch,
                saliency_path
            )

            # Generate Grad-CAM overlay.

            gradcam_name = (
                f"{uuid.uuid4().hex}_gradcam.png"
            )

            gradcam_path = os.path.join(
                HEATMAP_FOLDER,
                gradcam_name
            )

            create_gradcam_overlay(
                batch,
                resized,
                gradcam_path
            )

            result = {

                "prediction": label,

                "confidence": round(
                    confidence,
                    2
                ),

                "probability_damage": round(
                    (1.0 - probability_no_damage) * 100.0,
                    2
                ),

                "probability_no_damage": round(
                    probability_no_damage * 100.0,
                    2
                ),

                "image_url": url_for(
                    "static",
                    filename=f"uploads/{unique_name}"
                ),

                "saliency_url": url_for(
                    "static",
                    filename=f"heatmaps/{saliency_name}"
                ),

                "gradcam_url": url_for(
                    "static",
                    filename=f"heatmaps/{gradcam_name}"
                )
            }

        except Exception as exc:

            app.logger.exception(
                "Prediction failed"
            )

            flash(
                f"Prediction failed: {exc}",
                "danger"
            )

    return render_template(
        "prediction.html",
        result=result
    )


# --------------------------------------------------
# LOGOUT
# --------------------------------------------------

@app.route("/logout")
def logout():

    session.clear()

    flash(
        "You have been logged out.",
        "success"
    )

    return redirect(url_for("login"))


# --------------------------------------------------
# FILE SIZE ERROR HANDLER
# --------------------------------------------------

@app.errorhandler(413)
def too_large(_):

    flash(
        "Image is too large. Maximum upload size is 10 MB.",
        "danger"
    )

    return redirect(url_for("prediction"))


# --------------------------------------------------
# APPLICATION ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    app.run(debug=False, use_reloader=False)
