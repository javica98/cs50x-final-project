# ===== REQUIRED IMPORTS =====

# Flask web framework
from flask import Flask, render_template, request, redirect, url_for, send_file

# Data handling
import pandas as pd
import os

# Scikit-learn: model training, preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Classifier models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

# Regression models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Clustering models
from sklearn.cluster import KMeans, DBSCAN

# Utilities
from datetime import datetime
import joblib               # For saving/loading models
import uuid                 # For generating unique IDs
import io                   # In-memory file I/O
import matplotlib.pyplot as plt
import seaborn as sns
import base64               # For image encoding
import json
import pickle               # For saving models (especially regression)

# ===== FLASK APP INITIALIZATION =====

app = Flask(__name__)  # Create Flask app instance

# ===== CONFIGURATION FOR FILE UPLOADS AND MODEL STORAGE =====

app.config['UPLOAD'] = 'uploads'   # Directory to save uploaded CSV files
app.config['MODEL'] = 'models'     # Directory to save trained models

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD'], exist_ok=True)
os.makedirs(app.config['MODEL'], exist_ok=True)

# ===== AVAILABLE CLASSIFICATION MODELS =====

model_dict = {
    'KNN': KNeighborsClassifier(),
    'DECISION_TREE': DecisionTreeClassifier(),
    'RANDOM_FOREST': RandomForestClassifier(),
    'LOGISTIC_REGRESSION': LogisticRegression(max_iter=1000),
    'SVM': SVC(probability=True),  # Needed for predict_proba
    'NAIVE_BAYES': GaussianNB(),
    'GRADIENT_BOOSTING': GradientBoostingClassifier()
}

# ===== AVAILABLE REGRESSION MODELS =====

regression_models = {
    "LinearRegression": LinearRegression,
    "RandomForestRegressor": RandomForestRegressor,
    "DecisionTreeRegressor": DecisionTreeRegressor
}

# ===== AVAILABLE CLUSTERING MODELS =====

clustering_models = {
    "KMeans": KMeans,
    "DBSCAN": DBSCAN
}

# ===== HOME PAGE ROUTE =====
@app.route('/')
def index():
    return render_template('index.html')  # Load the homepage

# ===== TRAINING A CLUSTERING MODEL =====
@app.route("/train_clustering", methods=["GET", "POST"])
def train_clustering():
    if request.method == "POST":
        # Retrieve uploaded file and form fields
        file = request.files["file"]
        form_model_name = request.form["model_name"]
        form_model_description = request.form["model_description"]
        form_model = request.form["model"]

        # Load data from CSV
        df = pd.read_csv(file)
        X = df.copy()  # No target column in clustering

        # Standardize features (recommended for clustering)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Create clustering model
        model_class = clustering_models.get(form_model)
        model = model_class()
        labels = model.fit_predict(X_scaled)  # Fit model and assign clusters

        # Save model, scaler, and column names to a .pkl file
        model_filename = f"{form_model_name}.pkl"
        model_path = os.path.join(app.config['MODEL'], model_filename)
        joblib.dump((model, scaler, X.columns.tolist()), model_path)

        # Load or create metadata file
        meta_file = os.path.join(app.config['MODEL'], 'metadata.json')
        metadata = []
        if os.path.exists(meta_file):
            try:
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
            except json.JSONDecodeError:
                pass  # Skip if file is empty or broken

        # Store metadata for the new clustering model
        new_model = {
            "filename": model_filename,
            "name": form_model_name,
            "description": form_model_description,
            "model_type": "Clustering",
            "type": form_model,
            "created": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "features": X.columns.tolist(),
            "n_clusters": int(getattr(model, 'n_clusters', -1))  # -1 for models like DBSCAN
        }
        metadata.append(new_model)

        # Save metadata to disk
        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=2)

        # Render results page
        return render_template("train_result_clustering.html",
                               model_name=form_model_name,
                               model_description=form_model_description,
                               model_filename=model_filename,
                               features=list(X.columns),
                               n_clusters=new_model["n_clusters"])

    # GET request: render form
    return render_template("train_clustering.html", models=clustering_models.keys())

# ===== PREDICTING WITH A CLUSTERING MODEL =====
@app.route("/predict_clustering", methods=["GET", "POST"])
def predict_clustering():
    # Load metadata to get available clustering models
    meta_path = os.path.join(app.config['MODEL'], 'metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = []  # Prevent errors if file does not exist

    # Filter only clustering models from metadata
    av_models = [m["filename"] for m in metadata if m.get("model_type") == "Clustering"]

    if request.method == "POST":
        # Get uploaded file and selected model
        file = request.files["file"]
        model_file = request.form["model"]

        # Load model, scaler, and expected columns
        model, scaler, features = joblib.load(os.path.join(app.config['MODEL'], model_file))

        # Read input data and apply scaler
        df = pd.read_csv(file)
        X = df[features]
        X_scaled = scaler.transform(X)

        # Predict cluster labels and append to DataFrame
        labels = model.predict(X_scaled)
        df['cluster'] = labels

        # Create downloadable CSV with predictions
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return send_file(io.BytesIO(output.getvalue().encode()),
                         mimetype='text/csv',
                         as_attachment=True,
                         download_name='clusters.csv')

    # GET request: render prediction page
    return render_template("predict_clustering.html", models=av_models)

# ===== TRAINING A REGRESSION MODEL =====
@app.route("/train_regression", methods=["GET", "POST"])
def train_regression():
    if request.method == "POST":
        # Get uploaded file and form data
        file = request.files["file"]
        form_model_name = request.form["model_name"]
        form_model_description = request.form["model_description"]
        form_model = request.form["model"]
        target = request.form["target"]

        # Load dataset
        df = pd.read_csv(file)
        X = df.drop(columns=[target])  # Features
        y = df[target]                 # Target variable

        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize and train regression model
        model_class = regression_models.get(form_model)
        model = model_class()
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate regression metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Save the trained model to file
        model_filename = f"{form_model_name}.pkl"
        model_path = os.path.join("models", model_filename)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Load or initialize metadata
        meta_file = os.path.join(app.config['MODEL'], 'metadata.json')
        if os.path.exists(meta_file):
            try:
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
            except json.JSONDecodeError:
                metadata = []
        else:
            metadata = []

        # Create metadata entry for the model
        new_model = {
            "filename": os.path.basename(model_path),
            "name": form_model_name,
            "description": form_model_description,
            "model_type": "Regression",
            "type": form_model,
            "created": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "mse": round(mse, 4),
            "r2_score": round(r2, 4),
            "features": X.columns.tolist(),
            "train_size": len(X_train),
            "test_size": len(X_test)
        }

        metadata.append(new_model)

        # Save updated metadata
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Render results page
        return render_template(
            "train_result_regression.html",
            model_name=form_model_name,
            model_description=form_model_description,
            mse=round(mse, 4),
            r2=round(r2, 4),
            model_filename=model_filename,
            features=list(X.columns),
            train_size=len(X_train),
            test_size=len(X_test)
        )

    # GET request: render training form
    return render_template("train_regression.html", models=regression_models.keys())

# ===== MAKING PREDICTIONS WITH A REGRESSION MODEL =====
@app.route("/predict_regression", methods=["GET", "POST"])
def predict_regression():
    # Load metadata to get available regression models
    meta_path = os.path.join(app.config['MODEL'], 'metadata.json')

    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = []  # Safe fallback if metadata file doesn't exist

    # Filter regression models from metadata
    model_files = [m["filename"] for m in metadata if m.get("model_type") == "Regression"]

    if request.method == "POST":
        # Read uploaded data and selected model
        file = request.files["file"]
        selected_model = request.form["model"]

        df = pd.read_csv(file)

        # Load the trained model
        with open(f"models/{selected_model}", "rb") as f:
            model = pickle.load(f)

        # Make predictions on input data
        predictions = model.predict(df)

        # Return predictions as list to render in template
        return render_template("predict_result_regression.html", predictions=predictions.tolist())

    # GET request: render prediction form
    return render_template("predict_regression.html", models=model_files)
## TRAIN CLASSIFIER
@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        # Retrieve form fields
        form_model_name = request.form.get('model_name', 'Unnamed Model')
        form_model_description = request.form.get('model_description', '')
        file = request.files['file']
        form_model = request.form['model']
        form_target = request.form['target']

        # Validate file upload
        if not file:
            return 'You should upload your .csv file'

        df = pd.read_csv(file)

        # Validate target column
        if form_target not in df.columns:
            return f'Column "{form_target}" not found in dataset'

        # Split features and target
        X = df.drop(columns=[form_target])
        y = df[form_target]

        # Scale features (important for many classifiers)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Retrieve model from dictionary
        model = model_dict.get(form_model)
        if model is None:
            return 'You should choose a valid model'

        # Train the model
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_prediction = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_prediction)
        report = classification_report(y_test, y_prediction, output_dict=True)
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1 = report['weighted avg']['f1-score']
        matrix = confusion_matrix(y_test, y_prediction)
        train_size = len(X_train)
        test_size = len(X_test)

        # Save the trained model, scaler, and feature names
        model_id = str(uuid.uuid4())
        model_path = os.path.join(app.config['MODEL'], f"{form_model}_{model_id}.joblib")
        joblib.dump((model, scaler, X.columns.tolist()), model_path)

        # Generate confusion matrix image
        img = io.BytesIO()
        plt.figure(figsize=(6, 4))
        sns.heatmap(matrix, annot=True, fmt='d')
        plt.title("Confusion Matrix")
        plt.savefig(img, format='png')
        img.seek(0)
        confusion_image = base64.b64encode(img.getvalue()).decode()

        # Load or initialize metadata
        meta_file = os.path.join(app.config['MODEL'], 'metadata.json')
        if os.path.exists(meta_file):
            try:
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
            except json.JSONDecodeError:
                metadata = []
        else:
            metadata = []

        # Create metadata entry
        new_model = {
            "filename": os.path.basename(model_path),
            "name": form_model_name,
            "description": form_model_description,
            "model_type": "Classifier",
            "type": form_model,
            "created": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "features": X.columns.tolist(),
            "train_size": train_size,
            "test_size": test_size,
            "report": report
        }

        metadata.append(new_model)

        # Save confusion matrix image to file
        img_path = os.path.join(app.config['MODEL'], f"{form_model}_{model_id}.png")
        plt.savefig(img_path)

        # Save updated metadata
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Show training result
        return render_template('train_result.html',
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            features=X.columns.tolist(),
            train_size=train_size,
            test_size=test_size,
            report=report,
            model_filename=os.path.basename(model_path),
            model_type="Classifier",
            model_name=form_model_name,
            model_description=form_model_description,
            confusion_image=confusion_image
        )

    # GET request: show the training form
    return render_template('train.html', models=model_dict.keys())


## PREDICT CLASSIFIER
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    meta_path = os.path.join(app.config['MODEL'], 'metadata.json')

    # Load model metadata
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = []  # Avoid errors if no metadata yet

    if request.method == 'POST':
        file = request.files['file']
        form_model = request.form['model']

        if not file:
            return 'You should upload your .csv file'

        df = pd.read_csv(file)

        # Load selected model and associated preprocessor
        model_path = os.path.join(app.config['MODEL'], form_model)
        model, scaler, feature_columns = joblib.load(model_path)

        # Validate input columns
        if not all(col in df.columns for col in feature_columns):
            return 'File columns are not the model ones'

        # Apply same preprocessing and predict
        X = df[feature_columns]
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)

        df['prediction'] = predictions

        # Prepare downloadable CSV with predictions
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name='predicciones.csv'
        )

    # GET: show prediction page with available models
    av_models = [m["filename"] for m in metadata if m.get("model_type") == "Classifier"]
    return render_template('predict.html', models=av_models)

## MODEL LIST VIEW
@app.route('/models')
def list_models():
    meta_file = os.path.join(app.config['MODEL'], 'metadata.json')
    models = []
    if os.path.exists(meta_file):
        try:
            with open(meta_file, 'r') as f:
                models = json.load(f)
        except json.JSONDecodeError:
            models = []
    # Only show models that still exist in the filesystem
    models = [m for m in models if os.path.exists(os.path.join(app.config['MODEL'], m['filename']))]
    return render_template('models.html', models=models)

## MODEL DETAIL VIEW
@app.route('/models/<filename>')
def model_detail(filename):
    meta_file = os.path.join(app.config['MODEL'], 'metadata.json')
    if not os.path.exists(meta_file):
        return 'No metadata file found', 404

    with open(meta_file, 'r') as f:
        try:
            metadata = json.load(f)
        except json.JSONDecodeError:
            return 'Error reading metadata', 500

    # Find the matching model entry
    model_info = next((m for m in metadata if m['filename'] == filename), None)
    if not model_info:
        return 'Model not found', 404

    model_type = model_info.get('model_type')

    # Render detail view based on model type
    if model_type == "Classifier":
        img_path = os.path.join(app.config['MODEL'], filename.replace('.joblib', '.png'))
        confusion_image = None
        if os.path.exists(img_path):
            with open(img_path, 'rb') as img_file:
                confusion_image = base64.b64encode(img_file.read()).decode()

        return render_template(
            'train_result.html',
            model_name=model_info['name'],
            model_description=model_info['description'],
            model_filename=model_info['filename'],
            accuracy=model_info.get('accuracy'),
            precision=model_info.get('precision'),
            recall=model_info.get('recall'),
            f1_score=model_info.get('f1_score'),
            features=model_info.get('features'),
            train_size=model_info.get('train_size'),
            test_size=model_info.get('test_size'),
            report=model_info.get('report'),
            confusion_image=confusion_image
        )

    elif model_type == "Regression":
        return render_template(
            'train_result_regression.html',
            model_name=model_info['name'],
            model_description=model_info['description'],
            model_filename=model_info['filename'],
            mse=model_info.get('mse'),
            r2=model_info.get('r2_score'),
            features=model_info.get('features'),
            train_size=model_info.get('train_size'),
            test_size=model_info.get('test_size')
        )

    elif model_type == "Clustering":
        return render_template(
            'train_result_clustering.html',
            model_name=model_info['name'],
            model_description=model_info['description'],
            model_filename=model_info['filename'],
            features=model_info.get('features'),
            n_clusters=model_info.get('n_clusters', -1)
        )

    else:
        return 'Unknown model type', 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
