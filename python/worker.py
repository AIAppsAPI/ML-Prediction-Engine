#!/usr/bin/env python3
# ML worker for the ML Prediction Engine. The PHP app sends this worker one
# JSON payload per job: train a model from a dataset file, predict with a
# trained model, or run a DBSCAN clustering pass. The worker answers with a
# JSON result, and errors come back as {"error": "..."} instead of raising.
#
# Two ways to run it, both use the exact same handler:
#   python3 worker.py run                reads one payload from stdin, prints the result (the PHP "cli" driver)
#   python3 worker.py serve --port 8750  small HTTP service, POST the payload to / (the PHP "service" driver)

import argparse
import json
import os
import pickle
import sys

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, LocalOutlierFactor
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingRegressor, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVR
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture


# ---------------------------------------------------------------------------------
# Small helper transformers and estimators that live inside pickled pipelines.
# They must stay in this file: pickle finds them here again at load time.

class DenseTransformer(BaseEstimator, TransformerMixin):
    # Turns sparse tf-idf output into a dense array for estimators that need one.
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "toarray"):
            return X.toarray()
        return np.asarray(X)


class RobustZScore(BaseEstimator):
    # Anomaly detector using the modified z-score (median and MAD). Follows the
    # sklearn convention: predict returns 1 for normal and -1 for an anomaly.
    def __init__(self, threshold=3.5):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.median_ = np.median(X, axis=0)
        mad = np.median(np.abs(X - self.median_), axis=0)
        self.mad_ = np.maximum(mad, 1e-9)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        scores = np.abs(0.6745 * (X - self.median_) / self.mad_)
        worst = np.max(scores, axis=1)
        return np.where(worst > self.threshold, -1, 1)


# ---------------------------------------------------------------------------------
# Parameter helpers.

def parm_int(parm, key, default):
    try:
        return int(parm.get(key, default))
    except (TypeError, ValueError):
        return default


def parm_float(parm, key, default):
    try:
        return float(parm.get(key, default))
    except (TypeError, ValueError):
        return default


def parm_layers(parm, key, default):
    layers = parm.get(key, default)
    if isinstance(layers, str):
        layers = [p for p in layers.replace("[", "").replace("]", "").split(",") if p.strip()]
    try:
        out = tuple(int(v) for v in layers)
    except (TypeError, ValueError):
        out = tuple(default)
    if len(out) < 1:
        out = tuple(default)
    return out


# ---------------------------------------------------------------------------------
# Builds the estimator for a model class. Returns (estimator, wants_scaler,
# wants_dense). DBSCAN is not built here, it clusters at predict time.

def build_estimator(model_class, parm, data_type):
    text = (data_type == "text")

    if model_class == "knn":
        return KNeighborsClassifier(n_neighbors=parm_int(parm, "k", 5)), False, False

    if model_class == "naiveBayes":
        if text:
            return MultinomialNB(), False, False
        return GaussianNB(), False, False

    if model_class == "dnn":
        return MLPClassifier(
            hidden_layer_sizes=parm_layers(parm, "layers", (128, 64)),
            max_iter=parm_int(parm, "epochs", 300),
            learning_rate_init=parm_float(parm, "learningRate", 0.001),
            alpha=parm_float(parm, "alpha", 0.0001),
        ), True, False

    if model_class == "randomForest":
        depth = parm_int(parm, "treeMaxHeight", 0)
        return RandomForestClassifier(
            n_estimators=parm_int(parm, "estimators", 100),
            max_depth=(depth if depth > 0 else None),
        ), False, False

    if model_class == "tree":
        depth = parm_int(parm, "maxHeight", 10)
        return DecisionTreeClassifier(max_depth=(depth if depth > 0 else None)), False, False

    if model_class == "adaBoost":
        return AdaBoostClassifier(
            n_estimators=parm_int(parm, "estimators", 100),
            learning_rate=parm_float(parm, "rate", 1.0),
        ), False, False

    if model_class == "logisticRegression":
        return LogisticRegression(
            C=parm_float(parm, "c", 1.0),
            max_iter=parm_int(parm, "maxIter", 200),
        ), True, False

    if model_class == "ridge":
        return Ridge(alpha=parm_float(parm, "alpha", 1.0)), True, False

    if model_class == "knnRegressor":
        return KNeighborsRegressor(n_neighbors=parm_int(parm, "k", 5)), True, False

    if model_class == "svr":
        return SVR(C=parm_float(parm, "c", 1.0), epsilon=parm_float(parm, "epsilon", 0.1)), True, False

    if model_class == "mlpRegressor":
        return MLPRegressor(
            hidden_layer_sizes=parm_layers(parm, "layers", (128, 64)),
            max_iter=parm_int(parm, "epochs", 300),
            learning_rate_init=parm_float(parm, "learningRate", 0.001),
            alpha=parm_float(parm, "alpha", 0.0001),
        ), True, False

    if model_class == "gradientBoost":
        return GradientBoostingRegressor(
            n_estimators=parm_int(parm, "estimators", 100),
            learning_rate=parm_float(parm, "rate", 0.1),
            max_depth=parm_int(parm, "treeMaxHeight", 3),
        ), False, False

    if model_class == "kmeans":
        return KMeans(n_clusters=parm_int(parm, "k", 4)), False, False

    if model_class == "gaussianMixture":
        return GaussianMixture(n_components=parm_int(parm, "k", 4)), False, True

    if model_class == "isolationForest":
        return IsolationForest(
            n_estimators=parm_int(parm, "estimators", 100),
            contamination=parm_float(parm, "contamination", 0.1),
        ), False, False

    if model_class == "localOutlierFactor":
        return LocalOutlierFactor(
            n_neighbors=parm_int(parm, "k", 20),
            contamination=parm_float(parm, "contamination", 0.1),
            novelty=True,
        ), True, True

    if model_class == "robustZScore":
        if text:
            raise ValueError("robustZScore works on numeric data only.")
        return RobustZScore(threshold=parm_float(parm, "threshold", 3.5)), False, False

    raise ValueError("Unknown model class: " + str(model_class))


# ---------------------------------------------------------------------------------
# Sample parsing. Text samples are plain strings. Numeric samples become float
# lists, accepting a real array, a comma separated string, or a single number.

def parse_numeric(value):
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip() != ""]
        return [float(p) for p in parts]
    return [float(value)]


def prep_sample(value, data_type):
    if data_type == "text":
        return str(value)
    return parse_numeric(value)


def load_dataset(dataset_path, category, data_type):
    if not os.path.isfile(dataset_path):
        raise ValueError("The dataset is empty. Add training rows first.")

    needs_labels = category in ("classifier", "regressor")
    samples = []
    labels = []

    with open(dataset_path, "r") as f:
        for line in f:
            line = line.strip()
            if len(line) < 1:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not isinstance(row, dict) or "input" not in row:
                continue
            if needs_labels:
                if "label" not in row or row["label"] is None:
                    continue
                labels.append(float(row["label"]) if category == "regressor" else str(row["label"]))
            samples.append(prep_sample(row["input"], data_type))

    if len(samples) < 2:
        raise ValueError("Training needs at least 2 valid dataset rows.")
    return samples, labels, needs_labels


def build_matrix(samples, data_type):
    if data_type == "text":
        return samples
    width = max(len(s) for s in samples)
    fixed = [s + [0.0] * (width - len(s)) for s in samples]
    return np.asarray(fixed, dtype=float)


def to_native(value):
    if hasattr(value, "item"):
        return value.item()
    return value


# ---------------------------------------------------------------------------------
# Operations.

def op_train(payload):
    model_class = str(payload.get("modelClass", ""))
    category = str(payload.get("category", "classifier"))
    data_type = str(payload.get("dataType", "text"))
    model_path = str(payload.get("modelPath", ""))
    dataset_path = str(payload.get("datasetPath", ""))
    parm = payload.get("modelParm", {})
    if not isinstance(parm, dict):
        try:
            parm = json.loads(str(parm))
        except ValueError:
            parm = {}
    if not isinstance(parm, dict):
        parm = {}

    if model_class == "dbscan":
        return {"error": "DBSCAN does not train. It clusters at prediction time from its dataset."}
    if len(model_path) < 1 or len(dataset_path) < 1:
        return {"error": "modelPath and datasetPath are required."}

    estimator, wants_scaler, wants_dense = build_estimator(model_class, parm, data_type)
    samples, labels, needs_labels = load_dataset(dataset_path, category, data_type)
    X = build_matrix(samples, data_type)

    steps = []
    if data_type == "text":
        steps.append(("tfidf", TfidfVectorizer(max_features=10000)))
    if wants_dense and data_type == "text":
        steps.append(("dense", DenseTransformer()))
    if wants_scaler:
        steps.append(("scaler", StandardScaler(with_mean=(data_type != "text"))))
    steps.append(("model", estimator))
    pipe = Pipeline(steps)

    if needs_labels:
        if category == "classifier" and len(set(labels)) < 2:
            return {"error": "Classifier training needs at least 2 different labels in the dataset."}
        pipe.fit(X, labels)
    else:
        pipe.fit(X)

    numeric_width = 0 if data_type == "text" else X.shape[1]
    bundle = {
        "pipeline": pipe,
        "modelClass": model_class,
        "category": category,
        "dataType": data_type,
        "numericWidth": numeric_width,
        "samples": len(samples),
    }

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    return {"message": "Model trained from " + str(len(samples)) + " rows.", "samples": len(samples)}


def op_predict(payload):
    model_path = str(payload.get("modelPath", ""))
    if not os.path.isfile(model_path):
        return {"error": "This model has not been trained yet."}

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)

    pipe = bundle["pipeline"]
    category = bundle.get("category", "classifier")
    data_type = bundle.get("dataType", "text")

    sample = prep_sample(payload.get("input", ""), data_type)
    if data_type == "text":
        X = [sample]
    else:
        width = int(bundle.get("numericWidth", len(sample)))
        if len(sample) < width:
            sample = sample + [0.0] * (width - len(sample))
        X = np.asarray([sample[:width]], dtype=float)

    pred = to_native(pipe.predict(X)[0])

    if category == "anomaly":
        return {"output": 1 if pred == -1 else 0}
    if category == "regressor":
        return {"output": float(pred)}
    if category == "clusterer":
        return {"output": int(pred)}
    return {"output": str(pred)}


def op_dbscan(payload):
    dataset_path = str(payload.get("datasetPath", ""))
    data_type = str(payload.get("dataType", "text"))
    parm = payload.get("modelParm", {})
    if not isinstance(parm, dict):
        parm = {}

    samples = []
    if os.path.isfile(dataset_path):
        with open(dataset_path, "r") as f:
            for line in f:
                line = line.strip()
                if len(line) < 1:
                    continue
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if isinstance(row, dict) and "input" in row:
                    samples.append(prep_sample(row["input"], data_type))

    samples.append(prep_sample(payload.get("input", ""), data_type))
    if len(samples) < 2:
        return {"output": 0, "notice": "DBSCAN needs reference data, add rows to this step's dataset."}

    if data_type == "text":
        X = TfidfVectorizer(max_features=10000).fit_transform(samples)
    else:
        X = build_matrix(samples, "numeric")

    dbscan = DBSCAN(eps=parm_float(parm, "radius", 0.5), min_samples=parm_int(parm, "minDensity", 5))
    predictions = dbscan.fit_predict(X)
    return {"output": int(predictions[-1])}


def handle(payload):
    try:
        op = str(payload.get("op", ""))
        if op == "train":
            return op_train(payload)
        if op == "predict":
            return op_predict(payload)
        if op == "dbscan":
            return op_dbscan(payload)
        if op == "health":
            return {"ok": 1}
        return {"error": "Unknown op. Valid ops: train, predict, dbscan, health."}
    except Exception as e:  # noqa: BLE001, every failure must come back as JSON
        return {"error": str(e)}


# ---------------------------------------------------------------------------------
# Entry points.

def run_stdin():
    try:
        payload = json.loads(sys.stdin.read())
    except ValueError:
        print(json.dumps({"error": "Worker received invalid JSON on stdin."}))
        return
    if not isinstance(payload, dict):
        payload = {}
    print(json.dumps(handle(payload)))


def serve(host, port):
    from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

    class Handler(BaseHTTPRequestHandler):
        def _send(self, result):
            body = json.dumps(result).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send({"ok": 1})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
            except ValueError:
                self._send({"error": "Worker received invalid JSON."})
                return
            if not isinstance(payload, dict):
                payload = {}
            self._send(handle(payload))

        def log_message(self, fmt, *args):
            pass  # stay quiet, PHP reports errors to the user

    server = ThreadingHTTPServer((host, port), Handler)
    print("ML worker listening on " + host + ":" + str(port))
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ML Prediction Engine worker")
    parser.add_argument("mode", choices=["run", "serve"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8750)
    args = parser.parse_args()

    if args.mode == "run":
        run_stdin()
    else:
        serve(args.host, args.port)
