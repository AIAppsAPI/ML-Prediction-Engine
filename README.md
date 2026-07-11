# ML Prediction Engine

A self-hosted machine learning API: train classifiers, regressors, clusterers, and anomaly detectors on your own data, then call them from anything over HTTPS. PHP serves the API and admin area, a small Python worker runs the actual scikit-learn models, and everything lives in one SQLite file plus plain dataset and model files. No ML expertise needed, pick a model, add rows, click train.

ML Prediction Engine is the open source release of Data Aggregator, the same idea that runs as a hosted app on aiappsapi.com. If you would rather not run your own server, the [Data Aggregator app](https://www.aiappsapi.com/apps/dataaggregator.php) is ready to go.

## What you get

- **18 model classes**: classifiers (K-Nearest Neighbors, Naive Bayes, neural network MLP, Random Forest, Decision Tree, AdaBoost, Logistic Regression), regressors (Ridge, KNN, Support Vector, MLP, Gradient Boost), clusterers (K-Means, DBSCAN, Gaussian Mixture), and anomaly detectors (Isolation Forest, Local Outlier Factor, Robust Z-Score).
- **Text or numeric inputs**: text runs through tf-idf automatically, numeric takes lists of numbers. Send "refund my order please" or "1500, 3, 0.7", both just work.
- **Pipelines**: chain a short sequence of models where each step's output feeds the next. Most pipelines are a single model, sequences are there when you need them.
- **Datasets you can see**: every model's training data is a visible, editable table in the admin, and a jsonL file on disk. Add rows over the API or in the browser, retrain any time, the dataset is always the source of truth.
- **Live training**: optionally let a pipeline learn from its own predictions. New rows are appended as they happen and the model retrains itself on a schedule you set.
- **Admin area**: build pipelines, tune model parameters with inline help, manage datasets, train with one click, and test predictions right on the page.
- **AI data summaries (optional)**: a /summarize endpoint that has an LLM write a factual summary of any data you send, stored in a reports history. Works with Anthropic or any OpenAI compatible API, and the ML side never needs an LLM key.

## How it is put together

PHP owns the API, the admin, the pipeline configs, and the dataset files. The Python worker (one file, scikit-learn only) does the training and predicting. PHP talks to it two ways, set by `mlDriver` in config.php:

- **service** (recommended): the worker runs as a tiny HTTP service on localhost. Docker compose starts it automatically as a second container.
- **cli**: PHP runs `python3 worker.py` per request. No daemon to manage, fine for light traffic on any box with Python.

## Quick start (Docker)

```
cp config.sample.php config.php
# edit config.php: set apiKey and adminPassword,
# and set mlServiceUrl to "http://ml:8750"
docker compose up --build
```

The app is at `http://localhost:8080/`, log into `admin.php`, create a pipeline, add a few rows, train, and test a prediction on the same page.

## Quick start (local, no Docker)

Requires PHP 8.1+ with pdo_sqlite, and Python 3.9+ with scikit-learn (`pip install scikit-learn`).

```
cp config.sample.php config.php
# edit config.php: set apiKey and adminPassword
python3 python/worker.py serve &
php -S localhost:8080 -t public
```

Or skip the worker service entirely by setting `mlDriver` to `cli` in config.php.

## Project structure

```
config.sample.php   copy to config.php, holds keys and settings
public/
  api.php           the JSON API your apps call
  admin.php         admin area for pipelines, datasets, training, testing
lib/
  ml.php            pipeline orchestration and the worker drivers
  llm.php           chat model drivers for the optional AI summaries
  db.php            SQLite storage layer
python/
  worker.py         the scikit-learn worker (HTTP service or CLI, same file)
data/               SQLite database, datasets, and trained models (gitignored)
```

## Configuration

Everything is in `config.php` (copied from `config.sample.php`):

| Setting | What it does |
| --- | --- |
| apiKey | the key clients must send with every API call |
| adminPassword | password for admin.php |
| corsOrigin | which browser origins may call the API |
| dbPath | SQLite file location |
| dataPath | folder for datasets and trained models |
| mlDriver | "service" or "cli", how PHP reaches the Python worker |
| mlServiceUrl | worker service address, `http://127.0.0.1:8750` local, `http://ml:8750` in Docker |
| pythonPath | python executable for the cli driver |
| retrainEvery | default live training retrain interval (rows) |
| chatProvider + keys | optional, only for the /summarize endpoint |

## The workflow

1. Create a pipeline in the admin and add a model step (say, a KNN classifier on text).
2. Add labeled rows to its dataset, in the admin or through `dataset/add`.
3. Train, one click or one API call.
4. Call `predict` from your app. The response is the label, number, cluster, or anomaly flag.
5. Add rows and retrain whenever you like, or turn on live training and let it learn as it runs.

## API

All endpoints are POST with a JSON body. Send your key in the `X-API-Key` header (or as an `apiKey` field in the body). Paths use path info, and if your server does not support that, `api.php?action=predict` works the same as `api.php/predict`.

### POST api.php/predict

```
curl -X POST http://localhost:8080/api.php/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{"pipelineID": 1, "input": "why was i charged twice"}'
```

```
{"output": "billing", "steps": [{"step": 0, "class": "knn", "output": "billing"}]}
```

Fields: `pipelineID`, `input` (a string for text models, a number list or comma string for numeric models). `output` is the final answer, `steps` shows each step's output when the pipeline has several. Anomaly detectors answer 1 (anomaly) or 0 (normal), clusterers answer a cluster number (DBSCAN uses -1 for noise).

### POST api.php/train

Train one step from its stored dataset. Fields: `pipelineID`, `modelIndex` (0 for the first step). Returns `{"message": "Model trained from 120 rows.", "samples": 120}`.

### POST api.php/dataset/add

Append a training row. Fields: `pipelineID`, `modelIndex`, `input`, `label` (required for classifiers and regressors), and optional `train: "1"` to retrain immediately after adding.

### POST api.php/dataset/set

Replace a step's whole dataset. Fields: `pipelineID`, `modelIndex`, `filecontent` as jsonL text, one `{"input": ..., "label": ...}` per line.

### POST api.php/dataset/rows and dataset/clear

List a step's dataset rows, or empty it. Fields: `pipelineID`, `modelIndex`.

### POST api.php/pipelines

Lists every pipeline with its steps and training status, handy for wiring this engine into workflow tools.

### POST api.php/summarize

Optional AI summary of any data you send. Fields: `data` (an array of items), optional `prompt` and `label`. Needs an LLM key in config.php, and the summary is stored in the admin's AI Reports page. The ML endpoints never touch an LLM.

## Model classes and parameters

Set parameters per step as JSON in the admin, each class shows inline help. The most useful knobs:

| Class | Category | Parameters |
| --- | --- | --- |
| knn | classifier | k: neighbors to consider |
| naiveBayes | classifier | none (Multinomial for text, Gaussian for numeric) |
| dnn | classifier | layers, epochs, learningRate, alpha |
| randomForest | classifier | estimators, treeMaxHeight (0 = unlimited) |
| tree | classifier | maxHeight |
| adaBoost | classifier | estimators, rate |
| logisticRegression | classifier | c, maxIter |
| ridge | regressor | alpha |
| knnRegressor | regressor | k |
| svr | regressor | c, epsilon |
| mlpRegressor | regressor | layers, epochs, learningRate, alpha |
| gradientBoost | regressor | estimators, rate, treeMaxHeight |
| kmeans | clusterer | k |
| dbscan | clusterer | radius, minDensity (clusters at predict time, no training) |
| gaussianMixture | clusterer | k |
| isolationForest | anomaly | estimators, contamination |
| localOutlierFactor | anomaly | k, contamination |
| robustZScore | anomaly | threshold (numeric data only) |

Classifiers and regressors need labeled rows. Clusterers and anomaly detectors train on inputs alone, and anomaly detectors should be trained on normal data only so they can flag anything different. DBSCAN is special: it never trains, it clusters each prediction against its reference dataset on the spot.

## Live training

Turn it on per pipeline in the admin. Every prediction is appended to each step's dataset, labeled with the step's own output, and after `retrainEvery` new rows the step retrains itself automatically. The models learn from what they predict, so use it for data without an exact right answer (grouping, drift-following, engagement patterns), and keep it off when you want full control of the training data. A failed live retrain never breaks the prediction, it just reports a notice.

## Ideas to steal

- **Ticket routing**: classifier on subject lines, labels are your queues. Train on last month's solved tickets.
- **Lead scoring**: regressor on `[pages_viewed, email_opens, days_since_signup]`, label is closed revenue.
- **Spam and intent filtering**: classifier in front of your inbound SMS or email replies.
- **Ops anomaly alerts**: isolation forest on `[requests, errors, latency]` trained on normal days, alert on output 1.
- **Feedback clustering**: k-means over incoming reviews to watch new themes appear.

## Security notes

- Set `apiKey` and `adminPassword` to long random strings before exposing the server.
- Keep the worker service on localhost or a private network. It has no authentication of its own, it should only ever hear from your PHP app (the Docker setup already isolates it).
- Trained model files are Python pickles created only by this app. Do not drop model files from untrusted sources into the data folder, loading a hostile pickle can execute code.
- Serve over HTTPS in production.

## Roadmap

- Optional incremental training fast path for the model classes that support it, so live training on large datasets skips the full rebuild.
- Prediction probability and confidence scores in the predict response.
- Dataset import from CSV in the admin.

## Author

Paul Crinigan, [https://www.aiappsapi.com/](https://www.aiappsapi.com/)

## More from the author

- [Email Campaign Engine](https://github.com/AIAppsAPI/Email-Campaign-Engine), a self-hosted email campaign engine: broadcasts, drips, and webhooks over plain SMTP.
- [SMS Campaign Engine](https://github.com/AIAppsAPI/SMS-Campaign-Engine), the same idea for SMS, with AI chatbot replies.
- [AbraTabia AI Storyteller](https://github.com/AIAppsAPI/AbraTabia-AI-Storyteller), a self-hosted AI storyteller and game host API for games.
- [AbraTabia Game Server](https://github.com/AIAppsAPI/AbraTabia-Game-Server), a self-hosted matchmaking and turn sync server for turn-based multiplayer games.
- [Auto Learning Agents](https://www.autolearningagents.com/get-started/), AI agents that learn your business and get better every day.
- [Adaptive Recall](https://www.adaptiverecall.com/features.php), a persistent memory system for AI assistants.
- [ACP Payments Module](https://www.afcommerce.com/free-acp-payments-module/), a free agentic commerce payments module.

## License

MIT, see LICENSE.
