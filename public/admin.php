<?php

session_start();

require_once __DIR__ . "/../lib/ml.php";

// Admin area: build pipelines, configure model steps, manage datasets, train,
// and test predictions. Login uses the adminPassword value from config.php.

$config = GetConfig();
$adminPassword = (string)($config["adminPassword"] ?? "");

function h($text) { return htmlspecialchars((string)$text, ENT_QUOTES, "UTF-8"); }

function AgoText($ts) {
$ts = (int)$ts;
if ($ts < 1) return "-";
$diff = time() - $ts;
if ($diff < 0) $diff = 0;
if ($diff < 60) return $diff . "s ago";
if ($diff < 3600) return floor($diff / 60) . "m ago";
if ($diff < 86400) return floor($diff / 3600) . "h ago";
return floor($diff / 86400) . "d ago";
} // ends function

// ---------------------------------------------------------------------------------
// Model catalog: class lists per category, default parameters, and help text.

$modelClasses = array(
	"classifier" => array(
		"knn" => "K-Nearest Neighbors",
		"naiveBayes" => "Naive Bayes",
		"dnn" => "Neural Network (MLP)",
		"randomForest" => "Random Forest",
		"tree" => "Decision Tree",
		"adaBoost" => "AdaBoost",
		"logisticRegression" => "Logistic Regression",
	),
	"regressor" => array(
		"ridge" => "Ridge Regression",
		"knnRegressor" => "KNN Regressor",
		"svr" => "Support Vector Regression",
		"mlpRegressor" => "MLP Regressor",
		"gradientBoost" => "Gradient Boost",
	),
	"clusterer" => array(
		"kmeans" => "K-Means",
		"dbscan" => "DBSCAN (clusters at predict time)",
		"gaussianMixture" => "Gaussian Mixture (soft clustering)",
	),
	"anomaly" => array(
		"isolationForest" => "Isolation Forest",
		"localOutlierFactor" => "Local Outlier Factor",
		"robustZScore" => "Robust Z-Score (numeric only)",
	),
);

$defaultParms = array(
	"knn" => '{"k": 5}',
	"naiveBayes" => '{}',
	"dnn" => '{"layers": [128, 64], "epochs": 300, "learningRate": 0.001, "alpha": 0.0001}',
	"randomForest" => '{"estimators": 100, "treeMaxHeight": 0}',
	"tree" => '{"maxHeight": 10}',
	"adaBoost" => '{"estimators": 100, "rate": 1.0}',
	"logisticRegression" => '{"c": 1.0, "maxIter": 200}',
	"ridge" => '{"alpha": 1.0}',
	"knnRegressor" => '{"k": 5}',
	"svr" => '{"c": 1.0, "epsilon": 0.1}',
	"mlpRegressor" => '{"layers": [128, 64], "epochs": 300, "learningRate": 0.001, "alpha": 0.0001}',
	"gradientBoost" => '{"estimators": 100, "rate": 0.1, "treeMaxHeight": 3}',
	"kmeans" => '{"k": 4}',
	"dbscan" => '{"radius": 0.5, "minDensity": 5}',
	"gaussianMixture" => '{"k": 4}',
	"isolationForest" => '{"estimators": 100, "contamination": 0.1}',
	"localOutlierFactor" => '{"k": 20, "contamination": 0.1}',
	"robustZScore" => '{"threshold": 3.5}',
);

$parmHelp = array(
	"knn" => "k: number of nearest neighbors to consider",
	"naiveBayes" => "No parameters needed. Text data uses Multinomial NB, numeric uses Gaussian NB",
	"dnn" => "layers: hidden layer sizes. epochs: training iterations. learningRate: step size. alpha: regularization strength",
	"randomForest" => "estimators: number of trees. treeMaxHeight: max depth per tree, 0 for unlimited",
	"tree" => "maxHeight: maximum tree depth, 0 for unlimited",
	"adaBoost" => "estimators: number of weak learners. rate: learning rate",
	"logisticRegression" => "c: inverse regularization, higher fits tighter. maxIter: solver iterations",
	"ridge" => "alpha: regularization strength",
	"knnRegressor" => "k: number of nearest neighbors",
	"svr" => "c: error penalty. epsilon: tolerance margin",
	"mlpRegressor" => "layers, epochs, learningRate, alpha: same as the neural network classifier",
	"gradientBoost" => "estimators: boosting rounds. rate: learning rate. treeMaxHeight: tree depth",
	"kmeans" => "k: number of clusters",
	"dbscan" => "radius: neighborhood size. minDensity: minimum points per cluster. Returns -1 for noise points",
	"gaussianMixture" => "k: number of mixture components (clusters)",
	"isolationForest" => "estimators: number of trees. contamination: expected anomaly fraction (0 to 0.5)",
	"localOutlierFactor" => "k: neighbors for density comparison. contamination: expected anomaly fraction",
	"robustZScore" => "threshold: modified z-score above which a point is flagged as an anomaly",
);

// ---------------------------------------------------------------------------------
// Login and logout

if (($_GET["do"] ?? "") == "logout") {
	$_SESSION = array();
	session_destroy();
	header("Location: admin.php");
	exit;
}

if (($_POST["do"] ?? "") == "login") {
	if (strlen($adminPassword) < 8) $noticeError = "Set adminPassword in config.php (at least 8 characters) before logging in.";
	else if (hash_equals($adminPassword, (string)($_POST["password"] ?? ""))) {
		session_regenerate_id(true);
		$_SESSION["mlAdmin"] = 1;
		$_SESSION["csrf"] = bin2hex(random_bytes(16));
		header("Location: admin.php");
		exit;
	}
	else $noticeError = "Wrong password.";
}

$loggedIn = !empty($_SESSION["mlAdmin"]);
$notice = $_GET["ok"] ?? "";
$noticeError = $noticeError ?? "";

// ---------------------------------------------------------------------------------
// Shared page style

function PageTop($title, $loggedIn) {
?><!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title><?php echo h($title); ?></title>
<style>
body { font-family: Arial, Helvetica, sans-serif; margin: 0; background: #f4f5f7; color: #222; }
.topbar { background: #1f2733; color: #fff; padding: 12px 20px; display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }
.topbar strong { font-size: 18px; }
.topbar a { color: #cfd8e3; text-decoration: none; padding: 4px 8px; border-radius: 4px; }
.topbar a:hover { background: #334052; color: #fff; }
.wrap { max-width: 980px; margin: 20px auto; padding: 0 16px; }
.card { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 18px; margin-bottom: 18px; }
.card h2 { margin-top: 0; font-size: 18px; }
table { width: 100%; border-collapse: collapse; }
th, td { text-align: left; padding: 8px 6px; border-bottom: 1px solid #eee; font-size: 14px; vertical-align: top; }
input[type=text], input[type=password], input[type=number], select, textarea {
	width: 100%; padding: 8px; margin: 4px 0 12px; box-sizing: border-box; border: 1px solid #bbb; border-radius: 5px; font-size: 14px; }
textarea { font-family: monospace; }
label { font-weight: bold; font-size: 13px; }
button { background: #2b6cb0; color: #fff; border: 0; padding: 8px 16px; border-radius: 6px; cursor: pointer; font-size: 14px; }
button:hover { background: #234f80; }
button.danger { background: #b03030; }
button.danger:hover { background: #8a2424; }
button.small { padding: 4px 10px; font-size: 12px; }
.notice { padding: 10px 14px; border-radius: 6px; margin-bottom: 16px; }
.notice.ok { background: #e2f2e4; border: 1px solid #9ec9a3; }
.notice.err { background: #f7e3e3; border: 1px solid #d3a2a2; }
.mono { font-family: monospace; font-size: 13px; }
.small { font-size: 12px; color: #666; }
.badge { display: inline-block; background: #e2f2e4; border: 1px solid #9ec9a3; border-radius: 10px; font-size: 11px; padding: 2px 8px; margin-left: 6px; }
.badge.gray { background: #eee; border-color: #ccc; color: #666; }
.settingsrow { display: flex; gap: 14px; flex-wrap: wrap; }
.settingsrow > div { flex: 1; min-width: 200px; }
.result { margin-top: 8px; font-size: 13px; white-space: pre-wrap; }
.inlineform { display: inline; }
</style></head><body>
<div class="topbar"><strong>ML Prediction Engine</strong>
<?php if ($loggedIn) { ?>
<a href="admin.php">Pipelines</a>
<a href="admin.php?page=reports">AI Reports</a>
<a href="admin.php?do=logout">Log out</a>
<?php } ?>
</div><div class="wrap">
<?php
} // ends function

function PageBottom() {
echo "</div></body></html>";
} // ends function

// ---------------------------------------------------------------------------------
// Not logged in: show the login page and stop

if (!$loggedIn) {
	PageTop("ML Prediction Engine Login", false);
	if (strlen($noticeError) > 0) echo "<div class='notice err'>" . h($noticeError) . "</div>";
	?>
	<div class="card" style="max-width:420px;margin:40px auto;">
	<h2>Admin Login</h2>
	<form method="post" action="admin.php">
	<input type="hidden" name="do" value="login">
	<label>Password</label>
	<input type="password" name="password" autofocus>
	<button type="submit">Log in</button>
	</form>
	</div>
	<?php
	PageBottom();
	exit;
}

$csrf = $_SESSION["csrf"] ?? "";
if (strlen($csrf) < 10) { $_SESSION["csrf"] = bin2hex(random_bytes(16)); $csrf = $_SESSION["csrf"]; }

// ---------------------------------------------------------------------------------
// AJAX actions (train and test predict run long, so they answer inline as JSON)

if (($_POST["ajax"] ?? "") == "1") {
	header("Content-Type: application/json");
	if (!hash_equals($csrf, (string)($_POST["csrf"] ?? ""))) { echo json_encode(array("error" => "Invalid form token, reload the page.")); exit; }
	set_time_limit(0);

	if (($_POST["do"] ?? "") == "train") {
		echo json_encode(MLTrainStep((int)($_POST["pipelineID"] ?? 0), (int)($_POST["modelIndex"] ?? -1)));
		exit;
	}
	if (($_POST["do"] ?? "") == "testpredict") {
		echo json_encode(MLPredictPipeline(array("pipelineID" => (int)($_POST["pipelineID"] ?? 0), "input" => (string)($_POST["input"] ?? ""))));
		exit;
	}
	echo json_encode(array("error" => "Unknown ajax action."));
	exit;
}

// ---------------------------------------------------------------------------------
// Handle form POST actions, then redirect

if (($_SERVER["REQUEST_METHOD"] ?? "") == "POST" && ($_POST["do"] ?? "") != "login") {

if (!hash_equals($csrf, (string)($_POST["csrf"] ?? ""))) {
	$noticeError = "Invalid form token, please try again.";
} else {

$pipelineID = (int)($_POST["pipelineID"] ?? 0);
$modelIndex = (int)($_POST["modelIndex"] ?? -1);

switch ($_POST["do"] ?? "")
{
	case "createpipeline":
		$name = trim((string)($_POST["name"] ?? ""));
		if (strlen($name) < 1) $name = "New Pipeline";
		$newID = SavePipeline(0, array("name" => $name, "liveTraining" => "0", "retrainEvery" => 0, "models" => array()));
		header("Location: admin.php?pipeline=" . $newID . "&ok=" . urlencode("Pipeline created."));
		exit;
	break;

	case "deletepipeline":
		$pipeline = GetPipeline($pipelineID);
		if ($pipeline) {
			foreach (($pipeline["models"] ?? array()) as $idx => $step) MLDeleteStepFiles($pipelineID, $idx);
			DeletePipelineRow($pipelineID);
		}
		header("Location: admin.php?ok=" . urlencode("Pipeline deleted."));
		exit;
	break;

	case "savesettings":
		$pipeline = GetPipeline($pipelineID);
		if ($pipeline) {
			$name = trim((string)($_POST["name"] ?? ""));
			if (strlen($name) > 0) $pipeline["name"] = $name;
			$pipeline["liveTraining"] = (($_POST["liveTraining"] ?? "") == "1") ? "1" : "0";
			$retrainEvery = (int)($_POST["retrainEvery"] ?? 0);
			$pipeline["retrainEvery"] = ($retrainEvery > 0) ? $retrainEvery : 0;
			SavePipeline($pipelineID, $pipeline);
		}
		header("Location: admin.php?pipeline=" . $pipelineID . "&ok=" . urlencode("Settings saved."));
		exit;
	break;

	case "addstep":
		$pipeline = GetPipeline($pipelineID);
		if ($pipeline) {
			if (!isset($pipeline["models"]) || !is_array($pipeline["models"])) $pipeline["models"] = array();
			$pipeline["models"][] = array("class" => "knn", "category" => "classifier", "dataType" => "text", "modelParm" => array("k" => 5));
			SavePipeline($pipelineID, $pipeline);
		}
		header("Location: admin.php?pipeline=" . $pipelineID . "&ok=" . urlencode("Model step added."));
		exit;
	break;

	case "removestep":
		$pipeline = GetPipeline($pipelineID);
		if ($pipeline && isset($pipeline["models"][$modelIndex])) {
			MLDeleteStepFiles($pipelineID, $modelIndex);
			$pipeline["models"][$modelIndex] = array();
			SavePipeline($pipelineID, $pipeline);
		}
		header("Location: admin.php?pipeline=" . $pipelineID . "&ok=" . urlencode("Model step removed."));
		exit;
	break;

	case "savestep":
		$pipeline = GetPipeline($pipelineID);
		if ($pipeline && isset($pipeline["models"][$modelIndex]) && is_array($pipeline["models"][$modelIndex])) {
			$step = $pipeline["models"][$modelIndex];
			$category = (string)($_POST["category"] ?? "classifier");
			if (!isset($modelClasses[$category])) $category = "classifier";
			$class = (string)($_POST["class"] ?? "");
			if (!isset($modelClasses[$category][$class])) $class = array_keys($modelClasses[$category])[0];
			$dataType = ((string)($_POST["dataType"] ?? "text") == "numeric") ? "numeric" : "text";
			$parm = @json_decode((string)($_POST["modelParm"] ?? "{}"), true);
			if (!is_array($parm)) {
				header("Location: admin.php?pipeline=" . $pipelineID . "&ok=" . urlencode("Step NOT saved, the parameters box is not valid JSON."));
				exit;
			}
			$step["category"] = $category;
			$step["class"] = $class;
			$step["dataType"] = $dataType;
			$step["modelParm"] = $parm;
			$pipeline["models"][$modelIndex] = $step;
			SavePipeline($pipelineID, $pipeline);
		}
		header("Location: admin.php?pipeline=" . $pipelineID . "&ok=" . urlencode("Step saved. Retrain the model for changes to take effect."));
		exit;
	break;

	case "addrow":
		$pipeline = GetPipeline($pipelineID);
		$step = ($pipeline["models"] ?? array())[$modelIndex] ?? null;
		if (is_array($step)) {
			$input = (string)($_POST["input"] ?? "");
			$label = $_POST["label"] ?? null;
			$category = $step["category"] ?? "classifier";
			if (strlen(trim($input)) < 1) { header("Location: admin.php?pipeline=" . $pipelineID . "&step=" . $modelIndex . "&ok=" . urlencode("Input is required.")); exit; }
			if (MLNeedsLabels($category) && strlen(trim((string)$label)) < 1) { header("Location: admin.php?pipeline=" . $pipelineID . "&step=" . $modelIndex . "&ok=" . urlencode("A label is required for this model type.")); exit; }
			MLDatasetAppendRow($pipelineID, $modelIndex, $input, MLNeedsLabels($category) ? $label : null, $category);
		}
		header("Location: admin.php?pipeline=" . $pipelineID . "&step=" . $modelIndex . "&ok=" . urlencode("Row added."));
		exit;
	break;

	case "deleterow":
		MLDatasetDeleteRow($pipelineID, $modelIndex, (int)($_POST["rowIndex"] ?? -1));
		header("Location: admin.php?pipeline=" . $pipelineID . "&step=" . $modelIndex . "&ok=" . urlencode("Row deleted."));
		exit;
	break;

	case "cleardataset":
		MLDatasetClear($pipelineID, $modelIndex);
		header("Location: admin.php?pipeline=" . $pipelineID . "&step=" . $modelIndex . "&ok=" . urlencode("Dataset cleared."));
		exit;
	break;
}
} // ends csrf check
}

// ---------------------------------------------------------------------------------
// AI reports page

if (($_GET["page"] ?? "") == "reports") {

PageTop("AI Reports", true);
echo "<div class='card'><h2>AI Data Summaries</h2>";
echo "<p class='small'>Summaries created through the /summarize API endpoint. This feature needs an LLM key in config.php, the ML side never uses one.</p>";

$reports = ReportList(50);
if (count($reports) < 1) echo "<p class='small'>No AI reports yet.</p>";
else {
	foreach ($reports as $report) {
		echo "<h3>" . h($report["label"]) . " <span class='small'>(" . AgoText($report["created"]) . ")</span></h3>";
		echo "<p style='white-space: pre-wrap;'>" . h($report["summary"]) . "</p><hr>";
	}
}
echo "</div>";
PageBottom();
exit;
}

// ---------------------------------------------------------------------------------
// Dataset page (?pipeline=ID&step=N)

$viewPipeline = (int)($_GET["pipeline"] ?? 0);
$viewStep = ($_GET["step"] ?? "");

if ($viewPipeline > 0 && strlen((string)$viewStep) > 0) {

$modelIndex = (int)$viewStep;
$pipeline = GetPipeline($viewPipeline);
$step = ($pipeline["models"] ?? array())[$modelIndex] ?? null;

PageTop("Dataset", true);
if (strlen($notice) > 0) echo "<div class='notice ok'>" . h($notice) . "</div>";

if (!$pipeline || !is_array($step) || strlen($step["class"] ?? "") < 1) {
	echo "<div class='card'><h2>Step not found</h2><p><a href='admin.php'>Back to pipelines</a></p></div>";
	PageBottom();
	exit;
}

$category = $step["category"] ?? "classifier";
$dataType = $step["dataType"] ?? "text";
$needsLabels = MLNeedsLabels($category);
$isDBSCAN = (($step["class"] ?? "") == "dbscan");
$rows = MLDatasetRows($viewPipeline, $modelIndex);

echo "<div class='card'><h2>" . h($pipeline["name"] ?? "") . ", Step " . ($modelIndex + 1) . " (" . h($step["class"]) . ") Dataset</h2>";
echo "<p><a href='admin.php?pipeline=" . $viewPipeline . "'>Back to pipeline</a></p>";
if ($isDBSCAN) echo "<p class='small'>DBSCAN clusters each prediction against this reference dataset, there is no training step. Add the reference rows it should cluster against.</p>";
else echo "<p class='small'>The dataset is the source of truth for this model. Add or upload rows here, then train. Training always rebuilds the model from the complete dataset.</p>";
echo "</div>";

echo "<div class='card'><h2>Add Training Row</h2>";
echo "<form method='post' action='admin.php'>";
echo "<input type='hidden' name='do' value='addrow'><input type='hidden' name='csrf' value='" . h($csrf) . "'>";
echo "<input type='hidden' name='pipelineID' value='" . $viewPipeline . "'><input type='hidden' name='modelIndex' value='" . $modelIndex . "'>";
echo "<label>Input</label>";
if ($dataType == "text") echo "<input type='text' name='input' placeholder='Enter text input'>";
else echo "<input type='text' name='input' placeholder='Comma separated numbers, like 1.5, 2.3, 4.0'>";
if ($needsLabels) {
	if ($category == "regressor") { echo "<label>Label (number)</label><input type='text' name='label' placeholder='like 350000'>"; }
	else { echo "<label>Label (category)</label><input type='text' name='label' placeholder='like spam'>"; }
}
echo "<button type='submit'>Add Row</button>";
echo "</form></div>";

if (!$isDBSCAN) {
	echo "<div class='card'><h2>Train</h2>";
	echo "<p class='small'>Rebuilds the model from all " . count($rows) . " dataset rows.</p>";
	echo "<button onclick='trainStep(" . $viewPipeline . ", " . $modelIndex . ", this)'>Train Model Now</button>";
	echo "<div class='result' id='trainResult'></div>";
	echo "</div>";
}

echo "<div class='card'><h2>Dataset Rows (" . count($rows) . ")</h2>";
if (count($rows) < 1) echo "<p class='small'>No rows yet.</p>";
else {
	echo "<table><tr><th>#</th><th>Input</th>";
	if ($needsLabels) echo "<th>Label</th>";
	echo "<th></th></tr>";
	foreach ($rows as $i => $row) {
		$inp = $row["input"] ?? "";
		if (is_array($inp)) $inp = json_encode($inp);
		echo "<tr><td>" . ($i + 1) . "</td><td class='mono'>" . h(substr((string)$inp, 0, 120)) . "</td>";
		if ($needsLabels) echo "<td>" . h((string)($row["label"] ?? "")) . "</td>";
		echo "<td><form method='post' action='admin.php' class='inlineform'>";
		echo "<input type='hidden' name='do' value='deleterow'><input type='hidden' name='csrf' value='" . h($csrf) . "'>";
		echo "<input type='hidden' name='pipelineID' value='" . $viewPipeline . "'><input type='hidden' name='modelIndex' value='" . $modelIndex . "'><input type='hidden' name='rowIndex' value='" . $i . "'>";
		echo "<button type='submit' class='danger small'>Delete</button></form></td></tr>";
	}
	echo "</table><br/>";

	echo "<form method='post' action='admin.php' onsubmit=\"return confirm('Clear the entire dataset for this step?');\">";
	echo "<input type='hidden' name='do' value='cleardataset'><input type='hidden' name='csrf' value='" . h($csrf) . "'>";
	echo "<input type='hidden' name='pipelineID' value='" . $viewPipeline . "'><input type='hidden' name='modelIndex' value='" . $modelIndex . "'>";
	echo "<button type='submit' class='danger'>Clear Entire Dataset</button></form>";
}
echo "</div>";

echo AdminScripts($csrf);
PageBottom();
exit;
}

// ---------------------------------------------------------------------------------
// Pipeline edit page (?pipeline=ID)

if ($viewPipeline > 0) {

$pipeline = GetPipeline($viewPipeline);

PageTop("Edit Pipeline", true);
if (strlen($notice) > 0) echo "<div class='notice ok'>" . h($notice) . "</div>";
if (strlen($noticeError) > 0) echo "<div class='notice err'>" . h($noticeError) . "</div>";

if (!$pipeline) {
	echo "<div class='card'><h2>Pipeline not found</h2><p><a href='admin.php'>Back to pipelines</a></p></div>";
	PageBottom();
	exit;
}

$models = $pipeline["models"] ?? array();
$liveTraining = (($pipeline["liveTraining"] ?? "0") == "1");
$configRetrain = (int)($config["retrainEvery"] ?? 20);

echo "<div class='card'><h2>" . h($pipeline["name"] ?? "") . "</h2>";
echo "<p class='small'>A pipeline is a short sequence of model steps. Predictions flow through in order, each step's output becomes the next step's input, and the last output is the answer. Most pipelines are a single step. For branching logic across many pipelines and APIs, chain them with a workflow tool and call /predict as one step.</p>";
echo "<p><a href='admin.php'>Back to all pipelines</a></p>";
echo "</div>";

// settings
echo "<div class='card'><h2>Pipeline Settings</h2>";
echo "<form method='post' action='admin.php'>";
echo "<input type='hidden' name='do' value='savesettings'><input type='hidden' name='csrf' value='" . h($csrf) . "'><input type='hidden' name='pipelineID' value='" . $viewPipeline . "'>";
echo "<div class='settingsrow'>";
echo "<div><label>Pipeline Name</label><input type='text' name='name' value='" . h($pipeline["name"] ?? "") . "'></div>";
echo "<div><label>Retrain Every (live rows)</label><input type='number' name='retrainEvery' min='0' value='" . (int)($pipeline["retrainEvery"] ?? 0) . "' placeholder='0'><span class='small'>0 uses the config default (" . $configRetrain . ")</span></div>";
echo "</div>";
echo "<label><input type='checkbox' name='liveTraining' value='1'" . ($liveTraining ? " checked" : "") . " style='width:auto;'> Enable live training</label>";
echo "<p class='small'>With live training on, every prediction is appended to each step's dataset with the step's own output as its label, and the step retrains automatically after enough new rows. The models learn from what they predict, so only use this where there is no exact right answer. To control exactly what the models learn, leave it off and train from curated datasets.</p>";
echo "<button type='submit'>Save Settings</button>";
echo "</form></div>";

// model steps
$stepNumber = 0;
foreach ($models as $si => $step) {
	if (!is_array($step) || strlen($step["class"] ?? "") < 1) continue;
	$stepNumber++;

	$stepClass = $step["class"];
	$stepCat = $step["category"] ?? "classifier";
	$stepDataType = $step["dataType"] ?? "text";
	$stepParm = $step["modelParm"] ?? array();
	$stepParmJson = is_array($stepParm) ? json_encode($stepParm) : (string)$stepParm;
	if (strlen($stepParmJson) < 2) $stepParmJson = $defaultParms[$stepClass] ?? "{}";
	$isDBSCAN = ($stepClass == "dbscan");
	$trained = (int)($step["trained"] ?? 0);
	$rowCount = count(MLDatasetRows($viewPipeline, $si));

	echo "<div class='card'><h2>Step " . $stepNumber . " (model index " . $si . ")";
	if ($isDBSCAN) echo "<span class='badge gray'>clusters at predict time, " . $rowCount . " reference rows</span>";
	else if ($trained > 0) echo "<span class='badge'>trained " . AgoText($trained) . ", " . (int)($step["trainedSamples"] ?? 0) . " rows</span>";
	else echo "<span class='badge gray'>not trained yet</span>";
	echo "</h2>";

	echo "<form method='post' action='admin.php'>";
	echo "<input type='hidden' name='do' value='savestep'><input type='hidden' name='csrf' value='" . h($csrf) . "'>";
	echo "<input type='hidden' name='pipelineID' value='" . $viewPipeline . "'><input type='hidden' name='modelIndex' value='" . $si . "'>";

	echo "<div class='settingsrow'>";
	echo "<div><label>Data Type</label><select name='dataType'>";
	echo "<option value='text'" . ($stepDataType == "text" ? " selected" : "") . ">Text (natural language)</option>";
	echo "<option value='numeric'" . ($stepDataType == "numeric" ? " selected" : "") . ">Numeric (list of numbers)</option>";
	echo "</select></div>";

	echo "<div><label>Category</label><select name='category' id='category_" . $si . "' onchange='updateClassOptions(" . $si . ")'>";
	foreach (array("classifier" => "Classifier (predicts a label)", "regressor" => "Regressor (predicts a number)", "clusterer" => "Clusterer (groups data)", "anomaly" => "Anomaly Detector") as $catVal => $catLabel) {
		echo "<option value='" . $catVal . "'" . ($stepCat == $catVal ? " selected" : "") . ">" . $catLabel . "</option>";
	}
	echo "</select></div>";

	echo "<div><label>Model Class</label><select name='class' id='class_" . $si . "' onchange='updateParmDefaults(" . $si . ")'>";
	$catClasses = $modelClasses[$stepCat] ?? $modelClasses["classifier"];
	foreach ($catClasses as $cls => $clsLabel) {
		echo "<option value='" . $cls . "'" . ($stepClass == $cls ? " selected" : "") . ">" . $clsLabel . "</option>";
	}
	echo "</select></div>";
	echo "</div>";

	echo "<label>Parameters (JSON)</label>";
	echo "<div class='small' id='parmHelp_" . $si . "'>" . h($parmHelp[$stepClass] ?? "") . "</div>";
	echo "<textarea name='modelParm' id='modelParm_" . $si . "' rows='2'>" . h($stepParmJson) . "</textarea>";

	echo "<button type='submit'>Save Step Config</button> ";
	echo "</form>";

	echo "<p>";
	echo "<a href='admin.php?pipeline=" . $viewPipeline . "&step=" . $si . "'>" . ($isDBSCAN ? "Manage Reference Dataset" : "Manage Dataset (" . $rowCount . " rows)") . "</a>";
	if (!$isDBSCAN) echo " &nbsp; <button class='small' onclick='trainStep(" . $viewPipeline . ", " . $si . ", this)'>Train Now</button><span class='result' id='trainResult_" . $si . "' style='margin-left:8px;'></span>";
	echo "</p>";

	echo "<form method='post' action='admin.php' class='inlineform' onsubmit=\"return confirm('Remove this step? Its trained model and dataset files will be deleted.');\">";
	echo "<input type='hidden' name='do' value='removestep'><input type='hidden' name='csrf' value='" . h($csrf) . "'>";
	echo "<input type='hidden' name='pipelineID' value='" . $viewPipeline . "'><input type='hidden' name='modelIndex' value='" . $si . "'>";
	echo "<button type='submit' class='danger small'>Remove Step</button></form>";

	echo "</div>";
}

if ($stepNumber < 1) echo "<div class='card'><p class='small'>No model steps yet. Add your first step below.</p></div>";

echo "<div class='card'>";
echo "<form method='post' action='admin.php' class='inlineform'>";
echo "<input type='hidden' name='do' value='addstep'><input type='hidden' name='csrf' value='" . h($csrf) . "'><input type='hidden' name='pipelineID' value='" . $viewPipeline . "'>";
echo "<button type='submit'>+ Add Model Step</button></form>";
echo "</div>";

// test prediction
echo "<div class='card'><h2>Test Prediction</h2>";
echo "<label>Input</label><input type='text' id='testInput' placeholder='Text input, or comma separated numbers for numeric pipelines'>";
echo "<button onclick='testPredict(" . $viewPipeline . ", this)'>Run Prediction</button>";
echo "<div class='result' id='testResult'></div>";
echo "</div>";

// delete pipeline
echo "<div class='card'><h2>Delete Pipeline</h2>";
echo "<form method='post' action='admin.php' onsubmit=\"return confirm('Delete this entire pipeline? All trained models and datasets will be permanently deleted.');\">";
echo "<input type='hidden' name='do' value='deletepipeline'><input type='hidden' name='csrf' value='" . h($csrf) . "'><input type='hidden' name='pipelineID' value='" . $viewPipeline . "'>";
echo "<button type='submit' class='danger'>Delete This Pipeline</button></form>";
echo "</div>";

echo AdminScripts($csrf, $modelClasses, $defaultParms, $parmHelp);
PageBottom();
exit;
}

// ---------------------------------------------------------------------------------
// Pipeline list page (default)

PageTop("ML Prediction Engine", true);
if (strlen($notice) > 0) echo "<div class='notice ok'>" . h($notice) . "</div>";

echo "<div class='card'><h2>Pipelines</h2>";
echo "<p class='small'>Pipelines are custom machine learning models you train on your own data, then call from anything over the API. Classifiers predict a category label, regressors predict a number, clusterers group similar items, and anomaly detectors flag unusual data after training on normal data only.</p>";

$pipelines = GetAllPipelines();
if (count($pipelines) < 1) echo "<p class='small'>No pipelines yet. Create your first one below.</p>";
else {
	echo "<table><tr><th>Pipeline</th><th>Steps</th><th>Live Training</th><th>Updated</th></tr>";
	foreach ($pipelines as $pipeline) {
		$stepCount = 0;
		foreach (($pipeline["models"] ?? array()) as $step) { if (is_array($step) && strlen($step["class"] ?? "") > 0) $stepCount++; }
		echo "<tr><td><a href='admin.php?pipeline=" . (int)$pipeline["pipelineID"] . "'>" . h($pipeline["name"] ?? ("Pipeline " . $pipeline["pipelineID"])) . "</a></td>";
		echo "<td>" . $stepCount . "</td><td>" . ((($pipeline["liveTraining"] ?? "0") == "1") ? "On" : "Off") . "</td><td>" . AgoText($pipeline["updated"] ?? 0) . "</td></tr>";
	}
	echo "</table>";
}
echo "</div>";

echo "<div class='card'><h2>Create New Pipeline</h2>";
echo "<form method='post' action='admin.php'>";
echo "<input type='hidden' name='do' value='createpipeline'><input type='hidden' name='csrf' value='" . h($csrf) . "'>";
echo "<label>Pipeline Name</label><input type='text' name='name' placeholder='Lead Scorer'>";
echo "<button type='submit'>Create Pipeline</button>";
echo "</form></div>";

PageBottom();

// ---------------------------------------------------------------------------------
// Shared admin JavaScript

function AdminScripts($csrf, $modelClasses = array(), $defaultParms = array(), $parmHelp = array()) {
$out = "<script>\n";
$out .= "var csrfToken = " . json_encode($csrf) . ";\n";
$out .= "var modelClasses = " . json_encode($modelClasses) . ";\n";
$out .= "var defaultParms = " . json_encode($defaultParms) . ";\n";
$out .= "var parmHelp = " . json_encode($parmHelp) . ";\n";
$out .= <<<'JS'

async function ajaxPost(data) {
	var fd = new FormData();
	fd.append("ajax", "1");
	fd.append("csrf", csrfToken);
	for (var k in data) fd.append(k, data[k]);
	var response = await fetch("admin.php", { method: "POST", body: fd });
	try { return await response.json(); } catch (err) { return { error: "Invalid response from the server." }; }
}

async function trainStep(pipelineID, modelIndex, btn) {
	var el = document.getElementById("trainResult_" + modelIndex) || document.getElementById("trainResult");
	el.textContent = "Training ...";
	btn.disabled = true;
	var r = await ajaxPost({ do: "train", pipelineID: pipelineID, modelIndex: modelIndex });
	btn.disabled = false;
	el.textContent = r.error ? ("Error: " + r.error) : (r.message || "Trained.");
}

async function testPredict(pipelineID, btn) {
	var input = document.getElementById("testInput").value;
	var el = document.getElementById("testResult");
	if (input.trim().length < 1) { el.textContent = "Enter an input first."; return; }
	el.textContent = "Running ...";
	btn.disabled = true;
	var r = await ajaxPost({ do: "testpredict", pipelineID: pipelineID, input: input });
	btn.disabled = false;
	if (r.error) { el.textContent = "Error: " + r.error; return; }
	var text = "Output: " + r.output;
	if (r.steps && r.steps.length > 1) {
		text += "\n";
		for (var i = 0; i < r.steps.length; i++) text += "\nStep " + (i + 1) + " (" + r.steps[i].class + "): " + r.steps[i].output;
	}
	if (r.notice) text += "\n\nNote: " + r.notice;
	el.textContent = text;
}

function updateClassOptions(stepIndex) {
	var cat = document.getElementById("category_" + stepIndex).value;
	var classSelect = document.getElementById("class_" + stepIndex);
	var classes = modelClasses[cat] || {};
	classSelect.innerHTML = "";
	for (var cls in classes) {
		var opt = document.createElement("option");
		opt.value = cls;
		opt.textContent = classes[cls];
		classSelect.appendChild(opt);
	}
	updateParmDefaults(stepIndex);
}

function updateParmDefaults(stepIndex) {
	var cls = document.getElementById("class_" + stepIndex).value;
	var pf = document.getElementById("modelParm_" + stepIndex);
	if (pf && defaultParms[cls]) pf.value = defaultParms[cls];
	var hf = document.getElementById("parmHelp_" + stepIndex);
	if (hf && parmHelp[cls]) hf.textContent = parmHelp[cls];
}

JS;
$out .= "</script>";
return $out;
} // ends function

?>
