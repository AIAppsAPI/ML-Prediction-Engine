<?php

require_once __DIR__ . "/db.php";
require_once __DIR__ . "/llm.php";

// ML engine core. PHP owns the pipeline configs, datasets, and API, and hands
// every train or predict job to the Python worker (python/worker.py), either
// through its small HTTP service or by running it as a CLI per request. The
// worker owns everything scikit-learn.
//
// A pipeline is a short ordered list of model steps. Predictions flow through
// in order, each step's output becomes the next step's input, and the final
// output is the pipeline's answer. Datasets and trained models are plain files
// under dataPath, one pair per step, shared with the worker.

// ---------------------------------------------------------------------------------
// File locations.

function MLDataDir($sub) {
$config = GetConfig();
$base = rtrim($config["dataPath"] ?? (dirname(__DIR__) . "/data"), "/");
$dir = $base . "/" . $sub;
if (!is_dir($dir)) @mkdir($dir, 0775, true);
return $dir;
} // ends function

// ---------------------------------------------------------------------------------

function MLModelPath($pipelineID, $modelIndex) {
return MLDataDir("models") . "/" . (int)$pipelineID . "-" . (int)$modelIndex . ".model";
} // ends function

// ---------------------------------------------------------------------------------

function MLDatasetPath($pipelineID, $modelIndex) {
return MLDataDir("datasets") . "/" . (int)$pipelineID . "-" . (int)$modelIndex . ".jsonl";
} // ends function

// ---------------------------------------------------------------------------------

function MLNeedsLabels($category) {
return in_array($category, array("classifier", "regressor"));
} // ends function

// ---------------------------------------------------------------------------------
// Datasets. One jsonL file per step: {"input": ..., "label": ...} per line.

function MLDatasetAppendRow($pipelineID, $modelIndex, $input, $label, $category) {

$row = array("input" => $input);
if (MLNeedsLabels($category) && $label !== null) {
	$row["label"] = ($category == "regressor") ? (float)$label : (string)$label;
}
$line = json_encode($row, JSON_INVALID_UTF8_SUBSTITUTE) . "\n";
return @file_put_contents(MLDatasetPath($pipelineID, $modelIndex), $line, FILE_APPEND | LOCK_EX) !== false;
} // ends function

// ---------------------------------------------------------------------------------
// Replaces the whole dataset from a jsonL string, keeping only valid rows.
// Returns how many rows were kept.

function MLDatasetReplace($pipelineID, $modelIndex, $filecontent) {

$kept = 0;
$out = "";
foreach (array_filter(explode("\n", trim((string)$filecontent))) as $line) {
	$row = @json_decode(trim($line), true);
	if (!is_array($row) || !array_key_exists("input", $row)) continue;
	$out .= json_encode($row, JSON_INVALID_UTF8_SUBSTITUTE) . "\n";
	$kept++;
}
@file_put_contents(MLDatasetPath($pipelineID, $modelIndex), $out, LOCK_EX);
return $kept;
} // ends function

// ---------------------------------------------------------------------------------

function MLDatasetRows($pipelineID, $modelIndex) {

$path = MLDatasetPath($pipelineID, $modelIndex);
if (!file_exists($path)) return array();

$rows = array();
foreach (array_filter(explode("\n", trim((string)@file_get_contents($path)))) as $line) {
	$row = @json_decode(trim($line), true);
	if (is_array($row)) $rows[] = $row;
}
return $rows;
} // ends function

// ---------------------------------------------------------------------------------

function MLDatasetDeleteRow($pipelineID, $modelIndex, $rowIndex) {

$rows = MLDatasetRows($pipelineID, $modelIndex);
$rowIndex = (int)$rowIndex;
if ($rowIndex < 0 || $rowIndex >= count($rows)) return false;

array_splice($rows, $rowIndex, 1);
$out = "";
foreach ($rows as $row) $out .= json_encode($row, JSON_INVALID_UTF8_SUBSTITUTE) . "\n";
@file_put_contents(MLDatasetPath($pipelineID, $modelIndex), $out, LOCK_EX);
return true;
} // ends function

// ---------------------------------------------------------------------------------

function MLDatasetClear($pipelineID, $modelIndex) {
@file_put_contents(MLDatasetPath($pipelineID, $modelIndex), "", LOCK_EX);
return true;
} // ends function

// ---------------------------------------------------------------------------------
// Deletes a step's trained model and dataset files.

function MLDeleteStepFiles($pipelineID, $modelIndex) {
@unlink(MLModelPath($pipelineID, $modelIndex));
@unlink(MLDatasetPath($pipelineID, $modelIndex));
return true;
} // ends function

// ---------------------------------------------------------------------------------
// Sends one job to the Python worker and returns its decoded response.
// mlDriver "service" posts to the worker's HTTP service, "cli" runs
// python3 worker.py per job with the payload on stdin.

function MLWorkerCall($payload) {

$config = GetConfig();
$driver = $config["mlDriver"] ?? "service";

if ($driver == "cli") {
	$python = $config["pythonPath"] ?? "python3";
	$workerPath = dirname(__DIR__) . "/python/worker.py";
	if (!file_exists($workerPath)) return array("error" => "python/worker.py not found.");

	$descriptors = array(0 => array("pipe", "r"), 1 => array("pipe", "w"), 2 => array("pipe", "w"));
	$proc = @proc_open(array($python, $workerPath, "run"), $descriptors, $pipes);
	if (!is_resource($proc)) return array("error" => "Could not start the Python worker. Check pythonPath in config.php.");

	fwrite($pipes[0], json_encode($payload, JSON_INVALID_UTF8_SUBSTITUTE));
	fclose($pipes[0]);
	$stdout = stream_get_contents($pipes[1]);
	$stderr = stream_get_contents($pipes[2]);
	fclose($pipes[1]);
	fclose($pipes[2]);
	proc_close($proc);

	$data = json_decode(trim((string)$stdout), true);
	if (!is_array($data)) return array("error" => "Worker returned no result. " . substr(trim((string)$stderr), 0, 300));
	return $data;
}

// service driver
$url = rtrim($config["mlServiceUrl"] ?? "http://127.0.0.1:8750", "/") . "/";
$curl = curl_init($url);
curl_setopt($curl, CURLOPT_RETURNTRANSFER, true);
curl_setopt($curl, CURLOPT_POST, true);
curl_setopt($curl, CURLOPT_POSTFIELDS, json_encode($payload, JSON_INVALID_UTF8_SUBSTITUTE));
curl_setopt($curl, CURLOPT_HTTPHEADER, array("Content-Type: application/json"));
curl_setopt($curl, CURLOPT_TIMEOUT, 600);
curl_setopt($curl, CURLOPT_CONNECTTIMEOUT, 10);
$response = curl_exec($curl);
$curlError = curl_error($curl);
curl_close($curl);

if ($response === false) return array("error" => "Could not reach the ML worker service at " . $url . " (" . $curlError . "). Start it with: python3 python/worker.py serve. Or set mlDriver to cli in config.php.");

$data = json_decode($response, true);
if (!is_array($data)) return array("error" => "Worker service returned a non JSON response.");
return $data;
} // ends function

// ---------------------------------------------------------------------------------
// Trains one step of a pipeline from its stored dataset, then records the
// training on the step config. Returns the worker's response.

function MLTrainStep($pipelineID, $modelIndex) {

$pipeline = GetPipeline($pipelineID);
if (!$pipeline) return array("error" => "Pipeline not found.");

$models = $pipeline["models"] ?? array();
if (!isset($models[$modelIndex]) || !is_array($models[$modelIndex]) || strlen($models[$modelIndex]["class"] ?? "") < 1) {
	return array("error" => "Model step not found.");
}
$step = $models[$modelIndex];

if (($step["class"] ?? "") == "dbscan") {
	return array("error" => "DBSCAN does not train. It clusters at prediction time from this step's dataset, just add reference rows.");
}

$payload = array(
	"op" => "train",
	"modelClass" => $step["class"],
	"category" => $step["category"] ?? "classifier",
	"dataType" => $step["dataType"] ?? "text",
	"modelParm" => $step["modelParm"] ?? array(),
	"modelPath" => MLModelPath($pipelineID, $modelIndex),
	"datasetPath" => MLDatasetPath($pipelineID, $modelIndex),
);

$response = MLWorkerCall($payload);
if (strlen($response["error"] ?? "") > 0) return $response;

$pipeline["models"][$modelIndex]["trained"] = time();
$pipeline["models"][$modelIndex]["trainedSamples"] = (int)($response["samples"] ?? 0);
$pipeline["models"][$modelIndex]["liveRows"] = 0;
SavePipeline($pipelineID, $pipeline);

return $response;
} // ends function

// ---------------------------------------------------------------------------------
// Runs an input through a pipeline's model sequence. Each step's output becomes
// the next step's input. With live training on, every step's input and output
// are appended to that step's dataset, and the step retrains automatically
// after retrainEvery new live rows.

function MLPredictPipeline($jsonData) {

$config = GetConfig();

$pipelineID = (int)($jsonData["pipelineID"] ?? 0);
$input = $jsonData["input"] ?? null;
if ($pipelineID < 1) return array("error" => "pipelineID is required.");
if ($input === null || (is_string($input) && strlen(trim($input)) < 1)) return array("error" => "input is required.");

$pipeline = GetPipeline($pipelineID);
if (!$pipeline) return array("error" => "Pipeline not found for pipelineID " . $pipelineID . ".");

$models = $pipeline["models"] ?? array();
$liveTraining = (($pipeline["liveTraining"] ?? "0") == "1");
$retrainEvery = (int)($pipeline["retrainEvery"] ?? 0);
if ($retrainEvery < 1) $retrainEvery = (int)($config["retrainEvery"] ?? 20);
if ($retrainEvery < 1) $retrainEvery = 20;

$current = $input;
$stepOutputs = array();
$notices = array();
$dirty = false;
$ranSteps = 0;

foreach ($models as $stepIndex => $step) {
	if (!is_array($step) || strlen($step["class"] ?? "") < 1) continue;

	$class = $step["class"];
	$category = $step["category"] ?? "classifier";
	$dataType = $step["dataType"] ?? "text";

	if ($class == "dbscan") {
		$payload = array(
			"op" => "dbscan",
			"dataType" => $dataType,
			"modelParm" => $step["modelParm"] ?? array(),
			"datasetPath" => MLDatasetPath($pipelineID, $stepIndex),
			"input" => $current,
		);
		$response = MLWorkerCall($payload);
		if (strlen($response["error"] ?? "") > 0) return array("error" => "Step " . $stepIndex . ": " . $response["error"]);
		if (strlen($response["notice"] ?? "") > 0) $notices[] = "Step " . $stepIndex . ": " . $response["notice"];

		// live training: the input joins the reference dataset for future clustering
		if ($liveTraining) MLDatasetAppendRow($pipelineID, $stepIndex, $current, null, $category);

		$current = $response["output"];
	}

	else {
		$payload = array(
			"op" => "predict",
			"modelPath" => MLModelPath($pipelineID, $stepIndex),
			"input" => $current,
		);
		$response = MLWorkerCall($payload);
		if (strlen($response["error"] ?? "") > 0) return array("error" => "Step " . $stepIndex . ": " . $response["error"]);

		$output = $response["output"];

		// live training: record this step's input and output, retrain on schedule
		if ($liveTraining) {
			MLDatasetAppendRow($pipelineID, $stepIndex, $current, MLNeedsLabels($category) ? $output : null, $category);
			$liveRows = (int)($pipeline["models"][$stepIndex]["liveRows"] ?? 0) + 1;
			$pipeline["models"][$stepIndex]["liveRows"] = $liveRows;
			$dirty = true;

			if ($liveRows >= $retrainEvery) {
				$trainResponse = MLWorkerCall(array(
					"op" => "train",
					"modelClass" => $class,
					"category" => $category,
					"dataType" => $dataType,
					"modelParm" => $step["modelParm"] ?? array(),
					"modelPath" => MLModelPath($pipelineID, $stepIndex),
					"datasetPath" => MLDatasetPath($pipelineID, $stepIndex),
				));
				if (strlen($trainResponse["error"] ?? "") > 0) {
					// a failed live retrain never breaks the prediction
					$notices[] = "Step " . $stepIndex . " live retrain failed: " . $trainResponse["error"];
				} else {
					$pipeline["models"][$stepIndex]["trained"] = time();
					$pipeline["models"][$stepIndex]["trainedSamples"] = (int)($trainResponse["samples"] ?? 0);
					$pipeline["models"][$stepIndex]["liveRows"] = 0;
				}
			}
		}

		$current = $output;
	}

	$stepOutputs[] = array("step" => (int)$stepIndex, "class" => $class, "output" => $current);
	$ranSteps++;
}

if ($dirty) SavePipeline($pipelineID, $pipeline);
if ($ranSteps < 1) return array("error" => "Pipeline has no configured model steps.");

$result = array("output" => $current, "steps" => $stepOutputs);
if (count($notices) > 0) $result["notice"] = implode(" ", $notices);
return $result;
} // ends function

// ---------------------------------------------------------------------------------
// Optional AI data summary. Sends an inline data array to the configured chat
// model and stores the summary in the reports table. Disabled until an LLM key
// is set in config.php, the ML features never need one.

function MLSummarizeData($jsonData) {

$config = GetConfig();
$provider = $config["chatProvider"] ?? "openai";
$hasKey = ($provider == "anthropic") ? (strlen($config["anthropicApiKey"] ?? "") > 9) : (strlen($config["openaiApiKey"] ?? "") > 9);
if (!$hasKey) return array("error" => "AI summaries are disabled. Set chatProvider and an API key in config.php to enable them.");

$data = $jsonData["data"] ?? array();
$label = substr((string)($jsonData["label"] ?? "summary"), 0, 200);
$prompt = (string)($jsonData["prompt"] ?? "");
if (strlen($prompt) < 1) $prompt = "Make a summary of my current data. If there are many different types of responses or common groups of answers (such as answers to static questions), include the most popular responses and what percentage of the data pool gave that answer.";

if (empty($data)) return array("error" => "data is empty. Send an array of items to summarize.");

$rules = array();
$rules[] = "You are a data analysis manager who reads current chunks of data and produces a complete, factual summary. Return only the summary, no suggestions, no commentary, no preamble. The response will be stored directly in a database so extra information will corrupt the record.";

$messages = array();
$messages[] = array("role" => "user", "message" => "Here is my current data.");
$messages[] = array("role" => "assistant", "message" => json_encode($data, JSON_INVALID_UTF8_SUBSTITUTE));

$response = SendChatQuery($prompt, $rules, $messages, array());
if (strlen($response["error"] ?? "") > 0) return array("error" => $response["error"]);

$summary = $response["message"] ?? "";
if (strlen($summary) > 5) ReportAdd($label, $summary);

return array("message" => $summary, "billing" => (int)($response["billing"] ?? 0));
} // ends function

?>
