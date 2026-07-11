<?php

require_once __DIR__ . "/../lib/ml.php";

// JSON API entry point. Routes use path info, for example:
//   POST api.php/predict        run an input through a pipeline
//   POST api.php/train          train one pipeline step from its dataset
//   POST api.php/dataset/add    append a training row to a step's dataset
//   POST api.php/dataset/set    replace a step's dataset with jsonL content
//   POST api.php/dataset/rows   list a step's dataset rows
//   POST api.php/dataset/clear  empty a step's dataset
//   POST api.php/pipelines      list pipelines and their steps
//   POST api.php/summarize      optional AI data summary (needs an LLM key)
// If your server does not pass path info, use api.php?action=predict instead.
// Every call must send the config apiKey in the X-API-Key header, or as an
// apiKey field in the JSON body.

$config = GetConfig();

$origin = $config["corsOrigin"] ?? "*";
header("Access-Control-Allow-Origin: " . $origin);
header("Access-Control-Allow-Headers: Content-Type, X-API-Key");
header("Access-Control-Allow-Methods: POST, OPTIONS");
if (($_SERVER["REQUEST_METHOD"] ?? "") == "OPTIONS") exit;

header("Content-Type: application/json");

$rawBody = file_get_contents("php://input");
$jsonData = json_decode($rawBody, true);
if (!is_array($jsonData)) $jsonData = $_POST;
if (!is_array($jsonData)) $jsonData = array();

// Authentication
$apiKey = (string)($config["apiKey"] ?? "");
if (strlen($apiKey) < 12) {
	http_response_code(500);
	echo json_encode(array("error" => "Set apiKey in config.php to a random string of at least 12 characters before using the API."));
	exit;
}

$sentKey = (string)($_SERVER["HTTP_X_API_KEY"] ?? ($jsonData["apiKey"] ?? ""));
if (!hash_equals($apiKey, $sentKey)) {
	http_response_code(401);
	echo json_encode(array("error" => "Invalid or missing API key. Send it in the X-API-Key header."));
	exit;
}

$action = trim((string)($_SERVER["PATH_INFO"] ?? ($_GET["action"] ?? "")), "/");

// Shared lookups for the dataset and train routes.
function RequireStep($jsonData) {
	$pipelineID = (int)($jsonData["pipelineID"] ?? 0);
	$modelIndex = (int)($jsonData["modelIndex"] ?? -1);
	if ($pipelineID < 1) return array(null, null, null, array("error" => "pipelineID is required."));
	if ($modelIndex < 0) return array(null, null, null, array("error" => "modelIndex is required."));
	$pipeline = GetPipeline($pipelineID);
	if (!$pipeline) return array(null, null, null, array("error" => "Pipeline not found."));
	$step = ($pipeline["models"] ?? array())[$modelIndex] ?? null;
	if (!is_array($step) || strlen($step["class"] ?? "") < 1) return array(null, null, null, array("error" => "Model step not found."));
	return array($pipelineID, $modelIndex, $step, null);
} // ends function

switch ($action)
{
	case "predict":
		$response = MLPredictPipeline($jsonData);
	break;

	case "train":
		list($pipelineID, $modelIndex, $step, $err) = RequireStep($jsonData);
		if ($err) { $response = $err; break; }
		set_time_limit(0);
		$response = MLTrainStep($pipelineID, $modelIndex);
	break;

	case "dataset/add":
		list($pipelineID, $modelIndex, $step, $err) = RequireStep($jsonData);
		if ($err) { $response = $err; break; }
		$input = $jsonData["input"] ?? null;
		if ($input === null || (is_string($input) && strlen(trim($input)) < 1)) { $response = array("error" => "input is required."); break; }
		$category = $step["category"] ?? "classifier";
		$label = $jsonData["label"] ?? null;
		if (MLNeedsLabels($category) && ($label === null || (is_string($label) && strlen(trim($label)) < 1))) {
			$response = array("error" => "label is required for " . $category . " models.");
			break;
		}
		MLDatasetAppendRow($pipelineID, $modelIndex, $input, $label, $category);
		$response = array("message" => "Row added.", "rows" => count(MLDatasetRows($pipelineID, $modelIndex)));
		if ((string)($jsonData["train"] ?? "") == "1") {
			set_time_limit(0);
			$trainResponse = MLTrainStep($pipelineID, $modelIndex);
			if (strlen($trainResponse["error"] ?? "") > 0) $response["trainError"] = $trainResponse["error"];
			else $response["message"] = "Row added. " . ($trainResponse["message"] ?? "");
		}
	break;

	case "dataset/set":
		list($pipelineID, $modelIndex, $step, $err) = RequireStep($jsonData);
		if ($err) { $response = $err; break; }
		$filecontent = (string)($jsonData["filecontent"] ?? "");
		if (strlen($filecontent) < 2) { $response = array("error" => "filecontent is required (jsonL, one {\"input\": ..., \"label\": ...} per line)."); break; }
		$kept = MLDatasetReplace($pipelineID, $modelIndex, $filecontent);
		$response = array("message" => "Dataset replaced with " . $kept . " rows.", "rows" => $kept);
	break;

	case "dataset/rows":
		list($pipelineID, $modelIndex, $step, $err) = RequireStep($jsonData);
		if ($err) { $response = $err; break; }
		$rows = MLDatasetRows($pipelineID, $modelIndex);
		$response = array("rows" => $rows, "count" => count($rows));
	break;

	case "dataset/clear":
		list($pipelineID, $modelIndex, $step, $err) = RequireStep($jsonData);
		if ($err) { $response = $err; break; }
		MLDatasetClear($pipelineID, $modelIndex);
		$response = array("message" => "Dataset cleared.");
	break;

	case "pipelines":
		$list = array();
		foreach (GetAllPipelines() as $pipeline) {
			$steps = array();
			foreach (($pipeline["models"] ?? array()) as $idx => $step) {
				if (!is_array($step) || strlen($step["class"] ?? "") < 1) continue;
				$steps[] = array(
					"modelIndex" => (int)$idx,
					"class" => $step["class"],
					"category" => $step["category"] ?? "classifier",
					"dataType" => $step["dataType"] ?? "text",
					"trained" => (int)($step["trained"] ?? 0),
					"trainedSamples" => (int)($step["trainedSamples"] ?? 0),
				);
			}
			$list[] = array(
				"pipelineID" => (int)$pipeline["pipelineID"],
				"name" => $pipeline["name"] ?? "",
				"liveTraining" => $pipeline["liveTraining"] ?? "0",
				"steps" => $steps,
			);
		}
		$response = array("pipelines" => $list);
	break;

	case "summarize":
		$response = MLSummarizeData($jsonData);
	break;

	default:
		http_response_code(404);
		$response = array("error" => "Unknown action. Valid actions: predict, train, dataset/add, dataset/set, dataset/rows, dataset/clear, pipelines, summarize.");
	break;
}

echo json_encode($response, JSON_INVALID_UTF8_SUBSTITUTE);

?>
