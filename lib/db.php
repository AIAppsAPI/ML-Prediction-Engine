<?php

// SQLite storage layer. Pipeline configs and AI report history live in one
// database file set by dbPath in config.php. Datasets and trained models are
// plain files under dataPath, shared with the Python worker.

// ---------------------------------------------------------------------------------

function GetConfig() {
static $config = null;
if ($config !== null) return $config;

$path = dirname(__DIR__) . "/config.php";
if (!file_exists($path)) {
	http_response_code(500);
	header("Content-Type: application/json");
	echo json_encode(array("error" => "config.php not found. Copy config.sample.php to config.php and fill in your settings."));
	exit;
}

$config = require $path;
if (!is_array($config)) $config = array();
return $config;
} // ends function

// ---------------------------------------------------------------------------------

function GetDB() {
static $db = null;
if ($db !== null) return $db;

$config = GetConfig();
$dbPath = $config["dbPath"] ?? (dirname(__DIR__) . "/data/mlengine.sqlite");

$dir = dirname($dbPath);
if (!is_dir($dir)) @mkdir($dir, 0775, true);

$db = new PDO("sqlite:" . $dbPath);
$db->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
$db->exec("PRAGMA journal_mode = WAL");
$db->exec("PRAGMA busy_timeout = 5000");

$db->exec("CREATE TABLE IF NOT EXISTS pipelines (
	pipelineID INTEGER PRIMARY KEY AUTOINCREMENT,
	config TEXT NOT NULL,
	created INTEGER,
	updated INTEGER
)");

$db->exec("CREATE TABLE IF NOT EXISTS reports (
	id INTEGER PRIMARY KEY AUTOINCREMENT,
	label TEXT,
	summary TEXT,
	created INTEGER
)");

return $db;
} // ends function

// ---------------------------------------------------------------------------------
// Pipelines. Each pipeline is one row, all settings and model steps in a JSON blob.

function GetPipeline($pipelineID) {

$db = GetDB();
$stmt = $db->prepare("SELECT config FROM pipelines WHERE pipelineID = ?");
$stmt->execute(array((int)$pipelineID));
$row = $stmt->fetch(PDO::FETCH_ASSOC);
if (!$row) return false;

$pipeline = json_decode($row["config"], true);
if (!is_array($pipeline)) return false;
$pipeline["pipelineID"] = (int)$pipelineID;
return $pipeline;
} // ends function

// ---------------------------------------------------------------------------------
// Pass pipelineID 0 to create a new pipeline. Returns the pipelineID.

function SavePipeline($pipelineID, $pipelineConfig) {

$db = GetDB();
unset($pipelineConfig["pipelineID"]);
$json = json_encode($pipelineConfig, JSON_INVALID_UTF8_SUBSTITUTE);
$now = time();

if ((int)$pipelineID > 0) {
	$stmt = $db->prepare("UPDATE pipelines SET config = ?, updated = ? WHERE pipelineID = ?");
	$stmt->execute(array($json, $now, (int)$pipelineID));
	return (int)$pipelineID;
}

$stmt = $db->prepare("INSERT INTO pipelines (config, created, updated) VALUES (?, ?, ?)");
$stmt->execute(array($json, $now, $now));
return (int)$db->lastInsertId();
} // ends function

// ---------------------------------------------------------------------------------

function DeletePipelineRow($pipelineID) {

$db = GetDB();
$stmt = $db->prepare("DELETE FROM pipelines WHERE pipelineID = ?");
$stmt->execute(array((int)$pipelineID));
return true;
} // ends function

// ---------------------------------------------------------------------------------

function GetAllPipelines() {

$db = GetDB();
$pipelines = array();
foreach ($db->query("SELECT pipelineID, config, updated FROM pipelines ORDER BY pipelineID") as $row) {
	$pipeline = json_decode($row["config"], true);
	if (!is_array($pipeline)) $pipeline = array();
	$pipeline["pipelineID"] = (int)$row["pipelineID"];
	$pipeline["updated"] = $row["updated"];
	$pipelines[] = $pipeline;
}
return $pipelines;
} // ends function

// ---------------------------------------------------------------------------------
// AI reports. One row per saved data summary.

function ReportAdd($label, $summary) {

$db = GetDB();
$stmt = $db->prepare("INSERT INTO reports (label, summary, created) VALUES (?, ?, ?)");
$stmt->execute(array(substr((string)$label, 0, 200), (string)$summary, time()));
return true;
} // ends function

// ---------------------------------------------------------------------------------

function ReportList($limit = 25) {

$limit = (int)$limit;
if ($limit < 1) $limit = 25;
if ($limit > 200) $limit = 200;

$db = GetDB();
$rows = array();
foreach ($db->query("SELECT id, label, summary, created FROM reports ORDER BY id DESC LIMIT " . $limit) as $row) {
	$rows[] = $row;
}
return $rows;
} // ends function

?>
