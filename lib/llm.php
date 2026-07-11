<?php

require_once __DIR__ . "/db.php";

// Chat model drivers. One entry point, SendChatQuery, with a driver for
// Anthropic and a driver for any OpenAI compatible API. Only used by the
// optional AI data summary feature, the ML side never touches an LLM.

// ---------------------------------------------------------------------------------

function SendChatQuery($prompt, $rules, $existingMessages, $options) {

$config = GetConfig();
$provider = $config["chatProvider"] ?? "openai";

$system = "";
foreach ($rules as $rule) {
	$rule = trim((string)$rule);
	if (strlen($rule) < 1) continue;
	$system .= $rule . "\n\n";
}
$system = trim($system);

// Build the message list, merging consecutive same role messages since some
// providers require strict user and assistant alternation.
$messages = array();
foreach ($existingMessages as $mess) {
	$role = (strtolower($mess["role"] ?? "user") == "assistant") ? "assistant" : "user";
	$text = trim((string)($mess["message"] ?? ""));
	if (strlen($text) < 1) continue;

	$lastIndex = count($messages) - 1;
	if ($lastIndex >= 0 && $messages[$lastIndex]["role"] == $role) $messages[$lastIndex]["content"] .= "\n\n" . $text;
	else $messages[] = array("role" => $role, "content" => $text);
}

$prompt = trim((string)$prompt);
$lastIndex = count($messages) - 1;
if ($lastIndex >= 0 && $messages[$lastIndex]["role"] == "user") $messages[$lastIndex]["content"] .= "\n\n" . $prompt;
else $messages[] = array("role" => "user", "content" => $prompt);

// Conversations must start with a user message.
if ($messages[0]["role"] == "assistant") array_shift($messages);

$model = trim((string)($options["model"] ?? ""));

if ($provider == "anthropic") return CallAnthropicChat($system, $messages, $model);
return CallOpenAIChat($system, $messages, $model);
} // ends function

// ---------------------------------------------------------------------------------

function CallAnthropicChat($system, $messages, $model) {

$config = GetConfig();
$apiKey = $config["anthropicApiKey"] ?? "";
if (strlen($apiKey) < 10) return array("message" => "", "billing" => 0, "error" => "anthropicApiKey is not set in config.php.");

if (strlen($model) < 1) $model = $config["anthropicModel"] ?? "claude-sonnet-5";

$body = array(
	"model" => $model,
	"max_tokens" => (int)($config["maxTokens"] ?? 4096),
	"messages" => $messages,
);
if (strlen($system) > 0) $body["system"] = $system;

$headers = array(
	"Content-Type: application/json",
	"x-api-key: " . $apiKey,
	"anthropic-version: 2023-06-01",
);

list($status, $data) = HttpJsonRequest("https://api.anthropic.com/v1/messages", $headers, $body);

if ($status != 200 || !is_array($data)) {
	$errMess = $data["error"]["message"] ?? ("Anthropic API returned HTTP " . $status . ".");
	return array("message" => "", "billing" => 0, "error" => $errMess);
}

$text = "";
foreach (($data["content"] ?? array()) as $block) {
	if (($block["type"] ?? "") == "text") $text .= $block["text"] ?? "";
}

$billing = (int)($data["usage"]["input_tokens"] ?? 0) + (int)($data["usage"]["output_tokens"] ?? 0);

if (strlen($text) < 1) return array("message" => "", "billing" => $billing, "error" => "Anthropic API returned no text.");
return array("message" => $text, "billing" => $billing, "error" => "");
} // ends function

// ---------------------------------------------------------------------------------

function CallOpenAIChat($system, $messages, $model) {

$config = GetConfig();
$apiKey = $config["openaiApiKey"] ?? "";
if (strlen($apiKey) < 10) return array("message" => "", "billing" => 0, "error" => "openaiApiKey is not set in config.php.");

if (strlen($model) < 1) $model = $config["openaiModel"] ?? "gpt-4o-mini";
$baseUrl = rtrim($config["openaiBaseUrl"] ?? "https://api.openai.com/v1", "/");

$allMessages = array();
if (strlen($system) > 0) $allMessages[] = array("role" => "system", "content" => $system);
foreach ($messages as $mess) $allMessages[] = $mess;

$body = array(
	"model" => $model,
	"max_tokens" => (int)($config["maxTokens"] ?? 4096),
	"messages" => $allMessages,
);

$headers = array(
	"Content-Type: application/json",
	"Authorization: Bearer " . $apiKey,
);

list($status, $data) = HttpJsonRequest($baseUrl . "/chat/completions", $headers, $body);

// Newer OpenAI models replaced max_tokens with max_completion_tokens, while many
// OpenAI compatible servers only accept max_tokens, so retry once on that error.
if ($status == 400 && isset($body["max_tokens"]) && substr_count(strtolower((string)($data["error"]["message"] ?? "")), "max_completion_tokens") > 0) {
	$body["max_completion_tokens"] = $body["max_tokens"];
	unset($body["max_tokens"]);
	list($status, $data) = HttpJsonRequest($baseUrl . "/chat/completions", $headers, $body);
}

if ($status != 200 || !is_array($data)) {
	$errMess = $data["error"]["message"] ?? ("Chat API returned HTTP " . $status . ".");
	return array("message" => "", "billing" => 0, "error" => $errMess);
}

$text = $data["choices"][0]["message"]["content"] ?? "";
$billing = (int)($data["usage"]["total_tokens"] ?? 0);

if (strlen($text) < 1) return array("message" => "", "billing" => $billing, "error" => "Chat API returned no text.");
return array("message" => $text, "billing" => $billing, "error" => "");
} // ends function

// ---------------------------------------------------------------------------------

function HttpJsonRequest($url, $headers, $body, $timeout = 120) {

$curl = curl_init($url);
curl_setopt($curl, CURLOPT_RETURNTRANSFER, true);
curl_setopt($curl, CURLOPT_POST, true);
curl_setopt($curl, CURLOPT_POSTFIELDS, json_encode($body, JSON_INVALID_UTF8_SUBSTITUTE));
curl_setopt($curl, CURLOPT_HTTPHEADER, $headers);
curl_setopt($curl, CURLOPT_TIMEOUT, $timeout);
curl_setopt($curl, CURLOPT_CONNECTTIMEOUT, 15);

$response = curl_exec($curl);
$status = (int)curl_getinfo($curl, CURLINFO_HTTP_CODE);
$curlError = curl_error($curl);
curl_close($curl);

if ($response === false) return array(0, array("error" => array("message" => "Connection failed: " . $curlError)));

$data = json_decode($response, true);
if (!is_array($data)) $data = array("error" => array("message" => "Non JSON response: " . substr($response, 0, 200)));
return array($status, $data);
} // ends function

?>
