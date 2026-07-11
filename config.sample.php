<?php

// ML Prediction Engine configuration.
// Copy this file to config.php in the same folder and fill in your values.
// config.php is gitignored so your keys never end up in the repo.

return array(

// Client API key. Every /api call must send this in the X-API-Key header,
// or as an apiKey field in the JSON body. Set a long random string.
"apiKey" => "",

// Password for admin.php. At least 8 characters.
"adminPassword" => "",

// Browser origins allowed to call the API. "*" allows any site, or set it
// to your app's origin like "https://myapp.example.com".
"corsOrigin" => "*",

// SQLite database file location (pipeline configs and AI report history).
"dbPath" => __DIR__ . "/data/mlengine.sqlite",

// Folder for datasets and trained model files. When running with Docker,
// keep the default: the PHP app and the Python worker share this folder.
"dataPath" => __DIR__ . "/data",

// How PHP talks to the Python ML worker:
//   "service"  the worker runs as a small HTTP service (recommended, and
//              what docker compose starts automatically)
//   "cli"      PHP runs python3 worker.py per request, no daemon needed
"mlDriver" => "service",

// Where the worker service listens. Local install: http://127.0.0.1:8750
// Docker compose: http://ml:8750
"mlServiceUrl" => "http://127.0.0.1:8750",

// Python executable for the "cli" driver.
"pythonPath" => "python3",

// Live training: after this many new live rows, a pipeline step is
// automatically retrained. Each pipeline can override this in the admin.
"retrainEvery" => 20,

// ---------------------------------------------------------------------
// Optional AI data summaries (the /summarize endpoint). Leave the keys
// blank to disable, the ML features work fine without any LLM.

// Chat provider: "anthropic" or "openai".
// The openai driver works with any OpenAI compatible API (OpenAI, OpenRouter,
// Ollama, LM Studio) by changing openaiBaseUrl.
"chatProvider" => "openai",

// Anthropic settings.
"anthropicApiKey" => "",
"anthropicModel" => "claude-sonnet-5",

// OpenAI compatible settings.
"openaiApiKey" => "",
"openaiBaseUrl" => "https://api.openai.com/v1",
"openaiModel" => "gpt-4o-mini",

// Max tokens per AI summary response.
"maxTokens" => 4096,

);
