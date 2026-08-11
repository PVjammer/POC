package config

import (
	"strings"
	"testing"

	"github.com/pelletier/go-toml/v2"
)

func TestConfig_ModelsTable_ParsesNamedPresets(t *testing.T) {
	src := `
[llm]
provider = "ollama"
model = "llama3.2"
endpoint = "http://localhost:11434"

[models."qwen3.6"]
provider = "openai"
model = "unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_XL"
endpoint = "http://192.168.1.88:30000/v1"

[models.local]
provider = "ollama"
model = "llama3.2:latest"
`
	var cfg Config
	if err := toml.Unmarshal([]byte(src), &cfg); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if cfg.LLM.Provider != "ollama" || cfg.LLM.Model != "llama3.2" {
		t.Errorf("LLM default not parsed correctly: %+v", cfg.LLM)
	}

	if len(cfg.Models) != 2 {
		t.Fatalf("expected 2 presets, got %d: %+v", len(cfg.Models), cfg.Models)
	}

	qwen, ok := cfg.Models["qwen3.6"]
	if !ok {
		t.Fatal("expected preset \"qwen3.6\" to be present")
	}
	if qwen.Provider != "openai" || qwen.Endpoint != "http://192.168.1.88:30000/v1" {
		t.Errorf("qwen3.6 preset not parsed correctly: %+v", qwen)
	}

	local, ok := cfg.Models["local"]
	if !ok {
		t.Fatal("expected preset \"local\" to be present")
	}
	if local.Provider != "ollama" || local.Model != "llama3.2:latest" {
		t.Errorf("local preset not parsed correctly: %+v", local)
	}
}

func TestConfig_ModelsTable_NilWhenAbsent(t *testing.T) {
	cfg := Defaults()
	if cfg.Models != nil {
		t.Errorf("expected nil Models map in defaults, got %+v", cfg.Models)
	}

	data, err := toml.Marshal(cfg)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if got := string(data); strings.Contains(got, "models") {
		t.Errorf("expected no [models] section for nil map, got:\n%s", got)
	}
}
