package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadDefaultAgentConfig_NoFile(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	cfg, err := LoadDefaultAgentConfig()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.Model != "" || cfg.Endpoint != "" || cfg.Provider != "" || len(cfg.ExcludedSkills) != 0 {
		t.Fatalf("expected zero value, got %+v", cfg)
	}
}

func TestLoadDefaultAgentConfig_ReadsProviderModelEndpoint(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	dir := filepath.Dir(AgentsFile())
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}
	toml := `[agents.default]
model = "qwen3.6"
endpoint = "http://192.168.1.88:30000/v1"
provider = "openai"
excluded_skills = ["noisy-skill"]
`
	if err := os.WriteFile(AgentsFile(), []byte(toml), 0644); err != nil {
		t.Fatal(err)
	}

	cfg, err := LoadDefaultAgentConfig()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.Model != "qwen3.6" {
		t.Errorf("Model = %q, want %q", cfg.Model, "qwen3.6")
	}
	if cfg.Endpoint != "http://192.168.1.88:30000/v1" {
		t.Errorf("Endpoint = %q, want %q", cfg.Endpoint, "http://192.168.1.88:30000/v1")
	}
	if cfg.Provider != "openai" {
		t.Errorf("Provider = %q, want %q", cfg.Provider, "openai")
	}
	if len(cfg.ExcludedSkills) != 1 || cfg.ExcludedSkills[0] != "noisy-skill" {
		t.Errorf("ExcludedSkills = %v, want [noisy-skill]", cfg.ExcludedSkills)
	}
}

func TestLoadAgentFiles_StillRejectsDefaultFromDirectory(t *testing.T) {
	dir := t.TempDir()
	toml := `[agents.default]
model = "should-be-ignored"

[agents.mine]
model = "should-load"
`
	if err := os.WriteFile(filepath.Join(dir, "agents.toml"), []byte(toml), 0644); err != nil {
		t.Fatal(err)
	}

	registry, warnings, err := LoadAgentFiles(dir)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, ok := registry["default"]; ok {
		t.Error("expected \"default\" to be rejected from directory-scanned files")
	}
	if registry["mine"].Model != "should-load" {
		t.Errorf("expected named agent \"mine\" to load, got %+v", registry["mine"])
	}
	if len(warnings) != 1 {
		t.Errorf("expected 1 warning about the reserved name, got %v", warnings)
	}
}

func TestSaveAgentDefault_PreservesOtherFields(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	dir := filepath.Dir(AgentsFile())
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatal(err)
	}
	toml := `[agents.default]
model = "qwen3.6"
provider = "openai"
`
	if err := os.WriteFile(AgentsFile(), []byte(toml), 0644); err != nil {
		t.Fatal(err)
	}

	if err := SaveAgentDefault(AgentConfig{ExcludedSkills: []string{"x"}}); err != nil {
		t.Fatalf("SaveAgentDefault: %v", err)
	}

	cfg, err := LoadDefaultAgentConfig()
	if err != nil {
		t.Fatalf("LoadDefaultAgentConfig: %v", err)
	}
	if cfg.Model != "qwen3.6" || cfg.Provider != "openai" {
		t.Errorf("expected model/provider preserved, got %+v", cfg)
	}
	if len(cfg.ExcludedSkills) != 1 || cfg.ExcludedSkills[0] != "x" {
		t.Errorf("ExcludedSkills = %v, want [x]", cfg.ExcludedSkills)
	}
}
