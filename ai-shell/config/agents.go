package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/pelletier/go-toml/v2"
)

// AgentConfig defines a named agent's runtime configuration.
// Fields left zero/empty inherit from the global shell config.
type AgentConfig struct {
	Description            string   `toml:"description"`
	Model                  string   `toml:"model"`
	Endpoint               string   `toml:"endpoint"`
	Provider               string   `toml:"provider"` // "ollama" | "openai"; empty = inherit shell default
	Tools                  []string `toml:"tools"`           // nil = all act tools; named agents: nil = none
	Skills                 []string `toml:"skills"`          // nil = all (default agent); named agents: nil = none
	ExcludedSkills         []string `toml:"excluded_skills"` // subtractive; only meaningful for default agent
	SystemPrompt           string   `toml:"system_prompt"`
	AdditionalInstructions string   `toml:"additional_instructions"`
	MaxRounds              int      `toml:"max_rounds"`
}

// reserved agent names that users cannot define.
var reservedAgentNames = map[string]bool{
	"default":  true,
	"one_shot": true,
}

type agentsFile struct {
	Agents map[string]AgentConfig `toml:"agents"`
}

// LoadAgentFiles reads all *.toml files from each dir in order.
// Later dirs (and later files within a dir) override earlier ones on name collision.
// Returns the merged registry, a slice of non-fatal warnings, and any fatal error.
func LoadAgentFiles(dirs ...string) (map[string]AgentConfig, []string, error) {
	registry := make(map[string]AgentConfig)
	var warnings []string

	for _, dir := range dirs {
		entries, err := os.ReadDir(dir)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return nil, warnings, fmt.Errorf("agents: read dir %s: %w", dir, err)
		}
		for _, e := range entries {
			if e.IsDir() || !strings.HasSuffix(e.Name(), ".toml") {
				continue
			}
			path := filepath.Join(dir, e.Name())
			data, err := os.ReadFile(path)
			if err != nil {
				warnings = append(warnings, fmt.Sprintf("agents: read %s: %v", path, err))
				continue
			}
			var af agentsFile
			if err := toml.Unmarshal(data, &af); err != nil {
				warnings = append(warnings, fmt.Sprintf("agents: parse %s: %v", path, err))
				continue
			}
			for name, cfg := range af.Agents {
				if reservedAgentNames[name] {
					warnings = append(warnings, fmt.Sprintf("agents: %s: %q is a reserved name and will be ignored", path, name))
					continue
				}
				registry[name] = cfg
			}
		}
	}

	return registry, warnings, nil
}

// AgentsDir returns the directory where user-global agent configs live.
func AgentsDir() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".config", "baish", "agents")
}

// AgentsFile returns the path to the single-file agents config (alternative to the dir).
func AgentsFile() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".config", "baish", "agents.toml")
}

// LoadDefaultAgentConfig reads the [agents.default] section from AgentsFile() —
// the one place "default" is allowed (LoadAgentFiles rejects it everywhere
// else as a reserved name). This is the file /agent edit default opens and
// SaveAgentDefault writes to; it lets users override the built-in default
// agent's model/endpoint/provider/system_prompt/etc. Returns the zero
// AgentConfig if the file or section doesn't exist.
func LoadDefaultAgentConfig() (AgentConfig, error) {
	data, err := os.ReadFile(AgentsFile())
	if os.IsNotExist(err) {
		return AgentConfig{}, nil
	}
	if err != nil {
		return AgentConfig{}, fmt.Errorf("agents: read %s: %w", AgentsFile(), err)
	}
	var af agentsFile
	if err := toml.Unmarshal(data, &af); err != nil {
		return AgentConfig{}, fmt.Errorf("agents: parse %s: %w", AgentsFile(), err)
	}
	return af.Agents["default"], nil
}

// ProjectAgentsDir returns the project-local agents config directory.
func ProjectAgentsDir() string {
	return filepath.Join(".baish", "agents")
}

// SkillsDir returns the user-global skills directory.
func SkillsDir() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".config", "baish", "skills")
}

// SaveAgentDefault persists the "default" agent config entry to agents.toml.
// Only ExcludedSkills is written (the fields that users can mutate at runtime).
func SaveAgentDefault(cfg AgentConfig) error {
	path := AgentsFile()

	// Read existing file to preserve other agents.
	var af agentsFile
	if data, err := os.ReadFile(path); err == nil {
		_ = toml.Unmarshal(data, &af)
	}
	if af.Agents == nil {
		af.Agents = make(map[string]AgentConfig)
	}

	// Merge: only update ExcludedSkills; leave other fields in the file as-is.
	existing := af.Agents["default"]
	existing.ExcludedSkills = cfg.ExcludedSkills
	af.Agents["default"] = existing

	data, err := toml.Marshal(af)
	if err != nil {
		return fmt.Errorf("marshal agents: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}
