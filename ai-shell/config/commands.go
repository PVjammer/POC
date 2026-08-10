package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/pelletier/go-toml/v2"
)

// CommandConfig defines a user-invocable /command.
// Either Skill or Prompt must be set, not both.
type CommandConfig struct {
	Name        string `toml:"name"`
	Description string `toml:"description"`
	Skill       string `toml:"skill"`    // references an installed skill by name
	Prompt      string `toml:"prompt"`   // inline prompt; mutually exclusive with Skill
	Agent       string `toml:"agent"`    // named agent, "default", or "one_shot"
	Model       string `toml:"model"`    // optional per-command model override
	Endpoint    string `toml:"endpoint"` // optional per-command endpoint override
}

type commandsFile struct {
	Commands []CommandConfig `toml:"commands"`
}

// LoadCommandFiles reads commands from each path, which may be a *.toml file or
// a directory of *.toml files. Later paths override earlier commands with the
// same name. Returns the merged list, non-fatal warnings, and any fatal error.
func LoadCommandFiles(paths ...string) ([]CommandConfig, []string, error) {
	byName := make(map[string]CommandConfig)
	var order []string
	var warnings []string

	loadFile := func(path string) {
		data, err := os.ReadFile(path)
		if err != nil {
			warnings = append(warnings, fmt.Sprintf("commands: read %s: %v", path, err))
			return
		}
		var cf commandsFile
		if err := toml.Unmarshal(data, &cf); err != nil {
			warnings = append(warnings, fmt.Sprintf("commands: parse %s: %v", path, err))
			return
		}
		for _, cmd := range cf.Commands {
			if cmd.Name == "" {
				warnings = append(warnings, fmt.Sprintf("commands: %s: command missing name field, skipping", path))
				continue
			}
			if cmd.Skill != "" && cmd.Prompt != "" {
				warnings = append(warnings, fmt.Sprintf("commands: %s/%s: both skill and prompt set; using skill", path, cmd.Name))
				cmd.Prompt = ""
			}
			if _, exists := byName[cmd.Name]; !exists {
				order = append(order, cmd.Name)
			}
			byName[cmd.Name] = cmd
		}
	}

	for _, p := range paths {
		info, err := os.Stat(p)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			warnings = append(warnings, fmt.Sprintf("commands: stat %s: %v", p, err))
			continue
		}
		if !info.IsDir() {
			// Single file (e.g. commands.toml).
			if strings.HasSuffix(p, ".toml") {
				loadFile(p)
			}
			continue
		}
		// Directory — read all *.toml files inside.
		entries, err := os.ReadDir(p)
		if err != nil {
			warnings = append(warnings, fmt.Sprintf("commands: read dir %s: %v", p, err))
			continue
		}
		for _, e := range entries {
			if !e.IsDir() && strings.HasSuffix(e.Name(), ".toml") {
				loadFile(filepath.Join(p, e.Name()))
			}
		}
	}

	out := make([]CommandConfig, 0, len(order))
	for _, name := range order {
		out = append(out, byName[name])
	}
	return out, warnings, nil
}

// CommandsDir returns the user-global commands config directory.
func CommandsDir() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".config", "baish", "commands")
}

// CommandsFile returns the path to the single-file commands config.
func CommandsFile() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".config", "baish", "commands.toml")
}

// ProjectCommandsDir returns the project-local commands config directory.
func ProjectCommandsDir() string {
	return filepath.Join(".baish", "commands")
}

// AppendCommand adds or updates a command entry in the user-global commands.toml.
// If a command with the same name already exists it is replaced.
func AppendCommand(cmd CommandConfig) error {
	path := CommandsFile()

	var cf commandsFile
	if data, err := os.ReadFile(path); err == nil {
		_ = toml.Unmarshal(data, &cf) // ignore parse errors; we'll overwrite
	}

	replaced := false
	for i, c := range cf.Commands {
		if c.Name == cmd.Name {
			cf.Commands[i] = cmd
			replaced = true
			break
		}
	}
	if !replaced {
		cf.Commands = append(cf.Commands, cmd)
	}

	data, err := toml.Marshal(cf)
	if err != nil {
		return fmt.Errorf("marshal commands: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// RemoveCommand removes a command entry by name from the user-global commands.toml.
func RemoveCommand(name string) error {
	path := CommandsFile()
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	var cf commandsFile
	if err := toml.Unmarshal(data, &cf); err != nil {
		return fmt.Errorf("parse commands.toml: %w", err)
	}
	filtered := cf.Commands[:0]
	for _, c := range cf.Commands {
		if c.Name != name {
			filtered = append(filtered, c)
		}
	}
	cf.Commands = filtered
	out, err := toml.Marshal(cf)
	if err != nil {
		return err
	}
	return os.WriteFile(path, out, 0644)
}
