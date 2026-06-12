// Package config manages baish's persistent settings.
// Settings are stored in ~/.config/baish/config.toml and loaded on startup.
package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/pelletier/go-toml/v2"
)

// ToolOverflow controls what happens when a tool result exceeds ToolOutputMaxChars.
type ToolOverflow string

const (
	OverflowTruncate  ToolOverflow = "truncate"
	OverflowSummarize ToolOverflow = "summarize"
)

// Config holds tuneable shell settings.
type Config struct {
	MaxHistoryMessages int          `toml:"max_history_messages"`
	ToolOutputMaxChars int          `toml:"tool_output_max_chars"`
	ToolOverflow       ToolOverflow `toml:"tool_output_overflow"`
	CtxMaxInjectChars  int          `toml:"ctx_max_inject_chars"`
}

// Defaults returns the baseline configuration.
func Defaults() Config {
	return Config{
		MaxHistoryMessages: 20,
		ToolOutputMaxChars: 4000,
		ToolOverflow:       OverflowTruncate,
		CtxMaxInjectChars:  8000,
	}
}

// Path returns the path to the config file.
func Path() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".config", "baish", "config.toml")
}

// Load reads the config file. If it doesn't exist the defaults are written
// to disk and returned so future runs have an editable template.
func Load() (Config, error) {
	cfg := Defaults()
	p := Path()

	data, err := os.ReadFile(p)
	if os.IsNotExist(err) {
		if saveErr := Save(cfg); saveErr != nil {
			// Non-fatal — just return defaults without persisting.
			return cfg, nil
		}
		return cfg, nil
	}
	if err != nil {
		return cfg, fmt.Errorf("read config: %w", err)
	}

	if err := toml.Unmarshal(data, &cfg); err != nil {
		return Defaults(), fmt.Errorf("parse config: %w", err)
	}

	// Clamp values to sane ranges.
	if cfg.MaxHistoryMessages < 2 {
		cfg.MaxHistoryMessages = 2
	}
	if cfg.ToolOutputMaxChars < 100 {
		cfg.ToolOutputMaxChars = 100
	}
	if cfg.ToolOverflow != OverflowSummarize {
		cfg.ToolOverflow = OverflowTruncate
	}
	if cfg.CtxMaxInjectChars < 500 {
		cfg.CtxMaxInjectChars = 500
	}

	return cfg, nil
}

// Save writes the config to disk.
func Save(cfg Config) error {
	p := Path()
	if err := os.MkdirAll(filepath.Dir(p), 0755); err != nil {
		return fmt.Errorf("create config dir: %w", err)
	}
	data, err := toml.Marshal(cfg)
	if err != nil {
		return fmt.Errorf("marshal config: %w", err)
	}
	return os.WriteFile(p, data, 0644)
}

// ContextDir returns the directory where context slots are persisted.
func ContextDir() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".config", "baish", "contexts")
}

// SaveContext persists a named context slot to disk.
func SaveContext(name, content string) error {
	dir := ContextDir()
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("create context dir: %w", err)
	}
	return os.WriteFile(filepath.Join(dir, name+".md"), []byte(content), 0644)
}

// DeleteContext removes a named context slot from disk.
func DeleteContext(name string) error {
	err := os.Remove(filepath.Join(ContextDir(), name+".md"))
	if os.IsNotExist(err) {
		return nil
	}
	return err
}

// ClearContexts removes all persisted context slots.
func ClearContexts() error {
	dir := ContextDir()
	entries, err := os.ReadDir(dir)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	for _, e := range entries {
		if !e.IsDir() && filepath.Ext(e.Name()) == ".md" {
			os.Remove(filepath.Join(dir, e.Name()))
		}
	}
	return nil
}

// LoadContexts reads all persisted context slots from disk.
func LoadContexts() (map[string]string, error) {
	dir := ContextDir()
	entries, err := os.ReadDir(dir)
	if os.IsNotExist(err) {
		return map[string]string{}, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read context dir: %w", err)
	}
	slots := make(map[string]string, len(entries))
	for _, e := range entries {
		if e.IsDir() || filepath.Ext(e.Name()) != ".md" {
			continue
		}
		data, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			continue
		}
		name := strings.TrimSuffix(e.Name(), ".md")
		slots[name] = string(data)
	}
	return slots, nil
}
