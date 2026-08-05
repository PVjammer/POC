package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// WithsDir returns the directory where interrupted /with buffers are persisted.
func WithsDir() string { return filepath.Join(DataDir(), "withs") }

func withDir(name string) string { return filepath.Join(WithsDir(), name) }

// WithRecord is the persisted metadata for a saved /with buffer.
type WithRecord struct {
	Name     string    `json:"name"`
	OnError  bool      `json:"on_error"`
	CmdCount int       `json:"cmd_count"`
	Started  time.Time `json:"started"`
	Saved    time.Time `json:"saved"`
}

// SaveWith persists a /with buffer's metadata and content to disk.
// Called on baish shutdown when a capture context was never closed.
func SaveWith(name, content string, r WithRecord) error {
	dir := withDir(name)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("create with dir: %w", err)
	}
	meta, err := json.Marshal(r)
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "meta.json"), meta, 0644); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "content"), []byte(content), 0644)
}

// LoadWiths returns metadata for all saved /with buffers. not-exist → nil, nil.
func LoadWiths() ([]WithRecord, error) {
	entries, err := os.ReadDir(WithsDir())
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var records []WithRecord
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		data, err := os.ReadFile(filepath.Join(WithsDir(), e.Name(), "meta.json"))
		if err != nil {
			continue
		}
		var rec WithRecord
		if err := json.Unmarshal(data, &rec); err != nil {
			continue
		}
		records = append(records, rec)
	}
	return records, nil
}

// LoadWith returns the metadata and content for a named saved buffer.
func LoadWith(name string) (WithRecord, string, error) {
	data, err := os.ReadFile(filepath.Join(withDir(name), "meta.json"))
	if err != nil {
		return WithRecord{}, "", fmt.Errorf("load with %q: %w", name, err)
	}
	var rec WithRecord
	if err := json.Unmarshal(data, &rec); err != nil {
		return WithRecord{}, "", fmt.Errorf("parse with %q metadata: %w", name, err)
	}
	content, err := os.ReadFile(filepath.Join(withDir(name), "content"))
	if err != nil {
		return rec, "", fmt.Errorf("load with %q content: %w", name, err)
	}
	return rec, string(content), nil
}

// DeleteWith removes a saved /with buffer directory.
func DeleteWith(name string) error {
	return os.RemoveAll(withDir(name))
}
