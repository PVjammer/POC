package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

// DataDir returns the XDG data home directory for baish.
func DataDir() string {
	if d := os.Getenv("XDG_DATA_HOME"); d != "" {
		return filepath.Join(d, "baish")
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".local", "share", "baish")
}

// JobsDir returns the directory where completed job records are persisted.
func JobsDir() string { return filepath.Join(DataDir(), "jobs") }

// SessionsDir returns the directory where session histories are persisted.
func SessionsDir() string { return filepath.Join(DataDir(), "sessions") }

// JobRecord is the persisted metadata for a completed job.
type JobRecord struct {
	ID      int           `json:"id"`
	Name    string        `json:"name"`       // empty if unnamed
	Display string        `json:"display"`    // original prompt or command
	Failed  bool          `json:"failed"`
	Elapsed time.Duration `json:"elapsed_ns"` // stored as nanoseconds
	Started time.Time     `json:"started"`
	Size    int           `json:"size"` // byte length of output
}

func jobDir(id int) string {
	return filepath.Join(JobsDir(), strconv.Itoa(id))
}

// SaveJob persists a completed job record and its output to disk.
func SaveJob(id int, name, display, output string, failed bool, elapsed time.Duration, started time.Time) error {
	dir := jobDir(id)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("create job dir: %w", err)
	}
	rec := JobRecord{
		ID: id, Name: name, Display: display, Failed: failed,
		Elapsed: elapsed, Started: started, Size: len(output),
	}
	meta, err := json.Marshal(rec)
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, "meta.json"), meta, 0644); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "output"), []byte(output), 0644)
}

// LoadJobs returns all persisted job records sorted by ID descending (newest first).
func LoadJobs() ([]JobRecord, error) {
	entries, err := os.ReadDir(JobsDir())
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var records []JobRecord
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		data, err := os.ReadFile(filepath.Join(JobsDir(), e.Name(), "meta.json"))
		if err != nil {
			continue
		}
		var rec JobRecord
		if err := json.Unmarshal(data, &rec); err != nil {
			continue
		}
		records = append(records, rec)
	}
	sort.Slice(records, func(i, j int) bool { return records[i].ID > records[j].ID })
	return records, nil
}

// LoadJobOutput reads the output of a job by numeric ID or by name.
// Returns (output, found, error).
func LoadJobOutput(idOrName string) (string, bool, error) {
	id := 0
	if n, err := strconv.Atoi(idOrName); err == nil {
		id = n
	} else {
		id = FindJobByName(idOrName)
	}
	if id == 0 {
		return "", false, nil
	}
	path := filepath.Join(jobDir(id), "output")
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return "", false, nil
	}
	if err != nil {
		return "", true, err
	}
	return string(data), true, nil
}

// MaxJobID returns the highest job ID persisted to disk, or 0 if none.
func MaxJobID() int {
	entries, err := os.ReadDir(JobsDir())
	if err != nil {
		return 0
	}
	max := 0
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		if n, err := strconv.Atoi(e.Name()); err == nil && n > max {
			max = n
		}
	}
	return max
}

// FindJobByName returns the ID of the most recent job with the given name, or 0.
func FindJobByName(name string) int {
	recs, err := LoadJobs() // already sorted newest-first
	if err != nil {
		return 0
	}
	for _, r := range recs {
		if strings.EqualFold(r.Name, name) {
			return r.ID
		}
	}
	return 0
}

// SessionPath returns the file path for a named session's persisted history.
func SessionPath(name string) string {
	return filepath.Join(SessionsDir(), name+".json")
}

