package skills

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"gopkg.in/yaml.v3"
)

// Loader discovers and caches skills from watched directories.
// All methods are safe for concurrent use.
type Loader struct {
	mu     sync.RWMutex
	skills map[string]Record // keyed by skill name
}

// NewLoader returns an empty Loader. Call Scan to populate it.
func NewLoader() *Loader {
	return &Loader{skills: make(map[string]Record)}
}

// Scan walks each path, discovers skill directories (those containing SKILL.md),
// parses their frontmatter, and updates the in-memory catalog.
// On name collision the later path wins (higher-priority paths should come last).
// Returns a slice of non-fatal parse warnings; a missing directory is not an error.
func (l *Loader) Scan(paths []string) []error {
	found := make(map[string]Record)
	var errs []error

	for _, base := range paths {
		entries, err := os.ReadDir(base)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			errs = append(errs, fmt.Errorf("skills: read dir %s: %w", base, err))
			continue
		}
		for _, e := range entries {
			if !e.IsDir() {
				continue
			}
			skillDir := filepath.Join(base, e.Name())
			mdPath := filepath.Join(skillDir, "SKILL.md")
			data, err := os.ReadFile(mdPath)
			if os.IsNotExist(err) {
				continue
			}
			if err != nil {
				errs = append(errs, fmt.Errorf("skills: read %s: %w", mdPath, err))
				continue
			}
			rec, err := parseFrontmatter(data, skillDir)
			if err != nil {
				errs = append(errs, fmt.Errorf("skills: parse %s: %w", mdPath, err))
				continue
			}
			found[rec.Name] = rec
		}
	}

	l.mu.Lock()
	l.skills = found
	l.mu.Unlock()
	return errs
}

// All returns a snapshot of all loaded skills, sorted by name.
func (l *Loader) All() []Record {
	l.mu.RLock()
	defer l.mu.RUnlock()
	out := make([]Record, 0, len(l.skills))
	for _, r := range l.skills {
		out = append(out, r)
	}
	// stable sort by name
	for i := 1; i < len(out); i++ {
		for j := i; j > 0 && out[j].Name < out[j-1].Name; j-- {
			out[j], out[j-1] = out[j-1], out[j]
		}
	}
	return out
}

// Get returns the Record for a named skill.
func (l *Loader) Get(name string) (Record, bool) {
	l.mu.RLock()
	defer l.mu.RUnlock()
	r, ok := l.skills[name]
	return r, ok
}

// Body reads and returns the full SKILL.md body (everything after the frontmatter block).
func (l *Loader) Body(name string) (string, error) {
	l.mu.RLock()
	rec, ok := l.skills[name]
	l.mu.RUnlock()
	if !ok {
		return "", fmt.Errorf("skill %q not found", name)
	}
	data, err := os.ReadFile(filepath.Join(rec.Dir, "SKILL.md"))
	if err != nil {
		return "", fmt.Errorf("skill %q: read SKILL.md: %w", name, err)
	}
	return extractBody(data), nil
}

// ApplySubstitution replaces $ARGUMENTS, $ARGUMENTS[N], $N, and ${SKILL_DIR}
// in the skill body. Shell injection (!`cmd`) is not performed in phase 1.
func (l *Loader) ApplySubstitution(body string, args []string, skillDir string) string {
	joined := strings.Join(args, " ")
	body = strings.ReplaceAll(body, "$ARGUMENTS", joined)
	body = strings.ReplaceAll(body, "${SKILL_DIR}", skillDir)
	for i, a := range args {
		body = strings.ReplaceAll(body, fmt.Sprintf("$ARGUMENTS[%d]", i), a)
		body = strings.ReplaceAll(body, fmt.Sprintf("$%d", i), a)
	}
	return body
}

// ── internal ──────────────────────────────────────────────────────────────────

type skillFrontmatter struct {
	Name                   string   `yaml:"name"`
	Description            string   `yaml:"description"`
	ArgumentHint           string   `yaml:"argument-hint"`
	AllowedTools           string   `yaml:"allowed-tools"` // space-delimited per spec
	DisableModelInvocation bool     `yaml:"disable-model-invocation"`
}

func parseFrontmatter(data []byte, dir string) (Record, error) {
	body := bytes.TrimSpace(data)
	if !bytes.HasPrefix(body, []byte("---")) {
		return Record{}, fmt.Errorf("SKILL.md has no YAML frontmatter (expected leading ---)")
	}
	// strip opening ---
	body = bytes.TrimPrefix(body, []byte("---"))
	// find closing ---
	idx := bytes.Index(body, []byte("\n---"))
	if idx < 0 {
		return Record{}, fmt.Errorf("SKILL.md frontmatter is not closed (missing closing ---)")
	}
	header := body[:idx]

	var fm skillFrontmatter
	if err := yaml.Unmarshal(header, &fm); err != nil {
		return Record{}, fmt.Errorf("parse YAML frontmatter: %w", err)
	}
	if fm.Name == "" {
		return Record{}, fmt.Errorf("SKILL.md frontmatter missing required field: name")
	}
	if fm.Description == "" {
		return Record{}, fmt.Errorf("SKILL.md frontmatter missing required field: description")
	}

	var allowedTools []string
	for _, t := range strings.Fields(fm.AllowedTools) {
		if t != "" {
			allowedTools = append(allowedTools, t)
		}
	}

	return Record{
		Name:                   fm.Name,
		Description:            fm.Description,
		ArgumentHint:           fm.ArgumentHint,
		AllowedTools:           allowedTools,
		DisableModelInvocation: fm.DisableModelInvocation,
		Dir:                    dir,
	}, nil
}

// extractBody returns the markdown body of a SKILL.md — everything after the
// closing --- of the frontmatter block. Returns the full file if no frontmatter found.
func extractBody(data []byte) string {
	s := string(data)
	// Find second --- (closing delimiter)
	first := strings.Index(s, "---")
	if first < 0 {
		return strings.TrimSpace(s)
	}
	rest := s[first+3:]
	second := strings.Index(rest, "---")
	if second < 0 {
		return strings.TrimSpace(s)
	}
	return strings.TrimSpace(rest[second+3:])
}
