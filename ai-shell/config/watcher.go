package config

import (
	"context"
	"path/filepath"
	"strings"
	"time"

	"github.com/fsnotify/fsnotify"
)

// DynamicConfig watches a set of paths and fires onChange after a short debounce
// whenever any watched file or directory changes.
type DynamicConfig struct {
	watcher  *fsnotify.Watcher
	onChange func(path string)
}

// NewDynamicConfig creates a watcher for the given paths (files or directories).
// Missing paths are silently skipped so the shell starts cleanly even if
// ~/.config/baish/agents/ doesn't exist yet.
// onChange is called with the changed path after a 200ms debounce.
func NewDynamicConfig(paths []string, onChange func(path string)) (*DynamicConfig, error) {
	w, err := fsnotify.NewWatcher()
	if err != nil {
		return nil, err
	}
	for _, p := range paths {
		_ = w.Add(p) // ignore errors for non-existent paths
	}
	return &DynamicConfig{watcher: w, onChange: onChange}, nil
}

// Start runs the watch loop in a goroutine until ctx is cancelled.
func (d *DynamicConfig) Start(ctx context.Context) {
	go d.loop(ctx)
}

// Stop releases the underlying watcher resources.
func (d *DynamicConfig) Stop() {
	_ = d.watcher.Close()
}

// AddPath adds a new path to the watch list. Safe to call after Start.
func (d *DynamicConfig) AddPath(path string) {
	_ = d.watcher.Add(path)
}

// isEditorTemp returns true for files that editors write as part of their
// save/backup mechanics and that should not trigger a config reload.
func isEditorTemp(path string) bool {
	base := filepath.Base(path)
	// vim/neovim swap files: .file.swp, .file.swx, .file.swo, .file.swn, etc.
	// vim cycles through .sw[a-z] when the first slot is taken.
	if strings.HasPrefix(base, ".") {
		ext := filepath.Ext(base)
		if len(ext) == 4 && strings.HasPrefix(ext, ".sw") {
			return true
		}
	}
	// emacs backup/lock files: file~  #file#
	if strings.HasSuffix(base, "~") || (strings.HasPrefix(base, "#") && strings.HasSuffix(base, "#")) {
		return true
	}
	// common editor temp patterns: *.tmp, *.bak
	ext := strings.ToLower(filepath.Ext(base))
	return ext == ".tmp" || ext == ".bak"
}

func (d *DynamicConfig) loop(ctx context.Context) {
	var (
		debounce  = 200 * time.Millisecond
		timer     *time.Timer
		lastPath  string
	)

	fire := func() {
		if d.onChange != nil {
			d.onChange(lastPath)
		}
	}

	for {
		select {
		case <-ctx.Done():
			if timer != nil {
				timer.Stop()
			}
			return

		case event, ok := <-d.watcher.Events:
			if !ok {
				return
			}
			if isEditorTemp(event.Name) {
				continue
			}
			lastPath = event.Name
			if timer != nil {
				timer.Stop()
			}
			timer = time.AfterFunc(debounce, fire)

		case _, ok := <-d.watcher.Errors:
			if !ok {
				return
			}
			// watcher errors are non-fatal; ignore
		}
	}
}
