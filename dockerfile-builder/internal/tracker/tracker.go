// Package tracker records environment-modifying commands during a session
// and detects filesystem changes outside the workspace directory.
package tracker

// InstallEvent records a single detected package install or environment change.
type InstallEvent struct {
	Command   string // raw command as typed
	Manager   string // apt, pip, npm, cargo, curl-installer, etc.
	Packages  []string
	Timestamp int64
}

// Tracker collects InstallEvents in real time and computes the environment
// diff from the overlay upperdir on session end.
type Tracker struct {
	Events []InstallEvent
}

// New returns a ready Tracker.
func New() *Tracker {
	return &Tracker{}
}

// Record checks a command string for known install patterns and logs a
// matching event. Returns true if the command was recognized as an install.
func (t *Tracker) Record(cmd string) bool {
	// TODO: implement pattern matching for apt/pip/npm/cargo/curl-installer
	_ = cmd
	return false
}

// DiffUpperdir walks upperdir and splits entries into workspace changes
// (inside workspaceRoot) and environment changes (everything else).
// Environment changes are returned as a flat list of affected paths.
func DiffUpperdir(upperdir, workspaceRoot string) (envPaths []string, workspacePaths []string, err error) {
	// TODO: walk upperdir; handle whiteout files (char dev 0/0 or .wh. prefix)
	_ = upperdir
	_ = workspaceRoot
	return nil, nil, nil
}
