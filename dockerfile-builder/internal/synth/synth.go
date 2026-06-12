// Package synth converts raw install events and overlay diffs into clean
// Dockerfile RUN instructions via LLM synthesis.
package synth

import "github.com/pvjammer/dockerfile-builder-poc/internal/tracker"

// Input is the full picture of what happened in a session.
type Input struct {
	BaseImage    string
	Events       []tracker.InstallEvent // real-time tracked installs
	EnvPaths     []string               // files changed outside workspace (overlay diff)
}

// Result holds the synthesized Dockerfile content.
type Result struct {
	Dockerfile string // complete Dockerfile text
	Layers     []string // individual RUN instructions (for incremental append)
}

// Synthesize calls an LLM to convert raw install data into clean Dockerfile
// instructions. The LLM prompt includes the install events, the changed paths,
// and the base image so it can make informed decisions about grouping and cleanup.
//
// TODO: wire up to an LLM provider (Ollama / Anthropic) once the core
// detection logic is validated.
func Synthesize(input Input) (Result, error) {
	_ = input
	return Result{}, nil
}
