// dockerfile-builder: interactive Dockerfile builder POC.
//
// Run a Docker container interactively, track environment changes
// (package installs, tool setup, config files), and synthesize a
// clean Dockerfile from the session.
//
// Usage:
//
//	dockerfile-builder [image]           # start interactive session
//	dockerfile-builder --from session-id # resume / export from saved session
//	dockerfile-builder export session-id # print synthesized Dockerfile to stdout
//
// Detection mechanisms:
//
//  1. Real-time: intercept known install commands (apt, pip, npm, cargo, etc.)
//     as they are typed and log them to a session recipe.
//
//  2. Post-session overlay diff: compare container filesystem changes against
//     the base image layer. Files that changed outside the workspace directory
//     are environment changes; files inside the workspace are project changes.
//     Whiteout entries (deleted files) are preserved.
//
// Synthesis:
//
//	The recorded install commands + overlay diff are passed to an LLM that
//	produces clean, cache-friendly Dockerfile RUN instructions: grouped apt
//	installs, proper cleanup, DEBIAN_FRONTEND=noninteractive, etc.
package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Fprintln(os.Stderr, "dockerfile-builder: not yet implemented")
	os.Exit(1)
}
