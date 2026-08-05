package main

// Version and Commit are injected at build time via -ldflags.
// When built without ldflags (e.g. go run), both report "dev".
var (
	Version = "dev"
	Commit  = "dev"
)
