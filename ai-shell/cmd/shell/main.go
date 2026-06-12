package main

import (
	"fmt"
	"os"

	"github.com/pvjammer/ai-shell-poc/config"
	"github.com/pvjammer/ai-shell-poc/shell"
)

func main() {
	appCfg, err := config.Load()
	if err != nil {
		fmt.Fprintf(os.Stderr, "ai-shell: config warning: %v (using defaults)\n", err)
		appCfg = config.Defaults()
	}

	cfg := shell.Config{
		Model:    env("AI_SHELL_MODEL", "llama3.2"),
		Endpoint: env("AI_SHELL_ENDPOINT", "http://localhost:11434"),
	}

	s, err := shell.New(cfg, appCfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "ai-shell: %v\n", err)
		os.Exit(1)
	}

	if err := s.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "ai-shell: %v\n", err)
		os.Exit(1)
	}
}

func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
