package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/pvjammer/ai-shell-poc/backend"
	"github.com/pvjammer/ai-shell-poc/cli"
	"github.com/pvjammer/ai-shell-poc/config"
	"github.com/pvjammer/ai-shell-poc/shell"
)

func main() {
	// If the first argument is a known CLI subcommand, dispatch to CLI mode.
	// Otherwise start the interactive shell as usual.
	if len(os.Args) > 1 && cli.Subcommands[os.Args[1]] {
		if err := cli.Run(os.Args[1:]); err != nil {
			fmt.Fprintf(os.Stderr, "baish: %v\n", err)
			os.Exit(1)
		}
		return
	}

	appCfg, err := config.Load()
	if err != nil {
		fmt.Fprintf(os.Stderr, "ai-shell: config warning: %v (using defaults)\n", err)
		appCfg = config.Defaults()
	}

	ocURL := env("AI_OPENCODE_URL", "")
	resumeID, err := parseResume(os.Args[1:], ocURL)
	if err != nil {
		fmt.Fprintf(os.Stderr, "baish: %v\n", err)
		os.Exit(1)
	}

	cfg := shell.Config{
		// Priority: hardcoded default < appCfg.LLM (~/.config/baish/config.toml) < env var.
		Model:         env("AI_SHELL_MODEL", firstNonEmpty(appCfg.LLM.Model, "llama3.2")),
		Endpoint:      env("AI_SHELL_ENDPOINT", firstNonEmpty(appCfg.LLM.Endpoint, "http://localhost:11434")),
		Provider:      env("AI_SHELL_PROVIDER", firstNonEmpty(appCfg.LLM.Provider, "ollama")), // "ollama" | "openai" (llama.cpp, vLLM, LM Studio, real OpenAI)
		APIKey:        env("AI_SHELL_API_KEY", appCfg.LLM.APIKey),
		OpenCodeURL:   ocURL,
		OpenCodeModel: env("AI_OPENCODE_MODEL", ""),
		ResumeSession: resumeID,
		Version:       Version,
	}

	s, err := shell.New(cfg, appCfg)
	if err != nil {
		fmt.Fprintf(os.Stderr, "baish: %v\n", err)
		os.Exit(1)
	}

	if err := s.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "baish: %v\n", err)
		os.Exit(1)
	}
}

// parseResume scans args for --resume [id] and returns the session ID to resume.
// With --resume <id> it returns that ID directly. With --resume alone it queries
// the OpenCode server and prompts the user to pick a session interactively.
func parseResume(args []string, ocURL string) (string, error) {
	resumeIdx := -1
	for i, a := range args {
		if a == "--resume" {
			resumeIdx = i
			break
		}
	}
	if resumeIdx < 0 {
		return "", nil
	}
	if ocURL == "" {
		return "", fmt.Errorf("--resume requires AI_OPENCODE_URL to be set")
	}

	// --resume <id>: next arg is the session ID (when it doesn't look like a flag)
	if resumeIdx+1 < len(args) && !strings.HasPrefix(args[resumeIdx+1], "-") {
		return args[resumeIdx+1], nil
	}

	// --resume with no ID: list sessions and let the user choose
	return pickSession(ocURL)
}

// pickSession fetches sessions from the OpenCode server, prints a numbered list,
// and reads the user's choice from stdin. Returns the chosen session ID.
func pickSession(ocURL string) (string, error) {
	client := backend.NewOpenCodeClient(ocURL, "")
	sessions, err := client.ListSessions(context.Background())
	if err != nil {
		return "", fmt.Errorf("--resume: could not list sessions: %w", err)
	}
	if len(sessions) == 0 {
		return "", fmt.Errorf("--resume: no sessions found on %s", ocURL)
	}

	now := time.Now()
	home, _ := os.UserHomeDir()
	fmt.Println("OpenCode sessions:")
	for i, s := range sessions {
		updated := time.Unix(s.Time.Updated/1000, 0)
		age := now.Sub(updated)
		dir := s.Directory
		if home != "" {
			dir = strings.Replace(dir, home, "~", 1)
		}
		title := s.Title
		if len(title) > 40 {
			title = title[:37] + "..."
		}
		fmt.Printf("  %2d  %-20s  %-42s  %-24s  %s ago\n",
			i+1, s.Slug, title, dir, formatAge(age))
	}
	fmt.Printf("\nSelect session (1-%d): ", len(sessions))

	scanner := bufio.NewScanner(os.Stdin)
	if !scanner.Scan() {
		return "", fmt.Errorf("--resume: no selection made")
	}
	choice, err := strconv.Atoi(strings.TrimSpace(scanner.Text()))
	if err != nil || choice < 1 || choice > len(sessions) {
		return "", fmt.Errorf("--resume: invalid selection %q", strings.TrimSpace(scanner.Text()))
	}
	return sessions[choice-1].ID, nil
}

func formatAge(d time.Duration) string {
	switch {
	case d < time.Minute:
		return fmt.Sprintf("%ds", int(d.Seconds()))
	case d < time.Hour:
		return fmt.Sprintf("%dm", int(d.Minutes()))
	case d < 24*time.Hour:
		return fmt.Sprintf("%dh", int(d.Hours()))
	default:
		return fmt.Sprintf("%dd", int(d.Hours()/24))
	}
}

func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}
