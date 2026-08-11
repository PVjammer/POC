package cli

import (
	"fmt"
	"os"
	"os/exec"
	"strings"

	"github.com/pvjammer/ai-shell-poc/agent"
	"github.com/pvjammer/ai-shell-poc/config"
	"github.com/pvjammer/ai-shell-poc/llmprovider"
)

// runCm implements `baish cm` — generate a commit message from staged changes.
func runCm(args []string) error {
	if len(args) > 0 && (args[0] == "--help" || args[0] == "-h") {
		fmt.Println("usage: baish cm")
		fmt.Println("  Generates a commit message from staged git changes.")
		fmt.Println("  Stage changes with 'git add' before running.")
		return nil
	}

	diff, err := stagedDiff()
	if err != nil {
		return fmt.Errorf("cm: %w", err)
	}
	if strings.TrimSpace(diff) == "" {
		return fmt.Errorf("cm: nothing staged (run 'git add' first)")
	}

	appCfg, _ := config.Load()
	// Priority: hardcoded default < appCfg.LLM (config.toml) < env var.
	model := envOr("AI_SHELL_MODEL", firstNonEmpty(appCfg.LLM.Model, "llama3.2"))
	endpoint := envOr("AI_SHELL_ENDPOINT", firstNonEmpty(appCfg.LLM.Endpoint, "http://localhost:11434"))
	providerKind := envOr("AI_SHELL_PROVIDER", firstNonEmpty(appCfg.LLM.Provider, "ollama"))
	apiKey := envOr("AI_SHELL_API_KEY", appCfg.LLM.APIKey)

	provider, err := llmprovider.New(providerKind, endpoint, model, apiKey)
	if err != nil {
		return fmt.Errorf("cm: create provider: %w", err)
	}

	loop := agent.New(provider, nil, nil)
	loop.SetConfig(agent.LoopConfig{MaxResponseTokens: appCfg.MaxResponseTokens})

	systemPrompt := `You are an expert software developer writing git commit messages.
Write a concise, informative commit message for the given diff.
- First line: imperative mood, ≤72 chars, no period
- If helpful, add a blank line then bullet points explaining why
- Detect and follow conventional commits style if the project uses it (feat/fix/refactor/chore/docs/test/etc.)
- Output ONLY the commit message — no preamble, no explanation, no code fences`

	userMsg := "Write a commit message for this diff:\n\n" + diff

	fmt.Fprint(os.Stderr, "\033[2mgenerating commit message…\033[0m\r")
	if err := loop.RunOneShot(nil, systemPrompt, userMsg, func(token string) {
		fmt.Print(token)
	}); err != nil {
		return fmt.Errorf("cm: %w", err)
	}
	fmt.Println()
	fmt.Fprint(os.Stderr, "\033[2m                            \033[0m\r") // clear status line
	return nil
}

func stagedDiff() (string, error) {
	// Check for git repo
	if err := exec.Command("git", "rev-parse", "--git-dir").Run(); err != nil {
		return "", fmt.Errorf("not a git repository")
	}
	// Get staged diff
	out, err := exec.Command("git", "diff", "--cached").Output()
	if err != nil {
		return "", fmt.Errorf("git diff failed: %w", err)
	}
	diff := string(out)
	if strings.TrimSpace(diff) != "" {
		return diff, nil
	}
	// Nothing staged — give a helpful message
	statusOut, _ := exec.Command("git", "status", "--porcelain").Output()
	status := string(statusOut)
	hasUnstaged := strings.ContainsAny(status, "MADRCU?")
	if hasUnstaged {
		return "", fmt.Errorf("nothing staged — you have unstaged changes (run 'git add' first)")
	}
	return "", fmt.Errorf("nothing staged and working tree is clean")
}
