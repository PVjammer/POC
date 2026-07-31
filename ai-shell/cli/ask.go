package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/pvjammer/ai-shell-poc/agent"
	"github.com/pvjammer/ai-shell-poc/config"
	"github.com/pvjammer/ai-shell-poc/tools"
	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

type askOpts struct {
	session  string
	model    string
	endpoint string
	act      bool // true = agentic (do), false = advisory (ask)
}

// runAsk implements `baish ask` and `baish do`.
func runAsk(args []string, act bool) error {
	opts, prompt, err := parseAskArgs(args, act)
	if err != nil {
		return err
	}
	if prompt == "" {
		if act {
			return fmt.Errorf("usage: baish do <prompt>")
		}
		return fmt.Errorf("usage: baish ask <prompt>")
	}

	appCfg, _ := config.Load()
	model := opts.model
	if model == "" {
		model = envOr("AI_SHELL_MODEL", "llama3.2")
	}
	endpoint := opts.endpoint
	if endpoint == "" {
		endpoint = envOr("AI_SHELL_ENDPOINT", "http://localhost:11434")
	}

	provider, err := llm.NewOllamaProvider(endpoint, model)
	if err != nil {
		return fmt.Errorf("create provider: %w", err)
	}

	var toolDefs []llm.ToolDef
	var handlers map[string]func(map[string]interface{}) (string, error)
	if act {
		toolDefs = tools.AllTools()
		handlers = tools.AllHandlers()
	}

	loop := agent.New(provider, toolDefs, handlers)
	loop.SetConfig(agent.LoopConfig{
		MaxHistoryMessages:     appCfg.MaxHistoryMessages,
		ToolOutputMaxChars:     appCfg.ToolOutputMaxChars,
		ToolOutputKeepRounds:   appCfg.ToolOutputKeepRounds,
		MaxContextTokens:       appCfg.MaxContextTokens,
		CompactionThreshold:    appCfg.CompactionThreshold,
		CompactionTailMessages: appCfg.CompactionTailMessages,
		MaxResponseTokens:      appCfg.MaxResponseTokens,
	})

	// Inject ctx slots into system prompt via loop context slots.
	rawSlots, _ := config.LoadContexts()
	agentSlots := make(map[string]agent.CtxSlot, len(rawSlots))
	for name, content := range rawSlots {
		if len(content) <= appCfg.CtxInlineThreshold {
			agentSlots[name] = agent.CtxSlot{Content: content}
		} else {
			agentSlots[name] = agent.CtxSlot{
				Content:     content,
				Description: fmt.Sprintf("%s document. Call read_context('%s') to retrieve.", humanSize(len(content)), name),
			}
		}
	}
	loop.SetContextSlots(agentSlots)

	loop.SetSpinnerEnabled(false)

	// Load persisted session history.
	sessName := opts.session
	if sessName == "" {
		sessName = "main"
	}
	if data, err := os.ReadFile(config.SessionPath(sessName)); err == nil {
		var hist []llm.ChatMessage
		if json.Unmarshal(data, &hist) == nil && len(hist) > 0 {
			loop.SetHistory(hist)
		}
	}

	// Run and stream to stdout.
	ctx := context.Background()
	if err := loop.Run(ctx, prompt, func(token string) {
		fmt.Print(token)
	}); err != nil {
		return err
	}
	fmt.Println() // trailing newline after streaming

	// Persist updated session history.
	hist := loop.CopyHistory()
	if data, err := json.Marshal(hist); err == nil {
		_ = os.MkdirAll(config.SessionsDir(), 0755)
		_ = os.WriteFile(config.SessionPath(sessName), data, 0644)
	}
	return nil
}

func parseAskArgs(args []string, act bool) (askOpts, string, error) {
	opts := askOpts{act: act}
	var promptParts []string
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--session", "-s":
			if i+1 >= len(args) {
				return opts, "", fmt.Errorf("--session requires a value")
			}
			i++
			opts.session = args[i]
		case "--model", "-m":
			if i+1 >= len(args) {
				return opts, "", fmt.Errorf("--model requires a value")
			}
			i++
			opts.model = args[i]
		case "--endpoint", "-e":
			if i+1 >= len(args) {
				return opts, "", fmt.Errorf("--endpoint requires a value")
			}
			i++
			opts.endpoint = args[i]
		default:
			promptParts = append(promptParts, args[i])
		}
	}
	return opts, strings.Join(promptParts, " "), nil
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func humanSize(n int) string {
	if n < 1024 {
		return fmt.Sprintf("%dB", n)
	}
	return fmt.Sprintf("%.1fKB", float64(n)/1024)
}
