package shell

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/chzyer/readline"
	"github.com/pvjammer/ai-shell-poc/agent"
	"github.com/pvjammer/ai-shell-poc/tools"
	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

// Config holds runtime configuration for the shell.
type Config struct {
	Model    string
	Endpoint string
}

// Shell is the main REPL.
type Shell struct {
	cfg   Config
	rl    *readline.Instance
	agent *agent.Loop
}

// New creates and wires up the shell. Returns an error if the LLM provider
// or readline cannot be initialised.
func New(cfg Config) (*Shell, error) {
	provider, err := llm.NewOllamaProvider(cfg.Endpoint, cfg.Model)
	if err != nil {
		return nil, fmt.Errorf("create llm provider: %w", err)
	}

	rl, err := readline.NewEx(&readline.Config{
		Prompt:          buildPrompt(),
		HistoryFile:     filepath.Join(os.Getenv("HOME"), ".ai_shell_history"),
		InterruptPrompt: "^C",
		EOFPrompt:       "exit",
		HistorySearchFold: true,
	})
	if err != nil {
		return nil, fmt.Errorf("create readline: %w", err)
	}

	agentLoop := agent.New(provider, tools.AllTools(), tools.AllHandlers())

	// Wire up tool call display callbacks.
	agentLoop.OnToolCall = func(name string, args map[string]interface{}) {
		if cmd, ok := args["command"]; ok {
			fmt.Fprintf(os.Stderr, "\033[2m  [%s] $ %v\033[0m\n", name, cmd)
		} else {
			fmt.Fprintf(os.Stderr, "\033[2m  [%s] %v\033[0m\n", name, args)
		}
	}
	agentLoop.OnToolResult = func(name string, result string) {
		// Don't print results by default — the agent will summarise them.
		// Uncomment to debug tool output:
		// fmt.Fprintf(os.Stderr, "\033[2m  → %s\033[0m\n", result)
		_ = name
		_ = result
	}

	return &Shell{cfg: cfg, rl: rl, agent: agentLoop}, nil
}

// Run starts the interactive loop and blocks until the user exits.
func (s *Shell) Run() error {
	defer s.rl.Close()

	fmt.Printf("ai-shell  model=%s  endpoint=%s\n", s.cfg.Model, s.cfg.Endpoint)
	fmt.Println("  <cmd>    run a shell command (ls, vim, htop, ...)")
	fmt.Println("  ?<msg>   ask the AI")
	fmt.Println("  /help    built-in commands")
	fmt.Println()

	for {
		s.rl.SetPrompt(buildPrompt())

		line, err := s.rl.Readline()
		if err != nil {
			// io.EOF (Ctrl+D) or readline.ErrInterrupt (Ctrl+C at empty prompt)
			fmt.Println("exit")
			break
		}

		in := Parse(line)
		if in.Content == "" {
			continue
		}

		switch in.Type {
		case InputDirect:
			s.runDirect(in.Content)
		case InputAgent:
			s.runAgent(in.Content)
		case InputMeta:
			if s.runMeta(in.Content) {
				return nil
			}
		}
	}
	return nil
}

// runDirect executes a command with the real terminal attached so interactive
// programs (vim, htop, less, ssh, etc.) work correctly.
// cd is handled as a builtin since it must change the shell process's own cwd.
func (s *Shell) runDirect(cmdStr string) {
	// Handle cd as a builtin — a subprocess cd would be invisible to us.
	if dir, ok := parseCd(cmdStr); ok {
		if dir == "" {
			dir = os.Getenv("HOME")
		}
		if err := os.Chdir(dir); err != nil {
			fmt.Fprintf(os.Stderr, "cd: %v\n", err)
		}
		return
	}

	c := exec.Command("sh", "-c", cmdStr)
	c.Stdin = os.Stdin
	c.Stdout = os.Stdout
	c.Stderr = os.Stderr

	// Forward SIGINT/SIGTERM to the child while it runs.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	defer signal.Stop(sigCh)

	if err := c.Start(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		return
	}

	done := make(chan error, 1)
	go func() { done <- c.Wait() }()

	select {
	case sig := <-sigCh:
		_ = c.Process.Signal(sig)
		<-done
	case err := <-done:
		if err != nil {
			if _, isExit := err.(*exec.ExitError); !isExit {
				fmt.Fprintf(os.Stderr, "error: %v\n", err)
			}
		}
	}
}

// runAgent sends the message to the AI and streams its response.
func (s *Shell) runAgent(msg string) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Allow Ctrl+C to cancel the in-flight agent request.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT)
	defer signal.Stop(sigCh)

	go func() {
		<-sigCh
		cancel()
	}()

	fmt.Println() // blank line before response

	err := s.agent.Run(ctx, msg, func(token string) {
		fmt.Print(token)
	})

	fmt.Println() // newline after response

	if err != nil && err != context.Canceled {
		fmt.Fprintf(os.Stderr, "\nagent error: %v\n", err)
	}
}

// runMeta handles /commands. Returns true if the shell should exit.
func (s *Shell) runMeta(cmd string) (exit bool) {
	parts := strings.Fields(cmd)
	if len(parts) == 0 {
		return false
	}

	switch parts[0] {
	case "help", "h", "?":
		fmt.Println()
		fmt.Println("  <cmd>         run a shell command — ls, vim, htop, ssh, etc.")
		fmt.Println("  ?<text>       ask the AI")
		fmt.Println()
		fmt.Println("  /help         show this help")
		fmt.Println("  /clear        clear AI conversation history")
		fmt.Println("  /model        show current model")
		fmt.Println("  /history      show number of messages in context")
		fmt.Println("  /exit         exit the shell")
		fmt.Println()

	case "clear":
		s.agent.ClearHistory()
		fmt.Println("conversation history cleared")

	case "model":
		fmt.Printf("model: %s  endpoint: %s\n", s.cfg.Model, s.cfg.Endpoint)

	case "history":
		fmt.Printf("messages in context: %d\n", s.agent.HistoryLen())

	case "exit", "quit", "q":
		return true

	default:
		fmt.Fprintf(os.Stderr, "unknown command: /%s  (try /help)\n", parts[0])
	}
	return false
}

// parseCd detects `cd [dir]` commands and returns the target directory.
// Returns ("", false) if the input is not a cd command.
func parseCd(cmd string) (string, bool) {
	cmd = strings.TrimSpace(cmd)
	if cmd == "cd" {
		return "", true
	}
	if strings.HasPrefix(cmd, "cd ") || strings.HasPrefix(cmd, "cd\t") {
		return strings.TrimSpace(cmd[3:]), true
	}
	return "", false
}

// buildPrompt builds a coloured shell prompt showing the current directory.
func buildPrompt() string {
	cwd, err := os.Getwd()
	if err != nil {
		cwd = "?"
	}
	if home := os.Getenv("HOME"); home != "" {
		cwd = strings.Replace(cwd, home, "~", 1)
	}
	// green cwd, reset, space, dollar, space
	return fmt.Sprintf("\033[32m%s\033[0m $ ", cwd)
}
