package shell

import (
	"bytes"
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
	"github.com/pvjammer/ai-shell-poc/functions"
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
	cfg      Config
	rl       *readline.Instance
	agent    *agent.Loop
	fnLoader *functions.Loader

	// Shell state tracked for agent context.
	lastCmd      string
	lastExitCode int
	lastStderr   string
}

// New creates and wires up the shell.
func New(cfg Config) (*Shell, error) {
	provider, err := llm.NewOllamaProvider(cfg.Endpoint, cfg.Model)
	if err != nil {
		return nil, fmt.Errorf("create llm provider: %w", err)
	}

	// Load builtin AIFunctions.
	fnLoader := functions.New(functions.ShellConfig{
		LLMEndpoint: cfg.Endpoint,
		LLMModel:    cfg.Model,
	})

	// Combine bash tool + all function tool defs.
	allTools := append(tools.AllTools(), fnLoader.ToolDefs()...)
	allHandlers := tools.AllHandlers()
	for k, v := range fnLoader.ToolHandlers() {
		allHandlers[k] = v
	}

	agentLoop := agent.New(provider, allTools, allHandlers)

	// Display tool calls inline.
	agentLoop.OnToolCall = func(name string, args map[string]interface{}) {
		if cmd, ok := args["command"]; ok {
			fmt.Fprintf(os.Stderr, "\033[2m  [%s] $ %v\033[0m\n", name, cmd)
		} else {
			fmt.Fprintf(os.Stderr, "\033[2m  [%s] %v\033[0m\n", name, args)
		}
	}

	// Build tab completer: slash commands (in-process) + bash delegation.
	completer := NewHybridCompleter(append(metaCommands(), fnLoader.Names()...))

	rl, err := readline.NewEx(&readline.Config{
		Prompt:            buildPrompt(),
		HistoryFile:       filepath.Join(os.Getenv("HOME"), ".ai_shell_history"),
		InterruptPrompt:   "^C",
		EOFPrompt:         "exit",
		HistorySearchFold: true,
		AutoComplete:      completer,
	})
	if err != nil {
		return nil, fmt.Errorf("create readline: %w", err)
	}

	return &Shell{
		cfg:      cfg,
		rl:       rl,
		agent:    agentLoop,
		fnLoader: fnLoader,
	}, nil
}

// Run starts the interactive loop and blocks until the user exits.
func (s *Shell) Run() error {
	defer s.rl.Close()

	fmt.Printf("ai-shell  model=%s  endpoint=%s\n", s.cfg.Model, s.cfg.Endpoint)
	fmt.Println("  <cmd>       shell command (ls, vim, git, ...)")
	fmt.Println("  ?<msg>      ask the AI")
	fmt.Println("  /<fn> ...   run an AI function  (try /tools)")
	fmt.Println("  /help       show all commands")
	fmt.Println()

	for {
		s.rl.SetPrompt(buildPrompt())

		line, err := s.rl.Readline()
		if err != nil {
			fmt.Println("exit")
			break
		}

		in := Parse(line)
		if in.Content == "" && in.Type != InputPipeline {
			continue
		}

		if os.Getenv("BAISH_DEBUG") != "" {
			fmt.Fprintf(os.Stderr, "[debug] parsed: type=%d content=%q pipeLeft=%q\n",
				in.Type, in.Content, in.PipeLeft)
		}

		// Update agent with current shell context before every turn.
		s.syncAgentContext()

		switch in.Type {
		case InputDirect:
			s.runDirect(in.Content)
		case InputAgent:
			s.runAgent(in.Content)
		case InputMeta:
			if s.runMeta(in.Content) {
				return nil
			}
		case InputPipeline:
			s.runPipeline(in.PipeLeft, in.PipeRight)
		}
	}
	return nil
}

// runDirect executes a command with the real terminal attached so interactive
// programs (vim, htop, less, ssh, etc.) work correctly.
// cd is handled as a builtin — subprocess cd would be invisible to us.
func (s *Shell) runDirect(cmdStr string) {
	if dir, ok := parseCd(cmdStr); ok {
		if dir == "" {
			dir = os.Getenv("HOME")
		}
		if err := os.Chdir(dir); err != nil {
			fmt.Fprintf(os.Stderr, "cd: %v\n", err)
			s.lastExitCode = 1
		} else {
			s.lastExitCode = 0
		}
		s.lastCmd = cmdStr
		s.lastStderr = ""
		return
	}

	s.lastCmd = cmdStr
	s.lastStderr = ""

	// Capture stderr for agent context (helps with "?why did that fail").
	var stderrBuf bytes.Buffer
	c := exec.Command("sh", "-c", cmdStr)
	c.Stdin = os.Stdin
	c.Stdout = os.Stdout

	// Tee stderr: terminal sees it AND we capture it for ?why style queries.
	c.Stderr = &teeWriter{w1: os.Stderr, w2: &stderrBuf}

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	defer signal.Stop(sigCh)

	if err := c.Start(); err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		s.lastExitCode = 1
		return
	}

	done := make(chan error, 1)
	go func() { done <- c.Wait() }()

	select {
	case sig := <-sigCh:
		_ = c.Process.Signal(sig)
		<-done
		s.lastExitCode = 130
	case err := <-done:
		if err != nil {
			if exit, ok := err.(*exec.ExitError); ok {
				s.lastExitCode = exit.ExitCode()
			} else {
				fmt.Fprintf(os.Stderr, "error: %v\n", err)
				s.lastExitCode = 1
			}
		} else {
			s.lastExitCode = 0
		}
	}

	s.lastStderr = stderrBuf.String()
}

// runAgent sends the message to the AI and streams its response.
func (s *Shell) runAgent(msg string) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT)
	defer signal.Stop(sigCh)
	go func() { <-sigCh; cancel() }()

	fmt.Println()

	err := s.agent.Run(ctx, msg, func(token string) {
		fmt.Print(token)
	})
	fmt.Println()

	if err != nil && err != context.Canceled {
		fmt.Fprintf(os.Stderr, "agent error: %v\n", err)
	}
}

// runMeta handles /commands. Returns true if the shell should exit.
func (s *Shell) runMeta(cmd string) (exit bool) {
	parts := splitWords(cmd)
	if len(parts) == 0 {
		return false
	}

	name := parts[0]
	args := parts[1:]

	// Check if it's a registered function name.
	for _, fn := range s.fnLoader.Names() {
		if fn == name {
			return s.runFunction(name, args)
		}
	}

	switch name {
	case "help", "h", "?":
		s.printHelp()

	case "tools":
		s.printTools()

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
		fmt.Fprintf(os.Stderr, "unknown command: /%s  (try /help)\n", name)
	}
	return false
}

// runFunction executes a registered AIFunction and prints its output.
// Returns true if the shell should exit (it never should from a function).
func (s *Shell) runFunction(name string, args []string) bool {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT)
	defer signal.Stop(sigCh)
	go func() { <-sigCh; cancel() }()

	result, err := s.fnLoader.Execute(ctx, name, args, "")
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: %v\n", name, err)
		return false
	}
	fmt.Println(result)
	return false
}

func (s *Shell) printHelp() {
	fmt.Println()
	fmt.Println("Shell commands (anything runs as bash):")
	fmt.Println("  ls, vim, git status, cd .., ...   — normal shell")
	fmt.Println()
	fmt.Println("AI:")
	fmt.Println("  ?<text>      ask the AI (agent can run commands for you)")
	fmt.Println()
	fmt.Println("AI functions (slash commands):")
	for _, name := range s.fnLoader.Names() {
		fmt.Println(formatEntry("/"+name, s.fnLoader.Describe(name)))
	}
	fmt.Println()
	fmt.Println("Built-in commands:")
	fmt.Println("  /tools       list all tools available to the AI agent")
	fmt.Println("  /clear       clear conversation history")
	fmt.Println("  /model       show current model")
	fmt.Println("  /history     show number of messages in context")
	fmt.Println("  /exit        exit the shell")
	fmt.Println()
}

func (s *Shell) printTools() {
	fmt.Println()
	fmt.Println("Tools available to the AI agent:")
	fmt.Println()
	for _, td := range append(tools.AllTools(), s.fnLoader.ToolDefs()...) {
		fmt.Println(formatEntry(td.Name, td.Description))
	}
	fmt.Println()
}

// runPipeline executes the left bash command, captures its stdout, and passes
// it as stdin to the right-side AI function or agent query.
func (s *Shell) runPipeline(leftCmd string, right *ParsedInput) {
	debug := os.Getenv("BAISH_DEBUG") != ""
	if debug {
		fmt.Fprintf(os.Stderr, "[debug] runPipeline: left=%q right.Type=%d right.Content=%q\n",
			leftCmd, right.Type, right.Content)
	}

	if right == nil {
		return
	}

	// Run left side, capture stdout; stderr goes to terminal as normal.
	cmd := exec.Command("sh", "-c", leftCmd)
	var outBuf bytes.Buffer
	cmd.Stdout = &outBuf
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		if _, isExit := err.(*exec.ExitError); !isExit {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			return
		}
		// ExitError but no output — command failed and produced nothing useful.
		if outBuf.Len() == 0 {
			fmt.Fprintf(os.Stderr, "pipeline: left side produced no output\n")
			return
		}
	}
	captured := outBuf.String()
	if debug {
		fmt.Fprintf(os.Stderr, "[debug] captured %d bytes from left side\n", len(captured))
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT)
	defer signal.Stop(sigCh)
	go func() { <-sigCh; cancel() }()

	switch right.Type {
	case InputMeta:
		parts := splitWords(right.Content)
		if len(parts) == 0 {
			return
		}
		// ExecuteWithStdin bypasses cobra and injects captured content directly
		// into the function's primary text field.
		result, err := s.fnLoader.ExecuteWithStdin(ctx, parts[0], captured, parts[1:])
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", parts[0], err)
			return
		}
		fmt.Println(result)

	case InputAgent:
		// Prepend captured output to the agent message so it has full context.
		msg := strings.TrimSpace(captured)
		if q := strings.TrimSpace(right.Content); q != "" {
			msg = msg + "\n\n" + q
		}
		s.syncAgentContext()
		s.runAgent(msg)
	}
}

// syncAgentContext pushes current shell state into the agent loop.
func (s *Shell) syncAgentContext() {
	cwd, _ := os.Getwd()
	s.agent.SetShellContext(agent.ShellContext{
		CWD:          cwd,
		LastCommand:  s.lastCmd,
		LastExitCode: s.lastExitCode,
		LastStderr:   s.lastStderr,
	})
}

// ── Helpers ───────────────────────────────────────────────────────────────────

// teeWriter writes to two io.Writers simultaneously (terminal + capture buffer).
type teeWriter struct {
	w1 *os.File
	w2 *bytes.Buffer
}

func (t *teeWriter) Write(p []byte) (int, error) {
	n, err := t.w1.Write(p)
	if err == nil {
		t.w2.Write(p)
	}
	return n, err
}

func buildPrompt() string {
	cwd, err := os.Getwd()
	if err != nil {
		cwd = "?"
	}
	if home := os.Getenv("HOME"); home != "" {
		cwd = strings.Replace(cwd, home, "~", 1)
	}
	return fmt.Sprintf("\033[32m%s\033[0m $ ", cwd)
}

func parseCd(cmd string) (string, bool) {
	cmd = strings.TrimSpace(cmd)
	if cmd == "cd" {
		return "", true
	}
	if after, ok := strings.CutPrefix(cmd, "cd "); ok {
		return strings.TrimSpace(after), true
	}
	if after, ok := strings.CutPrefix(cmd, "cd\t"); ok {
		return strings.TrimSpace(after), true
	}
	return "", false
}

// formatEntry formats a name+description pair so that description continuation
// lines are always aligned with where the first description line starts —
// matching the style of standard CLI help text.
//
//	  bash                   Execute a shell command and return output. Use for
//	                         file ops, searching, running scripts, etc.
func formatEntry(name, desc string) string {
	const (
		leftPad  = 2
		nameWidth = 22
		colWidth  = leftPad + nameWidth + 2 // total chars before description
	)

	// Detect terminal width; fall back to 80.
	termWidth := 80
	if n, err := fmt.Sscanf(os.Getenv("COLUMNS"), "%d", &termWidth); n == 0 || err != nil {
		termWidth = 80
	}

	descWidth := termWidth - colWidth
	if descWidth < 20 {
		descWidth = 40
	}

	nameField := fmt.Sprintf("%*s%-*s  ", leftPad, "", nameWidth, name)
	contIndent := strings.Repeat(" ", colWidth)

	lines := wordWrap(desc, descWidth)
	if len(lines) == 0 {
		return nameField
	}

	var sb strings.Builder
	sb.WriteString(nameField)
	sb.WriteString(lines[0])
	for _, l := range lines[1:] {
		sb.WriteByte('\n')
		sb.WriteString(contIndent)
		sb.WriteString(l)
	}
	return sb.String()
}

// wordWrap splits text into lines of at most width characters, breaking on spaces.
func wordWrap(text string, width int) []string {
	words := strings.Fields(text)
	if len(words) == 0 {
		return nil
	}

	var lines []string
	var cur strings.Builder

	for _, w := range words {
		switch {
		case cur.Len() == 0:
			cur.WriteString(w)
		case cur.Len()+1+len(w) <= width:
			cur.WriteByte(' ')
			cur.WriteString(w)
		default:
			lines = append(lines, cur.String())
			cur.Reset()
			cur.WriteString(w)
		}
	}
	if cur.Len() > 0 {
		lines = append(lines, cur.String())
	}
	return lines
}

// splitWords splits s into tokens respecting single and double quotes.
// Quoted spans may contain spaces; the enclosing quotes are stripped.
// Unmatched quotes are treated as literal characters.
func splitWords(s string) []string {
	var tokens []string
	var cur strings.Builder
	inSingle := false
	inDouble := false

	for i := 0; i < len(s); i++ {
		ch := s[i]
		switch {
		case ch == '\'' && !inDouble:
			inSingle = !inSingle
		case ch == '"' && !inSingle:
			inDouble = !inDouble
		case (ch == ' ' || ch == '\t') && !inSingle && !inDouble:
			if cur.Len() > 0 {
				tokens = append(tokens, cur.String())
				cur.Reset()
			}
		default:
			cur.WriteByte(ch)
		}
	}
	if cur.Len() > 0 {
		tokens = append(tokens, cur.String())
	}
	return tokens
}

// metaCommands returns the list of built-in /commands for autocomplete.
func metaCommands() []string {
	return []string{"help", "tools", "clear", "model", "history", "exit"}
}

