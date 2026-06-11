package shell

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"

	"github.com/chzyer/readline"
	"github.com/pvjammer/ai-shell-poc/agent"
	"github.com/pvjammer/ai-shell-poc/config"
	"github.com/pvjammer/ai-shell-poc/functions"
	"github.com/pvjammer/ai-shell-poc/permissions"
	"github.com/pvjammer/ai-shell-poc/tools"
	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

// Config holds runtime configuration for the shell (LLM connection).
type Config struct {
	Model    string
	Endpoint string
}

// Shell is the main REPL.
type Shell struct {
	cfg      Config
	appCfg   config.Config
	rl       *readline.Instance
	agent    *agent.Loop
	fnLoader *functions.Loader

	ctxSlots     map[string]string
	lastCmd      string
	lastExitCode int
	lastStderr   string
}

// New creates and wires up the shell.
func New(cfg Config, appCfg config.Config) (*Shell, error) {
	provider, err := llm.NewOllamaProvider(cfg.Endpoint, cfg.Model)
	if err != nil {
		return nil, fmt.Errorf("create llm provider: %w", err)
	}

	fnLoader := functions.New(functions.ShellConfig{
		LLMEndpoint: cfg.Endpoint,
		LLMModel:    cfg.Model,
	})

	allTools := append(tools.AllTools(), fnLoader.ToolDefs()...)
	allHandlers := tools.AllHandlers()
	for k, v := range fnLoader.ToolHandlers() {
		allHandlers[k] = v
	}

	agentLoop := agent.New(provider, allTools, allHandlers)
	agentLoop.SetConfig(agent.LoopConfig{
		MaxHistoryMessages: appCfg.MaxHistoryMessages,
		ToolOutputMaxChars: appCfg.ToolOutputMaxChars,
		ToolOverflow:       string(appCfg.ToolOverflow),
	})

	agentLoop.OnToolCall = func(name string, args map[string]interface{}) {
		if cmd, ok := args["command"]; ok {
			fmt.Fprintf(os.Stderr, "\033[2m  [%s] $ %v\033[0m\n", name, cmd)
		} else {
			fmt.Fprintf(os.Stderr, "\033[2m  [%s] %v\033[0m\n", name, args)
		}
	}

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

	ctxSlots, err := config.LoadContexts()
	if err != nil {
		ctxSlots = make(map[string]string)
	}

	return &Shell{
		cfg:      cfg,
		appCfg:   appCfg,
		rl:       rl,
		agent:    agentLoop,
		fnLoader: fnLoader,
		ctxSlots: ctxSlots,
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

// runDirect executes a command with the real terminal attached.
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

	var stderrBuf bytes.Buffer
	c := exec.Command("sh", "-c", cmdStr)
	c.Stdin = os.Stdin
	c.Stdout = os.Stdout
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

	case "ctx":
		s.runCtx(args, "")

	case "config":
		s.runConfig(args)

	case "clear":
		s.agent.ClearHistory()
		fmt.Println("conversation history cleared")

	case "model":
		fmt.Printf("model: %s  endpoint: %s\n", s.cfg.Model, s.cfg.Endpoint)

	case "history":
		fmt.Printf("messages in context: %d\n", s.agent.HistoryLen())

	case "permissions", "perm":
		s.runPermissions(args)

	case "exit", "quit", "q":
		return true

	default:
		fmt.Fprintf(os.Stderr, "unknown command: /%s  (try /help)\n", name)
	}
	return false
}

// runFunction executes a registered AIFunction and prints its output.
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

// runCtx handles /ctx subcommands and piped content.
func (s *Shell) runCtx(args []string, piped string) {
	if len(args) == 0 && strings.TrimSpace(piped) == "" {
		fmt.Println("usage:")
		fmt.Println("  cat file | /ctx add <name>   store piped content as a named slot")
		fmt.Println("  /ctx show <name>              print slot content")
		fmt.Println("  /ctx list                     list all slots with sizes")
		fmt.Println("  /ctx clear [name]             remove one slot or all")
		return
	}

	sub := ""
	if len(args) > 0 {
		sub = args[0]
	}
	slotName := "default"
	if len(args) > 1 {
		slotName = args[1]
	}

	switch sub {
	case "add":
		if strings.TrimSpace(piped) == "" {
			fmt.Fprintln(os.Stderr, "ctx: pipe content into /ctx add — e.g. cat file.md | /ctx add design")
			return
		}
		s.ctxSlots[slotName] = strings.TrimSpace(piped)
		size := len(s.ctxSlots[slotName])

		// Named slots persist across sessions; "default" is session-only.
		if slotName != "default" {
			if err := config.SaveContext(slotName, s.ctxSlots[slotName]); err != nil {
				fmt.Fprintf(os.Stderr, "ctx: warning: could not persist slot: %v\n", err)
			}
		}

		cap := s.appCfg.CtxMaxInjectChars
		if cap > 0 && size > cap {
			fmt.Printf("ctx: stored %q (%d chars) — only first %d chars will be injected per turn; use /ctx show %s for full content\n",
				slotName, size, cap, slotName)
		} else {
			fmt.Printf("ctx: stored %q (%d chars)\n", slotName, size)
		}

	case "show":
		if v, ok := s.ctxSlots[slotName]; ok {
			fmt.Println(v)
		} else {
			fmt.Fprintf(os.Stderr, "ctx: no slot %q  (try /ctx list)\n", slotName)
		}

	case "list":
		if len(s.ctxSlots) == 0 {
			fmt.Println("(no context slots)")
			return
		}
		for k, v := range s.ctxSlots {
			fmt.Printf("  %-20s %d chars\n", k, len(v))
		}

	case "clear":
		if len(args) > 1 {
			delete(s.ctxSlots, slotName)
			if slotName != "default" {
				if err := config.DeleteContext(slotName); err != nil {
					fmt.Fprintf(os.Stderr, "ctx: warning: could not remove persisted slot: %v\n", err)
				}
			}
			fmt.Printf("ctx: cleared %q\n", slotName)
		} else {
			s.ctxSlots = make(map[string]string)
			if err := config.ClearContexts(); err != nil {
				fmt.Fprintf(os.Stderr, "ctx: warning: could not clear persisted slots: %v\n", err)
			}
			fmt.Println("ctx: all slots cleared")
		}

	default:
		fmt.Fprintf(os.Stderr, "ctx: unknown subcommand %q  (try /ctx for usage)\n", sub)
	}
}

// runConfig handles /config subcommands.
func (s *Shell) runConfig(args []string) {
	if len(args) == 0 {
		fmt.Println()
		fmt.Printf("  %-30s %v\n", "max_history_messages", s.appCfg.MaxHistoryMessages)
		fmt.Printf("  %-30s %v\n", "tool_output_max_chars", s.appCfg.ToolOutputMaxChars)
		fmt.Printf("  %-30s %v\n", "tool_output_overflow", s.appCfg.ToolOverflow)
		fmt.Printf("  %-30s %v\n", "ctx_max_inject_chars", s.appCfg.CtxMaxInjectChars)
		fmt.Println()
		fmt.Println("  /config set <key> <value>   change a setting")
		fmt.Println("  /config reset               restore defaults")
		fmt.Println()
		return
	}

	switch args[0] {
	case "set":
		if len(args) < 3 {
			fmt.Fprintln(os.Stderr, "usage: /config set <key> <value>")
			return
		}
		key, val := args[1], args[2]
		switch key {
		case "max_history_messages":
			n, err := strconv.Atoi(val)
			if err != nil || n < 2 {
				fmt.Fprintln(os.Stderr, "config: max_history_messages must be an integer >= 2")
				return
			}
			s.appCfg.MaxHistoryMessages = n
		case "tool_output_max_chars":
			n, err := strconv.Atoi(val)
			if err != nil || n < 100 {
				fmt.Fprintln(os.Stderr, "config: tool_output_max_chars must be an integer >= 100")
				return
			}
			s.appCfg.ToolOutputMaxChars = n
		case "tool_output_overflow":
			if val != "truncate" && val != "summarize" {
				fmt.Fprintln(os.Stderr, "config: tool_output_overflow must be 'truncate' or 'summarize'")
				return
			}
			s.appCfg.ToolOverflow = config.ToolOverflow(val)
		case "ctx_max_inject_chars":
			n, err := strconv.Atoi(val)
			if err != nil || n < 500 {
				fmt.Fprintln(os.Stderr, "config: ctx_max_inject_chars must be an integer >= 500")
				return
			}
			s.appCfg.CtxMaxInjectChars = n
		default:
			fmt.Fprintf(os.Stderr, "config: unknown key %q\n", key)
			return
		}
		if err := config.Save(s.appCfg); err != nil {
			fmt.Fprintf(os.Stderr, "config: failed to save: %v\n", err)
		} else {
			fmt.Printf("config: %s = %s\n", key, val)
		}

	case "reset":
		s.appCfg = config.Defaults()
		if err := config.Save(s.appCfg); err != nil {
			fmt.Fprintf(os.Stderr, "config: failed to save: %v\n", err)
		} else {
			fmt.Println("config: reset to defaults")
		}

	default:
		fmt.Fprintf(os.Stderr, "config: unknown subcommand %q\n", args[0])
	}
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
	fmt.Println("  /tools             list tools available to the AI agent")
	fmt.Println("  /ctx add <name>    pipe content into a named context slot")
	fmt.Println("  /ctx show <name>   print a context slot")
	fmt.Println("  /ctx list          list all context slots")
	fmt.Println("  /ctx clear [name]  remove one slot or all")
	fmt.Println("  /config            show or change settings")
	fmt.Println("  /permissions [cmd] show permission tier for a command")
	fmt.Println("  /clear             clear conversation history")
	fmt.Println("  /model             show current model")
	fmt.Println("  /history           show number of messages in context")
	fmt.Println("  /exit              exit the shell")
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
// it through the right-side chain (which may itself be a pipeline of AI functions).
func (s *Shell) runPipeline(leftCmd string, right *ParsedInput) {
	debug := os.Getenv("BAISH_DEBUG") != ""
	if debug {
		fmt.Fprintf(os.Stderr, "[debug] runPipeline: left=%q right.Type=%d right.Content=%q\n",
			leftCmd, right.Type, right.Content)
	}

	if right == nil {
		return
	}

	cmd := exec.Command("sh", "-c", leftCmd)
	var outBuf bytes.Buffer
	cmd.Stdout = &outBuf
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		if _, isExit := err.(*exec.ExitError); !isExit {
			fmt.Fprintf(os.Stderr, "error: %v\n", err)
			return
		}
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

	s.applyRight(ctx, captured, right)
}

// applyRight applies the right side of a pipeline to already-captured content.
// right may be InputMeta (terminal function), InputPipeline (chained function),
// or InputAgent. This enables chains like: cat f | /summarize | /ctx add
func (s *Shell) applyRight(ctx context.Context, content string, right *ParsedInput) {
	if right == nil {
		return
	}

	switch right.Type {
	case InputMeta:
		parts := splitWords(right.Content)
		if len(parts) == 0 {
			return
		}
		if parts[0] == "ctx" {
			s.runCtx(parts[1:], content)
			return
		}
		result, err := s.fnLoader.ExecuteWithStdin(ctx, parts[0], content, parts[1:])
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", parts[0], err)
			return
		}
		fmt.Println(result)

	case InputPipeline:
		// PipeLeft is a /function to run with content; its output feeds PipeRight.
		metaParsed := Parse(right.PipeLeft)
		if metaParsed.Type != InputMeta {
			fmt.Fprintf(os.Stderr, "pipeline: expected /function, got %q\n", right.PipeLeft)
			return
		}
		parts := splitWords(metaParsed.Content)
		if len(parts) == 0 {
			return
		}
		intermediate, err := s.fnLoader.ExecuteWithStdin(ctx, parts[0], content, parts[1:])
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", parts[0], err)
			return
		}
		s.applyRight(ctx, intermediate, right.PipeRight)

	case InputAgent:
		msg := strings.TrimSpace(content)
		if q := strings.TrimSpace(right.Content); q != "" {
			msg = msg + "\n\n" + q
		}
		s.syncAgentContext()
		s.runAgent(msg)
	}
}

// runPermissions handles /permissions — shows the tier for one or more commands.
func (s *Shell) runPermissions(args []string) {
	if len(args) == 0 {
		fmt.Println()
		fmt.Println("  /permissions <cmd>   show the permission tier for a command")
		fmt.Println()
		fmt.Println("  Tiers:")
		fmt.Println("    auto     — runs without prompting")
		fmt.Println("    confirm  — agent must get your approval before running")
		fmt.Println("    deny     — always blocked")
		fmt.Println()
		fmt.Println("  Examples:")
		fmt.Println("    /permissions ls -la")
		fmt.Println("    /permissions rm -rf ./dist")
		fmt.Println("    /permissions git commit -m 'fix'")
		fmt.Println()
		return
	}
	cmd := strings.Join(args, " ")
	tier := permissions.Classify(cmd)
	fmt.Printf("  %-12s %s\n", tier, cmd)
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
	s.agent.SetContextSlots(s.ctxSlots)
	s.agent.SetConfig(agent.LoopConfig{
		MaxHistoryMessages: s.appCfg.MaxHistoryMessages,
		ToolOutputMaxChars: s.appCfg.ToolOutputMaxChars,
		ToolOverflow:       string(s.appCfg.ToolOverflow),
		CtxMaxInjectChars:  s.appCfg.CtxMaxInjectChars,
	})
}

// ── Helpers ───────────────────────────────────────────────────────────────────

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

// splitWords splits s into tokens respecting single and double quotes.
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

// formatEntry formats a name+description pair for help output.
func formatEntry(name, desc string) string {
	const (
		leftPad   = 2
		nameWidth = 22
		colWidth  = leftPad + nameWidth + 2
	)

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

func metaCommands() []string {
	return []string{"help", "tools", "clear", "ctx", "config", "permissions", "model", "history", "exit"}
}
