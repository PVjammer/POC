package shell

import (
	"bufio"
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
	"time"

	"github.com/ergochat/readline"
	"github.com/pvjammer/ai-shell-poc/agent"
	"github.com/pvjammer/ai-shell-poc/config"
	"github.com/pvjammer/ai-shell-poc/functions"
	"github.com/pvjammer/ai-shell-poc/permissions"
	"github.com/pvjammer/ai-shell-poc/tools"
	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

// shellCtxSlot is the shell-side representation of a named context slot.
type shellCtxSlot struct {
	content string
	desc    string // empty = auto-generate stub description; non-empty = user-provided
}

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
	fnLoader *functions.Loader
	jobs     *jobManager

	// actTools/actHandlers hold the shared tool definitions and stateless
	// handlers. Session-specific handlers (read_context, describe_tool) are
	// created per session in makeSessionHandlers.
	actTools    []llm.ToolDef
	actHandlers map[string]func(map[string]interface{}) (string, error)

	// sessions holds all named conversation sessions. "main" always exists.
	sessions      map[string]*sessionEntry
	activeSession string

	lastCmd      string
	lastExitCode int
	lastStderr   string
}

// New creates and wires up the shell.
func New(cfg Config, appCfg config.Config) (*Shell, error) {
	fnLoader := functions.New(functions.ShellConfig{
		LLMEndpoint: cfg.Endpoint,
		LLMModel:    cfg.Model,
	})

	// Build shared tool definitions. read_context/describe_tool defs are
	// included here; their handlers are session-specific (see makeSessionHandlers).
	allTools := append(tools.AllTools(), fnLoader.ToolDefs()...)
	allTools = append(allTools, tools.ReadContextToolDef(), tools.DescribeToolToolDef())

	// Base handlers — stateless tools only. Session-specific handlers are
	// created per session in makeSessionHandlers.
	baseHandlers := tools.AllHandlers()
	for k, v := range fnLoader.ToolHandlers() {
		baseHandlers[k] = v
	}

	completer := NewHybridCompleter(append(metaCommands(), fnLoader.Names()...))

	rl, err := readline.NewEx(&readline.Config{
		Prompt:            buildPrompt(nil, appCfg.Prompt, ""),
		HistoryFile:       filepath.Join(os.Getenv("HOME"), ".ai_shell_history"),
		InterruptPrompt:   "^C",
		EOFPrompt:         "exit",
		HistorySearchFold: true,
		AutoComplete:      completer,
	})
	if err != nil {
		return nil, fmt.Errorf("create readline: %w", err)
	}

	rawSlots, err := config.LoadContexts()
	if err != nil {
		rawSlots = make(map[string]string)
	}
	shellSlots := make(map[string]shellCtxSlot, len(rawSlots))
	for name, content := range rawSlots {
		shellSlots[name] = shellCtxSlot{content: content}
	}

	s := &Shell{
		cfg:           cfg,
		appCfg:        appCfg,
		rl:            rl,
		fnLoader:      fnLoader,
		jobs:          newJobManager(),
		actTools:      allTools,
		actHandlers:   baseHandlers,
		sessions:      make(map[string]*sessionEntry),
		activeSession: "main",
	}
	s.jobs.onComplete = s.maybeNotify

	// Create the main session.
	mainSess, err := s.createSession("main", "", nil, shellSlots, 0)
	if err != nil {
		return nil, fmt.Errorf("create main session: %w", err)
	}

	if os.Getenv("BAISH_DEBUG") != "" {
		logPath := config.Path()[:len(config.Path())-len("config.toml")] + "debug.log"
		if f, err := os.OpenFile(logPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644); err == nil {
			mainSess.loop.SetDebugLog(f)
			fmt.Fprintf(os.Stderr, "\033[2m[debug] logging to %s\033[0m\n", logPath)
		}
	}

	return s, nil
}

// Run starts the interactive loop and blocks until the user exits.
func (s *Shell) Run() error {
	defer s.rl.Close()

	fmt.Printf("ai-shell  model=%s  endpoint=%s\n", s.cfg.Model, s.cfg.Endpoint)
	fmt.Println("  <cmd>          shell command  (ls, vim, git, ...)")
	fmt.Println("  ?<msg>         ask the AI     (advisory — explains, no execution)")
	fmt.Println("  !\"<msg>\"       act mode       (AI executes commands; permissions apply)")
	fmt.Println("  /<fn> ...      AI function    (try /tools)")
	fmt.Println("  /help          show all commands")
	fmt.Println()

	for {
		// Drain notifications from completed background jobs.
		for _, msg := range s.jobs.drain() {
			fmt.Println(msg)
		}
		s.rl.SetPrompt(buildPrompt(s.jobs, s.appCfg.Prompt, s.activeSession))

		line, err := s.rl.Readline()
		if err != nil {
			fmt.Println("exit")
			break
		}

		// Detect trailing & for background execution, with optional job name.
		// Supported forms: "cmd &"  or  "cmd & jobname"
		line = strings.TrimSpace(line)
		background, jobName, line := parseBackgroundSuffix(line)

		in := Parse(line)
		if in.Content == "" && in.Type != InputPipeline {
			continue
		}

		if os.Getenv("BAISH_DEBUG") != "" {
			fmt.Fprintf(os.Stderr, "[debug] parsed: type=%d content=%q pipeLeft=%q background=%v\n",
				in.Type, in.Content, in.PipeLeft, background)
		}

		s.syncAgentContext()

		switch in.Type {
		case InputDirect:
			if background {
				s.runDirectBackground(in.Content, jobName)
			} else {
				s.runDirect(in.Content)
			}
		case InputAgent:
			if background {
				s.runAgentBackground(in.Content, jobName)
			} else {
				s.runAgent(in.Content)
			}
		case InputAgentAct:
			if background {
				s.runAgentActBackground(in.Content, jobName)
			} else {
				s.runAgentAct(in.Content)
			}
		case InputMeta:
			if s.runMeta(in.Content) {
				return nil
			}
		case InputPipeline:
			if background {
				s.runPipelineBackground(in.PipeLeft, in.PipeRight, jobName)
			} else {
				s.runPipeline(in.PipeLeft, in.PipeRight)
			}
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

// setupAgentForMode configures tools and system prompt for advisory (?) or
// agentic (!) mode. Must be called with agentMu held.
//
// Advisory mode gets read_context and describe_tool — enough to resolve
// stub context slots without granting execution capabilities.
func (s *Shell) setupAgentForMode(act bool) {
	sess := s.currentSession()
	handlers := s.makeSessionHandlers(sess)
	if act {
		sess.loop.SetTools(s.actTools, handlers)
		sess.loop.SetSystemPrompt("")
	} else {
		advisoryDefs := s.advisoryToolSet()
		sess.loop.SetTools(advisoryDefs, handlers)
		sess.loop.SetSystemPrompt(agent.AdvisorySystemPrompt)
	}
}

// advisoryToolAllowlist is the explicit set of tools permitted in advisory (?)
// mode. Tools must be reviewed before being added here — new tools loaded from
// the function registry are agentic-only by default.
var advisoryToolAllowlist = map[string]bool{
	// Context / introspection (baish built-ins, always safe).
	"read_context": true,
	"describe_tool": true,
	// Filesystem reads (no mutation).
	"read_file":  true,
	"list_files": true,
	// Web (read-only network access).
	"web_search": true,
	// Pure text computation (no I/O).
	"summarize":       true,
	"extract_data":    true,
	"text_transform":  true,
}

// advisoryToolSet returns the subset of actTools permitted in advisory mode.
func (s *Shell) advisoryToolSet() []llm.ToolDef {
	var out []llm.ToolDef
	for _, t := range s.actTools {
		if advisoryToolAllowlist[t.Name] {
			out = append(out, t)
		}
	}
	return out
}

// runAgentForeground is the shared foreground agent runner for both modes.
func (s *Shell) runAgentForeground(msg string, act bool) {
	s.sessionMu().Lock()
	defer s.sessionMu().Unlock()
	s.setupAgentForMode(act)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT)
	defer signal.Stop(sigCh)
	go func() { <-sigCh; cancel() }()

	fmt.Println()
	err := s.activeLoop().Run(ctx, msg, func(token string) { fmt.Print(token) })
	fmt.Println()

	if err != nil && err != context.Canceled {
		fmt.Fprintf(os.Stderr, "agent error: %v\n", err)
	}
}

func (s *Shell) runAgent(msg string)    { s.runAgentForeground(msg, false) }
func (s *Shell) runAgentAct(msg string) { s.runAgentForeground(msg, true) }

// runAgentCapture runs the agent in capture mode (no terminal output).
// Spinner is suppressed — runs in a background goroutine and would corrupt
// the foreground terminal cursor if allowed to write.
func (s *Shell) runAgentCapture(ctx context.Context, msg string, act bool) (string, error) {
	sess := s.currentSession() // snapshot at call time — safe for background jobs
	sess.mu.Lock()
	defer sess.mu.Unlock()
	sess.loop.SetSpinnerEnabled(false)
	defer sess.loop.SetSpinnerEnabled(true)
	// Suppress terminal callbacks — background output is retrieved via /jobs N.
	savedToolCall, savedToolResult := sess.loop.OnToolCall, sess.loop.OnToolResult
	sess.loop.OnToolCall, sess.loop.OnToolResult = nil, nil
	defer func() {
		sess.loop.OnToolCall, sess.loop.OnToolResult = savedToolCall, savedToolResult
	}()
	s.setupAgentForMode(act)
	var buf strings.Builder
	err := sess.loop.Run(ctx, msg, func(token string) { buf.WriteString(token) })
	return strings.TrimSpace(buf.String()), err
}

// runDirectBackground runs a bash command in the background, capturing output.
func (s *Shell) runDirectBackground(cmd, name string) {
	display := truncateDisplay(cmd, 40)
	id, err := s.jobs.start(display, name, func() (string, error) {
		c := exec.Command("sh", "-c", cmd)
		var out bytes.Buffer
		c.Stdout = &out
		c.Stderr = &out
		err := c.Run()
		return strings.TrimSpace(out.String()), err
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "jobs: %v\n", err)
		return
	}
	fmt.Printf("[%d] started: %s\n", id, display)
}

// runAgentBg is the shared background agent runner for both modes.
func (s *Shell) runAgentBg(msg, name string, act bool) {
	prefix := "?"
	if act {
		prefix = "!"
	}
	display := prefix + truncateDisplay(msg, 37)
	s.syncAgentContext()
	id, err := s.jobs.start(display, name, func() (string, error) {
		return s.runAgentCapture(context.Background(), msg, act)
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "jobs: %v\n", err)
		return
	}
	fmt.Printf("[%d] started: %s\n", id, display)
}

func (s *Shell) runAgentBackground(msg, name string)    { s.runAgentBg(msg, name, false) }
func (s *Shell) runAgentActBackground(msg, name string) { s.runAgentBg(msg, name, true) }

// runPipelineBackground runs a pipeline (bash | /fn or bash | ?msg) as a background job.
func (s *Shell) runPipelineBackground(leftCmd string, right *ParsedInput, name string) {
	display := truncateDisplay(leftCmd+" | "+rightDisplay(right), 40)
	s.syncAgentContext()
	id, err := s.jobs.start(display, name, func() (string, error) {
		content, err := s.resolveLeftContent(leftCmd)
		if err != nil {
			return "", err
		}
		if strings.TrimSpace(content) == "" {
			return "", fmt.Errorf("left side produced no output")
		}
		return s.applyRightCapture(context.Background(), content, right)
	})
	if err != nil {
		fmt.Fprintf(os.Stderr, "jobs: %v\n", err)
		return
	}
	fmt.Printf("[%d] started: %s\n", id, display)
}

// resolveLeftContent handles the left side of a pipeline. If leftCmd starts
// with "/" it is treated as a meta command and its output is captured
// programmatically. Otherwise it is run as a bash command.
func (s *Shell) resolveLeftContent(leftCmd string) (string, error) {
	if strings.HasPrefix(leftCmd, "/") {
		return s.runMetaCapture(leftCmd[1:])
	}
	c := exec.Command("sh", "-c", leftCmd)
	var outBuf bytes.Buffer
	c.Stdout = &outBuf
	c.Stderr = os.Stderr
	if err := c.Run(); err != nil {
		if outBuf.Len() == 0 {
			return "", err
		}
		// Non-zero exit but produced output — continue with what we got.
	}
	return outBuf.String(), nil
}

// runMetaCapture executes a meta command and returns its output as a string.
// Only commands that produce text suitable for piping are supported.
func (s *Shell) runMetaCapture(cmd string) (string, error) {
	parts := splitWords(cmd)
	if len(parts) == 0 {
		return "", fmt.Errorf("empty command")
	}
	switch parts[0] {
	case "job", "jobs":
		if len(parts) < 2 {
			return "", fmt.Errorf("/job: usage: /job <N|name>")
		}
		j := s.jobs.resolve(parts[1])
		if j == nil {
			return "", fmt.Errorf("/job: no job %q", parts[1])
		}
		switch j.status {
		case jobRunning:
			return "", fmt.Errorf("/job %q: still running — wait for it to finish first", parts[1])
		case jobFailed:
			return "", fmt.Errorf("/job %q: failed: %v", parts[1], j.err)
		case jobDone:
			return j.output, nil
		}
	case "ctx":
		// /ctx show <name>
		if len(parts) < 3 || parts[1] != "show" {
			return "", fmt.Errorf("/ctx: pipe usage: /ctx show <name>")
		}
		name := parts[2]
		slot, ok := s.currentSession().ctxSlots[name]
		if !ok {
			return "", fmt.Errorf("/ctx: no slot %q  (try /ctx list)", name)
		}
		return slot.content, nil
	}
	return "", fmt.Errorf("/%s: not pipeable — only /job and /ctx show can feed a pipeline", parts[0])
}

// applyRightCapture is the capture-mode version of applyRight — returns output
// as a string instead of printing it. Used by background pipeline jobs.
func (s *Shell) applyRightCapture(ctx context.Context, content string, right *ParsedInput) (string, error) {
	if right == nil {
		return "", nil
	}
	switch right.Type {
	case InputMeta:
		parts := splitWords(right.Content)
		if len(parts) == 0 {
			return "", nil
		}
		if parts[0] == "ctx" {
			s.runCtx(parts[1:], content)
			return fmt.Sprintf("stored in ctx (%d chars)", len(strings.TrimSpace(content))), nil
		}
		return s.fnLoader.ExecuteWithStdin(ctx, parts[0], content, parts[1:])

	case InputPipeline:
		metaParsed := Parse(right.PipeLeft)
		if metaParsed.Type != InputMeta {
			return "", fmt.Errorf("expected /function, got %q", right.PipeLeft)
		}
		parts := splitWords(metaParsed.Content)
		if len(parts) == 0 {
			return "", nil
		}
		intermediate, err := s.fnLoader.ExecuteWithStdin(ctx, parts[0], content, parts[1:])
		if err != nil {
			return "", err
		}
		return s.applyRightCapture(ctx, intermediate, right.PipeRight)

	case InputAgent:
		msg := strings.TrimSpace(content)
		if q := strings.TrimSpace(right.Content); q != "" {
			msg = msg + "\n\n" + q
		}
		return s.runAgentCapture(ctx, msg, false)
	case InputAgentAct:
		msg := strings.TrimSpace(content)
		if q := strings.TrimSpace(right.Content); q != "" {
			msg = msg + "\n\n" + q
		}
		return s.runAgentCapture(ctx, msg, true)
	case InputBash:
		cmd := exec.CommandContext(ctx, "sh", "-c", right.Content)
		cmd.Stdin = strings.NewReader(content)
		var outBuf bytes.Buffer
		cmd.Stdout = &outBuf
		cmd.Stderr = os.Stderr
		cmd.Run() //nolint:errcheck — non-zero exits (e.g. grep no match) are normal
		return outBuf.String(), nil
	}
	return "", nil
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

	// Global --help / -h: route to each command's usage display.
	if len(args) > 0 && (args[0] == "--help" || args[0] == "-h") {
		switch name {
		case "ctx":
			s.runCtx(nil, "")
			return false
		case "config":
			s.runConfig([]string{"--help"})
			return false
		case "jobs", "job":
			s.runJobs([]string{"--help"})
			return false
		case "session", "s":
			s.runSession([]string{"--help"})
			return false
		case "permissions", "perm":
			s.runPermissions(nil)
			return false
		case "model":
			fmt.Printf("usage: /model <name> [endpoint]\nmodel: %s  endpoint: %s\n", s.cfg.Model, s.cfg.Endpoint)
			return false
		case "commit-msg", "commit", "cm":
			fmt.Println("usage: /commit-msg")
			fmt.Println("  Generates a commit message from staged git changes.")
			fmt.Println("  Stage changes with 'git add' before running.")
			return false
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
		s.activeLoop().ClearHistory()
		fmt.Println("conversation history cleared")

	case "model":
		if len(args) == 0 {
			fmt.Printf("model: %s  endpoint: %s\n", s.cfg.Model, s.cfg.Endpoint)
		} else {
			model := args[0]
			endpoint := s.cfg.Endpoint
			if len(args) > 1 {
				endpoint = args[1]
			}
			if err := s.setModel(model, endpoint); err != nil {
				fmt.Fprintf(os.Stderr, "model: %v\n", err)
			} else {
				fmt.Printf("model: switched to %s  endpoint: %s\n", model, endpoint)
			}
		}

	case "history":
		fmt.Printf("messages in context: %d\n", s.activeLoop().HistoryLen())

	case "jobs", "job":
		s.runJobs(args)

	case "permissions", "perm":
		s.runPermissions(args)

	case "session", "s":
		s.runSession(args)

	case "commit-msg", "commit", "cm":
		s.runCommitMsg()

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
		fmt.Println("  cat file | /ctx set <name>   overwrite a slot with piped content")
		fmt.Println("  cat file | /ctx add <name>   append piped content to a slot")
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
	case "set":
		if strings.TrimSpace(piped) == "" {
			fmt.Fprintln(os.Stderr, "ctx: pipe content into /ctx set — e.g. cat file.md | /ctx set design")
			return
		}
		content := strings.TrimSpace(piped)
		s.currentSession().ctxSlots[slotName] = shellCtxSlot{content: content}

		if slotName != "default" {
			if err := config.SaveContext(slotName, content); err != nil {
				fmt.Fprintf(os.Stderr, "ctx: warning: could not persist slot: %v\n", err)
			}
		}
		s.printCtxFeedback("set", slotName, len(content))

	case "add":
		if strings.TrimSpace(piped) == "" {
			fmt.Fprintln(os.Stderr, "ctx: pipe content into /ctx add — e.g. cat file.md | /ctx add docs")
			return
		}
		incoming := strings.TrimSpace(piped)
		existing := s.currentSession().ctxSlots[slotName]
		var merged string
		if existing.content != "" {
			merged = existing.content + "\n\n" + incoming
		} else {
			merged = incoming
		}
		s.currentSession().ctxSlots[slotName] = shellCtxSlot{content: merged, desc: existing.desc}

		if slotName != "default" {
			if err := config.SaveContext(slotName, merged); err != nil {
				fmt.Fprintf(os.Stderr, "ctx: warning: could not persist slot: %v\n", err)
			}
		}
		s.printCtxFeedback("appended to", slotName, len(merged))

	case "show":
		if slot, ok := s.currentSession().ctxSlots[slotName]; ok {
			fmt.Println(slot.content)
		} else {
			fmt.Fprintf(os.Stderr, "ctx: no slot %q  (try /ctx list)\n", slotName)
		}

	case "list":
		slots := s.currentSession().ctxSlots
		if len(slots) == 0 {
			fmt.Println("(no context slots)")
			return
		}
		threshold := s.appCfg.CtxInlineThreshold
		for k, slot := range slots {
			mode := "inline"
			if len(slot.content) > threshold {
				mode = "stub"
			}
			fmt.Printf("  %-20s  %-8s  %s\n", k, humanSize(len(slot.content)), mode)
		}

	case "clear":
		ctxSlots := s.currentSession().ctxSlots
		if len(args) > 1 {
			delete(ctxSlots, slotName)
			if slotName != "default" {
				if err := config.DeleteContext(slotName); err != nil {
					fmt.Fprintf(os.Stderr, "ctx: warning: could not remove persisted slot: %v\n", err)
				}
			}
			fmt.Printf("ctx: cleared %q\n", slotName)
		} else {
			for k := range ctxSlots {
				delete(ctxSlots, k)
			}
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
		p := s.appCfg.Prompt
		fmt.Println()
		fmt.Printf("  %-30s %v\n", "max_history_messages", s.appCfg.MaxHistoryMessages)
		fmt.Printf("  %-30s %v\n", "tool_output_max_chars", s.appCfg.ToolOutputMaxChars)
		fmt.Printf("  %-30s %v\n", "tool_output_overflow", s.appCfg.ToolOverflow)
		fmt.Printf("  %-30s %v\n", "ctx_inline_threshold", s.appCfg.CtxInlineThreshold)
		fmt.Printf("  %-30s %v\n", "tool_output_keep_rounds", s.appCfg.ToolOutputKeepRounds)
		fmt.Printf("  %-30s %v\n", "max_context_tokens", s.appCfg.MaxContextTokens)
		fmt.Printf("  %-30s %v\n", "compaction_threshold", s.appCfg.CompactionThreshold)
		fmt.Printf("  %-30s %v\n", "compaction_tail_messages", s.appCfg.CompactionTailMessages)
		fmt.Printf("  %-30s %v\n", "max_response_tokens", s.appCfg.MaxResponseTokens)
		fmt.Printf("  %-30s %v\n", "notifications", s.appCfg.Notifications)
		fmt.Println()
		fmt.Printf("  %-30s %v\n", "prompt.path_max_depth", p.PathMaxDepth)
		fmt.Printf("  %-30s %v\n", "prompt.show_git_branch", p.ShowGitBranch)
		fmt.Printf("  %-30s %v\n", "prompt.path_color", p.PathColor)
		fmt.Printf("  %-30s %v\n", "prompt.branch_color", p.BranchColor)
		fmt.Printf("  %-30s %v\n", "prompt.job_color", p.JobColor)
		fmt.Printf("  %-30s %q\n", "prompt.suffix", p.Suffix)
		fmt.Println()
		fmt.Println("  /config set <key> <value>   change a setting")
		fmt.Println("  /config reset               restore defaults")
		fmt.Println("  colors: red green yellow blue magenta cyan white bold dim none")
		fmt.Println()
		return
	}

	switch args[0] {
	case "--help", "-h", "help":
		fmt.Println("usage:")
		fmt.Println("  /config                    show current settings")
		fmt.Println("  /config set <key> <value>  change a setting")
		fmt.Println("  /config reset              restore defaults")
		return
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
		case "ctx_inline_threshold":
			n, err := strconv.Atoi(val)
			if err != nil || n < 256 {
				fmt.Fprintln(os.Stderr, "config: ctx_inline_threshold must be an integer >= 256")
				return
			}
			s.appCfg.CtxInlineThreshold = n
		case "tool_output_keep_rounds":
			n, err := strconv.Atoi(val)
			if err != nil || n < 1 {
				fmt.Fprintln(os.Stderr, "config: tool_output_keep_rounds must be an integer >= 1")
				return
			}
			s.appCfg.ToolOutputKeepRounds = n
		case "max_context_tokens":
			n, err := strconv.Atoi(val)
			if err != nil || n < 1024 {
				fmt.Fprintln(os.Stderr, "config: max_context_tokens must be an integer >= 1024")
				return
			}
			s.appCfg.MaxContextTokens = n
		case "compaction_threshold":
			f, err := strconv.ParseFloat(val, 64)
			if err != nil || f <= 0 || f >= 1 {
				fmt.Fprintln(os.Stderr, "config: compaction_threshold must be a float between 0 and 1 (e.g. 0.75)")
				return
			}
			s.appCfg.CompactionThreshold = f
		case "compaction_tail_messages":
			n, err := strconv.Atoi(val)
			if err != nil || n < 4 {
				fmt.Fprintln(os.Stderr, "config: compaction_tail_messages must be an integer >= 4")
				return
			}
			s.appCfg.CompactionTailMessages = n
		case "max_response_tokens":
			n, err := strconv.Atoi(val)
			if err != nil || n < 256 {
				fmt.Fprintln(os.Stderr, "config: max_response_tokens must be an integer >= 256")
				return
			}
			s.appCfg.MaxResponseTokens = n
		case "notifications":
			switch val {
			case "true", "1", "yes":
				s.appCfg.Notifications = true
			case "false", "0", "no":
				s.appCfg.Notifications = false
			default:
				fmt.Fprintln(os.Stderr, "config: notifications must be true or false")
				return
			}
		case "prompt.path_max_depth":
			n, err := strconv.Atoi(val)
			if err != nil || n < 0 {
				fmt.Fprintln(os.Stderr, "config: prompt.path_max_depth must be a non-negative integer (0 = full path)")
				return
			}
			s.appCfg.Prompt.PathMaxDepth = n
		case "prompt.show_git_branch":
			switch val {
			case "true", "1", "yes":
				s.appCfg.Prompt.ShowGitBranch = true
			case "false", "0", "no":
				s.appCfg.Prompt.ShowGitBranch = false
			default:
				fmt.Fprintln(os.Stderr, "config: prompt.show_git_branch must be true or false")
				return
			}
		case "prompt.path_color":
			if !isValidColor(val) {
				fmt.Fprintln(os.Stderr, "config: valid colors: red green yellow blue magenta cyan white bold dim none")
				return
			}
			s.appCfg.Prompt.PathColor = val
		case "prompt.branch_color":
			if !isValidColor(val) {
				fmt.Fprintln(os.Stderr, "config: valid colors: red green yellow blue magenta cyan white bold dim none")
				return
			}
			s.appCfg.Prompt.BranchColor = val
		case "prompt.job_color":
			if !isValidColor(val) {
				fmt.Fprintln(os.Stderr, "config: valid colors: red green yellow blue magenta cyan white bold dim none")
				return
			}
			s.appCfg.Prompt.JobColor = val
		case "prompt.suffix":
			s.appCfg.Prompt.Suffix = val
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
	fmt.Println("AI (advisory — explains and guides, never executes):")
	fmt.Println("  ?<text>            ask a question or request guidance")
	fmt.Println("  ?<text> &          run in background")
	fmt.Println("  ?<text> & name     run in background, tag with name")
	fmt.Println()
	fmt.Println("AI (agentic — executes commands; permissions tier applies):")
	fmt.Println("  !\"<text>\"          ask the AI to act")
	fmt.Println("  !\"<text>\" &        act in background")
	fmt.Println()
	fmt.Println("Pipelines work with both modes:")
	fmt.Println("  cmd | ?<text>      pipe output to advisory AI")
	fmt.Println("  cmd | !\"<text>\"    pipe output to agentic AI")
	fmt.Println()
	fmt.Println("AI functions (slash commands):")
	for _, name := range s.fnLoader.Names() {
		fmt.Println(formatEntry("/"+name, s.fnLoader.Describe(name)))
	}
	fmt.Println()
	fmt.Println("Built-in commands:")
	fmt.Println("  /tools             list tools available to the AI agent")
	fmt.Println("  /ctx set <name>    overwrite a slot with piped content")
	fmt.Println("  /ctx add <name>    append piped content to a slot")
	fmt.Println("  /ctx show <name>   print a slot's content")
	fmt.Println("  /ctx list          list all slots with sizes")
	fmt.Println("  /ctx clear [name]  remove one slot or all")
	fmt.Println("  /config            show or change settings")
	fmt.Println("  /jobs              list background jobs")
	fmt.Println("  /job <N|name>      show output of job N or named job")
	fmt.Println("  /job <N|name> | /fn       pipe job output into a function")
	fmt.Println("  /job <N|name> | ?msg      pipe job output to advisory AI")
	fmt.Println("  /job <N|name> | grep foo  pipe job output through bash")
	fmt.Println("  /job <N|name> | /ctx add <name>  store job output in context")
	fmt.Println("  /commit-msg (/cm)  generate a commit message from staged git changes")
	fmt.Println("  /permissions [cmd] show permission tier for a command")
	fmt.Println("  /clear             clear conversation history")
	fmt.Println("  /model             show current model and endpoint")
	fmt.Println("  /model <name>      switch to a different model")
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

// runPipeline resolves the left side (bash command or /meta command) and passes
// its output through the right-side chain (/fn, ?query, or a further pipeline).
func (s *Shell) runPipeline(leftCmd string, right *ParsedInput) {
	debug := os.Getenv("BAISH_DEBUG") != ""
	if debug {
		fmt.Fprintf(os.Stderr, "[debug] runPipeline: left=%q right.Type=%d right.Content=%q\n",
			leftCmd, right.Type, right.Content)
	}

	if right == nil {
		return
	}

	content, err := s.resolveLeftContent(leftCmd)
	if err != nil {
		fmt.Fprintf(os.Stderr, "pipeline: %v\n", err)
		return
	}
	if debug {
		fmt.Fprintf(os.Stderr, "[debug] captured %d bytes from left side\n", len(content))
	}
	if strings.TrimSpace(content) == "" {
		fmt.Fprintf(os.Stderr, "pipeline: left side produced no output\n")
		return
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT)
	defer signal.Stop(sigCh)
	go func() { <-sigCh; cancel() }()

	s.applyRight(ctx, content, right)
}

// applyRight applies the right side of a pipeline to already-captured content.
// right may be InputMeta (terminal function), InputPipeline (chained function),
// or InputAgent. This enables chains like: cat f | /summarize | /ctx set summary
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
	case InputAgentAct:
		msg := strings.TrimSpace(content)
		if q := strings.TrimSpace(right.Content); q != "" {
			msg = msg + "\n\n" + q
		}
		s.syncAgentContext()
		s.runAgentAct(msg)
	case InputBash:
		cmd := exec.CommandContext(ctx, "sh", "-c", right.Content)
		cmd.Stdin = strings.NewReader(content)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		cmd.Run() //nolint:errcheck — non-zero exits (e.g. grep no match) are normal
	}
}

// runJobs handles /jobs (list all) and /job N|name (show output of a job).
func (s *Shell) runJobs(args []string) {
	if len(args) == 0 {
		s.listJobs()
		return
	}

	switch args[0] {
	case "--help", "-h", "help":
		fmt.Println("usage:")
		fmt.Println("  /jobs                  list all background jobs")
		fmt.Println("  /jobs <N|name>         show output of job N or named job")
		fmt.Println("  /jobs list             list all background jobs (explicit form)")
		fmt.Println("  /jobs output <N|name>  show output of a job (explicit form)")
		return
	case "list":
		s.listJobs()
		return
	case "output":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "jobs output: requires a job ID or name")
			return
		}
		args = args[1:]
	}

	j := s.jobs.resolve(args[0])
	if j == nil {
		fmt.Fprintf(os.Stderr, "jobs: no job %q\n", args[0])
		return
	}
	switch j.status {
	case jobRunning:
		fmt.Printf("[%d] still running (%.1fs)\n", j.id, time.Since(j.startedAt).Seconds())
	case jobFailed:
		fmt.Printf("[%d] failed: %v\n", j.id, j.err)
	case jobDone:
		if j.output != "" {
			fmt.Println(j.output)
		} else {
			fmt.Printf("[%d] done (no output)\n", j.id)
		}
	}
}

// listJobs prints all jobs and marks them read.
func (s *Shell) listJobs() {
	jobs := s.jobs.list()
	if len(jobs) == 0 {
		fmt.Println("(no jobs)")
		return
	}
	fmt.Println()
	for _, j := range jobs {
		nameTag := ""
		if j.name != "" {
			nameTag = " \033[36m[" + j.name + "]\033[0m"
		}
		switch j.status {
		case jobRunning:
			fmt.Printf("  %2d%s  \033[33mrunning\033[0m  %5.1fs  %s\n",
				j.id, nameTag, time.Since(j.startedAt).Seconds(), j.display)
		case jobDone:
			fmt.Printf("  %2d%s  \033[32mdone\033[0m     %5.1fs  %s%s\n",
				j.id, nameTag, j.elapsed.Seconds(), j.display, sizeLabel(len(j.output)))
		case jobFailed:
			fmt.Printf("  %2d%s  \033[31mfailed\033[0m   %5.1fs  %s  (%v)\n",
				j.id, nameTag, j.elapsed.Seconds(), j.display, j.err)
		}
	}
	fmt.Println()
	s.jobs.markRead()
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

// syncAgentContext pushes current shell state into the active session's loop.
func (s *Shell) syncAgentContext() {
	sess := s.currentSession()
	cwd, _ := os.Getwd()
	sess.loop.SetShellContext(agent.ShellContext{
		CWD:          cwd,
		LastCommand:  s.lastCmd,
		LastExitCode: s.lastExitCode,
		LastStderr:   s.lastStderr,
	})
	threshold := s.appCfg.CtxInlineThreshold
	agentSlots := make(map[string]agent.CtxSlot, len(sess.ctxSlots))
	for name, slot := range sess.ctxSlots {
		if len(slot.content) <= threshold {
			agentSlots[name] = agent.CtxSlot{Content: slot.content}
		} else {
			agentSlots[name] = agent.CtxSlot{
				Content:     slot.content,
				Description: s.slotDesc(name, slot),
			}
		}
	}
	sess.loop.SetContextSlots(agentSlots)
	sess.loop.SetConfig(agent.LoopConfig{
		MaxHistoryMessages:     s.appCfg.MaxHistoryMessages,
		ToolOutputMaxChars:     s.appCfg.ToolOutputMaxChars,
		ToolOverflow:           string(s.appCfg.ToolOverflow),
		CtxInlineThreshold:     s.appCfg.CtxInlineThreshold,
		ToolOutputKeepRounds:   s.appCfg.ToolOutputKeepRounds,
		MaxContextTokens:       s.appCfg.MaxContextTokens,
		CompactionThreshold:    s.appCfg.CompactionThreshold,
		CompactionTailMessages: s.appCfg.CompactionTailMessages,
		MaxResponseTokens:      s.appCfg.MaxResponseTokens,
	})
}

// setModel switches the active session's LLM provider and updates the functions
// loader. Other sessions keep their existing provider.
func (s *Shell) setModel(model, endpoint string) error {
	provider, err := llm.NewOllamaProvider(endpoint, model)
	if err != nil {
		return fmt.Errorf("create provider: %w", err)
	}
	s.activeLoop().SetProvider(provider)

	s.fnLoader = functions.New(functions.ShellConfig{
		LLMEndpoint: endpoint,
		LLMModel:    model,
	})
	allTools := append(tools.AllTools(), s.fnLoader.ToolDefs()...)
	allTools = append(allTools, tools.ReadContextToolDef(), tools.DescribeToolToolDef())
	baseHandlers := tools.AllHandlers()
	for k, v := range s.fnLoader.ToolHandlers() {
		baseHandlers[k] = v
	}
	s.actTools = allTools
	s.actHandlers = baseHandlers

	// Rewire tools on the active session with the updated tool set.
	sess := s.currentSession()
	sess.loop.SetTools(s.actTools, s.makeSessionHandlers(sess))

	s.cfg.Model = model
	s.cfg.Endpoint = endpoint
	return nil
}

// ── Commit message generation ─────────────────────────────────────────────────

// runCommitMsg generates a commit message from staged git changes.
func (s *Shell) runCommitMsg() {
	out, err := exec.Command("git", "diff", "--staged").Output()
	if err != nil {
		// Distinguish "not a git repo" from other errors.
		if _, statErr := exec.Command("git", "rev-parse", "--git-dir").Output(); statErr != nil {
			fmt.Fprintln(os.Stderr, "commit-msg: not a git repository")
		} else {
			fmt.Fprintf(os.Stderr, "commit-msg: git diff failed: %v\n", err)
		}
		return
	}

	diff := strings.TrimSpace(string(out))
	if diff == "" {
		// Help the user understand why nothing is staged.
		unstaged, _ := exec.Command("git", "diff", "--name-only").Output()
		untracked, _ := exec.Command("git", "ls-files", "--others", "--exclude-standard").Output()
		switch {
		case len(strings.TrimSpace(string(unstaged))) > 0:
			fmt.Fprintln(os.Stderr, "commit-msg: nothing staged — you have unstaged changes (run 'git add' first)")
		case len(strings.TrimSpace(string(untracked))) > 0:
			fmt.Fprintln(os.Stderr, "commit-msg: nothing staged — you have untracked files (run 'git add' first)")
		default:
			fmt.Fprintln(os.Stderr, "commit-msg: nothing staged and working tree is clean")
		}
		return
	}

	const maxDiffChars = 12000
	truncationNote := ""
	if len(diff) > maxDiffChars {
		truncationNote = fmt.Sprintf("\n[diff truncated — showing first %d of %d chars]\n", maxDiffChars, len(diff))
		diff = diff[:maxDiffChars]
	}

	const systemPrompt = `You are an expert at writing clear, informative git commit messages.
Analyse the staged diff and produce a commit message.
Rules:
- Subject line: imperative mood, ≤72 chars, no trailing period
- Detect and follow conventional commits style if the project uses it (feat/fix/refactor/chore/docs/test/etc.)
- If the change is non-trivial: blank line, then a short body explaining WHY (the diff already shows what)
- Output ONLY the commit message — no preamble, no explanation, no markdown fences`

	userMsg := fmt.Sprintf("Staged diff:%s\n\n%s", truncationNote, diff)

	s.sessionMu().Lock()
	defer s.sessionMu().Unlock()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT)
	defer signal.Stop(sigCh)
	go func() { <-sigCh; cancel() }()

	fmt.Println()
	var msgBuf strings.Builder
	err = s.activeLoop().RunOneShot(ctx, systemPrompt, userMsg, func(token string) {
		fmt.Print(token)
		msgBuf.WriteString(token)
	})
	fmt.Println()
	if err != nil && err != context.Canceled {
		fmt.Fprintf(os.Stderr, "commit-msg: %v\n", err)
		return
	}

	message := strings.TrimSpace(msgBuf.String())
	if message == "" {
		return
	}
	s.promptCommitAction(message)
}

// promptCommitAction shows a post-generation menu letting the user commit or
// copy the message. Runs via /dev/tty so it works even when stdout is piped.
func (s *Shell) promptCommitAction(message string) {
	tty, err := os.OpenFile("/dev/tty", os.O_RDWR, 0)
	if err != nil {
		return // non-interactive — just leave the message on screen
	}
	defer tty.Close()

	fmt.Fprintf(tty, "\n\033[2m[c]ommit  [e]dit  [x] copy to clipboard  [enter] dismiss\033[0m ")
	sc := bufio.NewScanner(tty)
	if !sc.Scan() {
		return
	}
	choice := strings.TrimSpace(strings.ToLower(sc.Text()))

	switch {
	case choice == "c" || choice == "commit":
		s.gitCommit(message)
	case choice == "e" || choice == "edit":
		if edited, ok := editInEditor(message); ok {
			s.gitCommit(edited)
		}
	case choice == "x" || choice == "copy":
		if err := copyToClipboard(message); err != nil {
			fmt.Fprintf(os.Stderr, "clipboard: %v\n", err)
		} else {
			fmt.Println("copied to clipboard")
		}
	}
}

// gitCommit runs git commit -m with the given message.
func (s *Shell) gitCommit(message string) {
	out, err := exec.Command("git", "commit", "-m", message).CombinedOutput()
	if len(out) > 0 {
		fmt.Print(string(out))
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "git commit: %v\n", err)
	}
}

// editInEditor opens message in $VISUAL/$EDITOR for the user to edit.
// Returns the edited text and true if the user saved a non-empty result;
// false if the editor exited non-zero or the file was emptied.
func editInEditor(message string) (string, bool) {
	tmp, err := os.CreateTemp("", "baish-commit-*.txt")
	if err != nil {
		fmt.Fprintf(os.Stderr, "edit: %v\n", err)
		return "", false
	}
	defer os.Remove(tmp.Name())

	if _, err := tmp.WriteString(message + "\n"); err != nil {
		fmt.Fprintf(os.Stderr, "edit: %v\n", err)
		return "", false
	}
	tmp.Close()

	editor := os.Getenv("VISUAL")
	if editor == "" {
		editor = os.Getenv("EDITOR")
	}
	if editor == "" {
		editor = "vi"
	}

	cmd := exec.Command(editor, tmp.Name())
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "edit: editor exited with error: %v\n", err)
		return "", false
	}

	data, err := os.ReadFile(tmp.Name())
	if err != nil {
		fmt.Fprintf(os.Stderr, "edit: %v\n", err)
		return "", false
	}
	edited := strings.TrimSpace(string(data))
	if edited == "" {
		fmt.Fprintln(os.Stderr, "edit: empty message, commit cancelled")
		return "", false
	}
	return edited, true
}

// copyToClipboard writes text to the system clipboard using whatever tool is
// available (wl-copy, xclip, xsel, pbcopy).
func copyToClipboard(text string) error {
	candidates := [][]string{
		{"wl-copy"},
		{"xclip", "-selection", "clipboard"},
		{"xsel", "--clipboard", "--input"},
		{"pbcopy"},
	}
	for _, args := range candidates {
		cmd := exec.Command(args[0], args[1:]...)
		cmd.Stdin = strings.NewReader(text)
		if err := cmd.Run(); err == nil {
			return nil
		}
	}
	return fmt.Errorf("no clipboard utility found (install wl-clipboard, xclip, or xsel)")
}


// slotDesc returns a stub description for a context slot, using the user-provided
// description if set, otherwise generating one from the slot's content size.
func (s *Shell) slotDesc(name string, slot shellCtxSlot) string {
	if slot.desc != "" {
		return slot.desc
	}
	return fmt.Sprintf("%s document. Call read_context(%q) to retrieve.", humanSize(len(slot.content)), name)
}

// printCtxFeedback prints a summary line after a ctx set/add, showing mode.
func (s *Shell) printCtxFeedback(verb, name string, size int) {
	threshold := s.appCfg.CtxInlineThreshold
	if size > threshold {
		fmt.Printf("ctx: %s %q (%s) — stub mode; agent will use read_context() to fetch\n",
			verb, name, humanSize(size))
	} else {
		fmt.Printf("ctx: %s %q (%s) — inline\n", verb, name, humanSize(size))
	}
}

// humanSize formats a byte count as a human-readable string.
func humanSize(n int) string {
	switch {
	case n < 1024:
		return fmt.Sprintf("%dB", n)
	case n < 1024*1024:
		return fmt.Sprintf("%.1fKB", float64(n)/1024)
	default:
		return fmt.Sprintf("%.1fMB", float64(n)/(1024*1024))
	}
}

// filterContent returns lines from content that match any word in query,
// with up to 2 lines of surrounding context around each match.
func filterContent(content, query string) string {
	terms := strings.Fields(strings.ToLower(query))
	if len(terms) == 0 {
		return content
	}
	lines := strings.Split(content, "\n")
	type span struct{ start, end int }
	var spans []span
	for i, line := range lines {
		lower := strings.ToLower(line)
		for _, term := range terms {
			if strings.Contains(lower, term) {
				s := i - 2
				if s < 0 {
					s = 0
				}
				e := i + 3
				if e > len(lines) {
					e = len(lines)
				}
				spans = append(spans, span{s, e})
				break
			}
		}
	}
	if len(spans) == 0 {
		return fmt.Sprintf("(no lines matching %q)", query)
	}
	var out []string
	prev := -1
	for _, sp := range spans {
		if sp.start > prev+1 && prev >= 0 {
			out = append(out, "---")
		}
		if sp.start <= prev {
			sp.start = prev + 1
		}
		if sp.start < sp.end {
			out = append(out, lines[sp.start:sp.end]...)
		}
		prev = sp.end - 1
	}
	return strings.Join(out, "\n")
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

func buildPrompt(jobs *jobManager, cfg config.PromptConfig, session string) string {
	cwd, err := os.Getwd()
	if err != nil {
		cwd = "?"
	}
	if home := os.Getenv("HOME"); home != "" {
		cwd = strings.Replace(cwd, home, "~", 1)
	}
	cwd = truncatePath(cwd, cfg.PathMaxDepth)

	var sb strings.Builder
	sb.WriteString(colorize(cwd, cfg.PathColor))

	if cfg.ShowGitBranch {
		if branch := gitBranch(); branch != "" {
			sb.WriteByte(' ')
			sb.WriteString(colorize("("+branch+")", cfg.BranchColor))
		}
	}

	if session != "" && session != "main" {
		sb.WriteByte(' ')
		sb.WriteString(colorize("["+session+"]", "cyan"))
	}

	if jobs != nil {
		running, done := jobs.activity()
		switch {
		case running > 0 && done > 0:
			sb.WriteByte(' ')
			sb.WriteString(colorize(fmt.Sprintf("[%d⋯ %d✓]", running, done), cfg.JobColor))
		case running > 0:
			sb.WriteByte(' ')
			sb.WriteString(colorize(fmt.Sprintf("[%d⋯]", running), "dim"))
		case done > 0:
			sb.WriteByte(' ')
			sb.WriteString(colorize(fmt.Sprintf("[%d✓]", done), cfg.JobColor))
		}
	}

	sb.WriteString(cfg.Suffix)
	return sb.String()
}

// truncatePath keeps the last maxDepth path segments, prefixed with "...".
// 0 means return the full path unchanged.
func truncatePath(path string, maxDepth int) string {
	if maxDepth <= 0 {
		return path
	}
	parts := strings.Split(path, "/")
	if len(parts) <= maxDepth {
		return path
	}
	return ".../" + strings.Join(parts[len(parts)-maxDepth:], "/")
}

// gitBranch returns the current git branch name, or "" if not in a repo.
func gitBranch() string {
	out, err := exec.Command("git", "rev-parse", "--abbrev-ref", "HEAD").Output()
	if err != nil {
		return ""
	}
	branch := strings.TrimSpace(string(out))
	if branch == "HEAD" {
		// detached HEAD — show short commit hash instead
		out2, err := exec.Command("git", "rev-parse", "--short", "HEAD").Output()
		if err != nil {
			return "HEAD"
		}
		return strings.TrimSpace(string(out2))
	}
	return branch
}

var ansiColors = map[string]string{
	"red":     "\033[31m",
	"green":   "\033[32m",
	"yellow":  "\033[33m",
	"blue":    "\033[34m",
	"magenta": "\033[35m",
	"cyan":    "\033[36m",
	"white":   "\033[37m",
	"bold":    "\033[1m",
	"dim":     "\033[2m",
}

// colorize wraps text in an ANSI color escape, or returns it unchanged for "none"/empty.
func colorize(text, color string) string {
	if code, ok := ansiColors[color]; ok {
		return code + text + "\033[0m"
	}
	return text
}

func isValidColor(s string) bool {
	if s == "none" || s == "" {
		return true
	}
	_, ok := ansiColors[s]
	return ok
}

// truncateDisplay shortens a command string for display in job listings.
func truncateDisplay(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-3] + "..."
}

// parseBackgroundSuffix detects a trailing & and optional job name.
// It handles:
//
//	"cmd &"           → bg=true, name="", stripped="cmd"
//	"cmd & auth"      → bg=true, name="auth", stripped="cmd"
//	"cmd"             → bg=false, name="", stripped="cmd"
func parseBackgroundSuffix(line string) (bg bool, name, stripped string) {
	// Look for the last occurrence of " &" in the line.
	if idx := strings.LastIndex(line, " &"); idx >= 0 {
		rest := strings.TrimSpace(line[idx+2:])
		base := strings.TrimSpace(line[:idx])
		if rest == "" {
			return true, "", base
		}
		if isValidJobName(rest) {
			return true, rest, base
		}
	}
	// No " &" found; check for a bare trailing "&" (no space before it).
	if strings.HasSuffix(line, "&") {
		return true, "", strings.TrimSpace(strings.TrimSuffix(line, "&"))
	}
	return false, "", line
}

// maybeNotify prints a job-completion line immediately (via readline's
// thread-safe stderr writer, which auto-refreshes the prompt) and then drains
// the notification queue so the main REPL loop doesn't double-print.
// Called from the job goroutine via jobs.onComplete.
func (s *Shell) maybeNotify(j *job) {
	nameTag := ""
	if j.name != "" {
		nameTag = " [" + j.name + "]"
	}
	var line string
	if j.status == jobDone {
		line = fmt.Sprintf("[%d]%s done: %s%s  %.1fs", j.id, nameTag, j.display, sizeLabel(len(j.output)), j.elapsed.Seconds())
	} else {
		line = fmt.Sprintf("[%d]%s failed: %s — %v", j.id, nameTag, j.display, j.err)
	}

	// Print via readline's wrapped stderr: clears prompt, writes line, redraws.
	fmt.Fprintln(s.rl.Stderr(), line)
	// Update prompt to reflect new job counts, then redraw.
	s.rl.SetPrompt(buildPrompt(s.jobs, s.appCfg.Prompt, s.activeSession))
	s.rl.Refresh()
	// Drain so the main loop's drain() finds nothing to re-print.
	s.jobs.drain()

	if s.appCfg.Notifications {
		sendDesktopNotification("baish", line)
	}
}

// rightDisplay returns a short display string for the right side of a pipeline.
func rightDisplay(right *ParsedInput) string {
	if right == nil {
		return ""
	}
	switch right.Type {
	case InputAgent:
		return "?" + right.Content
	case InputMeta:
		return "/" + right.Content
	case InputPipeline:
		return "/" + right.PipeLeft + " | " + rightDisplay(right.PipeRight)
	case InputBash:
		return right.Content
	}
	return right.Content
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
	return []string{"help", "tools", "clear", "ctx", "config", "jobs", "job", "permissions", "model", "history", "commit-msg", "cm", "exit"}
}
