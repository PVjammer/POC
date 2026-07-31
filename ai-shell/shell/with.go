package shell

import (
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"time"
)

// withModeConfig defines the behavior for a named /with capture mode.
// Adding a new mode requires a handcrafted prompt and a deliberate tool
// permission decision — the registry is intentionally explicit and finite.
type withModeConfig struct {
	description string
	agentic     bool // true = full tool set (agentic); false = advisory tools only
}

// withModes is the curated registry of capture modes. Each name here becomes
// a valid /with <name> argument, and /<name> acts as its closing trigger.
// Names must not conflict with existing meta-commands or registered functions.
var withModes = map[string]withModeConfig{
	"debug": {
		description: "analyze errors and trace root causes across output and source files",
		agentic:     true,
	},
}

// withContext holds the state for an active /with capture context.
type withContext struct {
	name     string
	onError  bool
	buf      strings.Builder
	cmdCount int
	started  time.Time
}

// runWith handles /with <mode> [--on-error] and management subcommands
// (status, clear, end).
func (s *Shell) runWith(args []string) {
	if len(args) == 0 || args[0] == "--help" || args[0] == "-h" {
		s.printWithHelp()
		return
	}

	switch args[0] {
	case "status":
		if s.withCtx == nil {
			fmt.Fprintln(os.Stderr, "with: no active capture context")
			return
		}
		fmt.Printf("[with:%s  %s  %d cmd(s)  %.1fs]\n",
			s.withCtx.name, humanSize(s.withCtx.buf.Len()),
			s.withCtx.cmdCount, time.Since(s.withCtx.started).Seconds())

	case "clear":
		if s.withCtx == nil {
			fmt.Fprintln(os.Stderr, "with: no active capture context")
			return
		}
		s.withCtx.buf.Reset()
		s.withCtx.cmdCount = 0
		fmt.Printf("[with:%s buffer cleared]\n", s.withCtx.name)

	case "end", "discard", "cancel":
		if s.withCtx == nil {
			fmt.Fprintln(os.Stderr, "with: no active capture context")
			return
		}
		fmt.Printf("[with:%s discarded]\n", s.withCtx.name)
		s.withCtx = nil

	default:
		name := args[0]
		if _, ok := withModes[name]; !ok {
			fmt.Fprintf(os.Stderr, "with: unknown mode %q  (available: %s)\n",
				name, strings.Join(s.withModeNames(), ", "))
			return
		}
		if s.withCtx != nil {
			fmt.Fprintf(os.Stderr, "with: already capturing %q — /%s to trigger, /with end to discard\n",
				s.withCtx.name, s.withCtx.name)
			return
		}
		onError := false
		for _, a := range args[1:] {
			if a == "--on-error" {
				onError = true
			}
		}
		s.withCtx = &withContext{
			name:    name,
			onError: onError,
			started: time.Now(),
		}
		suffix := ""
		if onError {
			suffix = " --on-error"
		}
		fmt.Printf("\033[2m[with:%s started%s — type /%s to trigger]\033[0m\n", name, suffix, name)
	}
}

// triggerWith fires the agent using the capture buffer for the active mode.
// It clears s.withCtx before dispatching so nested capture cannot occur.
func (s *Shell) triggerWith() {
	if s.withCtx == nil {
		return
	}
	ctx := s.withCtx
	s.withCtx = nil

	modeCfg := withModes[ctx.name] // name was validated at /with time
	content := ctx.buf.String()

	if strings.TrimSpace(content) == "" {
		fmt.Fprintln(os.Stderr, "with: no output captured")
		return
	}

	fmt.Printf("\n\033[2m[with:%s — %d cmd(s), %s, %.1fs]\033[0m\n\n",
		ctx.name, ctx.cmdCount, humanSize(len(content)), time.Since(ctx.started).Seconds())

	const inlineThreshold = 8000

	var prompt string
	if len(content) <= inlineThreshold {
		prompt = withPromptInline(ctx.name, content)
	} else {
		f, err := os.CreateTemp("", "baish-with-*.txt")
		if err != nil {
			fmt.Fprintf(os.Stderr, "with: could not write temp file: %v\n", err)
			return
		}
		_, _ = io.WriteString(f, content)
		f.Close()
		prompt = withPromptFile(ctx.name, f.Name())
	}

	s.syncAgentContext()
	if modeCfg.agentic {
		s.runAgentAct(prompt)
	} else {
		s.runAgent(prompt)
	}
}

func withPromptInline(mode, content string) string {
	switch mode {
	case "debug":
		return "Debug capture session:\n\n" + content + `

Analyze the output above for errors, exceptions, or failures.
If file paths or line numbers appear (tracebacks, compiler errors, Go panics), read those files at the relevant lines.
Explain the root cause and suggest a concrete fix.`
	}
	return content
}

func withPromptFile(mode, path string) string {
	switch mode {
	case "debug":
		return fmt.Sprintf(`Debug capture was written to: %s

The file may be large. Follow these steps:
1. Search for error patterns:
   grep -in "error\|exception\|panic\|traceback\|failed\|fatal" %s | head -60
2. Check recent output:
   tail -50 %s
3. Extract any file paths and line numbers from the output, then read those files at the relevant lines.
4. Identify the root cause and suggest a fix.`, path, path, path)
	}
	return fmt.Sprintf("Captured output is in %s — search or read it to complete the task.", path)
}

func (s *Shell) withModeNames() []string {
	names := make([]string, 0, len(withModes))
	for k := range withModes {
		names = append(names, k)
	}
	sort.Strings(names)
	return names
}

func (s *Shell) printWithHelp() {
	fmt.Println("usage: /with <mode> [--on-error]")
	fmt.Println("       /with status | clear | end")
	fmt.Println()
	fmt.Println("  Capture command output for AI analysis.")
	fmt.Println("  Type /<mode> to close the context and trigger the agent.")
	fmt.Println()
	fmt.Println("  Modes:")
	for _, name := range s.withModeNames() {
		cfg := withModes[name]
		fmt.Printf("    %-12s  %s\n", name, cfg.description)
	}
	fmt.Println()
	fmt.Println("  --on-error   auto-trigger when a command exits non-zero")
	fmt.Println()
	fmt.Println("  Management:")
	fmt.Println("    /with status   show buffer size and elapsed time")
	fmt.Println("    /with clear    reset buffer without closing")
	fmt.Println("    /with end      discard and close without triggering")
	fmt.Println()
	fmt.Println("  Examples:")
	fmt.Println("    /with debug             capture commands; /debug to trigger")
	fmt.Println("    /with debug --on-error  auto-trigger on first non-zero exit")
}
