package shell

import (
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/pvjammer/ai-shell-poc/config"
)

// withAction determines what happens when a /with context is triggered.
type withAction int

const (
	withActionAnalyze withAction = iota // dispatch to the AI agent
	withActionStore                     // write buffer into a ctx slot directly
)

// withModeConfig defines the behavior for a named /with capture mode.
// Adding a new mode requires a handcrafted prompt and a deliberate tool
// permission decision — the registry is intentionally explicit and finite.
type withModeConfig struct {
	description string
	agentic     bool       // true = full tool set; false = advisory tools only
	action      withAction // default: withActionAnalyze
}

// withModes is the curated registry of capture modes. Each name here becomes
// a valid /with <name> argument, and /<name> acts as its closing trigger.
// Names must not conflict with existing meta-commands or registered functions.
var withModes = map[string]withModeConfig{
	"debug": {
		description: "analyze errors and trace root causes across output and source files",
		agentic:     true,
	},
	"recap": {
		description: "summarize session activity for standups, handoffs, or picking up tomorrow",
		agentic:     false,
	},
	"scripts": {
		description: "identify repetitive patterns and propose automation scripts",
		agentic:     true,
	},
	"context": {
		description: "accumulate output and store it in a named context slot (no AI)",
		agentic:     false,
		action:      withActionStore,
	},
}

// withContext holds the in-memory state for one active /with capture buffer.
type withContext struct {
	name     string
	onError  bool
	buf      strings.Builder
	cmdCount int
	started  time.Time
}

// runWith handles /with <mode> [--on-error] and management subcommands.
func (s *Shell) runWith(args []string) {
	if len(args) == 0 || args[0] == "--help" || args[0] == "-h" {
		s.printWithHelp()
		return
	}

	switch args[0] {
	case "status":
		s.withStatus(args[1:])

	case "clear":
		s.withClear(args[1:])

	case "end", "discard", "cancel":
		s.withEnd(args[1:])

	case "recover":
		s.withRecover(args[1:])

	default:
		name := args[0]
		if _, ok := withModes[name]; !ok {
			fmt.Fprintf(os.Stderr, "with: unknown mode %q  (available: %s)\n",
				name, strings.Join(s.withModeNames(), ", "))
			return
		}
		if _, exists := s.withCtxs[name]; exists {
			fmt.Fprintf(os.Stderr, "with: %q is already active — /%s to trigger, /with end %s to discard\n",
				name, name, name)
			return
		}
		onError := false
		for _, a := range args[1:] {
			if a == "--on-error" {
				onError = true
			}
		}
		s.withCtxs[name] = &withContext{
			name:    name,
			onError: onError,
			started: time.Now(),
		}
		suffix := ""
		if onError {
			suffix = " --on-error"
		}
		closing := fmt.Sprintf("/%s", name)
		if withModes[name].action == withActionStore {
			closing = fmt.Sprintf("/%s [slot-name]", name)
		}
		fmt.Printf("\033[2m[with:%s started%s — type %s to close]\033[0m\n", name, suffix, closing)
	}
}

func (s *Shell) withStatus(args []string) {
	if len(s.withCtxs) == 0 {
		fmt.Fprintln(os.Stderr, "with: no active capture contexts")
		return
	}
	if len(args) > 0 {
		name := args[0]
		ctx, ok := s.withCtxs[name]
		if !ok {
			fmt.Fprintf(os.Stderr, "with: no active %q context\n", name)
			return
		}
		fmt.Printf("[with:%s  %s  %d cmd(s)  %.1fs]\n",
			name, humanSize(ctx.buf.Len()), ctx.cmdCount, time.Since(ctx.started).Seconds())
		return
	}
	for _, name := range s.withModeNames() {
		ctx, ok := s.withCtxs[name]
		if !ok {
			continue
		}
		fmt.Printf("[with:%s  %s  %d cmd(s)  %.1fs]\n",
			name, humanSize(ctx.buf.Len()), ctx.cmdCount, time.Since(ctx.started).Seconds())
	}
}

func (s *Shell) withClear(args []string) {
	if len(s.withCtxs) == 0 {
		fmt.Fprintln(os.Stderr, "with: no active capture contexts")
		return
	}
	if len(args) > 0 {
		name := args[0]
		ctx, ok := s.withCtxs[name]
		if !ok {
			fmt.Fprintf(os.Stderr, "with: no active %q context\n", name)
			return
		}
		ctx.buf.Reset()
		ctx.cmdCount = 0
		fmt.Printf("[with:%s buffer cleared]\n", name)
		return
	}
	// No name given: require one if multiple are active.
	if len(s.withCtxs) > 1 {
		fmt.Fprintf(os.Stderr, "with: multiple contexts active — specify one: /with clear <name>\n")
		return
	}
	for name, ctx := range s.withCtxs {
		ctx.buf.Reset()
		ctx.cmdCount = 0
		fmt.Printf("[with:%s buffer cleared]\n", name)
	}
}

func (s *Shell) withEnd(args []string) {
	if len(s.withCtxs) == 0 {
		fmt.Fprintln(os.Stderr, "with: no active capture contexts")
		return
	}
	if len(args) > 0 {
		name := args[0]
		if name == "all" {
			for name := range s.withCtxs {
				delete(s.withCtxs, name)
				fmt.Printf("[with:%s discarded]\n", name)
			}
			return
		}
		if _, ok := s.withCtxs[name]; !ok {
			fmt.Fprintf(os.Stderr, "with: no active %q context\n", name)
			return
		}
		delete(s.withCtxs, name)
		fmt.Printf("[with:%s discarded]\n", name)
		return
	}
	// No name given: discard the single active context, or require a name.
	if len(s.withCtxs) > 1 {
		fmt.Fprintf(os.Stderr, "with: multiple contexts active — specify one: /with end <name>  or  /with end all\n")
		return
	}
	for name := range s.withCtxs {
		delete(s.withCtxs, name)
		fmt.Printf("[with:%s discarded]\n", name)
	}
}

// withRecover loads a saved buffer and dispatches it immediately to the agent.
// Recovery is a one-shot trigger — it does not create an active capture context.
func (s *Shell) withRecover(args []string) {
	if len(args) == 0 {
		records, err := config.LoadWiths()
		if err != nil {
			fmt.Fprintf(os.Stderr, "with: recover: %v\n", err)
			return
		}
		if len(records) == 0 {
			fmt.Println("(no saved /with buffers)")
			return
		}
		fmt.Printf("  %-12s  %-8s  %s\n", "NAME", "CMDS", "SAVED")
		for _, r := range records {
			fmt.Printf("  %-12s  %-8d  %s\n", r.Name, r.CmdCount, r.Saved.Format("2006-01-02 15:04"))
		}
		return
	}

	name := args[0]
	if _, ok := withModes[name]; !ok {
		fmt.Fprintf(os.Stderr, "with: recover: unknown mode %q\n", name)
		return
	}

	rec, content, err := config.LoadWith(name)
	if err != nil {
		fmt.Fprintf(os.Stderr, "with: recover: %v\n", err)
		return
	}

	fmt.Printf("\n\033[2m[with:%s recovered — %d cmd(s), %s, originally %.0fs ago]\033[0m\n\n",
		name, rec.CmdCount, humanSize(len(content)), time.Since(rec.Started).Seconds())

	modeCfg := withModes[name]
	s.syncAgentContext()
	s.dispatchWith(name, modeCfg, rec.CmdCount, rec.Started, content, args[1:])

	// Delete saved file only after successful dispatch setup.
	_ = config.DeleteWith(name)
}

// triggerWith looks up the named active context, removes it, and dispatches.
func (s *Shell) triggerWith(name string, args []string) {
	ctx, ok := s.withCtxs[name]
	if !ok {
		return
	}
	delete(s.withCtxs, name)

	modeCfg := withModes[name]
	content := ctx.buf.String()

	if strings.TrimSpace(content) == "" {
		fmt.Fprintln(os.Stderr, "with: no output captured")
		return
	}

	fmt.Printf("\n\033[2m[with:%s — %d cmd(s), %s, %.1fs]\033[0m\n\n",
		name, ctx.cmdCount, humanSize(len(content)), time.Since(ctx.started).Seconds())

	s.syncAgentContext()
	s.dispatchWith(name, modeCfg, ctx.cmdCount, ctx.started, content, args)
}

// dispatchWith executes the configured action for a capture context.
// Shared by triggerWith (live context) and withRecover (saved snapshot).
func (s *Shell) dispatchWith(name string, cfg withModeConfig, cmdCount int, started time.Time, content string, args []string) {
	// Store action: write buffer directly into a ctx slot, no AI involved.
	if cfg.action == withActionStore {
		slotName := "capture"
		if len(args) > 0 && args[0] != "" {
			slotName = args[0]
		}
		s.currentSession().ctxSlots[slotName] = shellCtxSlot{content: content}
		if slotName != "default" {
			if err := config.SaveContext(slotName, content); err != nil {
				fmt.Fprintf(os.Stderr, "with: warning: could not persist slot: %v\n", err)
			}
		}
		s.printCtxFeedback("set", slotName, len(content))
		return
	}

	// Analyze action: build a mode-specific prompt and dispatch to the agent.
	const inlineThreshold = 8000

	var prompt string
	if len(content) <= inlineThreshold {
		prompt = withPromptInline(name, content)
	} else {
		f, err := os.CreateTemp("", "baish-with-*.txt")
		if err != nil {
			fmt.Fprintf(os.Stderr, "with: could not write temp file: %v\n", err)
			return
		}
		_, _ = io.WriteString(f, content)
		f.Close()
		prompt = withPromptFile(name, f.Name())
	}

	// Dispatch as isolated one-shot — must not pollute the main conversation history.
	s.runAgentDispatch(prompt, cfg.agentic)
}

func withPromptInline(name, content string) string {
	switch name {
	case "debug":
		return `Debug capture session:

` + content + `

First, identify the failure type from the output:
- Runtime error / traceback: read the source files at the mentioned file paths and line numbers
- Test failure (pytest / go test / jest / etc.): read BOTH the failing test AND the implementation it exercises — the fix may be in either
- Build or compiler error: read the source file at the reported line, check imports and types
- Docker layer failure: read the Dockerfile and identify the failing layer

Then:
1. Explain the root cause clearly
2. Show the corrected code`

	case "recap":
		return `Work session activity:

` + content + `

Summarize what happened. Focus on:
- What was being worked on (infer from commands, file paths, error messages)
- Key outcomes: what succeeded, what failed, what was left unresolved
- Any decisions or discoveries worth noting

Write 3–5 bullet points suitable for a standup update or end-of-day note.`

	case "scripts":
		return `Work session capture:

` + content + `

Analyze the commands above for patterns worth automating. For each opportunity:
1. Describe the pattern (what is repeated or error-prone)
2. Propose a shell script, function, or alias that automates it
3. Show the complete implementation

Focus on: sequences run more than once, long commands that could be wrapped, multi-step workflows that are easy to get wrong. If asked, write the scripts to an appropriate location (~/bin/, a local scripts/ directory, or as shell functions in ~/.bashrc).`
	}
	return content
}

func withPromptFile(name, path string) string {
	switch name {
	case "debug":
		return fmt.Sprintf(`Debug capture was written to: %s

The file may be large. Work through it in order:
1. Grep for error patterns to orient yourself:
   grep -in "error\|exception\|panic\|traceback\|failed\|fatal\|assert" %s | head -60
2. Check the tail for the most recent output:
   tail -50 %s
3. Identify the failure type:
   - Runtime error / traceback: read source files at the mentioned paths and line numbers
   - Test failure (pytest / go test / jest / etc.): read BOTH the failing test AND the implementation — the fix may be in either
   - Build or compiler error: read the source file at the reported line, check imports and types
   - Docker layer failure: read the Dockerfile and identify the failing layer
4. Explain the root cause and show the corrected code.`, path, path, path)

	case "recap":
		return fmt.Sprintf(`Work session activity was written to: %s

To orient yourself:
1. Extract the commands run:
   grep "^\$ " %s
2. Check for errors and failures:
   grep -in "error\|failed\|\[exit: [^0]\]" %s | head -30

Then summarize in 3–5 bullet points:
- What was worked on
- Key outcomes (successes, failures, unresolved issues)
- Anything worth noting for tomorrow`, path, path, path)

	case "scripts":
		return fmt.Sprintf(`Work session capture is in: %s

First, extract the commands that were run:
  grep "^\$ " %s

Look for repetition and multi-step sequences. For each automation opportunity:
1. Describe the pattern (what is repeated or error-prone)
2. Propose a shell script, function, or alias
3. Show the complete implementation

If asked, write the scripts to an appropriate location.`, path, path)
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
	fmt.Println("       /with status [name] | clear [name] | end [name|all] | recover [name]")
	fmt.Println()
	fmt.Println("  Capture command output for AI analysis or storage.")
	fmt.Println("  Type /<mode> to close that context and trigger its action.")
	fmt.Println("  Multiple different modes may be active simultaneously.")
	fmt.Println()
	fmt.Println("  Modes:")
	for _, name := range s.withModeNames() {
		cfg := withModes[name]
		closing := "/" + name
		if cfg.action == withActionStore {
			closing = "/" + name + " [slot]"
		}
		fmt.Printf("    %-12s  %-22s  %s\n", name, closing, cfg.description)
	}
	fmt.Println()
	fmt.Println("  --on-error   auto-trigger on first non-zero exit (analyze modes)")
	fmt.Println()
	fmt.Println("  Management:")
	fmt.Println("    /with status [name]    show buffer size(s) and elapsed time")
	fmt.Println("    /with clear [name]     reset buffer without closing")
	fmt.Println("    /with end [name|all]   discard and close without triggering")
	fmt.Println("    /with recover [name]   trigger a saved buffer from a previous session")
	fmt.Println()
	fmt.Println("  Examples:")
	fmt.Println("    /with recap                   start session capture")
	fmt.Println("    /with debug --on-error         auto-trigger on first error")
	fmt.Println("    /with recap  +  /with debug    both active simultaneously")
	fmt.Println("    /with context  →  /context notes   store output as ctx slot 'notes'")
}
