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

// withContext holds the state for an active /with capture context.
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
		closing := fmt.Sprintf("/%s", name)
		if withModes[name].action == withActionStore {
			closing = fmt.Sprintf("/%s [slot-name]", name)
		}
		fmt.Printf("\033[2m[with:%s started%s — type %s to close]\033[0m\n", name, suffix, closing)
	}
}

// triggerWith fires the configured action for the active capture context.
// args are passed from the closing command (e.g. slot name for /context notes).
// Clears s.withCtx before acting so nested capture cannot occur.
func (s *Shell) triggerWith(args []string) {
	if s.withCtx == nil {
		return
	}
	ctx := s.withCtx
	s.withCtx = nil

	modeCfg := withModes[ctx.name]
	content := ctx.buf.String()

	if strings.TrimSpace(content) == "" {
		fmt.Fprintln(os.Stderr, "with: no output captured")
		return
	}

	fmt.Printf("\n\033[2m[with:%s — %d cmd(s), %s, %.1fs]\033[0m\n\n",
		ctx.name, ctx.cmdCount, humanSize(len(content)), time.Since(ctx.started).Seconds())

	// Store action: write buffer directly into a ctx slot, no AI involved.
	if modeCfg.action == withActionStore {
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

func withPromptFile(mode, path string) string {
	switch mode {
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
	fmt.Println("       /with status | clear | end")
	fmt.Println()
	fmt.Println("  Capture command output for later AI analysis or storage.")
	fmt.Println("  Type /<mode> to close the context and trigger the action.")
	fmt.Println()
	fmt.Println("  Modes:")
	for _, name := range s.withModeNames() {
		cfg := withModes[name]
		closing := "/" + name
		if cfg.action == withActionStore {
			closing = "/" + name + " [slot]"
		}
		fmt.Printf("    %-12s  %-20s  %s\n", name, closing, cfg.description)
	}
	fmt.Println()
	fmt.Println("  --on-error   auto-trigger when a command exits non-zero (analyze modes only)")
	fmt.Println()
	fmt.Println("  Management:")
	fmt.Println("    /with status   show buffer size and elapsed time")
	fmt.Println("    /with clear    reset buffer without closing")
	fmt.Println("    /with end      discard and close without triggering")
	fmt.Println()
	fmt.Println("  Examples:")
	fmt.Println("    /with debug             capture commands; /debug to trigger")
	fmt.Println("    /with debug --on-error  auto-trigger on first non-zero exit")
	fmt.Println("    /with recap             capture session; /recap for standup summary")
	fmt.Println("    /with context           capture output; /context notes to store as ctx slot")
}
