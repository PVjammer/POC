package shell

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/pvjammer/ai-shell-poc/agent"
	"github.com/pvjammer/ai-shell-poc/config"
)

// runCustomCommand dispatches a user-defined /command from commandRegistry.
// piped is any stdin content that was piped into the shell before the command.
func (s *Shell) runCustomCommand(cmd config.CommandConfig, args []string, piped string) {
	// Resolve the prompt text.
	var prompt string
	if cmd.Skill != "" {
		body, err := s.skillLoader.Body(cmd.Skill)
		if err != nil {
			fmt.Fprintf(os.Stderr, "/%s: cannot load skill %q: %v\n", cmd.Name, cmd.Skill, err)
			return
		}
		body = s.skillLoader.ApplySubstitution(body, args, "")
		prompt = body
	} else {
		prompt = cmd.Prompt
		if len(args) > 0 {
			prompt = prompt + "\n\n" + strings.Join(args, " ")
		}
	}

	if piped != "" {
		prompt = prompt + "\n\n" + piped
	}

	agentName := cmd.Agent
	if agentName == "" {
		agentName = "default"
	}

	// one_shot: direct LLM call, no loop.
	if agentName == "one_shot" {
		model := firstNonEmpty(cmd.Model, s.cfg.Model)
		endpoint := firstNonEmpty(cmd.Endpoint, s.cfg.Endpoint)
		s.runCommandOneShot(prompt, "", model, endpoint)
		return
	}

	agentCfg := s.agentRegistry[agentName]
	if cmd.Model != "" {
		agentCfg.Model = cmd.Model
	}
	if cmd.Endpoint != "" {
		agentCfg.Endpoint = cmd.Endpoint
	}

	if agentName == "default" {
		// Run on the current session's loop directly.
		s.runLoopForeground(s.currentSession().loop, prompt)
		return
	}

	// Named agent: ephemeral loop.
	s.runAgentOneOff(agentName, agentCfg, prompt)
}

// runCommandOneShot makes a direct single LLM call with no agent loop, no tools, no history.
func (s *Shell) runCommandOneShot(prompt, piped, model, endpoint string) {
	if model == "" {
		model = s.cfg.Model
	}
	if endpoint == "" {
		endpoint = s.cfg.Endpoint
	}
	provider, err := s.newProvider(endpoint, model)
	if err != nil {
		fmt.Fprintf(os.Stderr, "one-shot: %v\n", err)
		return
	}

	// Reuse a bare loop in RunOneShot mode — no history, no tools.
	loop := agent.New(provider, nil, nil)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT)
	defer signal.Stop(sigCh)
	go func() { <-sigCh; cancel() }()

	userMsg := prompt
	if piped != "" {
		userMsg = userMsg + "\n\n" + piped
	}

	fmt.Println()
	if err := loop.RunOneShot(ctx, "", userMsg, func(t string) { fmt.Print(t) }); err != nil {
		fmt.Fprintf(os.Stderr, "\none-shot: %v\n", err)
	}
	fmt.Println()
}

// runLoopForeground runs a query on an existing loop foreground, with SIGINT handling.
func (s *Shell) runLoopForeground(loop *agent.Loop, msg string) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT)
	defer signal.Stop(sigCh)
	go func() { <-sigCh; cancel() }()

	fmt.Println()
	if err := loop.Run(ctx, msg, func(t string) { fmt.Print(t) }); err != nil {
		fmt.Fprintf(os.Stderr, "\ncommand: %v\n", err)
	}
	fmt.Println()
}
