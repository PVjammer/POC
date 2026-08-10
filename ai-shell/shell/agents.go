package shell

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/pvjammer/ai-shell-poc/agent"
	"github.com/pvjammer/ai-shell-poc/config"
	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

// runAgentCmd handles the /agent meta command.
//
//	/agent                         list available agents, show active
//	/agent reset                   return to "default"
//	/agent <name>                  persistent switch on current session
//	/agent <name> "quoted query"   one-off foreground run; no session change
func (s *Shell) runAgentCmd(args []string) {
	if len(args) == 0 || args[0] == "list" {
		s.agentList()
		return
	}
	name := args[0]
	query := strings.TrimSpace(strings.Join(args[1:], " "))

	if name == "reset" {
		if err := s.applyAgentConfig("default", s.agentRegistry["default"]); err != nil {
			fmt.Fprintf(os.Stderr, "agent reset: %v\n", err)
		} else {
			fmt.Println("agent reset to default")
		}
		return
	}

	if name == "--help" || name == "-h" {
		fmt.Println("usage:")
		fmt.Println("  /agent                    list agents, show active")
		fmt.Println("  /agent reset              return to default agent")
		fmt.Println("  /agent create <name>      scaffold a new agent config, open $EDITOR")
		fmt.Println("  /agent edit <name>        open agent config in $EDITOR")
		fmt.Println("  /agent <name>             persistent switch for this session")
		fmt.Println("  /agent <name> <query>     one-off run; session unchanged")
		return
	}

	if name == "create" {
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "usage: /agent create <name>")
			return
		}
		s.agentCreate(args[1])
		return
	}

	if name == "edit" {
		target := "default"
		if len(args) >= 2 {
			target = args[1]
		}
		s.agentEdit(target)
		return
	}

	if _, ok := s.agentRegistry[name]; !ok && name != "default" && name != "one_shot" {
		fmt.Fprintf(os.Stderr, "agent: unknown agent %q (try /agent to list)\n", name)
		return
	}

	cfg := s.agentRegistry[name]

	if query != "" {
		// One-off: run then return to current session state unchanged.
		if name == "one_shot" {
			model := firstNonEmpty(cfg.Model, s.cfg.Model)
			endpoint := firstNonEmpty(cfg.Endpoint, s.cfg.Endpoint)
			s.runCommandOneShot(query, "", model, endpoint)
			return
		}
		s.runAgentOneOff(name, cfg, query)
		return
	}

	// Persistent switch.
	if err := s.applyAgentConfig(name, cfg); err != nil {
		fmt.Fprintf(os.Stderr, "agent %q: %v\n", name, err)
		return
	}
	fmt.Printf("agent switched to %q\n", name)
}

// agentList prints available agents and marks the active one.
func (s *Shell) agentList() {
	fmt.Printf("%-22s %-8s %s\n", "NAME", "ACTIVE", "DESCRIPTION")
	fmt.Printf("%-22s %-8s %s\n", strings.Repeat("-", 22), "------", "-----------")

	active := s.activeAgentName
	mark := func(name string) string {
		if name == active {
			return "✓"
		}
		return ""
	}

	fmt.Printf("%-22s %-8s %s\n", "default", mark("default"), "Built-in default agent")
	fmt.Printf("%-22s %-8s %s\n", "one_shot", mark("one_shot"), "Direct LLM call — no tool loop, no history")

	for name, cfg := range s.agentRegistry {
		desc := cfg.Description
		if desc == "" {
			desc = "-"
		}
		fmt.Printf("%-22s %-8s %s\n", name, mark(name), desc)
	}
}

// applyAgentConfig applies the named agent config to the current session's loop in-place.
// History and ctx slots are preserved.
func (s *Shell) applyAgentConfig(name string, cfg config.AgentConfig) error {
	sess := s.currentSession()

	// Switch provider if model/endpoint differ from what is currently active.
	newModel := firstNonEmpty(cfg.Model, s.cfg.Model)
	newEndpoint := firstNonEmpty(cfg.Endpoint, s.cfg.Endpoint)
	if newModel != s.cfg.Model || newEndpoint != s.cfg.Endpoint {
		if err := s.setModel(newModel, newEndpoint); err != nil {
			return fmt.Errorf("switch model: %w", err)
		}
	}

	// System prompt.
	switch {
	case cfg.SystemPrompt != "":
		sess.loop.SetSystemPrompt(cfg.SystemPrompt)
	case cfg.AdditionalInstructions != "":
		sess.loop.SetSystemPrompt(agent.BaseSystemPrompt + "\n\n" + cfg.AdditionalInstructions)
	default:
		sess.loop.SetSystemPrompt("") // revert to base
	}

	// Max rounds.
	if cfg.MaxRounds > 0 {
		lc := sess.loop.GetConfig()
		lc.MaxRounds = cfg.MaxRounds
		sess.loop.SetConfig(lc)
	}

	s.activeAgentName = name
	s.syncSkillCatalog(sess)
	return nil
}

// runAgentOneOff runs a single query with a temporary loop configured for agentName.
// The current session and its history are not modified.
func (s *Shell) runAgentOneOff(agentName string, cfg config.AgentConfig, query string) {
	model := firstNonEmpty(cfg.Model, s.cfg.Model)
	endpoint := firstNonEmpty(cfg.Endpoint, s.cfg.Endpoint)
	provider, err := llm.NewOllamaProvider(endpoint, model)
	if err != nil {
		fmt.Fprintf(os.Stderr, "agent one-off: %v\n", err)
		return
	}

	tdefs, handlers := s.agentToolSetFor(cfg)
	loop := agent.New(provider, tdefs, handlers)

	switch {
	case cfg.SystemPrompt != "":
		loop.SetSystemPrompt(cfg.SystemPrompt)
	case cfg.AdditionalInstructions != "":
		loop.SetSystemPrompt(agent.BaseSystemPrompt + "\n\n" + cfg.AdditionalInstructions)
	}

	if cfg.MaxRounds > 0 {
		lc := loop.GetConfig()
		lc.MaxRounds = cfg.MaxRounds
		loop.SetConfig(lc)
	}

	// Wire skills.
	recs := s.effectiveSkillList(agentName)
	catalog := make([]agent.SkillCatalogEntry, len(recs))
	for i, r := range recs {
		catalog[i] = agent.SkillCatalogEntry{
			Name:                   r.Name,
			Description:            r.Description,
			DisableModelInvocation: r.DisableModelInvocation,
		}
	}
	loop.SetSkillCatalog(catalog, s.skillLoader)
	loop.SetSpinnerEnabled(true)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT)
	defer signal.Stop(sigCh)
	go func() { <-sigCh; cancel() }()

	fmt.Println()
	if err := loop.Run(ctx, query, func(t string) { fmt.Print(t) }); err != nil {
		fmt.Fprintf(os.Stderr, "\nagent one-off: %v\n", err)
	}
	fmt.Println()
}

// agentToolSetFor returns tool defs and handlers filtered by cfg.Tools.
// Empty cfg.Tools means all act tools.
func (s *Shell) agentToolSetFor(cfg config.AgentConfig) ([]llm.ToolDef, map[string]func(map[string]interface{}) (string, error)) {
	sess := s.currentSession()
	allHandlers := s.makeSessionHandlers(sess)

	if len(cfg.Tools) == 0 {
		return s.actTools, allHandlers
	}

	allowed := make(map[string]bool, len(cfg.Tools))
	for _, t := range cfg.Tools {
		allowed[t] = true
	}

	var defs []llm.ToolDef
	handlers := make(map[string]func(map[string]interface{}) (string, error))
	for _, d := range s.actTools {
		if allowed[d.Name] {
			defs = append(defs, d)
			if h, ok := allHandlers[d.Name]; ok {
				handlers[d.Name] = h
			}
		}
	}
	return defs, handlers
}

// agentCreate scaffolds a new agent TOML and opens it in $EDITOR.
func (s *Shell) agentCreate(name string) {
	if name == "default" || name == "one_shot" {
		fmt.Fprintf(os.Stderr, "agent create: %q is a reserved name\n", name)
		return
	}
	path := filepath.Join(config.AgentsDir(), name+".toml")
	template := fmt.Sprintf(`# Agent: %s
# Place this file in ~/.config/baish/agents/ or .baish/agents/

[agents.%s]
description = ""

# model    = ""   # leave empty to use the global model
# endpoint = ""   # leave empty to use the global endpoint

# tools = ["bash", "read_file", "write_file"]  # empty = all tools

# Skills available to this agent (opt-in for named agents).
# Empty = no skills. List skill names exactly as shown in /skill list.
# skills = ["my-skill"]

# system_prompt replaces the base prompt entirely (use sparingly).
# system_prompt = ""

# additional_instructions is appended to the base prompt.
# additional_instructions = "Always respond in markdown."

# max_rounds = 10  # cap on tool-call rounds; global default is 25
`, name, name)

	if err := ensureFile(path, template); err != nil {
		fmt.Fprintf(os.Stderr, "agent create: %v\n", err)
		return
	}
	fmt.Printf("agent: opening %s\n", path)
	if err := openInEditor(path); err != nil {
		fmt.Fprintf(os.Stderr, "agent: editor: %v\n", err)
		return
	}
	// Reload so the new agent is immediately available.
	s.onConfigChange(path)
}

// agentEdit opens the config file for the named agent in $EDITOR.
// For named agents it finds the file that declares them; for "default" it
// opens agents.toml (the single-file config).
func (s *Shell) agentEdit(name string) {
	if name == "one_shot" {
		fmt.Fprintln(os.Stderr, "agent edit: one_shot is built-in and has no config file")
		return
	}

	// Look for an existing file that contains [agents.<name>].
	if name != "default" {
		found := findAgentFile(name)
		if found != "" {
			fmt.Printf("agent: opening %s\n", found)
			if err := openInEditor(found); err != nil {
				fmt.Fprintf(os.Stderr, "agent: editor: %v\n", err)
				return
			}
			s.onConfigChange(found)
			return
		}
		// Not found — offer to create.
		fmt.Fprintf(os.Stderr, "agent: %q not found; use /agent create %s to scaffold it\n", name, name)
		return
	}

	// "default" — open the global agents.toml.
	path := config.AgentsFile()
	template := `# Global agent overrides — ~/.config/baish/agents.toml
# You can also place individual *.toml files in ~/.config/baish/agents/

# Override settings for the built-in default agent:
# [agents.default]
# excluded_skills = []   # skill names to hide from the ! agent
# additional_instructions = ""
# max_rounds = 25
`
	if err := ensureFile(path, template); err != nil {
		fmt.Fprintf(os.Stderr, "agent: %v\n", err)
		return
	}
	fmt.Printf("agent: opening %s\n", path)
	if err := openInEditor(path); err != nil {
		fmt.Fprintf(os.Stderr, "agent: editor: %v\n", err)
		return
	}
	s.onConfigChange(path)
}

// findAgentFile searches the agent config dirs for a file that declares [agents.<name>].
func findAgentFile(name string) string {
	dirs := []string{config.AgentsDir(), filepath.Join(".baish", "agents")}
	needle := "[agents." + name + "]"
	for _, dir := range dirs {
		entries, err := os.ReadDir(dir)
		if err != nil {
			continue
		}
		for _, e := range entries {
			if e.IsDir() || !strings.HasSuffix(e.Name(), ".toml") {
				continue
			}
			p := filepath.Join(dir, e.Name())
			data, err := os.ReadFile(p)
			if err == nil && strings.Contains(string(data), needle) {
				return p
			}
		}
	}
	return ""
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}
