package shell

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/pvjammer/ai-shell-poc/agent"
	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

// sessionEntry is one independent conversation. Each has its own agent loop,
// context slots, and mutex so sessions never block each other.
type sessionEntry struct {
	loop     *agent.Loop
	ctxSlots map[string]shellCtxSlot
	mu       sync.Mutex
	parent   string    // name of session this was forked from; "" for main
	created  time.Time
	forkIdx  int // len(parent.history) at fork time; used for merge delta
}

// currentSession returns the active session. Must be called with no locks held.
func (s *Shell) currentSession() *sessionEntry {
	return s.sessions[s.activeSession]
}

// activeLoop is a convenience accessor for the active session's agent loop.
func (s *Shell) activeLoop() *agent.Loop {
	return s.sessions[s.activeSession].loop
}

// sessionMu returns a pointer to the active session's mutex.
func (s *Shell) sessionMu() *sync.Mutex {
	return &s.sessions[s.activeSession].mu
}

// makeSessionHandlers builds the handler map for a session. Stateless handlers
// come from s.actHandlers; session-specific ones (read_context, describe_tool)
// close over the given sessionEntry.
func (s *Shell) makeSessionHandlers(sess *sessionEntry) map[string]func(map[string]interface{}) (string, error) {
	handlers := make(map[string]func(map[string]interface{}) (string, error), len(s.actHandlers)+2)
	for k, v := range s.actHandlers {
		handlers[k] = v
	}

	handlers["read_context"] = func(args map[string]interface{}) (string, error) {
		name, _ := args["name"].(string)
		slot, ok := sess.ctxSlots[name]
		if !ok {
			return "", fmt.Errorf("no context slot %q (try /ctx list)", name)
		}
		if query, _ := args["query"].(string); query != "" {
			return filterContent(slot.content, query), nil
		}
		content := slot.content
		limit := s.appCfg.ToolOutputMaxChars
		if limit <= 0 {
			limit = 4000
		}
		offset := 0
		if v, ok := args["offset"].(float64); ok {
			offset = int(v)
		}
		if offset < 0 || offset >= len(content) {
			return fmt.Sprintf("[offset %d is out of range; slot is %d bytes]", offset, len(content)), nil
		}
		chunk := content[offset:]
		if len(chunk) <= limit {
			if offset > 0 {
				return fmt.Sprintf("[bytes %d–%d of %d]\n\n%s", offset, offset+len(chunk), len(content), chunk), nil
			}
			return chunk, nil
		}
		end := offset + limit
		if nl := strings.LastIndexByte(content[offset:end], '\n'); nl > 0 {
			end = offset + nl + 1
		}
		page := content[offset:end]
		return fmt.Sprintf("%s\n\n[bytes %d–%d of %d — call read_context(%q, offset=%d) to continue]",
			page, offset, end, len(content), name, end), nil
	}

	handlers["describe_tool"] = func(args map[string]interface{}) (string, error) {
		name, _ := args["name"].(string)
		for _, td := range s.actTools {
			if td.Name == name {
				b, err := json.Marshal(td)
				if err != nil {
					return "", err
				}
				return string(b), nil
			}
		}
		return "", fmt.Errorf("unknown tool %q", name)
	}

	return handlers
}

// createSession allocates a new sessionEntry, registers it, and wires its tools.
// If hist is non-nil the new loop starts with that history (fork path).
func (s *Shell) createSession(name, parent string, hist []llm.ChatMessage, ctxCopy map[string]shellCtxSlot, forkIdx int) (*sessionEntry, error) {
	if _, exists := s.sessions[name]; exists {
		return nil, fmt.Errorf("session %q already exists", name)
	}
	if name == "" || strings.ContainsAny(name, " \t/") {
		return nil, fmt.Errorf("session name must be non-empty with no spaces or slashes")
	}

	provider, err := llm.NewOllamaProvider(s.cfg.Endpoint, s.cfg.Model)
	if err != nil {
		return nil, fmt.Errorf("create provider: %w", err)
	}

	loop := agent.New(provider, s.actTools, nil) // handlers set below
	loop.SetConfig(agent.LoopConfig{
		MaxHistoryMessages:     s.appCfg.MaxHistoryMessages,
		ToolOutputMaxChars:     s.appCfg.ToolOutputMaxChars,
		ToolOverflow:           string(s.appCfg.ToolOverflow),
		CtxInlineThreshold:     s.appCfg.CtxInlineThreshold,
		ToolOutputKeepRounds:   s.appCfg.ToolOutputKeepRounds,
		MaxContextTokens:       s.appCfg.MaxContextTokens,
		CompactionThreshold:    s.appCfg.CompactionThreshold,
		CompactionTailMessages: s.appCfg.CompactionTailMessages,
	})
	if hist != nil {
		loop.SetHistory(hist)
	}
	loop.OnToolCall = func(toolName string, args map[string]interface{}) {
		if cmd, ok := args["command"]; ok {
			fmt.Fprintf(os.Stderr, "\033[2m  [%s] $ %v\033[0m\n", toolName, cmd)
		} else {
			fmt.Fprintf(os.Stderr, "\033[2m  [%s] %v\033[0m\n", toolName, args)
		}
	}

	if ctxCopy == nil {
		ctxCopy = make(map[string]shellCtxSlot)
	}

	sess := &sessionEntry{
		loop:     loop,
		ctxSlots: ctxCopy,
		parent:   parent,
		created:  time.Now(),
		forkIdx:  forkIdx,
	}
	s.sessions[name] = sess
	sess.loop.SetTools(s.actTools, s.makeSessionHandlers(sess))
	return sess, nil
}

// copyCtxSlots returns a deep copy of a ctx slot map.
func copyCtxSlots(src map[string]shellCtxSlot) map[string]shellCtxSlot {
	dst := make(map[string]shellCtxSlot, len(src))
	for k, v := range src {
		dst[k] = v
	}
	return dst
}

// runSession handles the /session (or /s) meta command.
func (s *Shell) runSession(args []string) {
	if len(args) == 0 {
		args = []string{"list"}
	}
	sub := args[0]
	rest := args[1:]

	switch sub {
	case "list", "ls":
		s.sessionList()
	case "new":
		if len(rest) == 0 {
			fmt.Fprintln(os.Stderr, "usage: /session new <name>")
			return
		}
		if _, err := s.createSession(rest[0], "", nil, copyCtxSlots(s.currentSession().ctxSlots), 0); err != nil {
			fmt.Fprintf(os.Stderr, "session new: %v\n", err)
			return
		}
		fmt.Printf("session %q created (blank, inherits current ctx slots)\n", rest[0])
	case "fork":
		if len(rest) == 0 {
			fmt.Fprintln(os.Stderr, "usage: /session fork <name>")
			return
		}
		if err := s.sessionFork(rest[0]); err != nil {
			fmt.Fprintf(os.Stderr, "session fork: %v\n", err)
		}
	case "switch", "sw":
		if len(rest) == 0 {
			fmt.Fprintln(os.Stderr, "usage: /session switch <name>")
			return
		}
		if err := s.sessionSwitch(rest[0]); err != nil {
			fmt.Fprintf(os.Stderr, "session switch: %v\n", err)
		}
	case "show":
		name := s.activeSession
		if len(rest) > 0 {
			name = rest[0]
		}
		s.sessionShow(name)
	case "export":
		if len(rest) == 0 {
			fmt.Fprintln(os.Stderr, "usage: /session export <name> [file]")
			return
		}
		var path string
		if len(rest) > 1 {
			path = rest[1]
		}
		if err := s.sessionExport(rest[0], path); err != nil {
			fmt.Fprintf(os.Stderr, "session export: %v\n", err)
		}
	case "delete", "del", "rm":
		if len(rest) == 0 {
			fmt.Fprintln(os.Stderr, "usage: /session delete <name>")
			return
		}
		if err := s.sessionDelete(rest[0]); err != nil {
			fmt.Fprintf(os.Stderr, "session delete: %v\n", err)
		}
	default:
		fmt.Fprintf(os.Stderr, "unknown session subcommand %q\n  subcommands: list new fork switch show export delete\n", sub)
	}
}

func (s *Shell) sessionList() {
	names := make([]string, 0, len(s.sessions))
	for name := range s.sessions {
		names = append(names, name)
	}
	sort.Strings(names)

	fmt.Printf("  %-16s %-10s %5s\n", "NAME", "PARENT", "MSGS")
	for _, name := range names {
		sess := s.sessions[name]
		marker := "  "
		if name == s.activeSession {
			marker = "* "
		}
		parent := sess.parent
		if parent == "" {
			parent = "—"
		}
		fmt.Printf("%s%-16s %-10s %5d\n", marker, name, parent, sess.loop.HistoryLen())
	}
}

func (s *Shell) sessionFork(name string) error {
	src := s.currentSession()
	src.mu.Lock()
	hist := src.loop.CopyHistory()
	ctxCopy := copyCtxSlots(src.ctxSlots)
	forkIdx := len(hist)
	src.mu.Unlock()

	if _, err := s.createSession(name, s.activeSession, hist, ctxCopy, forkIdx); err != nil {
		return err
	}
	fmt.Printf("session %q forked from %q (%d messages)\n", name, s.activeSession, forkIdx)
	return nil
}

func (s *Shell) sessionSwitch(name string) error {
	if _, ok := s.sessions[name]; !ok {
		return fmt.Errorf("no session %q (try /session list)", name)
	}
	s.activeSession = name
	fmt.Printf("switched to session %q\n", name)
	return nil
}

func (s *Shell) sessionShow(name string) {
	sess, ok := s.sessions[name]
	if !ok {
		fmt.Fprintf(os.Stderr, "no session %q\n", name)
		return
	}
	hist := sess.loop.CopyHistory()
	if len(hist) == 0 {
		fmt.Println("(no messages)")
		return
	}
	for i, m := range hist {
		switch m.Role {
		case "user":
			fmt.Printf("[%d] user: %s\n\n", i, m.Content)
		case "assistant":
			fmt.Printf("[%d] assistant: %s\n\n", i, m.Content)
		case "tool":
			fmt.Printf("[%d] tool(%s): %s\n\n", i, m.ToolName, truncate80(m.Content))
		}
	}
}

func (s *Shell) sessionExport(name, path string) error {
	sess, ok := s.sessions[name]
	if !ok {
		return fmt.Errorf("no session %q", name)
	}
	hist := sess.loop.CopyHistory()

	var sb strings.Builder
	fmt.Fprintf(&sb, "# Session: %s\n\n", name)
	if sess.parent != "" {
		fmt.Fprintf(&sb, "_Forked from %s at message %d_\n\n", sess.parent, sess.forkIdx)
	}
	fmt.Fprintf(&sb, "---\n\n")
	for _, m := range hist {
		switch m.Role {
		case "user":
			fmt.Fprintf(&sb, "**User:** %s\n\n", m.Content)
		case "assistant":
			if m.Content != "" {
				fmt.Fprintf(&sb, "**Assistant:** %s\n\n", m.Content)
			}
		case "tool":
			fmt.Fprintf(&sb, "<details><summary>tool: %s</summary>\n\n```\n%s\n```\n\n</details>\n\n", m.ToolName, m.Content)
		}
	}

	if path == "" {
		fmt.Print(sb.String())
		return nil
	}
	if err := os.WriteFile(path, []byte(sb.String()), 0644); err != nil {
		return err
	}
	fmt.Printf("exported %d messages to %s\n", len(hist), path)
	return nil
}

func (s *Shell) sessionDelete(name string) error {
	if name == "main" {
		return fmt.Errorf("cannot delete the main session")
	}
	if _, ok := s.sessions[name]; !ok {
		return fmt.Errorf("no session %q", name)
	}
	if s.activeSession == name {
		s.activeSession = "main"
		fmt.Printf("switched to main session\n")
	}
	delete(s.sessions, name)
	fmt.Printf("session %q deleted\n", name)
	return nil
}

func truncate80(s string) string {
	s = strings.TrimSpace(s)
	if len(s) <= 80 {
		return s
	}
	return s[:80] + "…"
}
