package shell

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/pvjammer/ai-shell-poc/agent"
	"github.com/pvjammer/ai-shell-poc/config"
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

	provider, err := s.newProvider(s.cfg.Endpoint, s.cfg.Model)
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
		MaxResponseTokens:      s.appCfg.MaxResponseTokens,
	})
	if hist != nil {
		loop.SetHistory(hist)
	}
	rlErr := s.rl.Stderr()
	loop.OnToolCall = func(toolName string, args map[string]interface{}) {
		if cmd, ok := args["command"]; ok {
			fmt.Fprintf(rlErr, "\033[2m  [%s] $ %v\033[0m\n", toolName, cmd)
		} else {
			fmt.Fprintf(rlErr, "\033[2m  [%s] %v\033[0m\n", toolName, args)
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
	s.syncSkillCatalog(sess)
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
	case "--help", "-h", "help":
		fmt.Println("usage:")
		fmt.Println("  /session list             list all sessions")
		fmt.Println("  /session new <name>       create a new blank session")
		fmt.Println("  /session fork <name>      fork current session into <name>")
		fmt.Println("  /session switch <name>    switch to session <name>")
		fmt.Println("  /session merge [name] [--edit]  merge a fork into current: injects overview, makes it queryable")
		fmt.Println("  /session show [name]      show session details")
		fmt.Println("  /session export <name>    export session history")
		fmt.Println("  /session delete <name>    delete a session")
		return
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
	case "merge":
		name := ""
		edit := false
		for _, a := range rest {
			if a == "--edit" || a == "-e" {
				edit = true
			} else if name == "" {
				name = a
			}
		}
		if err := s.doMerge(name, edit); err != nil {
			fmt.Fprintf(os.Stderr, "session merge: %v\n", err)
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
		fmt.Fprintf(os.Stderr, "unknown session subcommand %q\n  subcommands: list new fork merge switch show export delete\n", sub)
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

// saveSession persists any named session's history to disk.
func (s *Shell) saveSession(name string) error {
	sess, ok := s.sessions[name]
	if !ok {
		return fmt.Errorf("no session %q", name)
	}
	hist := sess.loop.CopyHistory()
	if len(hist) == 0 {
		return nil
	}
	data, err := json.Marshal(hist)
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}
	if err := os.MkdirAll(config.SessionsDir(), 0755); err != nil {
		return fmt.Errorf("mkdir: %w", err)
	}
	return os.WriteFile(config.SessionPath(name), data, 0644)
}

// syncSessionQuerier wires the shell's QuerySession implementation into the
// given session's loop so query_session tool calls resolve correctly.
func (s *Shell) syncSessionQuerier(sess *sessionEntry) {
	if len(s.queryableSessions) == 0 {
		return
	}
	names := make([]string, 0, len(s.queryableSessions))
	for name := range s.queryableSessions {
		names = append(names, name)
	}
	sort.Strings(names)
	sess.loop.SetSessionQuerier(s, names)
}

// resolveMergeSource returns the session name to merge when none is specified.
// It looks for sessions forked from the current session; errors if zero or ambiguous.
func (s *Shell) resolveMergeSource() (string, error) {
	var forks []string
	for name, sess := range s.sessions {
		if sess.parent == s.activeSession {
			forks = append(forks, name)
		}
	}
	switch len(forks) {
	case 0:
		return "", fmt.Errorf("no forked sessions to merge (try /session fork <name> first, or specify: /session merge <name>)")
	case 1:
		return forks[0], nil
	default:
		sort.Strings(forks)
		return "", fmt.Errorf("multiple forks of %q — specify one: /session merge <%s>", s.activeSession, strings.Join(forks, "|"))
	}
}

// doMerge merges srcName into the current session:
//  1. Summarises the source session's delta (post-fork messages, or all messages).
//  2. If edit is true, opens the summary in $EDITOR before injecting.
//  3. Registers the session as queryable via query_session.
//  4. Injects the summary as context into the current session.
func (s *Shell) doMerge(srcName string, edit bool) error {
	if srcName == "" {
		var err error
		srcName, err = s.resolveMergeSource()
		if err != nil {
			return err
		}
	}
	src, ok := s.sessions[srcName]
	if !ok {
		return fmt.Errorf("no session %q (try /session list)", srcName)
	}
	if srcName == s.activeSession {
		return fmt.Errorf("cannot merge the active session into itself")
	}

	// Snapshot source (one lock at a time — no deadlock risk with background jobs).
	src.mu.Lock()
	srcHist := src.loop.CopyHistory()
	forkIdx := src.forkIdx
	srcParent := src.parent
	src.mu.Unlock()

	// Compute the delta: post-fork messages if forked from current, else all messages.
	isFork := srcParent == s.activeSession
	var delta []llm.ChatMessage
	if isFork {
		if forkIdx > len(srcHist) {
			fmt.Printf("  note: source was compacted since fork; merging all remaining messages\n")
			forkIdx = 0
		}
		delta = srcHist[forkIdx:]
	} else {
		delta = srcHist
	}

	if len(delta) == 0 {
		fmt.Printf("nothing to merge: %q has no new messages\n", srcName)
		return nil
	}

	// Summarise the delta.
	fmt.Printf("summarizing %q (%d messages)...\n", srcName, len(delta))
	provider, err := s.newProvider(s.cfg.Endpoint, s.cfg.Model)
	if err != nil {
		return fmt.Errorf("create provider: %w", err)
	}
	summaryMsgs := buildSummaryMessages(delta, srcName)
	ctx := context.Background()
	opts := llm.DefaultOptions().WithMaxTokens(512).WithTemperature(0.2)
	ch := provider.ChatMessages(ctx, summaryMsgs, opts)
	resp, err := llm.CollectResponse(ctx, ch)
	if err != nil {
		return fmt.Errorf("summarize: %w", err)
	}
	summary, queryWhen := parseSummaryResponse(resp.Text)

	// Optional: open summary in $EDITOR so the user can refine it before injection.
	if edit {
		tmp, err := os.CreateTemp("", "baish-merge-*.md")
		if err != nil {
			return fmt.Errorf("create temp file: %w", err)
		}
		tmpPath := tmp.Name()
		defer os.Remove(tmpPath)

		content := fmt.Sprintf("Summary: %s\nQuery when: %s\n", summary, queryWhen)
		if _, err := tmp.WriteString(content); err != nil {
			tmp.Close()
			return fmt.Errorf("write temp file: %w", err)
		}
		tmp.Close()

		if err := openInEditor(tmpPath); err != nil {
			return fmt.Errorf("editor: %w", err)
		}

		data, err := os.ReadFile(tmpPath)
		if err != nil {
			return fmt.Errorf("read temp file: %w", err)
		}
		summary, queryWhen = parseSummaryResponse(string(data))
		if summary == "" {
			return fmt.Errorf("summary is empty after editing — merge cancelled")
		}
	}

	// Register session as queryable (store full text for potential future use).
	if s.queryableSessions == nil {
		s.queryableSessions = make(map[string]string)
	}
	s.queryableSessions[srcName] = summary

	// Inject overview + query hint into current session history.
	var mergeNote strings.Builder
	fmt.Fprintf(&mergeNote, "[session merge: %q]\n\n%s", srcName, summary)
	if queryWhen != "" {
		fmt.Fprintf(&mergeNote, "\n\nQuery this session when: %s", queryWhen)
	}
	fmt.Fprintf(&mergeNote, "\n\nCall query_session(%q, \"your question\") to retrieve specific details.", srcName)

	dst := s.currentSession()
	dst.mu.Lock()
	dstHist := dst.loop.CopyHistory()
	dstHist = append(dstHist, llm.ChatMessage{
		Role:    "user",
		Content: mergeNote.String(),
	})
	dstHist = append(dstHist, llm.ChatMessage{
		Role:    "assistant",
		Content: fmt.Sprintf("Understood. I have the overview of session %q and will query it when relevant.", srcName),
	})
	dst.loop.SetHistory(dstHist)
	dst.mu.Unlock()

	// Wire query_session tool into the current session's loop.
	s.syncSessionQuerier(dst)

	// Persist the source session so it survives a restart.
	if err := s.saveSession(srcName); err != nil {
		fmt.Fprintf(os.Stderr, "  warning: could not persist %q: %v\n", srcName, err)
	}

	// Print a brief success line — the summary lives in the agent's context.
	if isFork {
		fmt.Printf("merged %q into %q (%d messages since fork)\n", srcName, s.activeSession, len(delta))
	} else {
		fmt.Printf("merged %q into %q (%d messages)\n", srcName, s.activeSession, len(delta))
	}

	return nil
}

// buildSummaryMessages builds the LLM prompt used to summarise a session delta.
// The response is expected in two labeled sections:
//
//	Summary: <2-4 sentence overview of what was done and concluded>
//	Query when: <one sentence describing what topics this session can answer>
func buildSummaryMessages(delta []llm.ChatMessage, srcName string) []llm.ChatMessage {
	var sb strings.Builder
	fmt.Fprintf(&sb, "The following is a conversation from a shell session named %q.\n\n", srcName)
	for _, m := range delta {
		switch m.Role {
		case "user":
			if m.Content != "" {
				fmt.Fprintf(&sb, "User: %s\n", m.Content)
			}
		case "assistant":
			if m.Content != "" {
				fmt.Fprintf(&sb, "Assistant: %s\n", m.Content)
			}
		}
	}
	sb.WriteString(`
Respond with exactly two labeled lines (no other text):

Summary: <2-4 sentences on what was explored and what was concluded. Focus on outcomes and key artifacts, not the process.>
Query when: <one sentence describing what kinds of questions or topics this session can answer, so another agent knows when to query it>`)

	return []llm.ChatMessage{
		{Role: "system", Content: "You summarize shell sessions for other AI agents. Be factual and concise."},
		{Role: "user", Content: sb.String()},
	}
}

// parseSummaryResponse splits an LLM summary response into (summary, queryWhen).
// Expects lines starting with "Summary:" and "Query when:".
// Falls back gracefully if the model didn't follow the format.
func parseSummaryResponse(raw string) (summary, queryWhen string) {
	for _, line := range strings.Split(raw, "\n") {
		line = strings.TrimSpace(line)
		if after, ok := strings.CutPrefix(line, "Summary:"); ok {
			summary = strings.TrimSpace(after)
		} else if after, ok := strings.CutPrefix(line, "Query when:"); ok {
			queryWhen = strings.TrimSpace(after)
		}
	}
	if summary == "" {
		summary = strings.TrimSpace(raw)
	}
	return
}
