// Package backend provides HTTP client adapters for external AI backends.
// Currently: OpenCode serve (http://localhost:4096 by default).
package backend

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// OpenCodeClient talks to a running `opencode serve` instance.
// It maintains one session per baish session so the OpenCode model
// accumulates conversational context across turns.
type OpenCodeClient struct {
	baseURL    string
	model      string // "providerID/modelID" e.g. "opencode/big-pickle"; empty = server default
	sessionID  string
	httpClient *http.Client
}

// NewOpenCodeClient returns a client pointed at baseURL (e.g. "http://127.0.0.1:4096").
// model is an optional "providerID/modelID" string (e.g. "opencode/big-pickle"); pass ""
// to use whatever model the OpenCode server is configured with by default.
func NewOpenCodeClient(baseURL, model string) *OpenCodeClient {
	return &OpenCodeClient{
		baseURL:    strings.TrimRight(baseURL, "/"),
		model:      model,
		httpClient: &http.Client{Timeout: 5 * time.Minute},
	}
}

// ReuseSession attempts to adopt an existing OpenCode session by ID.
// Returns true if the session exists and was adopted; false if it has been
// deleted (e.g. the server was restarted) and a new session should be created.
func (c *OpenCodeClient) ReuseSession(ctx context.Context, sessionID string) bool {
	if sessionID == "" {
		return false
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet,
		c.baseURL+"/session/"+sessionID, nil)
	if err != nil {
		return false
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return false
	}
	c.sessionID = sessionID
	return true
}

// EnsureSession creates an OpenCode session if one has not been established yet.
// Safe to call multiple times — subsequent calls are no-ops.
func (c *OpenCodeClient) EnsureSession(ctx context.Context) error {
	if c.sessionID != "" {
		return nil
	}
	sessBody := map[string]interface{}{"agent": "build"}
	if c.model != "" {
		providerID, modelID := c.splitModel()
		sessBody["model"] = map[string]interface{}{"id": modelID, "providerID": providerID}
	}
	body, _ := json.Marshal(sessBody)
	resp, err := c.post(ctx, "/session", bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("opencode: create session: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return fmt.Errorf("opencode: create session: HTTP %d: %s", resp.StatusCode, b)
	}
	var sess struct {
		ID string `json:"id"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&sess); err != nil {
		return fmt.Errorf("opencode: decode session: %w", err)
	}
	c.sessionID = sess.ID
	return nil
}

// SessionID returns the current OpenCode session ID, or "" if not yet established.
func (c *OpenCodeClient) SessionID() string { return c.sessionID }

// DeleteSession deletes the OpenCode session. Used by disposable sessions
// created for /with dispatches to avoid polluting the main conversation.
func (c *OpenCodeClient) DeleteSession(ctx context.Context) error {
	if c.sessionID == "" {
		return nil
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodDelete, c.baseURL+"/session/"+c.sessionID, nil)
	if err != nil {
		return err
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	resp.Body.Close()
	c.sessionID = ""
	return nil
}

// Abort signals the running OpenCode session to stop mid-response.
func (c *OpenCodeClient) Abort(ctx context.Context) error {
	if c.sessionID == "" {
		return nil
	}
	resp, err := c.post(ctx, "/session/"+c.sessionID+"/abort", nil)
	if err != nil {
		return err
	}
	resp.Body.Close()
	return nil
}

// Send posts text to the active OpenCode session and streams the response token
// by token. onToken is called for each text delta as it arrives via SSE. If SSE
// streaming fails or produces no output, Send falls back to the full text from the
// synchronous POST response.
//
// systemPrompt is appended to OpenCode's base system prompt when non-empty. Pass
// "" to use the default build-mode prompt.
func (c *OpenCodeClient) Send(ctx context.Context, text, systemPrompt string, onToken func(string)) error {
	if c.sessionID == "" {
		return fmt.Errorf("opencode: no active session — call EnsureSession first")
	}

	// Subscribe to the SSE event stream before posting so we don't miss early deltas.
	// The goroutine calls onToken in real-time and signals idleCh when the session
	// transitions to idle (model finished).
	idleCh := make(chan struct{})
	var sseErr error

	sseCtx, sseCancel := context.WithCancel(ctx)
	defer sseCancel()

	go func() {
		sseErr = c.streamResponse(sseCtx, c.sessionID, onToken, idleCh)
	}()

	// Build and POST the message.
	msgBody := map[string]interface{}{
		"role": "user",
		"parts": []map[string]interface{}{
			{"type": "text", "text": text},
		},
	}
	if systemPrompt != "" {
		msgBody["system"] = systemPrompt
	}
	if c.model != "" {
		providerID, modelID := c.splitModel()
		msgBody["model"] = map[string]interface{}{"id": modelID, "providerID": providerID}
	}
	encoded, _ := json.Marshal(msgBody)

	resp, err := c.post(ctx, "/session/"+c.sessionID+"/message", bytes.NewReader(encoded))
	if err != nil {
		return fmt.Errorf("opencode: send message: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
		return fmt.Errorf("opencode: HTTP %d: %s", resp.StatusCode, b)
	}

	var syncMsg ocMessage
	if err := json.NewDecoder(resp.Body).Decode(&syncMsg); err != nil {
		return fmt.Errorf("opencode: decode response: %w", err)
	}

	// Wait for the SSE goroutine to see session.idle (or context cancellation).
	select {
	case <-idleCh:
	case <-ctx.Done():
	case <-time.After(5 * time.Second):
	}
	sseCancel()

	// If the SSE goroutine produced no output (e.g. connection failed), fall
	// back to the text from the synchronous POST response.
	if sseErr != nil {
		for _, p := range syncMsg.Parts {
			if p.Type == "text" && p.Text != "" {
				onToken(p.Text)
			}
		}
	}

	return nil
}

// streamResponse connects to the OpenCode SSE event stream and calls onToken
// for each text delta belonging to sessionID. It closes idleCh when the session
// transitions to idle. Returns a non-nil error only if no deltas were received
// (so the caller can fall back to the synchronous response text).
func (c *OpenCodeClient) streamResponse(ctx context.Context, sessionID string, onToken func(string), idleCh chan struct{}) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/event", nil)
	if err != nil {
		return err
	}
	req.Header.Set("Accept", "text/event-stream")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	received := false
	scanner := bufio.NewScanner(resp.Body)
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		raw := strings.TrimPrefix(line, "data: ")

		var evt ocEvent
		if err := json.Unmarshal([]byte(raw), &evt); err != nil {
			continue
		}

		switch evt.Type {
		case "message.part.delta":
			if evt.Properties.SessionID == sessionID && evt.Properties.Field == "text" {
				onToken(evt.Properties.Delta)
				received = true
			}
		case "session.idle":
			if evt.Properties.SessionID == sessionID {
				select {
				case <-idleCh:
				default:
					close(idleCh)
				}
			}
		}

		if ctx.Err() != nil {
			break
		}
	}

	// Signal idle in case we exited the loop without seeing a session.idle event.
	select {
	case <-idleCh:
	default:
		close(idleCh)
	}

	if !received {
		return fmt.Errorf("no streaming deltas received")
	}
	return nil
}

// OCSessionInfo holds the metadata for a single OpenCode session.
type OCSessionInfo struct {
	ID        string `json:"id"`
	Slug      string `json:"slug"`
	Title     string `json:"title"`
	Directory string `json:"directory"`
	Time      struct {
		Created int64 `json:"created"`
		Updated int64 `json:"updated"`
	} `json:"time"`
}

// ListSessions returns all sessions known to the OpenCode server, newest first.
func (c *OpenCodeClient) ListSessions(ctx context.Context) ([]OCSessionInfo, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/session", nil)
	if err != nil {
		return nil, err
	}
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, fmt.Errorf("opencode: list sessions: HTTP %d: %s", resp.StatusCode, b)
	}
	var sessions []OCSessionInfo
	if err := json.NewDecoder(resp.Body).Decode(&sessions); err != nil {
		return nil, fmt.Errorf("opencode: decode sessions: %w", err)
	}
	return sessions, nil
}

// splitModel splits "providerID/modelID" into its two parts.
// If there is no "/" it treats the whole string as the modelID with providerID "opencode".
func (c *OpenCodeClient) splitModel() (providerID, modelID string) {
	if idx := strings.LastIndex(c.model, "/"); idx >= 0 {
		return c.model[:idx], c.model[idx+1:]
	}
	return "opencode", c.model
}

func (c *OpenCodeClient) post(ctx context.Context, path string, body io.Reader) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, body)
	if err != nil {
		return nil, err
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	return c.httpClient.Do(req)
}

// ── Response / event types ──────────────────────────────────────────────────

type ocPart struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type ocMessage struct {
	Parts []ocPart `json:"parts"`
}

type ocEventProperties struct {
	SessionID string `json:"sessionID"`
	MessageID string `json:"messageID"`
	PartID    string `json:"partID"`
	Field     string `json:"field"`
	Delta     string `json:"delta"`
}

type ocEvent struct {
	ID         string            `json:"id"`
	Type       string            `json:"type"`
	Properties ocEventProperties `json:"properties"`
}
