package llmprovider

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/pvjammer/ai-sdk-go/pkg/llm"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewOpenAIProvider(t *testing.T) {
	tests := []struct {
		name      string
		baseURL   string
		model     string
		wantError bool
		wantURL   string
	}{
		{name: "valid provider", baseURL: "http://localhost:30000/v1", model: "qwen3", wantURL: "http://localhost:30000/v1"},
		{name: "empty base URL uses default", baseURL: "", model: "gpt-4o-mini", wantURL: "https://api.openai.com/v1"},
		{name: "trailing slash trimmed", baseURL: "http://localhost:30000/v1/", model: "qwen3", wantURL: "http://localhost:30000/v1"},
		{name: "missing model", baseURL: "http://localhost:30000/v1", model: "", wantError: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, err := NewOpenAIProvider(tt.baseURL, tt.model)
			if tt.wantError {
				assert.Error(t, err)
				assert.Nil(t, p)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.wantURL, p.baseURL)
			assert.Equal(t, tt.model, p.model)
		})
	}
}

func TestOpenAIProvider_Name(t *testing.T) {
	p, err := NewOpenAIProvider("http://localhost:30000/v1", "qwen3")
	require.NoError(t, err)
	assert.Equal(t, "openai:qwen3", p.Name())
}

func TestOpenAIProvider_MaxTokens(t *testing.T) {
	p, err := NewOpenAIProvider("http://localhost:30000/v1", "qwen3")
	require.NoError(t, err)
	assert.Equal(t, 8192, p.MaxTokens())

	p2, err := NewOpenAIProvider("http://localhost:30000/v1", "qwen3", WithContextLength(131072))
	require.NoError(t, err)
	assert.Equal(t, 131072, p2.MaxTokens())
}

func TestOpenAIProvider_ChatMessages_NonStreaming(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/chat/completions", r.URL.Path)
		var req oaiChatRequest
		require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
		assert.False(t, req.Stream)
		assert.Equal(t, "qwen3", req.Model)

		resp := oaiChatResponse{Model: "qwen3"}
		resp.Choices = []struct {
			Message      oaiMessage `json:"message"`
			FinishReason string     `json:"finish_reason"`
		}{{
			Message:      oaiMessage{Role: "assistant", Content: strPtr("hello there")},
			FinishReason: "stop",
		}}
		resp.Usage.PromptTokens = 10
		resp.Usage.CompletionTokens = 5
		resp.Usage.TotalTokens = 15
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	p, err := NewOpenAIProvider(srv.URL, "qwen3")
	require.NoError(t, err)

	ch := p.ChatMessages(context.Background(), []llm.ChatMessage{{Role: "user", Content: "hi"}}, llm.DefaultOptions())
	resp, err := llm.CollectResponse(context.Background(), ch)
	require.NoError(t, err)
	assert.Equal(t, "hello there", resp.Text)
	assert.Equal(t, 15, resp.TokensUsed)
	assert.Equal(t, "stop", resp.FinishReason)
}

func TestOpenAIProvider_ChatMessages_Streaming(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req oaiChatRequest
		require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
		assert.True(t, req.Stream)

		w.Header().Set("Content-Type", "text/event-stream")
		flusher := w.(http.Flusher)
		events := []string{
			`{"model":"qwen3","choices":[{"delta":{"content":"hel"},"finish_reason":null}]}`,
			`{"model":"qwen3","choices":[{"delta":{"content":"lo"},"finish_reason":null}]}`,
			`{"model":"qwen3","choices":[{"delta":{},"finish_reason":"stop"}]}`,
		}
		for _, e := range events {
			fmt.Fprintf(w, "data: %s\n\n", e)
			flusher.Flush()
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer srv.Close()

	p, err := NewOpenAIProvider(srv.URL, "qwen3")
	require.NoError(t, err)

	opts := llm.DefaultOptions()
	opts.Stream = true
	ch := p.ChatMessages(context.Background(), []llm.ChatMessage{{Role: "user", Content: "hi"}}, opts)

	var text string
	var finishReason string
	for chunk := range ch {
		require.NoError(t, chunk.Error)
		text += chunk.Text
		if chunk.Done {
			finishReason = chunk.FinishReason
		}
	}
	assert.Equal(t, "hello", text)
	assert.Equal(t, "stop", finishReason)
}

func TestOpenAIProvider_ChatWithTools(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req oaiChatRequest
		require.NoError(t, json.NewDecoder(r.Body).Decode(&req))
		assert.False(t, req.Stream, "tool calls must always use non-streaming mode")
		require.Len(t, req.Tools, 1)
		assert.Equal(t, "get_weather", req.Tools[0].Function.Name)

		resp := oaiChatResponse{Model: "qwen3"}
		resp.Choices = []struct {
			Message      oaiMessage `json:"message"`
			FinishReason string     `json:"finish_reason"`
		}{{
			Message: oaiMessage{
				Role: "assistant",
				ToolCalls: []oaiToolCall{{
					ID:   "call_1",
					Type: "function",
					Function: oaiToolCallFunction{
						Name:      "get_weather",
						Arguments: `{"city":"Portland"}`,
					},
				}},
			},
			FinishReason: "tool_calls",
		}}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	p, err := NewOpenAIProvider(srv.URL, "qwen3")
	require.NoError(t, err)

	tools := []llm.ToolDef{{
		Name:        "get_weather",
		Description: "Get the weather for a city",
		Parameters:  map[string]interface{}{"type": "object"},
	}}
	ch := p.ChatWithTools(context.Background(), []llm.ChatMessage{{Role: "user", Content: "weather in Portland?"}}, tools, llm.DefaultOptions())

	var final llm.StreamChunk
	for chunk := range ch {
		require.NoError(t, chunk.Error)
		final = chunk
	}
	require.Len(t, final.ToolCalls, 1)
	assert.Equal(t, "get_weather", final.ToolCalls[0].Name)
	assert.Equal(t, "Portland", final.ToolCalls[0].Args["city"])
	assert.Equal(t, "tool_calls", final.FinishReason)
}

func TestOpenAIProvider_HTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		fmt.Fprint(w, `{"error":"invalid api key"}`)
	}))
	defer srv.Close()

	p, err := NewOpenAIProvider(srv.URL, "qwen3")
	require.NoError(t, err)

	ch := p.ChatMessages(context.Background(), []llm.ChatMessage{{Role: "user", Content: "hi"}}, llm.DefaultOptions())
	_, err = llm.CollectResponse(context.Background(), ch)
	require.Error(t, err)
	assert.ErrorIs(t, err, llm.ErrAuthenticationFailed)
}

func TestOpenAIProvider_ConnectionError(t *testing.T) {
	p, err := NewOpenAIProvider("http://127.0.0.1:1", "qwen3", WithHTTPClient(&http.Client{Timeout: time.Second}))
	require.NoError(t, err)

	ch := p.ChatMessages(context.Background(), []llm.ChatMessage{{Role: "user", Content: "hi"}}, llm.DefaultOptions())
	_, err = llm.CollectResponse(context.Background(), ch)
	require.Error(t, err)
	assert.ErrorIs(t, err, llm.ErrProviderUnavailable)
}

func TestOpenAIProvider_APIKeySentAsBearer(t *testing.T) {
	var gotAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		resp := oaiChatResponse{Model: "qwen3"}
		resp.Choices = []struct {
			Message      oaiMessage `json:"message"`
			FinishReason string     `json:"finish_reason"`
		}{{Message: oaiMessage{Role: "assistant", Content: strPtr("ok")}, FinishReason: "stop"}}
		json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	p, err := NewOpenAIProvider(srv.URL, "qwen3", WithAPIKey("secret-key"))
	require.NoError(t, err)

	ch := p.ChatMessages(context.Background(), []llm.ChatMessage{{Role: "user", Content: "hi"}}, llm.DefaultOptions())
	_, err = llm.CollectResponse(context.Background(), ch)
	require.NoError(t, err)
	assert.Equal(t, "Bearer secret-key", gotAuth)
}

func TestOpenAIProvider_ListModels(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "/models", r.URL.Path)
		fmt.Fprint(w, `{"data":[{"id":"qwen3-6b"},{"id":"llama-3.3-70b"}]}`)
	}))
	defer srv.Close()

	p, err := NewOpenAIProvider(srv.URL, "qwen3")
	require.NoError(t, err)

	models, err := p.ListModels(context.Background())
	require.NoError(t, err)
	assert.Equal(t, []string{"qwen3-6b", "llama-3.3-70b"}, models)
}

func strPtr(s string) *string { return &s }
