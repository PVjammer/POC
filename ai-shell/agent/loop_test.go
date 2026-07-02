package agent

import (
	"strings"
	"testing"

	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

func newLoopWithConfig(cfg LoopConfig) *Loop {
	return &Loop{history: make([]llm.ChatMessage, 0), cfg: cfg}
}

func TestEstimateTokens(t *testing.T) {
	tests := []struct {
		name string
		msgs []llm.ChatMessage
		want int
	}{
		{"empty", nil, 0},
		{"single short", []llm.ChatMessage{{Role: "user", Content: "hello"}}, 1},
		{"40 chars", []llm.ChatMessage{{Role: "user", Content: strings.Repeat("x", 40)}}, 10},
		{"two messages", []llm.ChatMessage{
			{Role: "user", Content: strings.Repeat("a", 20)},
			{Role: "assistant", Content: strings.Repeat("b", 40)},
		}, 15},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := estimateTokens(tt.msgs); got != tt.want {
				t.Errorf("estimateTokens() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestStripOldToolOutputs(t *testing.T) {
	longContent := strings.Repeat("x", 300)

	// Build a history with 4 rounds. Each round is: user + assistant + tool.
	hist := []llm.ChatMessage{
		{Role: "user", Content: "q1"},
		{Role: "assistant", Content: "a1"},
		{Role: "tool", Content: longContent, ToolName: "bash"},
		{Role: "user", Content: "q2"},
		{Role: "assistant", Content: "a2"},
		{Role: "tool", Content: longContent, ToolName: "bash"},
		{Role: "user", Content: "q3"},
		{Role: "assistant", Content: "a3"},
		{Role: "tool", Content: longContent, ToolName: "bash"},
		{Role: "user", Content: "q4"},
		{Role: "assistant", Content: "a4"},
	}

	t.Run("keep_rounds=3 strips only oldest", func(t *testing.T) {
		l := newLoopWithConfig(LoopConfig{ToolOutputKeepRounds: 3})
		out := l.stripOldToolOutputs(hist)
		if out[2].Content == longContent {
			t.Error("oldest tool output should have been stripped")
		}
		if out[5].Content != longContent {
			t.Error("round 2 tool output should be preserved")
		}
		if out[8].Content != longContent {
			t.Error("round 3 tool output should be preserved")
		}
	})

	t.Run("keep_rounds=0 is no-op", func(t *testing.T) {
		l := newLoopWithConfig(LoopConfig{ToolOutputKeepRounds: 0})
		out := l.stripOldToolOutputs(hist)
		if out[2].Content != longContent {
			t.Error("keep_rounds=0 should not strip anything")
		}
	})

	t.Run("does not mutate original", func(t *testing.T) {
		l := newLoopWithConfig(LoopConfig{ToolOutputKeepRounds: 2})
		_ = l.stripOldToolOutputs(hist)
		if hist[2].Content != longContent {
			t.Error("original slice should not be mutated")
		}
	})
}

func TestShouldCompact(t *testing.T) {
	t.Run("no compact when disabled", func(t *testing.T) {
		l := newLoopWithConfig(LoopConfig{MaxContextTokens: 0, CompactionThreshold: 0.75})
		l.history = []llm.ChatMessage{{Role: "user", Content: strings.Repeat("x", 40000)}}
		if l.shouldCompact() {
			t.Error("shouldCompact should return false when MaxContextTokens=0")
		}
	})

	t.Run("compact when over threshold", func(t *testing.T) {
		l := newLoopWithConfig(LoopConfig{MaxContextTokens: 100, CompactionThreshold: 0.75})
		l.history = []llm.ChatMessage{{Role: "user", Content: strings.Repeat("x", 400)}} // 100 tokens
		if !l.shouldCompact() {
			t.Error("shouldCompact should return true when at/over threshold")
		}
	})

	t.Run("no compact when under threshold", func(t *testing.T) {
		l := newLoopWithConfig(LoopConfig{MaxContextTokens: 1000, CompactionThreshold: 0.75})
		l.history = []llm.ChatMessage{{Role: "user", Content: strings.Repeat("x", 400)}} // 100 tokens
		if l.shouldCompact() {
			t.Error("shouldCompact should return false when under threshold")
		}
	})
}
