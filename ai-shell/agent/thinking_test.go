package agent

import "testing"

func TestStripThinking(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want string
	}{
		{"no think block", "just an answer", "just an answer"},
		{"leading think block", "<think>reasoning here</think>the answer", "the answer"},
		{"think block with newlines", "<think>\nline one\nline two\n</think>\n\nthe answer", "the answer"},
		{"unclosed think block yields empty", "<think>never finished", ""},
		{"empty after stripping", "<think>only reasoning, no answer</think>", ""},
		{"think tag not at start is left alone", "well, <think>this isn't a real block", "well, <think>this isn't a real block"},
		{"empty string", "", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := stripThinking(tt.in); got != tt.want {
				t.Errorf("stripThinking(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}
