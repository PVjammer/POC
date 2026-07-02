package shell

import "testing"

func TestParse(t *testing.T) {
	tests := []struct {
		raw      string
		wantType InputType
		wantContent string
	}{
		{"", InputDirect, ""},
		{"ls -la", InputDirect, "ls -la"},
		{"!ls -la", InputDirect, "ls -la"},
		{`!"run tests"`, InputAgentAct, "run tests"},
		{"?why is the sky blue", InputAgent, "why is the sky blue"},
		{`?"why is the sky blue"`, InputAgent, "why is the sky blue"},
		{"/help", InputMeta, "help"},
		{"/ctx show design", InputMeta, "ctx show design"},
		{"cat file | /summarize", InputPipeline, ""},
		{"cat file | ?explain this", InputPipeline, ""},
	}
	for _, tt := range tests {
		t.Run(tt.raw, func(t *testing.T) {
			got := Parse(tt.raw)
			if got.Type != tt.wantType {
				t.Errorf("Parse(%q).Type = %v, want %v", tt.raw, got.Type, tt.wantType)
			}
			if tt.wantContent != "" && got.Content != tt.wantContent {
				t.Errorf("Parse(%q).Content = %q, want %q", tt.raw, got.Content, tt.wantContent)
			}
		})
	}
}

func TestPipeToAI(t *testing.T) {
	tests := []struct {
		input string
		wantIdx int
	}{
		{"cat file | /summarize", 9},
		{"cat file | ?explain", 9},
		{"cat file | grep foo", -1},
		{"cat file | /usr/bin/grep foo", -1},
		{"no pipe here", -1},
	}
	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := pipeToAI(tt.input)
			if (got >= 0) != (tt.wantIdx >= 0) {
				t.Errorf("pipeToAI(%q) = %d, want idx presence %v", tt.input, got, tt.wantIdx >= 0)
			}
		})
	}
}

func TestPipeFromAI(t *testing.T) {
	tests := []struct {
		input   string
		wantHit bool
	}{
		{"/job 1 | grep foo", true},
		{"/ctx show arch | wc -l", true},
		{"cat file | /summarize", false},
		{"cat file | grep foo", false},
	}
	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			got := pipeFromAI(tt.input)
			if (got >= 0) != tt.wantHit {
				t.Errorf("pipeFromAI(%q) = %d, wantHit=%v", tt.input, got, tt.wantHit)
			}
		})
	}
}

func TestIsAISegment(t *testing.T) {
	tests := []struct {
		s    string
		want bool
	}{
		{"?hello", true},
		{"/summarize", true},
		{`!"run tests"`, true},
		{"/usr/bin/grep", false},
		{"grep foo", false},
		{"", false},
	}
	for _, tt := range tests {
		if got := isAISegment(tt.s); got != tt.want {
			t.Errorf("isAISegment(%q) = %v, want %v", tt.s, got, tt.want)
		}
	}
}

func TestStripOuterQuotes(t *testing.T) {
	tests := []struct {
		in   string
		want string
	}{
		{`"hello"`, "hello"},
		{`'hello'`, "hello"},
		{`"unclosed`, "unclosed"},
		{"no quotes", "no quotes"},
		{"", ""},
	}
	for _, tt := range tests {
		if got := stripOuterQuotes(tt.in); got != tt.want {
			t.Errorf("stripOuterQuotes(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func TestParseBackgroundSuffix(t *testing.T) {
	tests := []struct {
		line       string
		wantBg     bool
		wantName   string
		wantStripped string
	}{
		{"cmd &", true, "", "cmd"},
		{"cmd & myname", true, "myname", "cmd"},
		{"cmd & 123", false, "", "cmd & 123"}, // purely numeric name is invalid
		{"cmd", false, "", "cmd"},
		{"cmd&", true, "", "cmd"},
	}
	for _, tt := range tests {
		t.Run(tt.line, func(t *testing.T) {
			bg, name, stripped := parseBackgroundSuffix(tt.line)
			if bg != tt.wantBg {
				t.Errorf("bg = %v, want %v", bg, tt.wantBg)
			}
			if name != tt.wantName {
				t.Errorf("name = %q, want %q", name, tt.wantName)
			}
			if stripped != tt.wantStripped {
				t.Errorf("stripped = %q, want %q", stripped, tt.wantStripped)
			}
		})
	}
}
