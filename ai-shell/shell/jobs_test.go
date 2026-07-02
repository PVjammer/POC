package shell

import (
	"testing"
	"time"
)

func TestIsValidJobName(t *testing.T) {
	tests := []struct {
		name string
		want bool
	}{
		{"research", true},
		{"my-job", true},
		{"my_job", true},
		{"job1", true},
		{"", false},
		{"123", false},        // purely numeric
		{"a b", false},        // space
		{"a/b", false},        // slash
		{string(make([]byte, 33)), false}, // too long
	}
	for _, tt := range tests {
		if got := isValidJobName(tt.name); got != tt.want {
			t.Errorf("isValidJobName(%q) = %v, want %v", tt.name, got, tt.want)
		}
	}
}

func TestSizeLabel(t *testing.T) {
	tests := []struct {
		n    int
		want string
	}{
		{0, ""},
		{512, "  512B"},
		{1024, "  1.0KB"},
		{2048, "  2.0KB"},
	}
	for _, tt := range tests {
		if got := sizeLabel(tt.n); got != tt.want {
			t.Errorf("sizeLabel(%d) = %q, want %q", tt.n, got, tt.want)
		}
	}
}

func TestJobManagerStartDuplicateName(t *testing.T) {
	m := newJobManager()
	done := make(chan struct{})
	_, err := m.start("task1", "myname", func() (string, error) {
		<-done
		return "", nil
	})
	if err != nil {
		t.Fatalf("first start failed: %v", err)
	}
	_, err = m.start("task2", "myname", func() (string, error) { return "", nil })
	if err == nil {
		t.Error("expected error for duplicate name, got nil")
	}
	close(done)
}

func TestJobManagerResolveByIDAndName(t *testing.T) {
	m := newJobManager()
	done := make(chan struct{})
	id, err := m.start("task", "alpha", func() (string, error) {
		<-done
		return "output", nil
	})
	if err != nil {
		t.Fatalf("start failed: %v", err)
	}

	if j := m.resolve("alpha"); j == nil {
		t.Error("resolve by name returned nil")
	}
	if j := m.resolve(string(rune('0' + id))); j == nil {
		t.Errorf("resolve by ID %d returned nil", id)
	}
	if j := m.resolve("notexist"); j != nil {
		t.Error("resolve of unknown name should return nil")
	}

	close(done)
	// Give goroutine time to finish
	time.Sleep(50 * time.Millisecond)
}
