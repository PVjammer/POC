package shell

import (
	"fmt"
	"strconv"
	"sync"
	"time"
	"unicode"
)

type jobStatus int

const (
	jobRunning jobStatus = iota
	jobDone
	jobFailed
)

type job struct {
	id        int
	name      string // optional user-provided tag; "" if unnamed
	display   string
	status    jobStatus
	output    string
	err       error
	startedAt time.Time
	elapsed   time.Duration
}

type jobManager struct {
	mu            sync.Mutex
	jobs          []*job
	nextID        int
	names         map[string]int // name → job ID
	notifications []string
	unreadDone    int
	onComplete    func(*job) // called after each job finishes (outside lock)
}

func newJobManager() *jobManager {
	return &jobManager{names: make(map[string]int)}
}

// start launches fn in a goroutine, tracks the job, and returns its ID.
// name is optional; pass "" for an unnamed job. Returns an error if the name
// is already in use by another job in this session.
func (m *jobManager) start(display, name string, fn func() (string, error)) (int, error) {
	m.mu.Lock()
	if name != "" {
		if _, exists := m.names[name]; exists {
			m.mu.Unlock()
			return 0, fmt.Errorf("job name %q is already in use (try /jobs to see active jobs)", name)
		}
	}
	m.nextID++
	id := m.nextID
	j := &job{
		id:        id,
		name:      name,
		display:   display,
		status:    jobRunning,
		startedAt: time.Now(),
	}
	m.jobs = append(m.jobs, j)
	if name != "" {
		m.names[name] = id
	}
	m.mu.Unlock()

	go func() {
		output, err := fn()
		elapsed := time.Since(j.startedAt)

		m.mu.Lock()
		j.elapsed = elapsed
		nameTag := ""
		if j.name != "" {
			nameTag = fmt.Sprintf(" [%s]", j.name)
		}
		if err != nil {
			j.status = jobFailed
			j.err = err
			m.notifications = append(m.notifications,
				fmt.Sprintf("[%d]%s failed: %s — %v", id, nameTag, display, err))
		} else {
			j.status = jobDone
			j.output = output
			sizeStr := sizeLabel(len(output))
			m.notifications = append(m.notifications,
				fmt.Sprintf("[%d]%s done: %s%s  %.1fs", id, nameTag, display, sizeStr, elapsed.Seconds()))
		}
		m.unreadDone++
		cb := m.onComplete
		m.mu.Unlock()

		if cb != nil {
			cb(j)
		}
	}()

	return id, nil
}

// resolve looks up a job by numeric ID or by name. Returns nil if not found.
func (m *jobManager) resolve(idOrName string) *job {
	m.mu.Lock()
	defer m.mu.Unlock()
	if n, err := strconv.Atoi(idOrName); err == nil {
		for _, j := range m.jobs {
			if j.id == n {
				return j
			}
		}
		return nil
	}
	id, ok := m.names[idOrName]
	if !ok {
		return nil
	}
	for _, j := range m.jobs {
		if j.id == id {
			return j
		}
	}
	return nil
}

// drain returns and clears all pending notification strings.
func (m *jobManager) drain() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.notifications) == 0 {
		return nil
	}
	n := m.notifications
	m.notifications = nil
	return n
}

// activity returns counts of running and unread-done jobs for the prompt.
func (m *jobManager) activity() (running, unreadDone int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, j := range m.jobs {
		if j.status == jobRunning {
			running++
		}
	}
	return running, m.unreadDone
}

// markRead clears the unread-done counter (called after /jobs).
func (m *jobManager) markRead() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.unreadDone = 0
}

// list returns a snapshot of all jobs (safe to read after lock released).
func (m *jobManager) list() []*job {
	m.mu.Lock()
	defer m.mu.Unlock()
	cp := make([]*job, len(m.jobs))
	copy(cp, m.jobs)
	return cp
}

// isValidJobName reports whether s is a legal job tag: non-numeric,
// non-empty, ≤32 chars, containing only letters, digits, hyphens, underscores.
func isValidJobName(s string) bool {
	if s == "" || len(s) > 32 {
		return false
	}
	if _, err := strconv.Atoi(s); err == nil {
		return false // purely numeric would be ambiguous with job IDs
	}
	for _, c := range s {
		if !unicode.IsLetter(c) && !unicode.IsDigit(c) && c != '-' && c != '_' {
			return false
		}
	}
	return true
}

func sizeLabel(n int) string {
	if n == 0 {
		return ""
	}
	if n < 1024 {
		return fmt.Sprintf("  %dB", n)
	}
	return fmt.Sprintf("  %.1fKB", float64(n)/1024)
}
