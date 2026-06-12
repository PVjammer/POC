package shell

import (
	"fmt"
	"sync"
	"time"
)

type jobStatus int

const (
	jobRunning jobStatus = iota
	jobDone
	jobFailed
)

type job struct {
	id        int
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
	notifications []string
	unreadDone    int
}

func newJobManager() *jobManager {
	return &jobManager{}
}

// start launches fn in a goroutine, tracks the job, and returns its ID.
func (m *jobManager) start(display string, fn func() (string, error)) int {
	m.mu.Lock()
	m.nextID++
	id := m.nextID
	j := &job{
		id:        id,
		display:   display,
		status:    jobRunning,
		startedAt: time.Now(),
	}
	m.jobs = append(m.jobs, j)
	m.mu.Unlock()

	go func() {
		output, err := fn()
		elapsed := time.Since(j.startedAt)

		m.mu.Lock()
		defer m.mu.Unlock()
		j.elapsed = elapsed
		if err != nil {
			j.status = jobFailed
			j.err = err
			m.notifications = append(m.notifications,
				fmt.Sprintf("[%d] failed: %s — %v", id, display, err))
		} else {
			j.status = jobDone
			j.output = output
			sizeStr := sizeLabel(len(output))
			m.notifications = append(m.notifications,
				fmt.Sprintf("[%d] done: %s%s  %.1fs", id, display, sizeStr, elapsed.Seconds()))
		}
		m.unreadDone++
	}()

	return id
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

// get returns the job with the given ID, or nil.
func (m *jobManager) get(id int) *job {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, j := range m.jobs {
		if j.id == id {
			return j
		}
	}
	return nil
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
