package cli

import (
	"fmt"
	"os"

	"github.com/pvjammer/ai-shell-poc/config"
)

// runJobs implements `baish jobs` — lists persisted completed jobs.
func runJobs(args []string) error {
	if len(args) > 0 && (args[0] == "--help" || args[0] == "-h") {
		fmt.Println("usage: baish jobs")
		fmt.Println("  Lists completed background jobs from the persistent job store.")
		fmt.Println("  Running jobs (in an interactive shell) are not shown here.")
		return nil
	}

	records, err := config.LoadJobs()
	if err != nil {
		return fmt.Errorf("load jobs: %w", err)
	}
	if len(records) == 0 {
		fmt.Println("(no completed jobs)")
		return nil
	}

	fmt.Printf("  %-4s  %-8s  %-6s  %s\n", "ID", "STATUS", "SIZE", "COMMAND")
	for _, r := range records {
		status := "done"
		if r.Failed {
			status = "failed"
		}
		name := ""
		if r.Name != "" {
			name = " [" + r.Name + "]"
		}
		size := ""
		if r.Size > 0 {
			size = humanSize(r.Size)
		}
		display := r.Display
		if len(display) > 60 {
			display = display[:57] + "..."
		}
		fmt.Printf("  %-4d  %-8s  %-6s  %s%s\n", r.ID, status, size, display, name)
	}
	return nil
}

// runJob implements `baish job <id|name>` — prints a job's output.
func runJob(args []string) error {
	if len(args) == 0 || args[0] == "--help" || args[0] == "-h" {
		fmt.Fprintln(os.Stderr, "usage: baish job <id|name>")
		return fmt.Errorf("missing job id or name")
	}

	output, found, err := config.LoadJobOutput(args[0])
	if err != nil {
		return fmt.Errorf("read job: %w", err)
	}
	if !found {
		return fmt.Errorf("job %q not found", args[0])
	}
	fmt.Print(output)
	if len(output) > 0 && output[len(output)-1] != '\n' {
		fmt.Println()
	}
	return nil
}
