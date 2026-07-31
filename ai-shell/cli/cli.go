// Package cli implements baish's standalone subcommand interface.
// When the baish binary is invoked with a recognised subcommand as its first
// argument it dispatches here instead of starting the interactive shell.
//
// Usage:
//
//	baish ask "question"               advisory query, stream answer to stdout
//	baish do  "task"                   agentic task with tool use
//	baish cm                           generate a commit message
//	baish ctx set  <name> < file       set a context slot
//	baish ctx add  <name> < file       append to a context slot
//	baish ctx show <name>              print a context slot
//	baish ctx list                     list all slots
//	baish ctx clear [name]             remove one or all slots
//	baish jobs                         list persisted jobs
//	baish job  <id|name>               print output of a job
//	baish session list                 list persisted sessions
//	baish session clear [name]         remove session history
package cli

import (
	"fmt"
	"os"
)

// Subcommands is the set of first-argument values that trigger CLI mode.
var Subcommands = map[string]bool{
	"ask": true, "do": true, "cm": true,
	"ctx": true, "jobs": true, "job": true,
	"session": true, "help": true,
}

// Run dispatches the given args (starting with the subcommand name).
func Run(args []string) error {
	if len(args) == 0 {
		printHelp()
		return nil
	}
	sub := args[0]
	rest := args[1:]

	switch sub {
	case "ask":
		return runAsk(rest, false)
	case "do":
		return runAsk(rest, true)
	case "cm":
		return runCm(rest)
	case "ctx":
		return runCtx(rest)
	case "jobs":
		return runJobs(rest)
	case "job":
		return runJob(rest)
	case "session":
		return runSession(rest)
	case "help", "--help", "-h":
		printHelp()
		return nil
	default:
		fmt.Fprintf(os.Stderr, "baish: unknown subcommand %q\n", sub)
		printHelp()
		return fmt.Errorf("unknown subcommand")
	}
}

func printHelp() {
	fmt.Println("usage: baish <subcommand> [args]")
	fmt.Println()
	fmt.Println("Subcommands:")
	fmt.Println("  ask <prompt>              advisory query — stream answer to stdout")
	fmt.Println("  do  <prompt>              agentic — AI executes commands with tool use")
	fmt.Println("  cm                        generate a commit message from staged changes")
	fmt.Println("  ctx set  <name>           set a context slot from stdin")
	fmt.Println("  ctx add  <name>           append to a context slot from stdin")
	fmt.Println("  ctx show <name>           print a context slot")
	fmt.Println("  ctx list                  list all slots")
	fmt.Println("  ctx clear [name]          remove one or all slots")
	fmt.Println("  jobs                      list completed background jobs")
	fmt.Println("  job <id|name>             print output of a job")
	fmt.Println("  session list              list persisted sessions")
	fmt.Println("  session clear [name]      remove session history files")
	fmt.Println()
	fmt.Println("Flags (ask / do):")
	fmt.Println("  --session <name>          use a named session (default: main)")
	fmt.Println("  --model   <name>          override model")
	fmt.Println("  --endpoint <url>          override endpoint")
	fmt.Println()
	fmt.Println("Without a subcommand, baish starts the interactive shell.")
}
