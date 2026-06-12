// Package permissions classifies agent bash commands into three tiers and
// prompts the user before running confirm-tier commands.
//
// Tiers:
//
//	auto    — runs immediately (read-only commands, git inspection, etc.)
//	confirm — user must approve before the agent runs the command
//	deny    — blocked entirely; agent receives an error message
package permissions

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

// Tier indicates how a command should be handled before execution.
type Tier int

const (
	TierAuto    Tier = iota // run without asking
	TierConfirm             // prompt user before running
	TierDeny                // refuse; return error to agent
)

func (t Tier) String() string {
	switch t {
	case TierAuto:
		return "auto"
	case TierConfirm:
		return "confirm"
	case TierDeny:
		return "deny"
	default:
		return "unknown"
	}
}

// Classify returns the permission tier for a raw bash command string.
// Deny patterns are checked first, then auto, defaulting to confirm.
func Classify(command string) Tier {
	cmd := strings.TrimSpace(command)

	for _, pat := range denyPatterns {
		if matchPattern(cmd, pat) {
			return TierDeny
		}
	}

	for _, pat := range autoPatterns {
		if matchPattern(cmd, pat) {
			return TierAuto
		}
	}

	return TierConfirm
}

// Prompt shows a confirm prompt on /dev/tty and returns true if the user
// approves. Falls back to denying if the terminal cannot be opened.
func Prompt(command string) bool {
	tty, err := os.OpenFile("/dev/tty", os.O_RDWR, 0)
	if err != nil {
		// Can't reach the terminal — safe default is to deny.
		fmt.Fprintf(os.Stderr, "\npermissions: cannot open /dev/tty, denying: %s\n", command)
		return false
	}
	defer tty.Close()

	fmt.Fprintf(tty, "\n  \033[33m[agent]\033[0m about to run: \033[1m%s\033[0m\n  allow? [y/N] ", command)

	scanner := bufio.NewScanner(tty)
	if !scanner.Scan() {
		return false
	}
	answer := strings.TrimSpace(strings.ToLower(scanner.Text()))
	return answer == "y" || answer == "yes"
}

// Check classifies command and prompts if needed. Returns an error string for
// the agent if the command is denied or the user declines, or "" if the
// command should proceed.
func Check(command string) string {
	switch Classify(command) {
	case TierDeny:
		return fmt.Sprintf("permission denied: command is blocked by policy: %s", command)
	case TierConfirm:
		if !Prompt(command) {
			return fmt.Sprintf("user declined: %s", command)
		}
	}
	return "" // auto or user approved
}

// matchPattern returns true if cmd contains or starts with the pattern.
// Patterns starting with ^ match only at the start of the command.
func matchPattern(cmd, pat string) bool {
	if strings.HasPrefix(pat, "^") {
		return strings.HasPrefix(cmd, pat[1:])
	}
	return strings.Contains(cmd, pat)
}

// ── Built-in tier tables ──────────────────────────────────────────────────────

// denyPatterns matches commands that should never run.
// Checked before autoPatterns.
var denyPatterns = []string{
	"rm -rf /",
	"rm -rf /*",
	":(){ :|:& };:", // fork bomb
	"> /dev/sda",
	"dd if=",
	"^sudo ",
	"^su ",
	"^sudo\t",
}

// autoPatterns matches commands that are safe to run without prompting.
// Read-only operations, inspection, output formatting.
var autoPatterns = []string{
	// Navigation and listing
	"^ls", "^ll", "^la", "^l ",
	"^pwd", "^cd ",
	"^tree",
	// File reading
	"^cat ", "^head ", "^tail ", "^less ", "^more ",
	"^file ", "^stat ", "^wc ",
	// Search
	"^grep ", "^rg ", "^ripgrep ", "^find ", "^fd ",
	"^awk ", "^sed ",
	// Git read-only
	"^git status", "^git log", "^git diff", "^git show",
	"^git branch", "^git remote", "^git stash list",
	"^git tag", "^git describe",
	// Text processing
	"^echo ", "^printf ",
	"^sort ", "^uniq ", "^cut ", "^tr ",
	"^jq ", "^yq ",
	// System info (read-only)
	"^ps ", "^ps\n", "ps aux", "ps -",
	"^df ", "^du ",
	"^env", "^printenv", "^which ", "^type ",
	"^uname", "^hostname", "^whoami", "^id",
	"^date", "^uptime",
	// Process substitution / piped reads
	"^strings ", "^xxd ", "^hexdump ",
}
