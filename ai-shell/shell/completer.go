package shell

import (
	"os"
	"os/exec"
	"strings"
	"unicode"
)

// HybridCompleter handles tab completion for:
//   - /slash-commands  (in-process, instant)
//   - bash commands and their arguments (delegated to bash's completion system)
type HybridCompleter struct {
	slashItems []string // slash names without the leading /
}

// NewHybridCompleter creates a completer. slashItems should contain the bare
// names (e.g. "help", "tools", "summarize") — the / prefix is added on match.
func NewHybridCompleter(items []string) *HybridCompleter {
	return &HybridCompleter{slashItems: items}
}

// Do implements readline.AutoCompleter.
//
// readline calls this on Tab; it returns the list of completion suffixes and
// the number of characters before the cursor to replace.
func (c *HybridCompleter) Do(line []rune, pos int) ([][]rune, int) {
	lineStr := string(line[:pos])
	trimmed := strings.TrimSpace(lineStr)

	// /slash-command completion — handled in-process.
	if strings.HasPrefix(trimmed, "/") {
		partial := trimmed[1:]
		var matches [][]rune
		for _, item := range c.slashItems {
			if strings.HasPrefix(item, partial) {
				matches = append(matches, []rune(item[len(partial):]))
			}
		}
		return matches, len([]rune(partial))
	}

	// Don't complete ?queries or empty lines.
	if trimmed == "" || strings.HasPrefix(trimmed, "?") {
		return nil, 0
	}

	// Delegate everything else to bash's programmable completion.
	completions := getBashCompletions(trimmed)
	if len(completions) == 0 {
		return nil, 0
	}

	cur := currentWord(trimmed)
	var results [][]rune
	for _, comp := range completions {
		if strings.HasPrefix(comp, cur) {
			results = append(results, []rune(comp[len(cur):]))
		}
	}
	return results, len([]rune(cur))
}

// getBashCompletions spawns bash with its programmable completion framework
// loaded and returns the completions for the current word in lineStr.
//
// For the first word it uses `compgen -c` (command names).
// For subsequent words it loads and invokes the command's completion function
// (e.g. __git_wrap__git_main for git), falling back to file completion.
var bashCompletionScript = `
for f in /usr/share/bash-completion/bash_completion \
         /etc/bash_completion \
         /usr/local/share/bash-completion/bash_completion; do
    [[ -f "$f" ]] && source "$f" 2>/dev/null && break
done

line="$_COMP_LINE"
read -ra _all <<< "$line"

if [[ "$line" =~ [[:space:]]$ ]]; then
    cur=""
    prev="${_all[-1]:-}"
    COMP_WORDS=("${_all[@]}" "")
    COMP_CWORD=${#_all[@]}
else
    if [[ ${#_all[@]} -gt 0 ]]; then
        cur="${_all[-1]}"
        unset '_all[-1]'
    else
        cur=""
    fi
    prev="${_all[-1]:-}"
    COMP_WORDS=("${_all[@]}" "$cur")
    COMP_CWORD=${#_all[@]}
fi

COMP_LINE="$line"
COMP_POINT=${#line}
cmd="${COMP_WORDS[0]:-}"

if [[ -z "$cmd" || ( "${#COMP_WORDS[@]}" -eq 1 && -n "$cur" ) ]]; then
    compgen -c -- "$cur"
else
    _completion_loader "$cmd" 2>/dev/null
    _cfunc=$(complete -p "$cmd" 2>/dev/null | sed -n 's/.*-F \([^ ]*\).*/\1/p')
    if [[ -n "$_cfunc" ]]; then
        "$_cfunc" "$cmd" "$cur" "$prev" 2>/dev/null
        printf '%s\n' "${COMPREPLY[@]}"
    else
        compgen -f -- "$cur"
    fi
fi
`

func getBashCompletions(lineStr string) []string {
	cmd := exec.Command("bash", "--norc", "--noprofile", "-c", bashCompletionScript)
	cmd.Env = append(os.Environ(), "_COMP_LINE="+lineStr)

	out, err := cmd.Output()
	if err != nil || len(out) == 0 {
		return nil
	}

	seen := make(map[string]bool)
	var results []string
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		line = strings.TrimSpace(line)
		if line != "" && !seen[line] {
			seen[line] = true
			results = append(results, line)
		}
	}
	return results
}

// currentWord returns the word currently being typed (empty if line ends in whitespace).
func currentWord(s string) string {
	if len(s) == 0 || unicode.IsSpace(rune(s[len(s)-1])) {
		return ""
	}
	fields := strings.Fields(s)
	if len(fields) == 0 {
		return ""
	}
	return fields[len(fields)-1]
}
