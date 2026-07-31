package cli

import (
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/pvjammer/ai-shell-poc/config"
)

// runCtx implements `baish ctx <sub> [name]`.
func runCtx(args []string) error {
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "usage: baish ctx <set|add|show|list|clear> [name]")
		return fmt.Errorf("missing subcommand")
	}
	sub := args[0]
	name := "default"
	if len(args) > 1 {
		name = args[1]
	}

	switch sub {
	case "set", "add":
		stat, _ := os.Stdin.Stat()
		if stat.Mode()&os.ModeCharDevice != 0 {
			return fmt.Errorf("ctx %s: pipe content into this command — e.g. cat file.md | baish ctx %s %s", sub, sub, name)
		}
		incoming, err := io.ReadAll(os.Stdin)
		if err != nil {
			return fmt.Errorf("read stdin: %w", err)
		}
		content := strings.TrimSpace(string(incoming))
		if sub == "add" {
			existing, _ := config.LoadContexts()
			if prev, ok := existing[name]; ok && prev != "" {
				content = prev + "\n\n" + content
			}
		}
		if err := config.SaveContext(name, content); err != nil {
			return fmt.Errorf("save ctx: %w", err)
		}
		fmt.Printf("ctx: %s %q (%s)\n", sub, name, humanSize(len(content)))

	case "show":
		slots, err := config.LoadContexts()
		if err != nil {
			return err
		}
		slot, ok := slots[name]
		if !ok {
			return fmt.Errorf("ctx: no slot %q  (try: baish ctx list)", name)
		}
		fmt.Println(slot)

	case "list":
		slots, err := config.LoadContexts()
		if err != nil {
			return err
		}
		if len(slots) == 0 {
			fmt.Println("(no context slots)")
			return nil
		}
		appCfg, _ := config.Load()
		for k, v := range slots {
			mode := "inline"
			if len(v) > appCfg.CtxInlineThreshold {
				mode = "stub"
			}
			fmt.Printf("  %-20s  %-8s  %s\n", k, humanSize(len(v)), mode)
		}

	case "clear":
		if len(args) > 1 {
			if err := config.DeleteContext(name); err != nil {
				return err
			}
			fmt.Printf("ctx: cleared %q\n", name)
		} else {
			if err := config.ClearContexts(); err != nil {
				return err
			}
			fmt.Println("ctx: all slots cleared")
		}

	default:
		return fmt.Errorf("ctx: unknown subcommand %q  (set|add|show|list|clear)", sub)
	}
	return nil
}
