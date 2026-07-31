package cli

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/pvjammer/ai-shell-poc/config"
)

// runSession implements `baish session <sub>`.
func runSession(args []string) error {
	sub := "list"
	if len(args) > 0 {
		sub = args[0]
	}
	rest := args
	if len(args) > 1 {
		rest = args[1:]
	} else {
		rest = nil
	}

	switch sub {
	case "list", "ls":
		return sessionList()
	case "clear", "delete", "rm":
		if len(rest) == 0 {
			return fmt.Errorf("usage: baish session clear <name>  (or 'all')")
		}
		return sessionClear(rest[0])
	case "--help", "-h", "help":
		fmt.Println("usage: baish session <subcommand>")
		fmt.Println("  list              list persisted sessions")
		fmt.Println("  clear <name|all>  delete a session history file")
		return nil
	default:
		return fmt.Errorf("session: unknown subcommand %q  (list|clear)", sub)
	}
}

func sessionList() error {
	dir := config.SessionsDir()
	entries, err := os.ReadDir(dir)
	if os.IsNotExist(err) {
		fmt.Println("(no persisted sessions)")
		return nil
	}
	if err != nil {
		return err
	}
	found := false
	for _, e := range entries {
		if e.IsDir() || filepath.Ext(e.Name()) != ".json" {
			continue
		}
		info, _ := e.Info()
		name := e.Name()[:len(e.Name())-5]
		size := int64(0)
		if info != nil {
			size = info.Size()
		}
		fmt.Printf("  %-20s  %s\n", name, humanSize(int(size)))
		found = true
	}
	if !found {
		fmt.Println("(no persisted sessions)")
	}
	return nil
}

func sessionClear(name string) error {
	dir := config.SessionsDir()
	if name == "all" {
		entries, err := os.ReadDir(dir)
		if os.IsNotExist(err) {
			fmt.Println("(no sessions to clear)")
			return nil
		}
		if err != nil {
			return err
		}
		for _, e := range entries {
			if !e.IsDir() && filepath.Ext(e.Name()) == ".json" {
				os.Remove(filepath.Join(dir, e.Name()))
			}
		}
		fmt.Println("session: all histories cleared")
		return nil
	}
	path := config.SessionPath(name)
	if err := os.Remove(path); err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("session %q not found", name)
		}
		return err
	}
	fmt.Printf("session: cleared %q\n", name)
	return nil
}
