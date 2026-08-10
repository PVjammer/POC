package cli

import (
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/pvjammer/ai-shell-poc/config"
	"github.com/pvjammer/ai-shell-poc/skills"
)

// runSkillCLI handles `baish skill <subcommand>`.
//
//	baish skill install <path> [--no-command] [--agent <name>]
//	baish skill list
//	baish skill remove <name>
func runSkillCLI(args []string) error {
	if len(args) == 0 || args[0] == "help" || args[0] == "--help" {
		printSkillHelp()
		return nil
	}
	sub := args[0]
	rest := args[1:]

	var err error
	switch sub {
	case "install":
		if len(rest) == 0 {
			err = fmt.Errorf("usage: baish skill install <path> [--no-command] [--agent <name>]")
		} else {
			err = cliInstallSkill(rest[0], rest[1:])
		}
	case "list":
		err = cliListSkills()
	case "remove":
		if len(rest) == 0 {
			err = fmt.Errorf("usage: baish skill remove <name>")
		} else {
			err = cliRemoveSkill(rest[0])
		}
	default:
		err = fmt.Errorf("skill: unknown subcommand %q  (try: baish skill --help)", sub)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "baish skill %s: %v\n", sub, err)
		return err
	}
	return nil
}

func printSkillHelp() {
	fmt.Println("usage: baish skill <subcommand>")
	fmt.Println()
	fmt.Println("Subcommands:")
	fmt.Println("  install <path> [flags]   install a skill from a directory")
	fmt.Println("  list                     list installed skills")
	fmt.Println("  remove <name>            remove an installed skill")
	fmt.Println()
	fmt.Println("Install flags:")
	fmt.Println("  --no-command             install to agent catalog only (no /command)")
	fmt.Println("  --agent <name>           use named agent for the command (default: default)")
}

func cliInstallSkill(srcPath string, flags []string) error {
	noCommand := false
	agentName := "default"
	for i := 0; i < len(flags); i++ {
		switch flags[i] {
		case "--no-command":
			noCommand = true
		case "--agent":
			if i+1 < len(flags) {
				i++
				agentName = flags[i]
			}
		}
	}

	srcPath = filepath.Clean(srcPath)
	mdPath := filepath.Join(srcPath, "SKILL.md")
	data, err := os.ReadFile(mdPath)
	if err != nil {
		return fmt.Errorf("%s: no SKILL.md found", srcPath)
	}

	// Parse directly — don't use Scan so we don't accidentally pick a sibling skill.
	tmp := skills.NewLoader()
	_ = tmp.Scan([]string{filepath.Dir(srcPath)})
	rec, ok := tmp.Get(filepath.Base(srcPath))
	if !ok {
		// Fallback: parse frontmatter directly to extract name.
		recs := tmp.All()
		// Find the one whose Dir matches srcPath.
		for _, r := range recs {
			if r.Dir == srcPath {
				rec = r
				ok = true
				break
			}
		}
	}
	if !ok {
		// Last resort: name from directory name, validate SKILL.md exists.
		_ = data // already read
		return fmt.Errorf("cannot determine skill name from %s — ensure 'name' field in frontmatter matches directory name", srcPath)
	}

	destDir := filepath.Join(config.SkillsDir(), rec.Name)
	if err := os.MkdirAll(destDir, 0755); err != nil {
		return fmt.Errorf("create skill dir: %w", err)
	}
	if err := cliCopyDir(srcPath, destDir); err != nil {
		return fmt.Errorf("copy skill: %w", err)
	}
	fmt.Printf("✓ skill %q installed to %s\n", rec.Name, destDir)

	if !noCommand {
		cmd := config.CommandConfig{
			Name:        rec.Name,
			Description: rec.Description,
			Skill:       rec.Name,
			Agent:       agentName,
		}
		if err := config.AppendCommand(cmd); err != nil {
			fmt.Fprintf(os.Stderr, "  warning: could not register /%s as command: %v\n", rec.Name, err)
		} else {
			fmt.Printf("✓ command /%s registered\n", rec.Name)
		}
	}
	return nil
}

func cliListSkills() error {
	loader := skills.NewLoader()
	errs := loader.Scan([]string{
		config.SkillsDir(),
		filepath.Join(".baish", "skills"),
	})
	for _, e := range errs {
		fmt.Fprintf(os.Stderr, "warning: %v\n", e)
	}

	all := loader.All()
	if len(all) == 0 {
		fmt.Println("no skills installed")
		return nil
	}

	fmt.Printf("%-25s %s\n", "NAME", "DESCRIPTION")
	fmt.Printf("%-25s %s\n", "----", "-----------")
	for _, r := range all {
		desc := r.Description
		if len(desc) > 60 {
			desc = desc[:57] + "..."
		}
		fmt.Printf("%-25s %s\n", r.Name, desc)
	}
	return nil
}

func cliRemoveSkill(name string) error {
	destDir := filepath.Join(config.SkillsDir(), name)
	if _, err := os.Stat(destDir); os.IsNotExist(err) {
		return fmt.Errorf("skill %q not found", name)
	}
	if err := os.RemoveAll(destDir); err != nil {
		return fmt.Errorf("remove skill dir: %w", err)
	}
	fmt.Printf("✓ skill %q removed\n", name)

	if err := config.RemoveCommand(name); err != nil {
		fmt.Fprintf(os.Stderr, "  warning: could not remove command: %v\n", err)
	} else {
		fmt.Printf("✓ command /%s removed\n", name)
	}
	return nil
}

func cliCopyDir(src, dst string) error {
	return filepath.Walk(src, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(src, path)
		target := filepath.Join(dst, rel)
		if info.IsDir() {
			return os.MkdirAll(target, info.Mode())
		}
		return cliCopyFile(path, target)
	})
}

func cliCopyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer out.Close()
	_, err = io.Copy(out, in)
	return err
}
