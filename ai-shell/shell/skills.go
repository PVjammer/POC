package shell

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/pvjammer/ai-shell-poc/config"
	"github.com/pvjammer/ai-shell-poc/skills"
)

// runSkillCmd handles the /skill meta command.
//
//	/skill list                     list installed skills
//	/skill install <path> [flags]   install a skill from a directory
//	/skill remove <name>            remove an installed skill
//	/skill enable <name>            remove from default agent excluded_skills
//	/skill disable <name>           add to default agent excluded_skills
func (s *Shell) runSkillCmd(args []string) {
	if len(args) == 0 || args[0] == "list" {
		s.skillList()
		return
	}
	sub := args[0]
	rest := args[1:]

	switch sub {
	case "--help", "-h", "help":
		fmt.Println("usage:")
		fmt.Println("  /skill list                      list installed skills")
		fmt.Println("  /skill install <path> [--no-command] [--agent <name>]")
		fmt.Println("                                   install from directory")
		fmt.Println("  /skill create <name>             scaffold a new skill, open $EDITOR")
		fmt.Println("  /skill edit <name>               open skill's SKILL.md in $EDITOR")
		fmt.Println("  /skill remove <name>             remove skill")
		fmt.Println("  /skill enable <name>             enable for default agent")
		fmt.Println("  /skill disable <name>            disable for default agent")
	case "create":
		if len(rest) == 0 {
			fmt.Fprintln(os.Stderr, "usage: /skill create <name>")
			return
		}
		s.skillCreate(rest[0])
	case "edit":
		if len(rest) == 0 {
			fmt.Fprintln(os.Stderr, "usage: /skill edit <name>")
			return
		}
		s.skillEdit(rest[0])
	case "install":
		if len(rest) == 0 {
			fmt.Fprintln(os.Stderr, "usage: /skill install <path> [--no-command] [--agent <name>]")
			return
		}
		opts := parseSkillInstallFlags(rest[1:])
		if err := s.installSkill(rest[0], opts); err != nil {
			fmt.Fprintf(os.Stderr, "skill install: %v\n", err)
		}
	case "remove":
		if len(rest) == 0 {
			fmt.Fprintln(os.Stderr, "usage: /skill remove <name>")
			return
		}
		if err := s.removeSkill(rest[0]); err != nil {
			fmt.Fprintf(os.Stderr, "skill remove: %v\n", err)
		}
	case "enable":
		if len(rest) == 0 {
			fmt.Fprintln(os.Stderr, "usage: /skill enable <name>")
			return
		}
		if err := s.setSkillEnabled(rest[0], true); err != nil {
			fmt.Fprintf(os.Stderr, "skill enable: %v\n", err)
		} else {
			fmt.Printf("skill %q enabled for default agent\n", rest[0])
			s.onConfigChange(config.AgentsFile())
		}
	case "disable":
		if len(rest) == 0 {
			fmt.Fprintln(os.Stderr, "usage: /skill disable <name>")
			return
		}
		if err := s.setSkillEnabled(rest[0], false); err != nil {
			fmt.Fprintf(os.Stderr, "skill disable: %v\n", err)
		} else {
			fmt.Printf("skill %q disabled for default agent\n", rest[0])
			s.onConfigChange(config.AgentsFile())
		}
	default:
		fmt.Fprintf(os.Stderr, "skill: unknown subcommand %q  (try /skill --help)\n", sub)
	}
}

// skillList prints a table of all installed skills with their status.
func (s *Shell) skillList() {
	all := s.skillLoader.All()
	if len(all) == 0 {
		fmt.Println("no skills installed")
		return
	}

	// Determine which skills are registered as commands.
	cmdSkills := make(map[string]bool)
	for _, cmd := range s.commandRegistry {
		if cmd.Skill != "" {
			cmdSkills[cmd.Skill] = true
		}
	}

	// Determine which are excluded from the default agent.
	excluded := make(map[string]bool)
	if dflt, ok := s.agentRegistry["default"]; ok {
		for _, n := range dflt.ExcludedSkills {
			excluded[n] = true
		}
	}

	fmt.Printf("%-25s %-16s %s\n", "NAME", "STATUS", "DESCRIPTION")
	fmt.Printf("%-25s %-16s %s\n", strings.Repeat("-", 25), strings.Repeat("-", 16), "-----------")
	for _, r := range all {
		status := "catalog"
		if cmdSkills[r.Name] {
			if excluded[r.Name] {
				status = "command"
			} else {
				status = "catalog+command"
			}
		} else if excluded[r.Name] {
			status = "disabled"
		}
		desc := r.Description
		if len(desc) > 50 {
			desc = desc[:47] + "..."
		}
		fmt.Printf("%-25s %-16s %s\n", r.Name, status, desc)
	}
}

type skillInstallOpts struct {
	noCommand   bool
	commandOnly bool
	agentName   string
}

func parseSkillInstallFlags(args []string) skillInstallOpts {
	var opts skillInstallOpts
	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--no-command":
			opts.noCommand = true
		case "--command-only":
			opts.commandOnly = true
		case "--agent":
			if i+1 < len(args) {
				i++
				opts.agentName = args[i]
			}
		}
	}
	return opts
}

// installSkill copies the skill directory to ~/.config/baish/skills/<name>/
// and optionally registers it as a command in commands.toml.
func (s *Shell) installSkill(srcPath string, opts skillInstallOpts) error {
	srcPath = filepath.Clean(srcPath)
	if _, err := os.Stat(filepath.Join(srcPath, "SKILL.md")); err != nil {
		return fmt.Errorf("%s: no SKILL.md found", srcPath)
	}

	// Parse name from frontmatter; look up by dir name first.
	tmp := skills.NewLoader()
	_ = tmp.Scan([]string{filepath.Dir(srcPath)})
	rec, ok := tmp.Get(filepath.Base(srcPath))
	if !ok {
		for _, r := range tmp.All() {
			if r.Dir == srcPath {
				rec = r
				ok = true
				break
			}
		}
	}
	if !ok {
		return fmt.Errorf("cannot determine skill name from %s — ensure 'name' field in frontmatter matches directory name", srcPath)
	}

	// Destination.
	destDir := filepath.Join(config.SkillsDir(), rec.Name)
	if err := os.MkdirAll(destDir, 0755); err != nil {
		return fmt.Errorf("create skill dir: %w", err)
	}

	// Copy all files from srcPath to destDir.
	if err := copyDir(srcPath, destDir); err != nil {
		return fmt.Errorf("copy skill: %w", err)
	}

	fmt.Printf("✓ skill %q installed to %s\n", rec.Name, destDir)

	if !opts.noCommand {
		agentName := opts.agentName
		if agentName == "" {
			agentName = "default"
		}
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

	// Reload skills.
	s.onConfigChange(destDir)
	return nil
}

// removeSkill removes an installed skill and any associated command entry.
func (s *Shell) removeSkill(name string) error {
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

	s.onConfigChange(destDir)
	return nil
}

// setSkillEnabled adds or removes a skill from [agents.default] excluded_skills.
func (s *Shell) setSkillEnabled(name string, enable bool) error {
	if _, ok := s.skillLoader.Get(name); !ok {
		return fmt.Errorf("skill %q not found", name)
	}

	dflt := s.agentRegistry["default"]
	excluded := make([]string, 0, len(dflt.ExcludedSkills))
	for _, n := range dflt.ExcludedSkills {
		if n != name {
			excluded = append(excluded, n)
		}
	}
	if !enable {
		excluded = append(excluded, name)
	}
	dflt.ExcludedSkills = excluded
	s.agentRegistry["default"] = dflt

	// Persist to agents.toml.
	if err := config.SaveAgentDefault(dflt); err != nil {
		// Non-fatal: in-memory is updated, disk write failed.
		fmt.Fprintf(os.Stderr, "  warning: could not persist change: %v\n", err)
	}
	return nil
}

// skillCreate scaffolds a new skill directory and opens SKILL.md in $EDITOR.
func (s *Shell) skillCreate(name string) {
	dir := filepath.Join(config.SkillsDir(), name)
	mdPath := filepath.Join(dir, "SKILL.md")

	template := fmt.Sprintf(`---
name: %s
description: One-line description shown in /skill list and the agent catalog.
# argument-hint: <optional arg hint shown in tab completion>
# allowed-tools: bash read_file   # space-separated; parsed but not enforced yet
# disable-model-invocation: false  # true = only usable as /command, not by the agent
---

<!-- Full skill instructions go here. The agent reads this when it calls use_skill("%s"). -->
<!-- Use $ARGUMENTS for any args passed to a /command, ${SKILL_DIR} for this directory. -->

`, name, name)

	if err := ensureFile(mdPath, template); err != nil {
		fmt.Fprintf(os.Stderr, "skill create: %v\n", err)
		return
	}
	fmt.Printf("skill: opening %s\n", mdPath)
	if err := openInEditor(mdPath); err != nil {
		fmt.Fprintf(os.Stderr, "skill: editor: %v\n", err)
		return
	}
	s.onConfigChange(dir)
}

// skillEdit opens an installed skill's SKILL.md in $EDITOR.
func (s *Shell) skillEdit(name string) {
	rec, ok := s.skillLoader.Get(name)
	if !ok {
		fmt.Fprintf(os.Stderr, "skill: %q not found (try /skill list)\n", name)
		return
	}
	mdPath := filepath.Join(rec.Dir, "SKILL.md")
	fmt.Printf("skill: opening %s\n", mdPath)
	if err := openInEditor(mdPath); err != nil {
		fmt.Fprintf(os.Stderr, "skill: editor: %v\n", err)
		return
	}
	s.onConfigChange(mdPath)
}

// copyDir recursively copies src to dst.
func copyDir(src, dst string) error {
	return filepath.Walk(src, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(src, path)
		target := filepath.Join(dst, rel)
		if info.IsDir() {
			return os.MkdirAll(target, info.Mode())
		}
		return copyFile(path, target)
	})
}

func copyFile(src, dst string) error {
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
