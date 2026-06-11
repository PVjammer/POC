// Package functions loads ai-sdk-go builtin AIFunctions and exposes them in two ways:
//  1. As llm.ToolDef + handler pairs so the agent can call them as tools.
//  2. As executable /commands in the shell (via cobra CLI builder).
package functions

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"strings"

	"github.com/pvjammer/ai-sdk-go/pkg/cli"
	"github.com/pvjammer/ai-sdk-go/pkg/llm"
	"github.com/pvjammer/ai-sdk-go/pkg/registry"

	// Side-effect imports register each builtin into the global registry.
	_ "github.com/pvjammer/ai-sdk-go/pkg/builtin/calculator"
	_ "github.com/pvjammer/ai-sdk-go/pkg/builtin/extract"
	_ "github.com/pvjammer/ai-sdk-go/pkg/builtin/fileio"
	_ "github.com/pvjammer/ai-sdk-go/pkg/builtin/summarize"
	_ "github.com/pvjammer/ai-sdk-go/pkg/builtin/texttransform"
	_ "github.com/pvjammer/ai-sdk-go/pkg/builtin/websearch"
)

// ShellConfig is the subset of shell configuration needed to initialise functions.
type ShellConfig struct {
	LLMEndpoint string
	LLMModel    string
}

// mapConfig implements function.Config backed by a plain map.
type mapConfig map[string]interface{}

func (c mapConfig) Get(key string) interface{}         { return c[key] }
func (c mapConfig) IsSet(key string) bool              { v, ok := c[key]; return ok && v != nil }
func (c mapConfig) Set(key string, value interface{})  { c[key] = value }
func (c mapConfig) GetStringSlice(key string) []string { return nil }

func (c mapConfig) GetString(key string) string {
	if v, ok := c[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}

func (c mapConfig) GetInt(key string) int {
	if v, ok := c[key]; ok {
		switch i := v.(type) {
		case int:
			return i
		case float64:
			return int(i)
		}
	}
	return 0
}

func (c mapConfig) GetBool(key string) bool {
	if v, ok := c[key]; ok {
		if b, ok := v.(bool); ok {
			return b
		}
	}
	return false
}

// Loader provides access to registered AIFunctions as both agent tools and /commands.
type Loader struct {
	cfg   mapConfig
	names []string
}

// New creates a Loader wired to the given shell LLM config.
func New(sc ShellConfig) *Loader {
	return &Loader{
		cfg: mapConfig{
			"llm.provider":        "ollama",
			"llm.ollama.endpoint": sc.LLMEndpoint,
			"llm.ollama.model":    sc.LLMModel,
		},
		names: registry.Global().List(),
	}
}

// Names returns the sorted list of available function names.
func (l *Loader) Names() []string {
	return l.names
}

// ToolDefs returns an llm.ToolDef for every registered function so the agent
// can call them natively via tool calling.
func (l *Loader) ToolDefs() []llm.ToolDef {
	defs := make([]llm.ToolDef, 0, len(l.names))
	for _, name := range l.names {
		fn := registry.Global().Get(name, l.cfg)
		if fn == nil {
			continue
		}
		meta := fn.Metadata()
		defs = append(defs, llm.ToolDef{
			Name:        meta.GetName("tool"),
			Description: meta.GetDescription("tool"),
			Parameters:  fn.InputSchema(),
		})
	}
	return defs
}

// ToolHandlers returns a name→handler map for agent tool dispatch.
// Keys match the ToolDef names returned by ToolDefs().
func (l *Loader) ToolHandlers() map[string]func(map[string]interface{}) (string, error) {
	handlers := make(map[string]func(map[string]interface{}) (string, error), len(l.names))
	for _, name := range l.names {
		name := name // capture loop var
		fn := registry.Global().Get(name, l.cfg)
		if fn == nil {
			continue
		}
		toolName := fn.Metadata().GetName("tool")
		handlers[toolName] = func(args map[string]interface{}) (string, error) {
			// Create a fresh instance per call.
			f := registry.Global().Get(name, l.cfg)
			result, err := f.Execute(context.Background(), args)
			if err != nil {
				return "", err
			}
			return fmt.Sprintf("%v", result), nil
		}
	}
	return handlers
}

// Execute runs a function by its CLI name via the cobra CLI builder.
// Use this for interactive /command invocations where cobra flag parsing is needed.
// For pipeline use (cat file | /fn), use ExecuteWithStdin instead.
func (l *Loader) Execute(ctx context.Context, name string, args []string, _ string) (string, error) {
	fn := registry.Global().Get(name, l.cfg)
	if fn == nil {
		return "", fmt.Errorf("unknown function %q (try /tools)", name)
	}

	cmd := cli.AsCommand(fn)

	var outBuf, errBuf bytes.Buffer
	cmd.SetOut(&outBuf)
	cmd.SetErr(&errBuf)
	cmd.SilenceUsage = true
	cmd.SilenceErrors = true
	cmd.SetArgs(args)

	if err := cmd.ExecuteContext(ctx); err != nil {
		if errBuf.Len() > 0 {
			return "", fmt.Errorf("%s", strings.TrimSpace(errBuf.String()))
		}
		return "", err
	}

	return strings.TrimSpace(outBuf.String()), nil
}

// ExecuteWithStdin runs a function with piped content injected directly into
// its primary text/content/input field, bypassing cobra to avoid stdin/TTY
// complexity. Extra args are parsed as --key value pairs.
// This is the correct path for pipeline execution: cat file | /fn [--flags]
func (l *Loader) ExecuteWithStdin(ctx context.Context, name string, stdin string, args []string) (string, error) {
	debug := os.Getenv("BAISH_DEBUG") != ""

	fn := registry.Global().Get(name, l.cfg)
	if fn == nil {
		return "", fmt.Errorf("unknown function %q (try /tools)", name)
	}

	schema := fn.InputSchema()
	inputMap, changed := parseArgFlags(args)

	if debug {
		fmt.Fprintf(os.Stderr, "[debug] ExecuteWithStdin: fn=%q stdin_len=%d args=%v\n", name, len(stdin), args)
		if schema != nil {
			keys := make([]string, 0, len(schema.Properties))
			for k := range schema.Properties {
				keys = append(keys, k)
			}
			fmt.Fprintf(os.Stderr, "[debug] schema properties: %v\n", keys)
		} else {
			fmt.Fprintf(os.Stderr, "[debug] schema is nil\n")
		}
	}

	// Inject stdin into the first text-like field that hasn't been explicitly set.
	injected := ""
	if strings.TrimSpace(stdin) != "" && schema != nil {
		for _, key := range []string{"text", "content", "input", "query"} {
			if _, ok := schema.Properties[key]; ok {
				if !changed[key] {
					inputMap[key] = strings.TrimSpace(stdin)
					changed[key] = true
					injected = key
				}
				break
			}
		}
	}

	if debug {
		fmt.Fprintf(os.Stderr, "[debug] injected stdin into field %q\n", injected)
		fmt.Fprintf(os.Stderr, "[debug] inputMap keys: ")
		for k := range inputMap {
			fmt.Fprintf(os.Stderr, "%q ", k)
		}
		fmt.Fprintln(os.Stderr)
	}

	inputMap["_changed_flags"] = changed

	result, err := fn.Execute(ctx, inputMap)
	if debug {
		fmt.Fprintf(os.Stderr, "[debug] Execute returned: result=%T err=%v\n", result, err)
	}
	if err != nil {
		return "", err
	}

	out := extractOutput(result)
	if debug {
		fmt.Fprintf(os.Stderr, "[debug] extractOutput returned %d chars\n", len(out))
	}
	return out, nil
}

// parseArgFlags converts ["--key", "value", "--flag2", "v2"] into a map
// and a set of explicitly-set keys. Dashes in key names are replaced with
// underscores to match JSON schema field names.
func parseArgFlags(args []string) (map[string]interface{}, map[string]bool) {
	m := make(map[string]interface{})
	changed := make(map[string]bool)
	for i := 0; i < len(args); i++ {
		if strings.HasPrefix(args[i], "--") && i+1 < len(args) {
			key := strings.ReplaceAll(strings.TrimPrefix(args[i], "--"), "-", "_")
			m[key] = args[i+1]
			changed[key] = true
			i++
		}
	}
	return m, changed
}

// extractOutput formats a function result for terminal display.
// It prefers a single meaningful string field over a full JSON dump.
func extractOutput(result interface{}) string {
	if result == nil {
		return ""
	}
	if s, ok := result.(string); ok {
		return s
	}

	// Reflect over struct to find a primary output string field.
	v := reflect.ValueOf(result)
	if v.Kind() == reflect.Ptr {
		v = v.Elem()
	}
	if v.Kind() == reflect.Struct {
		for _, name := range []string{"Summary", "Output", "Result", "Text", "Content", "Response", "Value"} {
			if f := v.FieldByName(name); f.IsValid() && f.Kind() == reflect.String {
				if s := f.String(); s != "" {
					return s
				}
			}
		}
	}

	// Fall back to JSON.
	b, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return fmt.Sprintf("%v", result)
	}
	return string(b)
}

// Describe returns a one-line description of a function for /tools output.
func (l *Loader) Describe(name string) string {
	fn := registry.Global().Get(name, l.cfg)
	if fn == nil {
		return ""
	}
	return fn.Metadata().GetDescription("cli")
}
