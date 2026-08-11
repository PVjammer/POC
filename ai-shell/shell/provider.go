package shell

import (
	"github.com/pvjammer/ai-shell-poc/llmprovider"
	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

// newProvider builds an llm.Provider for (endpoint, model) using the shell's
// configured provider kind ("ollama" by default, "openai" for llama.cpp/vLLM/
// OpenAI-compatible servers — see llmprovider.New).
func (s *Shell) newProvider(endpoint, model string) (llm.ToolCallingProvider, error) {
	return s.newProviderKind("", endpoint, model)
}

// newProviderKind is like newProvider but lets the caller override the
// provider kind (e.g. a named agent configured with provider = "openai").
// An empty kind falls back to the shell's global s.cfg.Provider.
func (s *Shell) newProviderKind(kind, endpoint, model string) (llm.ToolCallingProvider, error) {
	if kind == "" {
		kind = s.cfg.Provider
	}
	return llmprovider.New(kind, endpoint, model, s.cfg.APIKey)
}

// providerOrDefault returns kind, or "ollama" if kind is empty — for display
// purposes only (llmprovider.New already treats "" as "ollama").
func providerOrDefault(kind string) string {
	if kind == "" {
		return "ollama"
	}
	return kind
}

// setModelByAliasOrLiteral implements "/model <name>" with a single argument:
// if name matches a named preset in [models.<name>] (config.toml), its
// model/endpoint/provider are applied, with empty fields falling back to
// startupCfg. This is deliberately independent of the agent-config system —
// it's a pure connection swap, unlike "/agent <name>" which also switches
// tools/skills/system_prompt. A preset's APIKey is not yet applied (baish
// doesn't thread a per-connection API key through /model switches yet).
//
//	[models."qwen3.6"]
//	model    = "unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_XL"
//	endpoint = "http://192.168.1.88:30000/v1"
//	provider = "openai"
//
// Only fires for names that match a known preset; anything else falls back
// to today's behavior of treating name as a literal model string against
// the currently active endpoint/provider.
func (s *Shell) setModelByAliasOrLiteral(name string) error {
	if preset, ok := s.appCfg.Models[name]; ok {
		model := firstNonEmpty(preset.Model, s.startupCfg.Model)
		endpoint := firstNonEmpty(preset.Endpoint, s.startupCfg.Endpoint)
		providerKind := firstNonEmpty(preset.Provider, s.startupCfg.Provider)
		return s.setModel(model, endpoint, providerKind)
	}
	return s.setModel(name, s.cfg.Endpoint, s.cfg.Provider)
}
