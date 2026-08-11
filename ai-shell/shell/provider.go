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
