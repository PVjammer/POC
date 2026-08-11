package llmprovider

import (
	"fmt"

	"github.com/pvjammer/ai-sdk-go/pkg/llm"
)

// New builds an llm.ToolCallingProvider for the given provider kind (baish's
// agent loop requires native tool calling). kind is normalized here:
//
//   - "", "ollama"                     → llm.NewOllamaProvider (native tool calling)
//   - "openai", "llamacpp", "llama.cpp" → NewOpenAIProvider (OpenAI-compatible
//     /chat/completions — llama.cpp llama-server, vLLM, LM Studio, real OpenAI)
//
// endpoint and model are passed straight through; apiKey is only used by the
// openai-compatible path and may be "" (most local servers don't require one).
func New(kind, endpoint, model, apiKey string) (llm.ToolCallingProvider, error) {
	switch normalizeKind(kind) {
	case "", "ollama":
		return llm.NewOllamaProvider(endpoint, model)
	case "openai":
		var opts []Option
		if apiKey != "" {
			opts = append(opts, WithAPIKey(apiKey))
		}
		return NewOpenAIProvider(endpoint, model, opts...)
	default:
		return nil, fmt.Errorf("unknown provider %q (expected \"ollama\" or \"openai\")", kind)
	}
}

func normalizeKind(kind string) string {
	switch kind {
	case "llamacpp", "llama.cpp", "openai-compatible", "openai-compat":
		return "openai"
	default:
		return kind
	}
}
