// test_pipeline exercises ExecuteWithStdin directly, without the shell REPL.
// Run with: go run ./cmd/test_pipeline
package main

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/pvjammer/ai-shell-poc/functions"
)

func env(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func main() {
	loader := functions.New(functions.ShellConfig{
		LLMEndpoint: env("AI_SHELL_ENDPOINT", "http://localhost:11434"),
		LLMModel:    env("AI_SHELL_MODEL", "llama3.2"),
	})

	fmt.Println("=== Registered functions ===")
	for _, name := range loader.Names() {
		fmt.Printf("  %s: %s\n", name, loader.Describe(name))
	}
	fmt.Println()

	testContent := strings.Repeat("The Go programming language was created at Google by Robert Griesemer, Rob Pike, and Ken Thompson. It was announced in 2009 and version 1.0 was released in 2012. Go is a statically typed, compiled language with garbage collection, limited structural typing, memory safety, and CSP-style concurrent programming features. ", 3)

	fmt.Printf("=== Testing /summarize with %d chars ===\n", len(testContent))
	fmt.Println("Calling ExecuteWithStdin...")

	result, err := loader.ExecuteWithStdin(context.Background(), "summarize", testContent, []string{})
	if err != nil {
		fmt.Fprintf(os.Stderr, "ERROR: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Result (%d chars):\n%s\n", len(result), result)
}
