BIN      := baish
BIN_DIR  ?= $(HOME)/.local/bin
GO       := go

VERSION  := $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
PKG      := main
LDFLAGS  := -X '$(PKG).Version=$(VERSION)'

.PHONY: all build install clean test

build:
	$(GO) build -ldflags "$(LDFLAGS)" -o $(BIN) ./cmd/shell/
	chmod +x $(BIN)

install: build
	mkdir -p $(BIN_DIR)
	mv $(BIN) $(BIN_DIR)/$(BIN)

clean:
	rm -f $(BIN) $(BIN_DIR)/$(BIN)

test:
	$(GO) test ./...
