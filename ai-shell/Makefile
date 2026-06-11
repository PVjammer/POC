BIN      := baish
BIN_DIR  ?= $(HOME)/.local/bin
GO       := go

.PHONY: all build install clean test

all: install

build:
	$(GO) build -o $(BIN) ./cmd/shell/
	chmod +x $(BIN)

install: build
	mkdir -p $(BIN_DIR)
	cp $(BIN) $(BIN_DIR)/$(BIN)

clean:
	rm -f $(BIN) $(BIN_DIR)/$(BIN)

test:
	$(GO) test ./...
