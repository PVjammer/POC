BIN      := dockerfile-builder
BIN_DIR  ?= $(HOME)/.local/bin

.PHONY: build install clean

build:
	go build -o $(BIN) ./cmd/builder/
	chmod +x $(BIN)

install: build
	cp $(BIN) $(BIN_DIR)/$(BIN)

clean:
	rm -f $(BIN)
