package shell

import (
	"os/exec"
	"runtime"
)

// sendDesktopNotification fires a system notification if a supported tool is
// available. Silently no-ops on unsupported platforms or when the tool is absent.
func sendDesktopNotification(title, body string) {
	switch runtime.GOOS {
	case "linux":
		// notify-send works on most Linux desktops and WSL2 with a notification bridge.
		if path, err := exec.LookPath("notify-send"); err == nil {
			exec.Command(path, "-a", "baish", title, body).Start() //nolint:errcheck
		}
	case "darwin":
		script := `display notification "` + body + `" with title "` + title + `"`
		exec.Command("osascript", "-e", script).Start() //nolint:errcheck
	}
}
