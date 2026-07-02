#!/usr/bin/env bash
# release-sdk.sh <version>
#
# Tags ai-sdk-go at the given version, pushes the tag, then updates baish's
# go.mod to reference the published module (removing the local replace directive)
# and runs go mod tidy.
#
# Run from anywhere. Paths are resolved relative to this script's location.
#
# Usage:
#   ./scripts/release-sdk.sh v0.2.0
#
# After this script succeeds the replace directive is gone and baish depends on
# the published SDK. Re-add it manually for local dev:
#   echo 'replace github.com/pvjammer/ai-sdk-go => ../../projects/ai-sdk-go' >> go.mod

set -euo pipefail

VERSION="${1:-}"
if [[ -z "$VERSION" ]]; then
    echo "usage: $0 <version>  (e.g. v0.2.0)" >&2
    exit 1
fi
if [[ ! "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[a-zA-Z0-9.]+)?$ ]]; then
    echo "error: version must be semver with a leading 'v' (got: $VERSION)" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BAISH_DIR="$(dirname "$SCRIPT_DIR")"
SDK_DIR="$(cd "$BAISH_DIR/../../projects/ai-sdk-go" && pwd)"

echo "==> baish:   $BAISH_DIR"
echo "==> sdk:     $SDK_DIR"
echo "==> version: $VERSION"
echo ""

# ── 1. Sanity checks ──────────────────────────────────────────────────────────

echo "--- checking sdk working tree is clean"
if ! git -C "$SDK_DIR" diff --quiet || ! git -C "$SDK_DIR" diff --cached --quiet; then
    echo "error: ai-sdk-go has uncommitted changes — commit or stash first" >&2
    exit 1
fi

echo "--- checking sdk tag does not already exist"
if git -C "$SDK_DIR" tag | grep -qx "$VERSION"; then
    echo "error: tag $VERSION already exists in ai-sdk-go" >&2
    exit 1
fi

echo "--- checking baish go.mod has replace directive"
if ! grep -q "replace github.com/pvjammer/ai-sdk-go" "$BAISH_DIR/go.mod"; then
    echo "error: replace directive not found in go.mod — is this the dev setup?" >&2
    exit 1
fi

# ── 2. Tag and push ai-sdk-go ─────────────────────────────────────────────────

echo ""
echo "--- tagging ai-sdk-go $VERSION"
git -C "$SDK_DIR" tag "$VERSION"

echo "--- pushing tag to origin"
git -C "$SDK_DIR" push origin "$VERSION"

# ── 3. Update baish go.mod ────────────────────────────────────────────────────

echo ""
echo "--- updating go.mod: require $VERSION"
# Update the require line
sed -i "s|github.com/pvjammer/ai-sdk-go v[^ ]*|github.com/pvjammer/ai-sdk-go $VERSION|g" "$BAISH_DIR/go.mod"

echo "--- updating go.mod: removing replace directive"
# Remove the replace line (handles trailing newlines cleanly)
sed -i "/^replace github.com\/pvjammer\/ai-sdk-go/d" "$BAISH_DIR/go.mod"

echo "--- running go mod tidy"
(cd "$BAISH_DIR" && go mod tidy)

echo ""
echo "--- verifying build"
(cd "$BAISH_DIR" && go build ./...)

# ── 4. Done ───────────────────────────────────────────────────────────────────

echo ""
echo "==> sdk tagged and pushed: $VERSION"
echo "==> baish go.mod updated and verified"
echo ""
echo "Next steps:"
echo "  1. Review go.mod / go.sum changes"
echo "  2. Commit: git -C $BAISH_DIR commit -am 'chore: bump ai-sdk-go to $VERSION'"
echo "  3. Merge fix-tool-parsing → main, tag baish release"
echo "  4. To restore local dev: add 'replace github.com/pvjammer/ai-sdk-go => ../../projects/ai-sdk-go' to go.mod"
