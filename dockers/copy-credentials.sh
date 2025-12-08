#!/bin/bash
#
# Copy credentials and configuration to a Docker container
# Usage: ./copy-credentials.sh [--dry-run] <container_name_or_id> <target_user>
#
# This script copies:
# - .codex directory
# - SSH keys with proper permissions
# - Git configuration
# - Claude and Codex aliases
# - Related environment variables (from host .bashrc)

set -e

# Parse options
DRY_RUN=false
if [ "$1" = "--dry-run" ]; then
    DRY_RUN=true
    shift
fi

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 [--dry-run] <container_name_or_id> <target_user>"
    echo "Example: $0 dev-5090-trtllm-stage-2-1 me"
    echo "Example: $0 --dry-run dev-5090-trtllm-stage-2-1 me"
    exit 1
fi

CONTAINER="$1"
TARGET_USER="$2"

if [ "$DRY_RUN" = true ]; then
    echo "==> DRY RUN: Showing what would be copied to container: $CONTAINER (user: $TARGET_USER)"
else
    echo "==> Copying credentials to container: $CONTAINER (user: $TARGET_USER)"
fi

# 1. Copy .codex directory
echo "[1/6] Copying .codex directory..."
if [ -d ~/.codex ]; then
    echo "  FROM: ~/.codex"
    echo "  TO:   $CONTAINER:/home/$TARGET_USER/.codex"
    if [ "$DRY_RUN" = false ]; then
        docker cp ~/.codex "$CONTAINER:/home/$TARGET_USER/"
        echo "  ✓ .codex copied"
    fi
else
    echo "  ⚠ ~/.codex not found, skipping"
fi

# 2. Copy SSH keys
echo "[2/6] Copying SSH keys..."
if [ -d ~/.ssh ]; then
    echo "  FROM: ~/.ssh"
    echo "  TO:   $CONTAINER:/home/$TARGET_USER/.ssh"
    echo "  Permissions: 700 (dir), 600 (private keys), 644 (public keys)"
    if [ "$DRY_RUN" = false ]; then
        docker cp ~/.ssh "$CONTAINER:/home/$TARGET_USER/"
        
        # Get target user's group
        TARGET_GROUP=$(docker exec "$CONTAINER" id -gn "$TARGET_USER")
        
        # Set proper ownership and permissions
        docker exec "$CONTAINER" chown -R "$TARGET_USER:$TARGET_GROUP" "/home/$TARGET_USER/.ssh"
        docker exec "$CONTAINER" chmod 700 "/home/$TARGET_USER/.ssh"
        docker exec "$CONTAINER" find "/home/$TARGET_USER/.ssh" -type f -name 'id_*' ! -name '*.pub' -exec chmod 600 {} \;
        docker exec "$CONTAINER" find "/home/$TARGET_USER/.ssh" -type f -name '*.pub' -exec chmod 644 {} \;
        
        echo "  ✓ SSH keys copied with proper permissions"
    fi
else
    echo "  ⚠ ~/.ssh not found, skipping"
fi

# 3. Configure Git
echo "[3/6] Configuring Git..."
GIT_NAME=$(git config --global user.name 2>/dev/null || echo "")
GIT_EMAIL=$(git config --global user.email 2>/dev/null || echo "")

if [ -n "$GIT_NAME" ] && [ -n "$GIT_EMAIL" ]; then
    echo "  Git user.name:  $GIT_NAME"
    echo "  Git user.email: $GIT_EMAIL"
    if [ "$DRY_RUN" = false ]; then
        docker exec -u "$TARGET_USER" "$CONTAINER" git config --global user.name "$GIT_NAME"
        docker exec -u "$TARGET_USER" "$CONTAINER" git config --global user.email "$GIT_EMAIL"
        echo "  ✓ Git configured"
    fi
else
    echo "  ⚠ Git user not configured on host, skipping"
fi

# 4. Extract and copy Claude/Codex aliases
echo "[4/6] Copying Claude/Codex aliases..."
if [ -f ~/.bashrc ]; then
    ALIASES_FILE="/tmp/claude_codex_aliases_$$.txt"
    
    # Extract aliases matching claude-* and codex-* patterns
    grep -E "^alias (claude-|codex-)" ~/.bashrc > "$ALIASES_FILE" 2>/dev/null || true
    
    # Also capture multi-line aliases (lines that end with backslash continuation)
    awk '/^alias (claude-|codex-)/ {
        print
        while (getline > 0) {
            print
            if (!/\\$/) break
        }
    }' ~/.bashrc > "$ALIASES_FILE" 2>/dev/null || true
    
    if [ -s "$ALIASES_FILE" ]; then
        # Count aliases
        ALIAS_COUNT=$(grep -c "^alias" "$ALIASES_FILE" 2>/dev/null || echo "0")
        echo "  Found $ALIAS_COUNT aliases in ~/.bashrc:"
        grep "^alias" "$ALIASES_FILE" | sed 's/=.*//' | sed 's/^/    /'
        echo "  TO: $CONTAINER:/home/$TARGET_USER/.bashrc (appended)"
        
        if [ "$DRY_RUN" = false ]; then
            # Add header
            sed -i '1i\n# Codex and Claude aliases' "$ALIASES_FILE"
            
            docker cp "$ALIASES_FILE" "$CONTAINER:/tmp/"
            docker exec -u "$TARGET_USER" "$CONTAINER" bash -c "cat /tmp/$(basename $ALIASES_FILE) >> /home/$TARGET_USER/.bashrc"
            echo "  ✓ Aliases copied"
        fi
        rm -f "$ALIASES_FILE"
    else
        echo "  ⚠ No claude-* or codex-* aliases found in ~/.bashrc"
        rm -f "$ALIASES_FILE"
    fi
else
    echo "  ⚠ ~/.bashrc not found, skipping aliases"
fi

# 5. Extract and copy environment variables (without exposing values)
echo "[5/6] Copying environment variables..."
if [ -f ~/.bashrc ]; then
    ENVVARS_FILE="/tmp/claude_codex_envvars_$$.txt"
    
    # Extract environment variables from host .bashrc
    grep -E "^export (TAVILY|YUNWU|TAOBAO|SILICONFLOW)_" ~/.bashrc > "$ENVVARS_FILE" 2>/dev/null || true
    
    if [ -s "$ENVVARS_FILE" ]; then
        ENVVAR_COUNT=$(wc -l < "$ENVVARS_FILE")
        echo "  Found $ENVVAR_COUNT environment variables in ~/.bashrc:"
        if [ "$DRY_RUN" = true ]; then
            sed 's/^/    /' "$ENVVARS_FILE"
        else
            sed 's/=.*/=***/' "$ENVVARS_FILE" | sed 's/^/    /'
        fi
        echo "  TO: $CONTAINER:/home/$TARGET_USER/.bashrc (appended)"
        
        if [ "$DRY_RUN" = false ]; then
            # Add header
            sed -i '1i\n# Environment variables for Codex and Claude' "$ENVVARS_FILE"
            
            docker cp "$ENVVARS_FILE" "$CONTAINER:/tmp/"
            docker exec -u "$TARGET_USER" "$CONTAINER" bash -c "cat /tmp/$(basename $ENVVARS_FILE) >> /home/$TARGET_USER/.bashrc"
            echo "  ✓ Environment variables copied"
        fi
        rm -f "$ENVVARS_FILE"
    else
        echo "  ⚠ No relevant environment variables found in ~/.bashrc"
        rm -f "$ENVVARS_FILE"
    fi
else
    echo "  ⚠ ~/.bashrc not found, skipping environment variables"
fi

# 6. Verify
echo "[6/6] Verifying setup..."
if [ "$DRY_RUN" = false ]; then
    docker exec -u "$TARGET_USER" "$CONTAINER" bash -c "
        echo '  SSH dir: \$([ -d ~/.ssh ] && echo '✓' || echo '✗')'
        echo '  .codex dir: \$([ -d ~/.codex ] && echo '✓' || echo '✗')'
        echo '  Git name: \$(git config --global user.name || echo 'not set')'
        echo '  Git email: \$(git config --global user.email || echo 'not set')'
    "
    
    echo ""
    echo "==> All credentials copied successfully!"
    echo "Note: The container user may need to restart their shell or run 'source ~/.bashrc' for changes to take effect."
else
    echo "  (Skipped in dry-run mode)"
    echo ""
    echo "==> Dry run complete. Use without --dry-run to actually copy the credentials."
fi
