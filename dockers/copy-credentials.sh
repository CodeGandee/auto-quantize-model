#!/bin/bash
#
# Copy credentials and configuration to a Docker container
# Usage: ./copy-credentials.sh <container_name_or_id> <target_user>
#
# This script copies:
# - .codex directory
# - SSH keys with proper permissions
# - Git configuration
# - Claude and Codex aliases
# - Related environment variables (from host .bashrc)

set -e

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <container_name_or_id> <target_user>"
    echo "Example: $0 dev-5090-trtllm-stage-2-1 me"
    exit 1
fi

CONTAINER="$1"
TARGET_USER="$2"

echo "==> Copying credentials to container: $CONTAINER (user: $TARGET_USER)"

# 1. Copy .codex directory
echo "[1/6] Copying .codex directory..."
if [ -d ~/.codex ]; then
    docker cp ~/.codex "$CONTAINER:/home/$TARGET_USER/"
    echo "  ✓ .codex copied"
else
    echo "  ⚠ ~/.codex not found, skipping"
fi

# 2. Copy SSH keys
echo "[2/6] Copying SSH keys..."
if [ -d ~/.ssh ]; then
    docker cp ~/.ssh "$CONTAINER:/home/$TARGET_USER/"
    
    # Get target user's group
    TARGET_GROUP=$(docker exec "$CONTAINER" id -gn "$TARGET_USER")
    
    # Set proper ownership and permissions
    docker exec "$CONTAINER" chown -R "$TARGET_USER:$TARGET_GROUP" "/home/$TARGET_USER/.ssh"
    docker exec "$CONTAINER" chmod 700 "/home/$TARGET_USER/.ssh"
    docker exec "$CONTAINER" find "/home/$TARGET_USER/.ssh" -type f -name 'id_*' ! -name '*.pub' -exec chmod 600 {} \;
    docker exec "$CONTAINER" find "/home/$TARGET_USER/.ssh" -type f -name '*.pub' -exec chmod 644 {} \;
    
    echo "  ✓ SSH keys copied with proper permissions"
else
    echo "  ⚠ ~/.ssh not found, skipping"
fi

# 3. Configure Git
echo "[3/6] Configuring Git..."
GIT_NAME=$(git config --global user.name 2>/dev/null || echo "")
GIT_EMAIL=$(git config --global user.email 2>/dev/null || echo "")

if [ -n "$GIT_NAME" ] && [ -n "$GIT_EMAIL" ]; then
    docker exec -u "$TARGET_USER" "$CONTAINER" git config --global user.name "$GIT_NAME"
    docker exec -u "$TARGET_USER" "$CONTAINER" git config --global user.email "$GIT_EMAIL"
    echo "  ✓ Git configured (name: $GIT_NAME, email: $GIT_EMAIL)"
else
    echo "  ⚠ Git user not configured on host, skipping"
fi

# 4. Extract and copy Claude/Codex aliases
echo "[4/6] Copying Claude/Codex aliases..."
ALIASES_FILE="/tmp/claude_codex_aliases_$$.txt"
cat > "$ALIASES_FILE" << 'EOF'

# Codex and Claude aliases
alias codex-skip-all='codex --dangerously-bypass-approvals-and-sandbox'

alias claude-yunwu='\
    ANTHROPIC_BASE_URL="${YUNWU_API_URL}" \
    ANTHROPIC_API_KEY="${YUNWU_API_KEY}" \
    claude --dangerously-skip-permissions'

alias claude-yunwu-adv='\
    ANTHROPIC_BASE_URL="${YUNWU_API_URL}" \
    ANTHROPIC_API_KEY="${YUNWU_API_KEY_ADV}" \
    claude --dangerously-skip-permissions'

alias claude-taobao='\
    ANTHROPIC_API_KEY="${TAOBAO_CC_API_KEY}" \
    ANTHROPIC_BASE_URL="${TAOBAO_CC_BASE_URL}" \
    claude --dangerously-skip-permissions'

alias claude-qwen='\
    ANTHROPIC_BASE_URL="${SILICONFLOW_API_URL}" \
    ANTHROPIC_API_KEY="${SILICONFLOW_API_KEY}" \
    ANTHROPIC_MODEL="${SILICONFLOW_MODEL}" \
    claude --dangerously-skip-permissions'
EOF

docker cp "$ALIASES_FILE" "$CONTAINER:/tmp/"
docker exec -u "$TARGET_USER" "$CONTAINER" bash -c "cat /tmp/$(basename $ALIASES_FILE) >> /home/$TARGET_USER/.bashrc"
rm -f "$ALIASES_FILE"
echo "  ✓ Aliases copied"

# 5. Extract and copy environment variables (without exposing values)
echo "[5/6] Copying environment variables..."
if [ -f ~/.bashrc ]; then
    ENVVARS_FILE="/tmp/claude_codex_envvars_$$.txt"
    
    # Extract environment variables from host .bashrc
    grep -E "^export (TAVILY|YUNWU|TAOBAO|SILICONFLOW)_" ~/.bashrc > "$ENVVARS_FILE" 2>/dev/null || true
    
    if [ -s "$ENVVARS_FILE" ]; then
        # Add header
        sed -i '1i\n# Environment variables for Codex and Claude' "$ENVVARS_FILE"
        
        docker cp "$ENVVARS_FILE" "$CONTAINER:/tmp/"
        docker exec -u "$TARGET_USER" "$CONTAINER" bash -c "cat /tmp/$(basename $ENVVARS_FILE) >> /home/$TARGET_USER/.bashrc"
        rm -f "$ENVVARS_FILE"
        echo "  ✓ Environment variables copied"
    else
        echo "  ⚠ No relevant environment variables found in ~/.bashrc"
        rm -f "$ENVVARS_FILE"
    fi
else
    echo "  ⚠ ~/.bashrc not found, skipping environment variables"
fi

# 6. Verify
echo "[6/6] Verifying setup..."
docker exec -u "$TARGET_USER" "$CONTAINER" bash -c "
    echo '  SSH dir: \$([ -d ~/.ssh ] && echo '✓' || echo '✗')'
    echo '  .codex dir: \$([ -d ~/.codex ] && echo '✓' || echo '✗')'
    echo '  Git name: \$(git config --global user.name || echo 'not set')'
    echo '  Git email: \$(git config --global user.email || echo 'not set')'
"

echo ""
echo "==> All credentials copied successfully!"
echo "Note: The container user may need to restart their shell or run 'source ~/.bashrc' for changes to take effect."
