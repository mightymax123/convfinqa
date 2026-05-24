# Initialise the project for your local user.
# Run this once after cloning: copies sample.env to .env, sets your host UID/GID,
# and creates required directories.
# After running, open .env and set OPENROUTER_API_KEY to your actual key.
init:
	@if [ -f .env ]; then echo ".env already exists - refusing to overwrite"; exit 1; fi
	@echo "# Host user identity — set automatically by \`make init\`, do not edit manually." > .env
	@echo "UID=$$(id -u)" >> .env
	@echo "GID=$$(id -g)" >> .env
	@echo "" >> .env
	@awk '/^#.*Host user identity/{next} /^(UID|GID)=/{next} {print}' sample.env >> .env
	@mkdir -p data outputs logs
	@echo "Wrote .env with UID=$$(id -u) GID=$$(id -g)"
	@echo "ACTION REQUIRED: open .env and set OPENROUTER_API_KEY to your OpenRouter API key"
