# Initialise the project for your local user.
# Run this once after cloning: copies sample.env to .env, sets your host UID/GID,
# and creates required directories.
# After running, open .env and set OPENAI_API_KEY to your actual key.
init:
	@if [ -f .env ]; then echo ".env already exists - refusing to overwrite"; exit 1; fi
	@echo "UID=$$(id -u)" > .env
	@echo "GID=$$(id -g)" >> .env
	@echo "" >> .env
	@awk 'BEGIN{skip=1} /^(UID|GID)=/{next} skip && /^[[:space:]]*$$/{next} {skip=0; print}' sample.env >> .env
	@mkdir -p data outputs
	@echo "Wrote .env with UID=$$(id -u) GID=$$(id -g)"
	@echo "ACTION REQUIRED: open .env and set OPENAI_API_KEY to your OpenAI API key"
