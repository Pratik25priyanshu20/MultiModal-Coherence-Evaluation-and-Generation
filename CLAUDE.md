# Claude Session Protocol

## Start of Session
At the **beginning of every conversation**, read all files in the `memory/` directory:
- `memory/decisions.md` — Architecture and design decisions made throughout the project
- `memory/people.md` — People involved, their roles, and relevant context
- `memory/preferences.md` — User's workflow, coding, and communication preferences
- `memory/user.md` — User profile, background, goals, and ongoing context

Use this context to inform your responses, maintain continuity across sessions, and avoid re-asking questions that have already been answered.

## End of Session
At the **end of every conversation** (or when the user is wrapping up), update the memory files with any new information learned during the session:
- New decisions or changed decisions → `memory/decisions.md`
- New people mentioned or role changes → `memory/people.md`
- New preferences discovered → `memory/preferences.md`
- Updated user context or goals → `memory/user.md`

Only update files when there is genuinely new or changed information. Do not rewrite unchanged content.

## Rules
- Never delete existing memory entries unless explicitly told to or the information is confirmed outdated
- Keep entries concise — bullet points preferred over paragraphs
- Date-stamp significant decisions and changes
- If conflicting information arises, flag it and ask the user before overwriting
- Memory files are the source of truth for cross-session continuity
