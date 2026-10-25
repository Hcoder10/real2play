# Publish Real2Play to the existing Roblox place

Build: `out/roblox/Real2Play.rbxlx`

Destination: **86276076528667** — [SquarianSquares Place 10252026-1](https://www.roblox.com/games/86276076528667/SquarianSquares-Place-10252026-1)

Agent Y prepared the place and server-only trace configuration. No Open Cloud
publish was attempted: `ROBLOX_API_KEY`, `ROBLOX_UNIVERSE_ID`, and
`ROBLOX_PLACE_ID` were absent from the environment. No new published version
is claimed.

1. Open **Roblox Studio** under the account that owns the existing place and its
   robot mesh assets. Choose **File > Open from File** and open
   `C:\Users\sarta\real2play\out\roblox\Real2Play.rbxlx`.
2. In Explorer, find **ServerStorage > LocalConfig**. The local build already
   includes this server-only ModuleScript. If preparing another build, create
   it with the following fields:

   ```lua
   return {
       TRACE_URL = "https://fairfield-explained-steve-constraint.trycloudflare.com",
       TRACE_KEY = "<contents of out/demo/trace_key.txt>",
   }
   ```

   Read the key privately from `out/demo/trace_key.txt`; do not put it in chat,
   screenshots, ReplicatedStorage, or a LocalScript. `TRACE_URL` is the base URL,
   with no `/trace` suffix. The source `roblox/src/ServerStorage/LocalConfig.lua`
   is ignored by Git. The place contains this private server configuration.
3. Choose **File > Publish to Roblox** (or **Publish to Roblox As…** when Studio
   asks for a destination), select the **existing** experience and place
   **86276076528667**, then publish/overwrite that place.
4. Open **Game Settings > Security**, enable **Allow HTTP Requests**, and save.
   Leave Studio API access disabled unless intentionally testing persistence.
5. In the experience settings, set maximum players to **30** or more to allow
   all 20 tables plus spectators. Enable the intended Computer, Phone, Tablet,
   and Console devices. Verify the robot mesh assets are permitted for this
   publishing owner. Save the settings and publish any pending place changes.
6. Join a private published server with two clients. Each player sits at a
   different blue chair, then selects **Place a cap — Start here**. Confirm the
   highlighted cell, white cap, independent controls, score/streak, and Scores
   leaderboard. Use **Watch table** to spectate; **Leave view** restores walking.
   Standing or disconnecting must free the seat.
7. Complete one round. The UI should report the upload and increment the
   sent-to-robot count. Confirm a new canonical trace on the receiver. The
   tunnel must remain running; its URL may change after restart. Update only
   `LocalConfig.TRACE_URL` if that happens, then republish.

To rebuild from the repository root:

```powershell
.venv/Scripts/python.exe roblox/tools/gen_presets.py
.venv/Scripts/python.exe roblox/tools/test.py
.venv/Scripts/python.exe -m pytest tests/roblox -q
.venv/Scripts/python.exe roblox/tools/build.py
```

`build.py` checks generated presets, syncs the open Studio window, runs actual
Studio parity, and writes `out/roblox/Real2Play.rbxlx` only after the gate passes.
Direct Rojo equivalent after validation:

```powershell
rojo build roblox/default.project.json -o out/roblox/Real2Play.rbxlx
```

Evidence and limitations: `roblox/AGENT_Y_ACCEPTANCE.md`. No hardware or Gemini
requests were used for these checks.
