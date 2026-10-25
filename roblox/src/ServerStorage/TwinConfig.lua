-- Server-only. Use an ignored LocalConfig ModuleScript for deployment secrets.
-- TRACE_URL is the base URL, without /trace. Never replicate TRACE_KEY.
local config={TRACE_URL="",TRACE_KEY="",OWNER_IDS={},UPLOAD_RETRIES=5,BOT_IDLE=true,BOT_INTERVAL=20,STUDIO_DATASTORES=false,REAL_ARM_IMAGE=""}
local localConfig=script.Parent:FindFirstChild("LocalConfig")
if localConfig then for k,v in pairs(require(localConfig)) do config[k]=v end end
return config
