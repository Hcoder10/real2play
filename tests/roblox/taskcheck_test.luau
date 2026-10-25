local Check=require("../../roblox/src/ReplicatedStorage/Twin/TaskCheck")
local vectors=require("./taskcheck_vectors")
local assertions=0
local function equal(actual,expected,path)
    if type(expected)=="table" then
        assert(type(actual)=="table",path..": table missing")
        for k,v in pairs(expected) do equal(actual[k],v,path.."."..tostring(k)) end
        if #expected>0 then assert(#actual==#expected,path..": array length mismatch") end
    else
        assertions+=1; assert(actual==expected,path..": "..tostring(actual).." != "..tostring(expected))
    end
end
for _,case in ipairs(vectors.cases) do equal(Check.check(case.task,case.state),case.expected,case.name) end
for _,case in ipairs(vectors.window_cases) do equal(Check.check_window(case.task,case.states,case.hz,case.hold_s),case.expected,case.name) end
print("task-check: "..#vectors.cases.." cases + "..#vectors.window_cases.." windows; "..assertions.." exact assertions passed")
