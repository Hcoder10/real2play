local Tasks=require(if script then script.Parent.Generated.Tasks else "./Generated/Tasks")
local Layout=Tasks.tasks.place_on_cell.board
local T={tolerance=.018,pieceSize=Tasks.tasks.place_on_cell.objects[1].size_m}
T.lines={{1,2,3},{4,5,6},{7,8,9},{1,4,7},{2,5,8},{3,6,9},{1,5,9},{3,5,7}}
function T.new() return table.create(9,0) end
function T.result(board)
    for _,line in ipairs(T.lines) do local a,b,c=unpack(line); if board[a]~=0 and board[a]==board[b] and board[b]==board[c] then return board[a] end end
    for i=1,9 do if board[i]==0 then return nil end end
    return 0
end
function T.legal(board)
    local out={}; if T.result(board)~=nil then return out end
    for i=1,9 do if board[i]==0 then table.insert(out,i) end end
    return out
end
function T.place(board,cell,player)
    if type(cell)~="number" or cell%1~=0 or cell<1 or cell>9 or (player~=1 and player~=2) or board[cell]~=0 or T.result(board)~=nil then return false end
    local one,two=0,0; for _,p in ipairs(board) do if p==1 then one+=1 elseif p==2 then two+=1 end end
    if (player==1 and one~=two) or (player==2 and one~=two+1) then return false end
    board[cell]=player; return true
end
local memo={}
local function search(board,turn)
    local result=T.result(board)
    if result~=nil then return if result==2 then 1 elseif result==1 then -1 else 0 end
    local key=table.concat(board)..turn; if memo[key]~=nil then return memo[key] end
    local best=if turn==2 then -2 else 2
    for _,cell in ipairs(T.legal(board)) do
        board[cell]=turn; local score=search(board,3-turn); board[cell]=0
        best=if turn==2 then math.max(best,score) else math.min(best,score)
    end
    memo[key]=best; return best
end
function T.choose(board,slip,random)
    random=random or math.random
    local legal=T.legal(board); if #legal==0 then return nil end
    -- An immediate winning move always outranks the slip branch.
    for _,cell in ipairs(legal) do board[cell]=2; local win=T.result(board)==2; board[cell]=0; if win then return cell end end
    if random()<(slip or .25) then return legal[math.min(#legal,1+math.floor(random()*#legal))] end
    local best,score=nil,-2
    for _,cell in ipairs(legal) do board[cell]=2; local s=search(board,1); board[cell]=0; if s>score then best,score=cell,s end end
    return best
end
function T.xy(cell)
    local row,col=math.floor((cell-1)/3),(cell-1)%3
    local p=assert(Layout.cells["r"..row.."c"..col],"unknown board cell")
    return {p[1],p[2]}
end
function T.cell(xyz,board)
    if T.result(board)~=nil then return nil end
    local nearest=nil; local distance=T.tolerance^2
    for i=1,9 do
        local p=T.xy(i); local d=(xyz[1]-p[1])^2+(xyz[2]-p[2])^2
        if d<=distance and (nearest==nil or d<distance) then nearest=i; distance=d end
    end
    return if nearest and board[nearest]==0 then nearest else nil
end
return T
