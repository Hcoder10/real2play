local base=Instance.new("Part"); base.Anchored=true; base.Position=Vector3.new(0,40,0); base.Parent=workspace
local link=Instance.new("Part"); link.Position=base.Position; link.CanCollide=false; link.Parent=workspace
local a=Instance.new("Attachment"); a.Parent=base
local b=Instance.new("Attachment"); b.Parent=link
local h=Instance.new("HingeConstraint"); h.Attachment0=a; h.Attachment1=b; h.ActuatorType=Enum.ActuatorType.Motor; h.MotorMaxTorque=10000; h.AngularVelocity=1; h.Parent=base
local begin=h.CurrentAngle; task.wait(.3)
return {start=begin,angle=h.CurrentAngle,velocity=link.AssemblyAngularVelocity.X}
