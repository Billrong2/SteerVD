import cpp

from Parameter p, Function f
where f = p.getFunction()
select "PARAM", p.getName(), p.getLocation().getStartLine(), f.getQualifiedName(), p.getIndex()
union
from LocalVariable v, Function f
where f = v.getEnclosingFunction()
select "LOCAL", v.getName(), v.getLocation().getStartLine(), f.getQualifiedName(), -1
