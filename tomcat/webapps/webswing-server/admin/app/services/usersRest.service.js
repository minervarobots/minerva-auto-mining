(function(e){e([],function(){function t(e,t,n,r){function i(){function r(e){return e.data}function i(e){return n.handleRestError("load user configuration",e,!0)}return t.get(e+"/rest/admin/users").then(r,i)}function s(i){function s(){r.success("User properties saved.")}function o(e){return n.handleRestError("save user configuration",e,!0)}return t.post(e+"/rest/admin/users",i).then(s,o)}return{getUsers:i,setUsers:s}}return t.$inject=["baseUrl","$http","errorHandler","messageService"],t})})(adminConsole.define);